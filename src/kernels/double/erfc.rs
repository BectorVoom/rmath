//! The complementary error function.
//!
//! `erfc` is not `1 - erf`, and the reason is worth stating because it decides
//! the whole shape of this kernel: `erfc(6)` is about `2e-17`, so computing it
//! as `1 - erf(6)` subtracts two numbers that agree to sixteen digits and
//! returns noise, and by `x = 27` the true value is `2^-1074` and the
//! subtraction returns zero. So there are two algorithms, and the vector
//! kernel runs whichever the data needs:
//!
//! * **Near** — `x <= 0x1.713786d9c7c09p+1`, where `erfc` is still `O(1)`.
//!   Shares [`super::erf`]'s double-double fast path and one `fast_two_sum`
//!   against 1. The two sides of zero differ only in a sign, so they are one
//!   blended path rather than two.
//! * **Asymptotic** — above it, `erfc(x) = exp(-x^2) p(1/x)`, with `exp(-x^2)`
//!   evaluated to 74 bits over two 64-entry double-double tables and `p` a
//!   Chebyshev fit per interval of `1/x`.
//!
//! A vector whose lanes all want the same branch runs only that branch — the
//! common case, since `erfc` is usually applied to a buffer that lives on one
//! side or the other. A mixed vector runs both and blends, which is the price
//! of not breaking the vector up.
//!
//! Then, as in [`super::erf`], a rounding test decides whether the
//! double-double settles the last bit, and the handful of lanes where it does
//! not go to the scalar accurate path through
//! [`crate::simd::patch_lanes`].
//!
//! # `Fast`
//!
//! Drops the rounding test: `h + l` rounded once, measured below 0.51 ulp. No
//! lane ever leaves the vector.
//!
//! `Finite` means `0x1.c5bf891b4ef6ap-55 < |x|`,
//! `-0x1.7744f8f74e94bp+2 < x` and `x < 0x1.9db1bb14e15cap+4`. Below that band
//! `erfc` is 1 or 2; above it the answer is under `2^-970`, where the
//! asymptotic branch's own low word underflows and only the scalar reference
//! is still accurate.

use crate::kernels::double::dd::{a_mul, d_mul, fast_sum, fast_two_sum, s_mul};
use crate::policy::{Accuracy, Domain};
use crate::reference::double::erfc as reference;
use crate::reference::double::erfc_parts as r;
use crate::simd::{Lanes, Mask, Simd, patch_lanes};
use crate::tables::double::erfc as t;

/// `-0x1.7744f8f74e94bp+2`, at or below which `erfc` rounds to 2.
const NEG_LIMIT: f64 = f64::from_bits(r::NEG_LIMIT & 0x7fff_ffff_ffff_ffff);
/// `0x1.9db1bb14e15cap+4`: the asymptotic branch's own low word underflows
/// above this, so those lanes are sent to the accurate path.
const ASYMPT_MAX: f64 = f64::from_bits(0x4039db1bb14e15ca);

/// `erfc(x)` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    let (h, l, err) = fast(x);

    if !A::BIT_EXACT {
        let y = h + l;
        return if D::CHECKED {
            patch_lanes(x, y, outside(x), reference)
        } else {
            y
        };
    }

    let left = h + (l - err);
    let right = h + (l + err);
    let mut repair = left.eq_mask(right).not();
    if D::CHECKED {
        repair = repair.or(outside(x));
    }
    patch_lanes(x, left, repair, reference)
}

/// Lanes outside the fast path's range, and NaN.
///
/// The upper end is `ASYMPT_MAX`, not `POS_LIMIT`: above it the asymptotic
/// branch's `exp(-x^2)` underflows and its double-double comes back zeroed,
/// so those lanes need the reference under *either* accuracy policy. The
/// bit-exact path would also catch them through the rounding test; the fast
/// path has no rounding test, and would return zero for an `erfc` of `1e-292`.
#[inline(always)]
fn outside<V: Simd<Elem = f64>>(x: V) -> V::Mask {
    x.gt_mask(V::splat(-NEG_LIMIT))
        .and(x.lt_mask(V::splat(ASYMPT_MAX)))
        .and(x.abs().gt_mask(V::splat(r::UNIT_LIMIT)))
        .not()
}

/// `erfc(x)` as `(h, l, err)`, with `err` an absolute error bound.
///
/// Lane-for-lane identical to
/// [`crate::reference::double::erfc_parts::erfc_fast`].
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V) -> (V, V, V) {
    let is_asympt = x.gt_mask(V::splat(r::THRESHOLD1));
    // A buffer of `erfc` arguments is usually all tail or all body, so the
    // uniform cases are worth taking: they halve the work without giving up
    // the vector.
    if is_asympt.none() {
        return near(x);
    }
    if is_asympt.all() {
        return asymptotic(x);
    }
    let (ah, al, ae) = asymptotic(x);
    let (nh, nl, ne) = near(x);
    (
        V::select(is_asympt, ah, nh),
        V::select(is_asympt, al, nl),
        V::select(is_asympt, ae, ne),
    )
}

/// `x <= 0x1.713786d9c7c09p+1`: `erfc(x) = 1 -+ erf(|x|)`.
///
/// One path for both signs. `erfc(x) = 1 + erf(-x)` below zero and
/// `1 - erf(x)` above it, so only the sign handed to the `fast_two_sum` and
/// the sign of the low word differ — a blend rather than a branch.
#[inline(always)]
fn near<V: Simd<Elem = f64>>(x: V) -> (V, V, V) {
    /// `0x1.e861fbb24c00ap-2`, above which `1 - erf(x)` is exact by Sterbenz.
    const STERBENZ: f64 = f64::from_bits(0x3fde861fbb24c00a);
    /// The reflection's extra absolute error: `0x1.4p-102`.
    const EPS_NEG: f64 = f64::from_bits(0x3994000000000000);
    /// The same below `STERBENZ`: `0x1.4p-104`.
    const EPS_POS: f64 = f64::from_bits(0x3974000000000000);

    let neg = x.lt_mask(V::splat(0.0));
    let (eh, el, eerr) = super::erf::fast_parts(x.abs());
    let err = eerr * eh;

    let (h, tt) = fast_two_sum(V::splat(1.0), V::select(neg, eh, -eh));
    let l = V::select(neg, tt + el, tt - el);

    // Above `STERBENZ` the `fast_two_sum` is exact and contributes nothing, so
    // the bound `erf_fast` reported stands unchanged.
    let bump = V::select(
        neg,
        V::splat(EPS_NEG),
        V::select(
            x.ge_mask(V::splat(STERBENZ)),
            V::splat(0.0),
            V::splat(EPS_POS),
        ),
    );
    (h, l, err + bump)
}

/// `x > 0x1.713786d9c7c09p+1`: `erfc(x) = exp(-x^2) p(1/x)`.
#[inline(always)]
fn asymptotic<V: Simd<Elem = f64>>(x: V) -> (V, V, V) {
    /// The asymptotic branch's relative error bound: `0x1.d9p-68`.
    const ERR: f64 = f64::from_bits(0x3bbd900000000000);
    /// Below this, `ERR * h` would underflow: `0x1.151b9a3fdd5c9p-955`.
    const UFLOW_GUARD: f64 = f64::from_bits(0x044151b9a3fdd5c9);
    /// `0x1p-1022`, the overestimate used instead.
    const MIN_NORMAL: f64 = f64::from_bits(0x0010000000000000);

    let (uh, ul) = a_mul(x, x);
    let (eh, el) = exp_neg(uh, ul);

    // `1/x` as a double-double: one divide, then a Newton step `y + y(1 - xy)`
    // whose residual the FMA computes exactly.
    let yh = V::splat(1.0) / x;
    let yl = yh * (-x).mul_add(yh, V::splat(1.0));

    let p = gather_t::<V>(&yh);

    let (uh, ul) = a_mul(yh, yh);
    let ul = (yh + yh).mul_add(yl, ul);

    let mut zh = p[12];
    zh = zh.mul_add(uh, p[11]);
    zh = zh.mul_add(uh, p[10]);
    let (h, l) = s_mul(zh, uh, ul);
    let (mut zh, mut zl) = fast_two_sum(p[9], h);
    zl = zl + l;

    let mut j = 15i32;
    while j >= 3 {
        let (h, l) = d_mul(zh, zl, uh, ul);
        let (nh, nl) = fast_two_sum(p[((j + 1) / 2) as usize], h);
        zh = nh;
        zl = nl + l;
        j -= 2;
    }
    let (h, l) = d_mul(zh, zl, uh, ul);
    let (zh, zl) = fast_two_sum(p[0], h);
    let zl = zl + (l + p[1]);

    let (uh, ul) = d_mul(zh, zl, yh, yl);
    let (h, l) = d_mul(uh, ul, eh, el);

    let err = V::select(
        h.ge_mask(V::splat(UFLOW_GUARD)),
        V::splat(ERR) * h,
        V::splat(MIN_NORMAL),
    );
    // Above `ASYMPT_MAX` the double-double itself underflows — the scale
    // `exp(-x^2)` needs is below `2^-1022` and the exponent surgery above
    // wraps — so those lanes are zeroed and given an error bound of 1. Zeroing
    // is the part that matters: leaving a huge wrong `h` in place would let
    // `h + (l - 1)` and `h + (l + 1)` round to the same value and the rounding
    // test would wave the lane through instead of repairing it.
    let bail = x.ge_mask(V::splat(ASYMPT_MAX));
    let zero = V::splat(0.0);
    (
        V::select(bail, zero, h),
        V::select(bail, zero, l),
        V::select(bail, V::splat(1.0), err),
    )
}

/// `exp(-(uh + ul))` as a double-double, to a relative accuracy of `2^-74`.
///
/// Two 64-entry tables rather than one 4096-entry table: `2^(K/4096)` factors
/// as `2^(K>>12) * 2^(i2/64) * 2^(i1/4096)`, which is the difference between
/// 128 doubles of table and 8192.
#[inline(always)]
fn exp_neg<V: Simd<Elem = f64>>(uh: V, ul: V) -> (V, V) {
    /// `2^12 / ln(2)`.
    const INVLOG2: f64 = f64::from_bits(0x40b71547652b82fe);
    /// `ln(2)/2^12`, high part.
    const LOG2H: f64 = f64::from_bits(0x3f262e42fefa39ef);
    /// `ln(2)/2^12`, low part.
    const LOG2L: f64 = f64::from_bits(0x3bbabc9e3b39803f);

    let xh = -uh;
    let xl = -ul;
    let k = (xh * V::splat(INVLOG2)).round_ties_even();
    let (kh, kl) = s_mul(k, V::splat(LOG2H), V::splat(LOG2L));
    let (yh, yl) = fast_two_sum(xh - kh, xl);
    let yl = yl - kl;

    // The gather, and the only part of this kernel that does not vectorise.
    let ks = k.to_array();
    let mut t1h = V::Floats::filled_default();
    let mut t1l = V::Floats::filled_default();
    let mut t2h = V::Floats::filled_default();
    let mut t2l = V::Floats::filled_default();
    let mut df = V::Bits::filled_default();
    for lane in 0..V::LANES {
        // `f64 as i64` saturates in Rust, so a lane holding an infinity or a
        // NaN lands at one end of the range instead of being undefined. Its
        // result is discarded downstream either way.
        let ki = ks.as_slice()[lane] as i64;
        let i2 = ((ki >> 6) & 0x3f) as usize;
        let i1 = (ki & 0x3f) as usize;
        t1h.as_mut_slice()[lane] = t::T1[i2][0];
        t1l.as_mut_slice()[lane] = t::T1[i2][1];
        t2h.as_mut_slice()[lane] = t::T2[i1][0];
        t2l.as_mut_slice()[lane] = t::T2[i1][1];
        df.as_mut_slice()[lane] = (((ki >> 12) + 0x3ff) as u64) << 52;
    }

    let (hi, lo) = d_mul(
        V::from_array(t2h),
        V::from_array(t2l),
        V::from_array(t1h),
        V::from_array(t1l),
    );
    let (qh, ql) = q_1(yh, yl);
    let (hi, lo) = d_mul(hi, lo, qh, ql);
    let scale = V::from_bits(df);
    (hi * scale, lo * scale)
}

/// `exp(z)` for `|z| < 2^-12.88`, as a double-double.
#[inline(always)]
fn q_1<V: Simd<Elem = f64>>(zh: V, zl: V) -> (V, V) {
    let z = zh + zl;
    let q = V::splat(t::Q1[4]).mul_add(zh, V::splat(t::Q1[3]));
    let q = q.mul_add(z, V::splat(t::Q1[2]));
    let (hi, lo) = fast_two_sum(V::splat(t::Q1[1]), q * z);
    let (hi, lo) = d_mul(zh, zl, hi, lo);
    fast_sum(V::splat(t::Q1[0]), hi, lo)
}

/// The thirteen coefficients of the Chebyshev fit covering each lane's `1/x`.
///
/// Six fits, chosen by a linear scan of the thresholds — a scan rather than
/// arithmetic because the intervals are not uniform, and six comparisons of a
/// scalar are cheaper than any index formula that would reproduce them.
#[inline(always)]
fn gather_t<V: Simd<Elem = f64>>(yh: &V) -> [V; 13] {
    /// Which fit covers a given `1/x`.
    const THRESHOLD: [f64; 6] = [
        f64::from_bits(0x3fbd500000000000),
        f64::from_bits(0x3fc59da6ca291ba6),
        f64::from_bits(0x3fcbc00000000000),
        f64::from_bits(0x3fd0c00000000000),
        f64::from_bits(0x3fd3800000000000),
        f64::from_bits(0x3fd6300000000000),
    ];

    let ys = yh.to_array();
    let mut out = [V::Floats::filled_default(); 13];
    for lane in 0..V::LANES {
        let y = ys.as_slice()[lane];
        let mut i = 0usize;
        while i < THRESHOLD.len() && y > THRESHOLD[i] {
            i += 1;
        }
        let row = &t::T[i.min(t::T.len() - 1)];
        for (k, o) in out.iter_mut().enumerate() {
            o.as_mut_slice()[lane] = row[k];
        }
    }
    out.map(V::from_array)
}
