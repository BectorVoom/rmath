//! `erfc`, and the double-double `exp` it needs.
//!
//! Like [`super::erf_parts`], this is CORE-MATH's correctly-rounded routine,
//! so matching it is not a claim about *this* platform's `libm`: any
//! correctly-rounded `erfc` returns the same bits.
//!
//! # Why `erfc` is a separate algorithm and not `1 - erf`
//!
//! Because that subtraction is catastrophic. `erfc(6)` is about `2e-17`, so
//! `1 - erf(6)` computes it as the difference of two numbers that agree to
//! sixteen digits and returns noise; by `x = 27` the true value is `2^-1074`
//! and the subtraction returns exactly zero. So the routine splits:
//!
//! * `x < 0` and `x <= 0x1.713786d9c7c09p+1` — where `erfc` is still `O(1)`
//!   and the cancellation is harmless — reuse `erf_parts::erf_fast`
//!   and one `fast_two_sum` against 1.
//! * above that, `erfc(x) = exp(-x^2) * p(1/x)` with a Chebyshev fit `p` per
//!   interval of `1/x`, and `exp(-x^2)` computed to 74 bits by a
//!   double-double exponential of its own — because a relative error of
//!   `2^-53` in `exp(-x^2)` is a relative error of `2^-53` in the answer, and
//!   the answer has to be correctly rounded.
//!
//! Both branches produce a double-double with an *absolute* error bound, and
//! the driver rounds it if the bound settles the rounding and calls the
//! accurate path if not.

use super::erf_parts::{
    a_mul, d_mul, erf_accurate, erf_fast, fast_sum, fast_two_sum, s_mul, two_sum,
};
use crate::tables::double::erfc as t;

/// `2^12 / ln(2)`, the reduction scale of [`exp_1`].
const INVLOG2: f64 = f64::from_bits(0x40b71547652b82fe);
/// `ln(2)/2^12`, high part.
const LOG2H: f64 = f64::from_bits(0x3f262e42fefa39ef);
/// `ln(2)/2^12`, low part.
const LOG2L: f64 = f64::from_bits(0x3bbabc9e3b39803f);

/// `0x1.713786d9c7c09p+1`: above this the asymptotic branch takes over.
pub(crate) const THRESHOLD1: f64 = f64::from_bits(0x400713786d9c7c09);
/// `0x1.e861fbb24c00ap-2`: at or above this, `1 - erf(x)` is exact by
/// Sterbenz, so the `fast_two_sum` contributes no error of its own.
const STERBENZ: f64 = f64::from_bits(0x3fde861fbb24c00a);
/// The extra absolute error the `x < 0` reflection costs: `0x1.4p-102`.
const EPS_NEG: f64 = f64::from_bits(0x3994000000000000);
/// The same for `0 <= x < STERBENZ`: `0x1.4p-104`.
const EPS_POS: f64 = f64::from_bits(0x3974000000000000);

/// `0x1.9db1bb14e15cap+4`: above this `erfc(x) < 2^-970` and the fast path's
/// low word would underflow, so it defers to the accurate path outright.
const ASYMPT_MAX: f64 = f64::from_bits(0x4039db1bb14e15ca);
/// The asymptotic branch's relative error bound: `0x1.d9p-68`.
const ERR_ASYMPT: f64 = f64::from_bits(0x3bbd900000000000);
/// Below this, `ERR_ASYMPT * h` would itself underflow: `0x1.151b9a3fdd5c9p-955`.
const UFLOW_GUARD: f64 = f64::from_bits(0x044151b9a3fdd5c9);
/// `0x1p-1022`, the overestimate used instead.
const MIN_NORMAL: f64 = f64::from_bits(0x0010000000000000);

/// Which Chebyshev fit of `erfc(x) exp(x^2) x` covers a given `1/x`.
const THRESHOLD: [f64; 6] = [
    f64::from_bits(0x3fbd500000000000),
    f64::from_bits(0x3fc59da6ca291ba6),
    f64::from_bits(0x3fcbc00000000000),
    f64::from_bits(0x3fd0c00000000000),
    f64::from_bits(0x3fd3800000000000),
    f64::from_bits(0x3fd6300000000000),
];

// ---------------------------------------------------------------------------
// The double-double exponential
// ---------------------------------------------------------------------------

/// `exp(z)` for `|z| < 2^-12.88`, as a double-double. CORE-MATH's `q_1`.
#[inline(always)]
fn q_1(zh: f64, zl: f64) -> (f64, f64) {
    let z = zh + zl;
    let q = t::Q1[4].mul_add(zh, t::Q1[3]);
    let q = q.mul_add(z, t::Q1[2]);
    let (hi, lo) = fast_two_sum(t::Q1[1], q * z);
    let (hi, lo) = d_mul(zh, zl, hi, lo);
    fast_sum(t::Q1[0], hi, lo)
}

/// `exp(xh + xl)` to a relative accuracy of `2^-74`, as a double-double.
///
/// Two 64-entry tables rather than one: `2^(K/4096)` splits as
/// `2^(K>>12) * 2^(i2/64) * 2^(i1/4096)`, so 4096 table entries become 128.
/// CORE-MATH's `exp_1`.
#[inline(always)]
fn exp_1(xh: f64, xl: f64) -> (f64, f64) {
    let k = (xh * INVLOG2).round_ties_even();
    let (kh, kl) = s_mul(k, LOG2H, LOG2L);
    let (yh, yl) = fast_two_sum(xh - kh, xl);
    let yl = yl - kl;

    let ki = k as i64;
    let m = (ki >> 12) + 0x3ff;
    let i2 = ((ki >> 6) & 0x3f) as usize;
    let i1 = (ki & 0x3f) as usize;

    let (hi, lo) = d_mul(t::T2[i1][0], t::T2[i1][1], t::T1[i2][0], t::T1[i2][1]);
    let (qh, ql) = q_1(yh, yl);
    let (hi, lo) = d_mul(hi, lo, qh, ql);
    let df = f64::from_bits((m as u64) << 52);
    (hi * df, lo * df)
}

/// `exp(xh + xl)` as `2^e * (h + l)` to about 104 bits, for the accurate path.
///
/// Separate from [`exp_1`] and not merely more terms of it: the argument here
/// reaches `-742`, so the result would underflow long before it is scaled, and
/// the exponent has to travel alongside the significand rather than in it.
fn exp_accurate(xh: f64, xl: f64) -> (f64, f64, i32) {
    /// `1/ln(2)`.
    const INVLOG2ACC: f64 = f64::from_bits(0x3ff71547652b82fe);
    /// `ln(2)`, high part.
    const LOG2HACC: f64 = f64::from_bits(0x3fe62e42fefa39ef);
    /// `ln(2) - LOG2HACC` to 38 bits, so `k * LOG2LACC` is exact for the
    /// 11-bit `k` this reduction produces.
    const LOG2LACC: f64 = f64::from_bits(0x3c7abc9e3b398000);
    /// What is left of `ln(2)` after the two above.
    const LOG2TINY: f64 = f64::from_bits(0x398f97b57a079a19);

    let k = (xh * INVLOG2ACC).round_ties_even() as i32;
    let kf = k as f64;
    // Exact by Sterbenz: `|xh| >= 2.92` forces `|k| >= 4`, which puts
    // `xh / (k ln 2)` inside `[1 - 1/(2|k|), 1 + 1/(2|k|)]`.
    let yh = (-kf).mul_add(LOG2HACC, xh);
    let (th, tl) = two_sum(-kf * LOG2LACC, xl);
    let (yh, yl) = fast_two_sum(yh, th);
    let yl = (-kf).mul_add(LOG2TINY, yl + tl);

    // Degrees 19 down to 16 ignore `yl`: its contribution there is below
    // 2^-104 and would be rounded away.
    let mut h = t::E2[19 + 8];
    for i in (16..=18).rev() {
        h = h.mul_add(yh, t::E2[i + 8]);
    }
    let (th, tl) = a_mul(h, yh);
    let tl = h.mul_add(yl, tl);
    let (mut h, mut l) = fast_two_sum(t::E2[15 + 8], th);
    l += tl;

    for i in (8..=14).rev() {
        let (th, tl) = a_mul(h, yh);
        let tl = h.mul_add(yl, tl);
        let tl = l.mul_add(yh, tl);
        let (nh, nl) = fast_two_sum(t::E2[i + 8], th);
        h = nh;
        l = nl + tl;
    }
    for i in (0..=7).rev() {
        let (th, tl) = a_mul(h, yh);
        let tl = h.mul_add(yl, tl);
        let tl = l.mul_add(yh, tl);
        let (nh, nl) = fast_two_sum(t::E2[2 * i], th);
        h = nh;
        l = nl + (tl + t::E2[2 * i + 1]);
    }
    (h, l, k)
}

// ---------------------------------------------------------------------------
// The fast path
// ---------------------------------------------------------------------------

/// `erfc(x)` as `(h, l, err)` for `x > 0x1.713786d9c7c09p+1`, by the
/// asymptotic formula. `err` is an *absolute* bound. CORE-MATH's
/// `erfc_asympt_fast`.
pub(crate) fn erfc_asympt_fast(x: f64) -> (f64, f64, f64) {
    if x >= ASYMPT_MAX {
        // Below 2^-970 the low word would underflow; hand the whole thing to
        // the accurate path by returning a bound that no rounding test passes.
        return (0.0, 0.0, 1.0);
    }

    let (uh, ul) = a_mul(x, x);
    let (eh, el) = exp_1(-uh, -ul);

    // `1/x` as a double-double: one divide, then one Newton step
    // `y -> y + y(1 - xy)`, which the FMA makes exact enough for 103 bits.
    let yh = 1.0 / x;
    let yl = yh * (-x).mul_add(yh, 1.0);

    let mut i = 0;
    while i < THRESHOLD.len() && yh > THRESHOLD[i] {
        i += 1;
    }
    let p = &t::T[i];

    let (uh, ul) = a_mul(yh, yh);
    let ul = (2.0 * yh).mul_add(yl, ul);

    let mut zh = p[12];
    zh = zh.mul_add(uh, p[11]);
    zh = zh.mul_add(uh, p[10]);
    let (h, l) = s_mul(zh, uh, ul);
    let (mut zh, mut zl) = fast_two_sum(p[9], h);
    zl += l;

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

    let err = if h >= UFLOW_GUARD {
        ERR_ASYMPT * h
    } else {
        MIN_NORMAL // an overestimate, but it cannot itself underflow
    };
    (h, l, err)
}

/// `erfc(x)` as `(h, l, err)` with `err` an absolute bound, for
/// `-0x1.7744f8f74e94bp+2 < x < 0x1.b39dc41e48bfdp+4`. CORE-MATH's
/// `cr_erfc_fast`.
pub(crate) fn erfc_fast(x: f64) -> (f64, f64, f64) {
    if x < 0.0 {
        // erfc(x) = 1 + erf(-x), and `h <= 2` keeps the sum's own error at
        // 2^-104, small enough that the stated constant absorbs it.
        let (h, l, err) = erf_fast(-x);
        let err = err * h;
        let (h, tt) = fast_two_sum(1.0, h);
        return (h, tt + l, err + EPS_NEG);
    }
    if x <= THRESHOLD1 {
        let (h, l, err) = erf_fast(x);
        let err = err * h;
        let (h, tt) = fast_two_sum(1.0, -h);
        let l = tt - l;
        // Above STERBENZ, `1 - h` is exact and `tt` is zero, so the only error
        // left is the one `erf_fast` reported.
        let err = if x >= STERBENZ { err } else { err + EPS_POS };
        return (h, l, err);
    }
    erfc_asympt_fast(x)
}

// ---------------------------------------------------------------------------
// The accurate path
// ---------------------------------------------------------------------------

/// `0x1.b59ffb450828cp+0`, where the accurate path switches to the asymptotic
/// formula. Higher than the fast path's threshold, because the accurate
/// polynomial keeps its digits further out.
const ACC_SPLIT: f64 = f64::from_bits(0x3ffb59ffb450828c);

/// The accurate asymptotic branch, for `1.70 < x < 27.3`.
fn erfc_asympt_accurate(x: f64) -> f64 {
    for e in t::EXCEPTIONS.iter() {
        if x == e[0] {
            return e[1] + e[2];
        }
    }
    // The one argument whose result is subnormal *and* a hard case.
    if x == f64::from_bits(0x403a8f7bfbd15495) {
        return f64::from_bits(1).mul_add(-0.25, f64::from_bits(0x000667bd620fd95b));
    }

    let (uh, ul) = a_mul(x, x);
    let (eh, el, e) = exp_accurate(-uh, -ul);

    let yh = 1.0 / x;
    let yl = yh * (-x).mul_add(yh, 1.0);

    /// Which of the ten accurate fits covers a given `1/x`.
    const THRESHOLD_ACC: [f64; 10] = [
        f64::from_bits(0x3fb4500000000000),
        f64::from_bits(0x3fbe000000000000),
        f64::from_bits(0x3fc3f00000000000),
        f64::from_bits(0x3fc9500000000000),
        f64::from_bits(0x3fcf500000000000),
        f64::from_bits(0x3fd3100000000000),
        f64::from_bits(0x3fd7100000000000),
        f64::from_bits(0x3fdbc00000000000),
        f64::from_bits(0x3fe0b00000000000),
        f64::from_bits(0x3fe3000000000000),
    ];
    let mut i = 0usize;
    while i < THRESHOLD_ACC.len() && yh > THRESHOLD_ACC[i] {
        i += 1;
    }
    let p = &t::TACC[i];

    let (uh, ul) = a_mul(yh, yh);
    let ul = (2.0 * yh).mul_add(yl, ul);

    // Degree 29 + 2i, whose leading coefficient is p[20 + i].
    let mut zh = p[14 + 6 + i];
    let mut zl = 0.0f64;
    let mut j = 27 + 2 * i as i32;
    while j >= 13 {
        let (h, l) = a_mul(zh, uh);
        let l = zh.mul_add(ul, l);
        let l = zl.mul_add(uh, l);
        let (nh, nl) = two_sum(p[((j - 1) / 2 + 6) as usize], h);
        zh = nh;
        zl = nl + l;
        j -= 2;
    }
    let mut j = 11i32;
    while j >= 1 {
        let (h, l) = a_mul(zh, uh);
        let l = zh.mul_add(ul, l);
        let l = zl.mul_add(uh, l);
        let (nh, nl) = two_sum(p[(j - 1) as usize], h);
        zh = nh;
        zl = nl + (l + p[j as usize]);
        j -= 2;
    }

    let (uh, ul) = a_mul(zh, yh);
    let ul = zh.mul_add(yl, ul);
    let ul = zl.mul_add(yh, ul);

    // Normalising before the last product keeps the number of hard cases down.
    let (uh, ul) = fast_two_sum(uh, ul);
    let (h, l) = a_mul(uh, eh);
    let l = uh.mul_add(el, l);
    let l = ul.mul_add(eh, l);

    let scale = |v: f64, k: i32| crate::kernels::exact::ldexp::scalar(v, k as f64);
    let res = scale(h + l, e);
    if res < MIN_NORMAL {
        // In the subnormal range the scaling above rounds twice. Recover the
        // discarded part and add it back at the right exponent.
        let corr = h - scale(res, -e) + l;
        return res + scale(corr, e);
    }
    res
}

/// `erfc(x)` for the lanes the rounding test could not settle. CORE-MATH's
/// `cr_erfc_accurate`.
pub(crate) fn erfc_accurate(x: f64) -> f64 {
    if x < 0.0 {
        for e in t::EXCEPTIONS_ACCURATE.iter() {
            if x == e[0] {
                return e[1] + e[2];
            }
        }
        let (h, l) = erf_accurate(-x);
        let (h, tt) = fast_two_sum(1.0, h);
        return h + (tt + l);
    }
    if x <= ACC_SPLIT {
        for e in t::EXCEPTIONS_ACCURATE_2.iter() {
            if x == e[0] {
                return e[1] + e[2];
            }
        }
        let (h, l) = erf_accurate(x);
        let (h, tt) = fast_two_sum(1.0, -h);
        return h + (tt - l);
    }
    erfc_asympt_accurate(x)
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// `-0x1.7744f8f74e94bp+2`: at or below this `erfc` rounds to 2.
pub(crate) const NEG_LIMIT: u64 = 0xc017744f8f74e94b;
/// `0x1.b39dc41e48bfdp+4`: at or above this `erfc(x) < 2^-1075`.
pub(crate) const POS_LIMIT: u64 = 0x403b39dc41e48bfd;
/// `0x1.c5bf891b4ef6ap-55`: below this magnitude `erfc` rounds to 1.
pub(crate) const UNIT_LIMIT: f64 = f64::from_bits(0x3c8c5bf891b4ef6a);

/// `erfc(x)`, correctly rounded — and so bit-identical to glibc's.
pub fn erfc(x: f64) -> f64 {
    let t_bits = x.to_bits();
    let at = t_bits & 0x7fff_ffff_ffff_ffff;

    if t_bits >= 0x8000_0000_0000_0000 {
        // x is negative, `-0.0`, or a negative NaN.
        if t_bits >= NEG_LIMIT {
            if t_bits >= 0xfff0_0000_0000_0000 {
                if t_bits == 0xfff0_0000_0000_0000 {
                    return 2.0; // -inf
                }
                return x + x; // NaN
            }
            return 2.0 - f64::from_bits(0x3c90000000000000); // 2, or below it
        }
        if -UNIT_LIMIT - UNIT_LIMIT <= x {
            // |x| below 0x1.c5bf891b4ef6ap-54: erfc(x) rounds to 1.
            return (-x).mul_add(f64::from_bits(0x3c90000000000000), 1.0);
        }
    } else {
        if at >= POS_LIMIT {
            if at >= 0x7ff0_0000_0000_0000 {
                if at == 0x7ff0_0000_0000_0000 {
                    return 0.0; // +inf
                }
                return x + x; // NaN
            }
            // Below 2^-1075: zero, or the smallest subnormal under a directed
            // rounding mode.
            return f64::from_bits(1) * 0.25;
        }
        if x <= UNIT_LIMIT {
            return (-x).mul_add(f64::from_bits(0x3c90000000000000), 1.0);
        }
    }

    let (h, l, err) = erfc_fast(x);
    let left = h + (l - err);
    let right = h + (l + err);
    if left == right {
        return left;
    }
    erfc_accurate(x)
}
