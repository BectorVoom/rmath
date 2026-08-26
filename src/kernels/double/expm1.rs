//! `e^x - 1`, accurate for small `x`.
//!
//! A port: the vector code replays glibc's `__expm1_fma` schedule
//! lane-parallel. Writing this function separately from `exp` is the whole
//! point of having it — for small `x`, `exp(x) - 1` cancels away every
//! significant bit, while here the reduction leaves `k == 0` and the answer is
//! the polynomial itself.
//!
//! The reconstruction is where the vector form differs most from the scalar
//! one. glibc branches five ways on `k`, and each arm is two or three
//! operations; branching per lane would serialise the vector, so all five are
//! evaluated and blended. That costs a handful of arithmetic ops and keeps
//! eight lanes moving together.
//!
//! The blend covers the *whole* real line, which is why there is no
//! `patch_lanes` here. The reconstruction arm glibc uses for `k < 0` is the
//! same one it uses for `k > 56`, so extending the vector path from `56 ln 2`
//! up to the overflow threshold costs one extra comparison rather than a
//! scalar fallback; past that there are only three constants to select --
//! `-1` below, `x * 2^1023` above, and `x + x` for a NaN. Repairing those
//! lane-by-lane used to cost more than the entire main path on any corpus
//! that contained them, because a single out-of-range lane drags all eight
//! through the scalar reference.

use crate::kernels::{dispatch, outside, pow2};
use crate::policy::{Accuracy, Domain};
use crate::reference::double::{
    self as reference, EXPM1_INVLN2, EXPM1_LN2HI, EXPM1_LN2LO, EXPM1_OTHRESHOLD, EXPM1_Q,
};
use crate::simd::{Mask, Simd};
use crate::tables::double::poly as p;

/// `56 * ln(2)`. Below `-this` the platform answers `-1` outright; above
/// `+this` it still runs the main path, up to the overflow threshold. Tested
/// on the high word alone, exactly as the platform does.
const MAIN_PATH_LIMIT: f64 = f64::from_bits(0x4043687a00000000);
/// `0.5 * ln(2)`, the point reduction starts.
const HALF_LN2: f64 = f64::from_bits(0x3fd62e4300000000);
/// `2^1023`, the multiplier the platform overflows through.
const P1023: f64 = f64::from_bits(0x7fe0000000000000);

/// `expm1(x)` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    if A::BIT_EXACT {
        return bit_exact::<V, D>(x);
    }
    dispatch::<V, A, D>(x, reference::expm1, fast, |x| outside(x, 512.0))
}

/// `2^(k-1)` per lane, for an integer-valued `k` inside the exponent range.
///
/// [`pow2`] with the float-to-int conversion taken out: adding `1.5 * 2^52`
/// leaves `k + 2^51` exactly in the low 52 bits, so the exponent field can be
/// built from `k`'s own mantissa with integer arithmetic alone. x86 has no
/// saturating fix-up sequence to emit for that, and this runs on every vector
/// the bit-exact path touches.
///
/// Off by one power of two on purpose — see the call site.
#[inline(always)]
fn half_pow2<V: Simd<Elem = f64>>(k: V) -> V {
    use crate::simd::Lanes;
    /// `1.5 * 2^52`: big enough to pin the binade, so that `k` lands in the
    /// low mantissa bits with the ulp equal to one.
    const SHIFT: f64 = 6755399441055744.0;
    let ks = (k + V::splat(SHIFT)).to_bits();
    let mut bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let m = ks.as_slice()[i] & 0x000f_ffff_ffff_ffff;
        bits.as_mut_slice()[i] = m.wrapping_sub(1u64 << 51).wrapping_add(1022) << 52;
    }
    V::from_bits(bits)
}

/// The rational core glibc's `__expm1_fma` shares between every reconstruction
/// arm: `(hxs, e)` for an already-reduced argument.
///
/// Split out so that [`unreduced`] — the `k == 0` case, which is all `cosh`
/// ever needs — can have the core without the reduction and the five-way
/// blend wrapped around it.
#[inline(always)]
fn core<V: Simd<Elem = f64>>(xr: V) -> (V, V) {
    let one = V::splat(1.0);
    let hfx = V::splat(0.5) * xr;
    let hxs = xr * hfx;
    let hxs2 = hxs * hxs;
    let c01 = hxs.mul_add(V::splat(EXPM1_Q[0]), one);
    let c23 = V::splat(EXPM1_Q[2]).mul_add(hxs, V::splat(EXPM1_Q[1]));
    let c45 = V::splat(EXPM1_Q[4]).mul_add(hxs, V::splat(EXPM1_Q[3]));
    let r1 = (hxs2 * hxs2).mul_add(c45, hxs2.mul_add(c23, c01));
    let t = (-r1).mul_add(hfx, V::splat(3.0));
    (hxs, hxs * ((r1 - t) / (-t).mul_add(xr, V::splat(6.0))))
}

/// `expm1(x)` for `|x| < 0.5 ln 2`, where the reduction leaves `k == 0`.
///
/// Bit-identical to [`eval`] on that interval — it *is* [`eval`]'s `k == 0`
/// arm, with the reduction, the exponent construction and the five-way blend
/// dropped, because none of them can do anything when `k` is zero. `cosh`'s
/// near-zero arm is bounded by exactly this interval, so it takes this
/// instead of paying for machinery it has already proved it cannot need.
#[inline(always)]
pub(crate) fn unreduced<V: Simd<Elem = f64>>(x: V) -> V {
    let (hxs, e) = core(x);
    x - e.mul_add(x, -hxs)
}

/// The ported path, over the whole real line.
///
/// `D::CHECKED` adds the three constant answers outside `|x| < 56 ln 2`; the
/// reduced range itself is the same schedule either way.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>, D: Domain>(x: V) -> V {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let half = V::splat(0.5);

    // Reduction. Outside the reduced range `k` is zero, which makes every
    // step below the identity — so the two cases need no select of their own.
    let reduce = x.abs().ge_mask(V::splat(HALF_LN2));
    let kf = (V::splat(EXPM1_INVLN2) * x + half.copysign(x)).trunc();
    let k = V::select(reduce, kf, zero);
    let hi = x - k * V::splat(EXPM1_LN2HI); // exact
    let lo = k * V::splat(EXPM1_LN2LO);
    let xr = hi - lo;
    let c = (hi - xr) - lo;

    // The rational core, Estrin as the platform evaluates it.
    let (hxs, e0) = core(xr);

    // All five reconstruction arms, then one blend chain.
    let at_zero = xr - e0.mul_add(xr, -hxs);
    let e = (e0 - c).mul_add(xr, -c) - hxs;
    // `2^(k-1)`, not `2^k`: the top of the range has `k == 1024`, where the
    // exponent field has no room left, and where the platform likewise
    // reaches its answer as `y * 2.0 * 2^1023`. Doubling is exact and so is
    // the scaling, so `(y + y) * 2^(k-1)` and `y * 2^k` are the same double.
    let halfpk = half_pow2(k);
    let twopk = halfpk + halfpk;
    // `2^-k` by reciprocal rather than a second `pow2`: exact for every `k`
    // the two arms that use it can hold (`0 ..= 56`), and one vector divide
    // is cheaper than a second trip through the per-lane exponent loop.
    let twomk = one / twopk;

    let small_k = (xr - e + (one - twomk)) * twopk; // 2 <= k < 20
    let large_k = (xr - (e + twomk) + one) * twopk; // 20 <= k <= 56
    // `k` outside `0..=56`, in both directions: one arm, as in the platform.
    let far_k = {
        let y = xr - e + one;
        (y + y) * halfpk - one
    };
    let at_minus_one = half * (xr - e) - half;
    let at_plus_one = V::select(
        xr.lt_mask(V::splat(-0.25)),
        V::splat(-2.0) * (e - (xr + half)),
        one + V::splat(2.0) * (xr - e),
    );

    let mut y = V::select(k.lt_mask(V::splat(20.0)), small_k, large_k);
    y = V::select(k.lt_mask(zero).or(k.gt_mask(V::splat(56.0))), far_k, y);
    y = V::select(k.eq_mask(one), at_plus_one, y);
    y = V::select(k.eq_mask(-one), at_minus_one, y);
    y = V::select(k.eq_mask(zero), at_zero, y);

    if !D::CHECKED {
        return y;
    }
    // Past `56 ln 2` the platform stops computing: `-1` downwards (`e^x` has
    // vanished under the 1), an overflowing multiply upwards, and a quieted
    // NaN for a NaN. Between `56 ln 2` and the overflow threshold it does
    // fall through to the main path — which is why `far_k` above had to be
    // right for `k > 56`, and why this tail is three selects and not a
    // scalar loop.
    y = V::select(x.le_mask(V::splat(-MAIN_PATH_LIMIT)), -one, y);
    y = V::select(
        x.gt_mask(V::splat(EXPM1_OTHRESHOLD)),
        x * V::splat(P1023),
        y,
    );
    V::select(x.is_nan(), x + x, y)
}

/// The table-free path.
///
/// Measured error: below 4 ulp over `|x| < 512`.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V) -> V {
    let k = (x * V::splat(p::LOG2E)).round_ties_even();
    let r = k.mul_add(V::splat(-p::LN2[1]), k.mul_add(V::splat(-p::LN2[0]), x));

    // (e^r - 1)/r, so the leading 1 never has to be added and cancelled.
    let er = r * crate::kernels::horner(r, &p::EXPM1);
    let s = pow2(k);
    // 2^k (1 + er) - 1, grouped so that k == 0 leaves exactly `er`.
    s.mul_add(er, s - V::splat(1.0))
}
