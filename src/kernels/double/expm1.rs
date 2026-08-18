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

use crate::kernels::{dispatch, outside, pow2};
use crate::policy::{Accuracy, Domain};
use crate::reference::double::{
    self as reference, EXPM1_INVLN2, EXPM1_LN2HI, EXPM1_LN2LO, EXPM1_Q,
};
use crate::simd::{Simd, patch_lanes};
use crate::tables::double::poly as p;

/// `56 * ln(2)`, above which the reference handles it.
const MAIN_PATH_LIMIT: f64 = f64::from_bits(0x4043687a00000000);
/// `0.5 * ln(2)`, the point reduction starts.
const HALF_LN2: f64 = f64::from_bits(0x3fd62e4300000000);

/// `expm1(x)` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    if A::BIT_EXACT {
        let y = bit_exact(x);
        if D::CHECKED {
            return patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::expm1);
        }
        return y;
    }
    dispatch::<V, A, D>(x, reference::expm1, fast, |x| outside(x, 512.0))
}

/// The ported path, valid for `|x| < 56 ln 2`.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
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
    let hfx = half * xr;
    let hxs = xr * hfx;
    let hxs2 = hxs * hxs;
    let c01 = hxs.mul_add(V::splat(EXPM1_Q[0]), one);
    let c23 = V::splat(EXPM1_Q[2]).mul_add(hxs, V::splat(EXPM1_Q[1]));
    let c45 = V::splat(EXPM1_Q[4]).mul_add(hxs, V::splat(EXPM1_Q[3]));
    let r1 = (hxs2 * hxs2).mul_add(c45, hxs2.mul_add(c23, c01));
    let t = (-r1).mul_add(hfx, V::splat(3.0));
    let e0 = hxs * ((r1 - t) / (-t).mul_add(xr, V::splat(6.0)));

    // All five reconstruction arms, then one blend chain.
    let at_zero = xr - e0.mul_add(xr, -hxs);
    let e = (e0 - c).mul_add(xr, -c) - hxs;
    let twopk = pow2(k);
    let twomk = pow2(-k);

    let small_k = (xr - e + (one - twomk)) * twopk; // 2 <= k < 20
    let large_k = (xr - (e + twomk) + one) * twopk; // 20 <= k <= 56
    let neg_k = (xr - e + one) * twopk - one; // k <= -2
    let at_minus_one = half * (xr - e) - half;
    let at_plus_one = V::select(
        xr.lt_mask(V::splat(-0.25)),
        V::splat(-2.0) * (e - (xr + half)),
        one + V::splat(2.0) * (xr - e),
    );

    let mut y = V::select(k.lt_mask(V::splat(20.0)), small_k, large_k);
    y = V::select(k.lt_mask(zero), neg_k, y);
    y = V::select(k.eq_mask(one), at_plus_one, y);
    y = V::select(k.eq_mask(-one), at_minus_one, y);
    V::select(k.eq_mask(zero), at_zero, y)
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
