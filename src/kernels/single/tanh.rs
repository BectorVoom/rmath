//! Hyperbolic tangent, single precision.
//!
//! `BitExact` delegates to the platform's `tanhf`, exactly as before this
//! kernel existed: widening into the double-precision `tanh` kernel measured
//! at 0.24x, so parity is the honest ceiling. `Fast` is native, in two bands:
//!
//! * `|x| <= 1` — `tanh(x) = x * P(x²)`, a direct minimax fit
//!   (`tables::single::poly::TANH`). No cancellation to route around here:
//!   unlike the f64 kernel's `expm1`-based identity (chosen specifically to
//!   avoid a *different* cancellation), a polynomial fit of `tanh(x)/x` never
//!   subtracts two close quantities, so there is nothing more elaborate to
//!   reach for.
//! * `1 < |x| < 9` — `tanh(x) = 1 - 2/(e^{2x} + 1)`, computed with the
//!   already-native f32 `exp2` kernel (`exp(2x) = 2^{2x log2(e)}`). Well
//!   conditioned once `x >= 1`: the quantity being subtracted from 1 is
//!   bounded away from 1 itself, so the subtraction costs no more than the
//!   ulp or two `exp2` already spends.
//!
//! `|x| >= 9` and every non-finite input go to the scalar reference. `tanh`
//! is within one `f32` ulp of `±1` from there on — `1 - tanh(9)` is 3.0e-8,
//! against a spacing of 6.0e-8 just below 1 — and rounds to exactly `±1`
//! from about 9.02. So neither band's arithmetic buys anything past 9, and
//! the crate's usual pattern (the rare lanes are the scalar routine's job,
//! not the vector path's) applies unchanged. Note the reference is what
//! makes the tail *correct*, not the saturation argument: at `|x|` just
//! above 9 the answer is still one ulp shy of 1, so returning a flat `±1`
//! there would be wrong.
//!
//! `Finite` means `|x| < 9`.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain, Fast, Finite};
use crate::reference::single as reference;
use crate::simd::{Simd, patch_lanes};

/// Widest `|x|` the vector main path is valid for.
///
/// `1 - tanh(9)` is 3.0e-8, inside the 6.0e-8 spacing just below 1, so past
/// this the two bands' arithmetic can no longer distinguish itself from what
/// the scalar reference returns — see the module docs.
const MAIN_PATH_LIMIT: f32 = 9.0;

/// `tanh(x)` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    if A::BIT_EXACT {
        return crate::simd::map_lanes(x, reference::tanh);
    }
    let y = fast(x);
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::tanh)
    } else {
        y
    }
}

/// The native path, valid for `|x| < 9`.
///
/// Measured error: below 2 ulp.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    use crate::tables::single::poly::TANH as P;

    let a = x.abs();
    let small = a.le_mask(V::splat(1.0));

    // Direct polynomial, |x| <= 1: no identity, no cancellation.
    let s = a * a;
    let poly = crate::kernels::horner(s, &P);
    let direct = a * poly;

    // 1 < |x| < 9: well-conditioned once the subtrahend is bounded from 1.
    // `LOG2_E` folds the `e^{2x} = 2^{2x log2(e)}` conversion into `exp2`'s
    // single argument multiply, so this costs one `exp2` and one divide.
    const LOG2_E: f32 = core::f32::consts::LOG2_E;
    let arg = (a + a) * V::splat(LOG2_E);
    let e2 = crate::kernels::single::exp2::eval::<V, Fast, Finite>(arg);
    let large = V::splat(1.0) - V::splat(2.0) / (e2 + V::splat(1.0));

    V::select(small, direct, large).copysign(x)
}
