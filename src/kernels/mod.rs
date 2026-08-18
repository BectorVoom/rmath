//! The vector kernels, one module per function.
//!
//! Every kernel exposes the same entry point:
//!
//! ```ignore
//! pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V
//! ```
//!
//! so the builder in [`crate::function`] can wrap any of them without knowing
//! what they do, and adding a function means adding a module here plus one
//! line of `math_fn!`. The policy parameters are resolved at compile time, so
//! each instantiation is a straight-line specialisation with no branches on
//! configuration.
//!
//! The shape is the same throughout: compute the main path across all lanes,
//! then — under [`crate::policy::FullRange`] — repair the rare lanes with the
//! scalar reference. That keeps the hard cases (overflow, subnormals, NaN) in
//! one place, [`crate::reference`], where they are written once against the
//! platform routine rather than re-derived in vector form per function.

pub mod cbrt;
pub mod exp;
pub mod exp2;
pub mod ln;
pub mod sqrt;

use crate::simd::{Mask, Simd};

/// Lanes that the vector main path must not be trusted with.
///
/// `!(|x| < limit)` rather than `|x| >= limit` so that NaN — which compares
/// false against everything — is caught by the negation.
#[inline(always)]
pub(crate) fn outside<V: Simd>(x: V, limit: f64) -> V::Mask {
    x.abs().lt_mask(V::splat(limit)).not()
}

/// Lanes that are not a *positive* normal number: negatives, zero,
/// subnormals, infinities and NaN.
///
/// For a function whose domain is the positive reals. Testing `x` rather than
/// `|x|` is the whole point — an earlier version used the magnitude, and every
/// negative input sailed into the main path and produced a plausible-looking
/// finite number instead of NaN.
#[inline(always)]
pub(crate) fn not_positive_normal<V: Simd>(x: V) -> V::Mask {
    x.ge_mask(V::splat(f64::MIN_POSITIVE))
        .and(x.lt_mask(V::splat(f64::INFINITY)))
        .not()
}
