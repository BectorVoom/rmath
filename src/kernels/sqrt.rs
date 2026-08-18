//! Square root.
//!
//! Both policy axes are no-ops here, and unusually that needs no apology:
//! IEEE-754 *requires* `sqrt` to be correctly rounded, and every target this
//! crate runs on implements it as a single instruction that already handles
//! zero, infinity, NaN and negatives to specification. There is no reference
//! to match — the hardware is the reference — no cheaper approximation worth
//! having, and no special case for `FullRange` to repair.
//!
//! It is included so that code applying a pipeline of these functions does not
//! have to break the pattern for this one, and so `eval_slice` works for it.

use crate::policy::{Accuracy, Domain};
use crate::simd::Simd;

/// `sqrt(x)` for a vector of lanes, correctly rounded.
#[inline(always)]
pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V {
    let _ = (A::BIT_EXACT, D::CHECKED);
    x.sqrt()
}
