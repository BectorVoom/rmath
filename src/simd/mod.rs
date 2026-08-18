//! The vector abstraction every kernel is written against.
//!
//! A kernel is generic over [`Simd`], so one source implementation serves
//! `f64`, `f64x2`, `f64x4` and `f64x8` — and a future backend only has to
//! implement this trait to inherit every function in the crate.
//!
//! The trait deliberately exposes *both* styles of access:
//!
//! * **Whole-vector arithmetic** (`mul_add`, `select`, the operators) for the
//!   parts that vectorise.
//! * **Per-lane arrays** ([`Simd::to_bits`] / [`Simd::from_bits`]) for the
//!   parts that cannot: table gathers and exponent surgery. These compile to
//!   a fixed-length, branch-free loop, which LLVM handles well — but they are
//!   the reason [`crate::policy::Fast`] exists, since avoiding the gather is
//!   the single biggest lever on throughput.

use core::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "wide")]
mod wide_backend;

mod scalar;

/// A fixed-length lane array, `[T; LANES]`.
///
/// Exists because `Simd::LANES` cannot size an array in a trait signature on
/// stable Rust; an associated type sidesteps that without `generic_const_exprs`.
pub trait Lanes<T: Copy>: Copy {
    /// The lanes, in order.
    fn as_slice(&self) -> &[T];
    /// The lanes, mutably.
    fn as_mut_slice(&mut self) -> &mut [T];
    /// An array of `T::default()`, to be filled in.
    fn filled_default() -> Self;
}

impl<T: Copy + Default, const N: usize> Lanes<T> for [T; N] {
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self
    }
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
    #[inline(always)]
    fn filled_default() -> Self {
        [T::default(); N]
    }
}

/// The per-lane boolean result of a comparison.
pub trait Mask: Copy {
    /// Lane count, equal to that of the vector that produced it.
    const LANES: usize;
    /// `[bool; LANES]`.
    type Bools: Lanes<bool>;

    /// True if any lane is set.
    fn any(self) -> bool;
    /// True if every lane is set.
    fn all(self) -> bool;
    /// Lane-wise conjunction.
    fn and(self, other: Self) -> Self;
    /// Lane-wise disjunction.
    fn or(self, other: Self) -> Self;
    /// Lane-wise negation.
    fn not(self) -> Self;
    /// Unpack to booleans, for driving a per-lane fallback.
    fn to_bools(self) -> Self::Bools;

    /// True if no lane is set.
    #[inline(always)]
    fn none(self) -> bool {
        !self.any()
    }
}

/// A vector of `f64` lanes.
///
/// `f64` itself implements this with `LANES == 1`, which is what lets the
/// generic kernels serve as their own scalar fallback — including for the
/// ragged tail of [`crate::Function::eval_slice`].
pub trait Simd:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Number of `f64` lanes.
    const LANES: usize;
    /// The comparison result type.
    type Mask: Mask<Bools = Self::Bools>;
    /// `[f64; LANES]`.
    type Floats: Lanes<f64>;
    /// `[u64; LANES]`.
    type Bits: Lanes<u64>;
    /// `[bool; LANES]`.
    type Bools: Lanes<bool>;

    /// Every lane set to `v`.
    fn splat(v: f64) -> Self;
    /// Unpack the lanes.
    fn to_array(self) -> Self::Floats;
    /// Pack lanes into a vector.
    fn from_array(a: Self::Floats) -> Self;
    /// Reinterpret each lane as `u64`.
    fn to_bits(self) -> Self::Bits;
    /// Reinterpret each `u64` as an `f64` lane.
    fn from_bits(b: Self::Bits) -> Self;

    /// `self * mul + add`, as one correctly-rounded fused operation.
    ///
    /// Must be a true FMA (one rounding). The bit-exact kernels depend on it:
    /// evaluating it as `a * b + c` rounds twice and changes the result.
    fn mul_add(self, mul: Self, add: Self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Correctly-rounded square root.
    fn sqrt(self) -> Self;

    /// Lane-wise `self < other`.
    fn lt_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self <= other`.
    fn le_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self > other`.
    fn gt_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self >= other`.
    fn ge_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self == other`.
    fn eq_mask(self, other: Self) -> Self::Mask;

    /// Lane-wise `if mask { if_true } else { if_false }`.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    /// Lane-wise "is NaN", derived from the IEEE rule that NaN != NaN.
    #[inline(always)]
    fn is_nan(self) -> Self::Mask {
        self.eq_mask(self).not()
    }
}

/// Recompute the lanes selected by `mask` with a scalar reference function.
///
/// The pattern every `FullRange` kernel ends with: compute the main path
/// across all lanes, then repair the rare ones. Lanes outside the mask keep
/// the vector result untouched, so this cannot perturb the common case, and
/// the loop is skipped entirely when no lane is set.
#[inline(always)]
pub fn patch_lanes<V: Simd>(x: V, y: V, mask: V::Mask, reference: impl Fn(f64) -> f64) -> V {
    if mask.none() {
        return y;
    }
    let xs = x.to_array();
    let mut ys = y.to_array();
    let flags = mask.to_bools();
    for i in 0..V::LANES {
        if flags.as_slice()[i] {
            ys.as_mut_slice()[i] = reference(xs.as_slice()[i]);
        }
    }
    V::from_array(ys)
}
