//! The two axes a function is configured on, as compile-time types.
//!
//! Both are *typestate*: the choice lives in the type parameters of the
//! function object, every policy type is a zero-sized struct, and the kernels
//! branch on associated `const`s. LLVM folds those branches away entirely, so
//! a configured function object costs exactly what hand-writing that variant
//! would — there is no runtime dispatch, no stored flags, and nothing to
//! inline through.
//!
//! ```
//! use rmath::prelude::*;
//!
//! let precise = Exp::new();                                  // BitExact + FullRange
//! let quick   = Exp::builder().accuracy(Fast).domain(Finite).build();
//!
//! assert_eq!(core::mem::size_of_val(&precise), 0);
//! assert_eq!(core::mem::size_of_val(&quick), 0);
//! ```

/// How accurate the result must be.
///
/// Sealed: the kernels only implement the variants defined here, so a
/// downstream type could not describe an algorithm they know how to run.
pub trait Accuracy: Copy + Default + private::Sealed {
    /// Take the reference algorithm rather than the cheap approximation.
    const BIT_EXACT: bool;
    /// Human-readable name, for `Debug` output.
    const NAME: &'static str;
}

/// Which inputs the caller promises to supply.
pub trait Domain: Copy + Default + private::Sealed {
    /// Check for, and correctly handle, inputs outside the fast path.
    const CHECKED: bool;
    /// Human-readable name, for `Debug` output.
    const NAME: &'static str;
}

/// Bit-for-bit identical to the platform's C library.
///
/// The kernels reproduce the reference algorithm's *operation schedule*, not
/// merely its mathematics: same table, same polynomial, same association, same
/// fused-multiply-add placement. Since every IEEE-754 operation rounds
/// identically regardless of vector width, running that schedule eight lanes
/// at a time gives the same bits as running it once.
///
/// This is the default because it is the only setting under which swapping
/// `rmath` in for the scalar libm cannot change a result — which is what makes
/// the swap reviewable. It is verified, not asserted: `tests/bit_exact.rs`
/// compares every lane against the platform libm over millions of inputs,
/// including every branch boundary and its neighbouring representable values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitExact;

/// A cheaper approximation, accurate to a few ulp.
///
/// Trades the reference algorithm's table lookup for a longer polynomial.
/// That is worth doing precisely because the lookup is the part that does not
/// vectorise: gathering eight table entries means eight scalar loads, whereas
/// a polynomial is pure vector arithmetic.
///
/// Each kernel documents its own measured error bound. Use this only where
/// you have decided the accuracy is adequate — and note that the derivative
/// of a formula can amplify a 1-ulp input error by orders of magnitude, so
/// "a few ulp here" is not "a few ulp downstream".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fast;

/// Inputs may be anything: infinities, NaN, subnormals, out-of-range values.
///
/// The kernel computes the main path across all lanes, then tests whether any
/// lane needs the reference treatment. That test is a single vector compare;
/// it only costs more when a lane actually is special, in which case those
/// lanes — and only those — are recomputed with the scalar reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FullRange;

/// The caller guarantees every input is finite and inside the main path.
///
/// Skips the range test entirely. Faster, and **unsound to use loosely**: an
/// out-of-range input does not trap or saturate, it silently produces a wrong
/// number. What "in range" means is documented per function.
///
/// This exists because in real vector workloads the guarantee is often already
/// available — a grid of physical densities, a normalised buffer — and paying
/// to re-establish it per call is waste.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Finite;

impl Accuracy for BitExact {
    const BIT_EXACT: bool = true;
    const NAME: &'static str = "BitExact";
}

impl Accuracy for Fast {
    const BIT_EXACT: bool = false;
    const NAME: &'static str = "Fast";
}

impl Domain for FullRange {
    const CHECKED: bool = true;
    const NAME: &'static str = "FullRange";
}

impl Domain for Finite {
    const CHECKED: bool = false;
    const NAME: &'static str = "Finite";
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::BitExact {}
    impl Sealed for super::Fast {}
    impl Sealed for super::FullRange {}
    impl Sealed for super::Finite {}
}
