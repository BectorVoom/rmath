//! Scalar reference implementations: what [`crate::policy::BitExact`] means.
//!
//! One module per precision. The double-precision names are also re-exported
//! here, so `reference::exp` continues to mean `reference::double::exp`.
//!
//! # What a reference is for
//!
//! Two things, and they pull in the same direction:
//!
//! 1. **The definition of correct.** A vector kernel under
//!    [`crate::policy::BitExact`] must agree with the function here on every
//!    input, and the function here must agree with the platform's scalar
//!    routine on every input. `tests/bit_exact.rs` checks both halves.
//! 2. **The fallback path.** A vector kernel handles rare lanes — overflow,
//!    subnormals, NaN — by calling the function here for those lanes only, so
//!    correctness at the edges never depends on the vector code getting the
//!    hard cases right.
//!
//! # Ported, delegated, or correctly rounded
//!
//! Three cases now, because `erf` and `erfc` are a third. glibc computes them
//! with CORE-MATH's routines, which are *correctly rounded* — they return the
//! representable value nearest the true result on every input. That makes them
//! the one family here whose reference is not a claim about this platform: any
//! correctly-rounded implementation returns the same bits, so
//! [`double::erf_parts`] and [`double::erfc_parts`] are bit-exact to glibc and
//! to every other correct `erf` anywhere. See those modules for how correct
//! rounding is actually reached — a double-double fast path, a test asking
//! whether its error bound settles the last bit, and an accurate path for the
//! rare input where it does not.
//!
//! The Bessel family is a plain port, of Sun's fdlibm — which is still what
//! glibc runs for it, verified by finding fdlibm's own constants in the
//! installed `libm.so` where `erf`'s and `lgamma`'s are no longer there. Those
//! routines call the platform's `sin`, `cos` and `log`, so they inherit the
//! delegation described below for the trigonometric part.
//!
//! # Ported, or delegated
//!
//! Most of these are faithful ports of the routine the platform's C library
//! actually runs, reproducing its *operation schedule* — which table, which
//! polynomial, which association, and critically where a fused multiply-add
//! is used and where two separate roundings are. Those are marked as ports in
//! their own documentation, and they are what makes a *vectorised* bit-exact
//! kernel possible: the schedule can be replayed lane-parallel.
//!
//! The rest — the trigonometric and inverse-trigonometric families, the
//! inverse hyperbolics, `log10`, `log1p` and `hypot` — delegate to the
//! platform routine. That is a deliberate, documented choice
//! rather than a stub: glibc implements those with the IBM Accurate Portable
//! Math Library routines, whose operation schedule depends on 400-plus-entry
//! tables and on per-expression fused-multiply-add placement chosen by the
//! compiler, and reproducing it is a much larger job than the rest of this
//! crate put together. Delegating keeps [`crate::policy::BitExact`] an honest
//! guarantee for those functions — it is bit-exact by construction — at the
//! cost of the kernel running one lane at a time. See the accuracy table in
//! the crate documentation for exactly which functions that covers, and use
//! [`crate::policy::Fast`] to get a vectorised path for them.

pub mod double;
pub mod single;

pub use double::*;
