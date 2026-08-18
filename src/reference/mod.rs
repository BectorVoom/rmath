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
