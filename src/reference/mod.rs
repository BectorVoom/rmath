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
//! `asin`, `acos`, `atan` and `atan2` ([`double::invtrig`]) are ports too now,
//! of `e_asin.c`/`s_atan.c`/`e_atan2.c` (the IBM Accurate Mathematical
//! Library's `asin`/`acos`/`atan`/`atan2`, disassembled the same way `sin`,
//! `cos`, `sincos` and the exponential/logarithm family were). `atan` and
//! `atan2` also have a vector kernel replaying that schedule lane-parallel
//! now (`src/kernels/double/invtrig.rs`'s `bit_exact` submodule); `asin` and
//! `acos` do not yet, so [`crate::policy::BitExact`] still runs those two
//! one lane at a time under [`crate::simd::map_lanes`], same cost as the
//! delegating functions below, for now. `acos`'s near-1 band also has one
//! known open bug — see `ROADMAP.md`'s A4 entry.
//!
//! `log10` (`crate::kernels::double::logx::log10`) is a port too: disassembly
//! shows `__log10_finite` is a thin wrapper around `__ieee754_log_fma` — the
//! exact table walk [`double::ln`] already replays lane-parallel — plus an
//! exponent reduction and a three-term combine in a specific, unfused order
//! (getting the order right mattered: it is not the order `e_log10.c`'s own
//! source grouping suggests). `log10` therefore has a full vector kernel
//! now, reusing `ln`'s table rather than re-deriving it.
//!
//! [`double::log1p()`] and [`double::hypot()`] are ports at
//! the scalar level — verified bit-exact against the live platform over tens
//! of millions of samples, including the NaN-payload-quieting and
//! signaling-NaN edge cases that a naive port gets wrong — but do not yet
//! have a vector kernel: [`crate::policy::BitExact`] still runs them one
//! lane at a time under [`crate::simd::map_lanes`]/[`crate::simd::map_lanes2`],
//! same cost as before, just no longer calling the platform underneath.
//! `hypot`'s algorithm is glibc's modern (2021+) Borges "MyHypot3" —
//! deliberately FMA-free even on an FMA-capable host, since its
//! error-free-transformation identities depend on separately-rounded
//! arithmetic — not the older Dekker/`dla.h`-style routine this crate's own
//! `ROADMAP.md` once assumed.
//!
//! `tan` and the inverse hyperbolics delegate to the platform routine. That
//! is a deliberate, documented choice rather than a stub: glibc implements
//! those with the IBM Accurate Portable Math Library routines, whose
//! operation schedule depends on large tables and on per-expression
//! fused-multiply-add placement chosen by the compiler, and reproducing it
//! is a much larger job than the rest of this crate put together. Delegating
//! keeps [`crate::policy::BitExact`] an honest guarantee for those
//! functions — it is bit-exact by construction — at the cost of the kernel
//! running one lane at a time. See the accuracy table in the crate
//! documentation for exactly which functions that covers, and use
//! [`crate::policy::Fast`] to get a vectorised path for them.

pub mod double;
pub mod single;

pub use double::*;
