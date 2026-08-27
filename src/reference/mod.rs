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
//! # Ported, or correctly rounded
//!
//! Two cases, and no third: **nothing in this module calls the platform any
//! more.** Every function here is either a port of the routine glibc runs, or
//! a port of a *correctly rounded* routine, and the difference is what the
//! bit-exactness claim rests on.
//!
//! A **port** reproduces the platform's *operation schedule* — which table,
//! which polynomial, which association, and critically where a fused
//! multiply-add is used and where two separate roundings are. That is what
//! makes a *vectorised* bit-exact kernel possible: the schedule can be
//! replayed lane-parallel. It is also a claim about a platform, which is why
//! `tests/bit_exact.rs` and `tests/single.rs` check every one of them against
//! the host and fail loudly on a machine whose libm differs.
//!
//! A **correctly rounded** routine returns the representable value nearest
//! the true result on every input. Reproducing one is *not* a claim about
//! this platform: any correctly-rounded implementation returns the same bits,
//! anywhere. `erf`, `erfc`, `asinh`, `acosh` and — in single precision —
//! `tanf`, `asinf`, `acosf`, `atanf`, `atan2f`, the whole hyperbolic family,
//! `expm1f`, `log1pf`, `log10f` and `cbrtf` are all of this kind, because
//! glibc took CORE-MATH's routines for them. They share a shape: a fast path
//! with a proven error bound, a bracket test asking whether that bound leaves
//! only one representable result in range, and a slower double-double or
//! triple-double path — sometimes plus a short table of the arguments where
//! even that lands on a halfway point — for when it does not.
//!
//! # What the ports cover
//!
//! * The exponential and logarithm families, `pow`, and (in single precision)
//!   `powf`, are ARM optimized-routines, ported from the `_fma` ifunc variant
//!   any FMA-capable x86-64 selects at load time.
//! * `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan` and `atan2` in
//!   double precision are the IBM Accurate Mathematical Library, disassembled
//!   rather than read off the C source — the FMA placement is the compiler's
//!   choice and the source does not show it. `__branred`, the Payne-Hanek
//!   reduction those four reach for past `|x| = 105414350`, is ported too
//!   ([`double::trig`]), so there is no huge-argument gap left; it is
//!   deliberately unfused, because glibc compiles `branred.c` as its own
//!   translation unit without `-mfma`.
//! * `sinf`, `cosf` and `sincosf` are ARM optimized-routines and are *not*
//!   correctly rounded (0.5607 ULP), so those three are schedule claims and
//!   were read out of the compiled `__sinf_fma` / `__cosf_fma` /
//!   `__sincosf_fma`.
//! * The Bessel family is Sun's fdlibm, which is still what glibc runs for it
//!   — verified by finding fdlibm's own constants in the installed `libm.so`
//!   where `erf`'s and `lgamma`'s are no longer there. Its trigonometry is
//!   now this crate's own [`double::trig`] / [`single::trig`], and the
//!   distinction between `__sincos` and a separate `__sin`/`__cos` pair is
//!   preserved: they are not the same computation.
//! * [`double::log1p()`] and [`double::hypot()`] are ports at the scalar
//!   level only. `hypot`'s algorithm is glibc's modern (2021+) Borges
//!   "MyHypot3" — deliberately FMA-free even on an FMA-capable host, since
//!   its error-free-transformation identities depend on separately-rounded
//!   arithmetic — not the older Dekker/`dla.h`-style routine this crate's own
//!   `ROADMAP.md` once assumed.
//!
//! # Which of these are vectorised
//!
//! A scalar port and a vector kernel are different pieces of work, and this
//! module only guarantees the first. Where no vector kernel exists,
//! [`crate::policy::BitExact`] runs the function here one lane at a time
//! under [`crate::simd::map_lanes`] — correct, and no faster than calling the
//! platform used to be, but no longer *calling* it. `src/kernels/` says which
//! functions have a replayed schedule and which are still scalar loops, and
//! the accuracy table in the crate documentation summarises it.
//!
//! `acos`'s near-1 band has one known open bug — see `ROADMAP.md`'s A4 entry.

pub mod double;
pub mod single;

pub use double::*;
