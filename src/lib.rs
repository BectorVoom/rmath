//! A SIMD `libm` for `f64` and `f32` — modular, and configured with a builder.
//!
//! Vectorising elementwise floating-point math usually stalls on the same
//! wall: LLVM will not vectorise a loop containing a call to `exp` or `log`,
//! because the call is opaque to it. The loop stays scalar no matter how well
//! the rest of it would have vectorised. `rmath` provides those functions as
//! vector code, so the loop can vectorise.
//!
//! # The two questions this crate makes explicit
//!
//! Reaching for a vector math library forces two decisions that a scalar
//! `libm` never asks you about, and getting either wrong is a silent bug. So
//! they are the two axes of the builder, and both are compile-time types:
//!
//! **How accurate?** Vector math libraries typically answer "about 1 ulp" and
//! move on. That is fine until it isn't: 1 ulp of `exp` amplified through a
//! derivative expression can be 1e-12 of the result, which is the difference
//! between agreeing with your reference implementation and not. So the default
//! here is [`BitExact`] — not close to the platform `libm`, *identical* to it,
//! reproducing the reference algorithm's exact operation schedule. Swapping
//! `rmath` in cannot change a result, which is what makes the swap reviewable.
//! [`Fast`] is available when you have decided you don't need that.
//!
//! **Which inputs?** Handling infinities, NaN and subnormals correctly costs a
//! test per call. Often the caller already knows the data is well-behaved — a
//! grid of physical quantities, a normalised buffer — and re-establishing that
//! per call is waste. [`FullRange`] is the default and is always safe;
//! [`Finite`] removes the test, and the guarantee becomes yours to keep.
//!
//! ```
//! use rmath::prelude::*;
//!
//! // Bit-identical to the platform libm, safe on anything.
//! let e = Exp::new();
//! assert_eq!(e.eval(1.0_f64), 1.0_f64.exp());
//!
//! // Configured: cheaper algorithm, and the caller vouches for the inputs.
//! let quick = Exp::builder().accuracy(Fast).domain(Finite).build();
//! assert!((quick.eval(1.0_f64) - 1.0_f64.exp()).abs() < 1e-15);
//!
//! // Neither object occupies any space; the configuration is in the type.
//! assert_eq!(size_of_val(&e), 0);
//! assert_eq!(size_of_val(&quick), 0);
//! ```
//!
//! # Using it on vectors
//!
//! A function object is not tied to a lane count. The same one evaluates a
//! scalar, any supported vector width, or a whole buffer:
//!
//! ```
//! use rmath::prelude::*;
//! # #[cfg(feature = "wide")] {
//! use wide::f64x8;
//!
//! let f = Ln::new();
//! let v = f.eval(f64x8::splat(2.0));
//! assert_eq!(v.to_array()[0], 2.0_f64.ln());
//!
//! let src: Vec<f64> = (1..=100).map(|i| i as f64).collect();
//! let mut dst = vec![0.0; src.len()];
//! f.eval_slice(&src, &mut dst);            // widest vectors, scalar tail
//! assert_eq!(dst[41], 42.0_f64.ln());
//! # }
//! ```
//!
//! # What "bit-exact" rests on
//!
//! Every IEEE-754 operation rounds identically regardless of vector width, so
//! running the reference algorithm's schedule eight lanes at a time gives the
//! same bits as running it once. The work is in reproducing the schedule
//! exactly — including where the compiled reference fuses a multiply-add and
//! where it rounds twice, which is not visible in its C source and differs
//! between functions and even between branches of one function. Those
//! placements were read from a disassembly of the compiled library; see
//! [`mod@reference`].
//!
//! It follows that bit-exactness is a claim about a platform, not a universal
//! one. The test suite checks it against the host's `libm` over millions of
//! inputs — every branch boundary and its neighbouring representable values,
//! subnormals, specials, and random bit patterns — and fails loudly rather
//! than silently degrading if the host differs.
//!
//! # Both precisions
//!
//! `f64` and `f32`, with the vector widths the backend provides: `f64x2`,
//! `f64x4`, `f64x8` and `f32x4`, `f32x8`. One function object serves all of
//! them — precision and width are properties of the data, not of the
//! configuration.
//!
//! ```
//! use rmath::prelude::*;
//!
//! let f = Exp::new();
//! assert_eq!(f.eval(1.0_f64), 1.0_f64.exp());
//! assert_eq!(f.eval(1.0_f32), 1.0_f32.exp());
//! ```
//!
//! Single precision is not the double-precision code with narrower lanes. The
//! platform's `expf`, `logf` and `log2f` are ports in their own right — and
//! they do their arithmetic in `double` over a small table, which is what lets
//! those kernels widen `f32x8` to `f64x8`, replay the schedule lane-parallel,
//! and round once. `f32` also admits a far stronger test than `f64` does:
//! there are only 2^32 inputs, so `tests/single.rs` checks every one of them
//! rather than a sample. That found a one-ulp difference in `expf` on two
//! inputs, and a double-rounding failure in an earlier design on one input in
//! four billion — neither of which any sampled test would have caught.
//!
//! # What each function actually gives you
//!
//! Three kinds of kernel, and the difference is worth knowing before you pick
//! a policy:
//!
//! | kind | `BitExact` | `Fast` |
//! |---|---|---|
//! | **Exact** — `floor`, `ceil`, `round`, `trunc`, `copysign`, `fmod`, `remainder`, `frexp`, `ldexp`, `nextafter`, `sqrt` | vectorised; exact by construction, on every platform | same code |
//! | **Ported** — `exp`, `exp2`, `expm1`, `ln`, `log2`, `pow`, `cbrt`, `sinh`, `cosh`, `tanh` | vectorised; replays the platform schedule | separate table-free vector path |
//! | **Delegating** — `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `asinh`, `acosh`, `atanh`, `log10`, `log1p`, `hypot` | one lane at a time; bit-exact, but no faster than the call it replaces | vectorised, with a measured ulp bound |
//!
//! The exact functions need no apology: IEEE-754 pins their results down
//! completely, so every correct implementation is bit-identical and both
//! policy axes are genuine no-ops.
//!
//! The delegating ones are the honest part. glibc implements those families
//! with the IBM Accurate Portable Math Library routines, whose operation
//! schedule turns on tables of several hundred entries and on per-expression
//! fused-multiply-add placement the compiler chooses and the C source does not
//! show. Reproducing that has not been done here, so under `BitExact` those
//! kernels call the platform routine per lane: still bit-exact — substituting
//! `rmath` cannot change your result — but at parity, not faster.
//! [`Fast`] is where their vector path lives, and it is a large win: 20x for
//! the trigonometric family on this machine.
//!
//! `lgamma` and `tgamma` are a fourth case. Rust has no `f64::tgamma`, so
//! there is no call for `BitExact` to be bit-exact *to*; both policies run one
//! vectorised implementation, documented by its measured error. See
//! [`kernels::double::gamma`].
//!
//! # Accuracy of `Fast`
//!
//! Measured against the platform, asserted in `tests/accuracy.rs`, and
//! therefore kept honest by the build: most kernels are within 4 ulp, the
//! inverse trigonometric and inverse hyperbolic ones within 8, and `pow`
//! within 40 because `y` multiplies the error in `log2 x` as well as its
//! value. Each kernel's module documentation states its own bound.
//!
//! # Requires `std`
//!
//! For `mul_add` and `sqrt`, which are not in `core`, and for the delegating
//! references, which are `std`'s own `sin`, `cos` and the rest. Supporting
//! `no_std` needs a correctly-rounded software FMA — evaluating `a * b + c`
//! instead would break every guarantee above — and would restrict the crate to
//! the exact and ported groups, since the delegating ones are *defined* as the
//! platform's routine.
//!
//! The ported references are deliberately self-contained, so that the
//! functions this crate claims to replace are not implemented by calling the
//! thing it replaces. The delegating ones are the stated exception, and the
//! table above says which they are.

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod function;
pub mod kernels;
pub mod policy;
pub mod reference;
pub mod simd;
pub mod tables;

pub use function::{
    Acos, AcosBuilder, Acosh, AcoshBuilder, Asin, AsinBuilder, Asinh, AsinhBuilder, Atan, Atan2,
    Atan2Builder, AtanBuilder, Atanh, AtanhBuilder, Cbrt, CbrtBuilder, Ceil, CeilBuilder, CopySign,
    CopySignBuilder, Cos, CosBuilder, Cosh, CoshBuilder, Exp, Exp2, Exp2Builder, ExpBuilder, Expm1,
    Expm1Builder, Floor, FloorBuilder, Fmod, FmodBuilder, Frexp, FrexpBuilder, Function, Function2,
    FunctionPair, Hypot, HypotBuilder, LGamma, LGammaBuilder, Ldexp, LdexpBuilder, Ln, LnBuilder,
    Log1p, Log1pBuilder, Log2, Log2Builder, Log10, Log10Builder, NextAfter, NextAfterBuilder, Pow,
    PowBuilder, Remainder, RemainderBuilder, Round, RoundBuilder, Sin, SinBuilder, SinCos,
    SinCosBuilder, Sinh, SinhBuilder, Sqrt, SqrtBuilder, TGamma, TGammaBuilder, Tan, TanBuilder,
    Tanh, TanhBuilder, Trunc, TruncBuilder, Widest, WidestF32, acos, acosh, asin, asinh, atan,
    atan2, atanh, cbrt, ceil, copysign, cos, cosh, exp, exp2, expm1, floor, fmod, frexp, hypot,
    ldexp, lgamma, ln, log1p, log2, log10, nextafter, pow, remainder, round, sin, sincos, sinh,
    sqrt, tan, tanh, tgamma, trunc,
};
pub use policy::{Accuracy, BitExact, Domain, Fast, Finite, FullRange};
pub use simd::{Real, Simd};

/// Everything needed to configure and call a function.
pub mod prelude {
    pub use crate::function::{
        Acos, Acosh, Asin, Asinh, Atan, Atan2, Atanh, Cbrt, Ceil, CopySign, Cos, Cosh, Exp, Exp2,
        Expm1, Floor, Fmod, Frexp, Hypot, LGamma, Ldexp, Ln, Log1p, Log2, Log10, NextAfter, Pow,
        Remainder, Round, Sin, SinCos, Sinh, Sqrt, TGamma, Tan, Tanh, Trunc,
    };
    pub use crate::function::{Function, Function2, FunctionPair};
    pub use crate::policy::{BitExact, Fast, Finite, FullRange};
    pub use crate::simd::{Real, Simd};
}
