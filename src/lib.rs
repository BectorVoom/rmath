//! A SIMD `libm` for `f64` — modular, and configured with a builder.
//!
//! Vectorising elementwise `f64` math usually stalls on the same wall: LLVM
//! will not vectorise a loop containing a call to `exp` or `log`, because the
//! call is opaque to it. The loop stays scalar no matter how well the rest of
//! it would have vectorised. `rmath` provides those functions as vector code,
//! so the loop can vectorise.
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
//! [`reference`].
//!
//! It follows that bit-exactness is a claim about a platform, not a universal
//! one. The test suite checks it against the host's `libm` over millions of
//! inputs — every branch boundary and its neighbouring representable values,
//! subnormals, specials, and random bit patterns — and fails loudly rather
//! than silently degrading if the host differs.
//!
//! Functions currently covered: [`exp`], [`exp2`], [`ln`], [`cbrt`], [`sqrt`].
//!
//! # Requires `std`
//!
//! Only for `f64::mul_add` and `f64::sqrt`, which are not in `core`.
//! Supporting `no_std` needs a correctly-rounded software FMA — evaluating
//! `a * b + c` instead would break every guarantee above. Nothing else in the
//! crate uses `std`: the reference implementations are self-contained ports,
//! precisely so that a `libm` replacement does not call the `libm` it
//! replaces.

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod function;
pub mod kernels;
pub mod policy;
pub mod reference;
pub mod simd;
pub mod tables;

pub use function::{
    Cbrt, CbrtBuilder, Exp, Exp2, Exp2Builder, ExpBuilder, Function, Ln, LnBuilder, Sqrt,
    SqrtBuilder, Widest, cbrt, exp, exp2, ln, sqrt,
};
pub use policy::{Accuracy, BitExact, Domain, Fast, Finite, FullRange};
pub use simd::Simd;

/// Everything needed to configure and call a function.
pub mod prelude {
    pub use crate::function::Function;
    pub use crate::function::{Cbrt, Exp, Exp2, Ln, Sqrt};
    pub use crate::policy::{BitExact, Fast, Finite, FullRange};
    pub use crate::simd::Simd;
}
