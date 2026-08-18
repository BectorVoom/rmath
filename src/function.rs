//! The builder: how a kernel becomes a configured, callable function object.
//!
//! Every function in the crate is generated from one macro invocation, which
//! is the point of the design rather than an economy. A function object is:
//!
//! * **A zero-sized type.** The configuration lives in its type parameters, so
//!   `size_of` is 0, `build()` compiles to nothing, and calling `eval` costs
//!   exactly what calling the kernel directly costs.
//! * **Generic at the call site, not at construction.** `Exp` is not tied to a
//!   lane count — one built object evaluates `f64`, `f64x2`, `f64x4` and
//!   `f64x8`. Width is a property of the data, not of the configuration, and
//!   pinning it at build time would force callers to build one object per
//!   width for no gain.
//!
//! Adding a function is a module in [`crate::kernels`] and one `math_fn!`
//! line here.

use crate::policy::{BitExact, FullRange};
use crate::simd::{Lanes, Simd};

/// The widest vector the enabled backend provides.
///
/// What [`Function::eval_slice`] steps a buffer with. Without the `wide`
/// feature this is `f64`, and `eval_slice` degenerates to a scalar loop that
/// still runs the same kernel source.
#[cfg(feature = "wide")]
pub type Widest = wide::f64x8;
/// The widest vector the enabled backend provides.
#[cfg(not(feature = "wide"))]
pub type Widest = f64;

/// What every function object implements.
///
/// Useful for writing code generic over *which* function, e.g. a routine that
/// applies any configured unary function to a buffer.
pub trait Function: Copy {
    /// Apply to one vector of lanes.
    fn eval<V: Simd>(&self, x: V) -> V;

    /// Apply to a whole buffer, `dst[i] = f(src[i])`.
    ///
    /// Steps [`Widest`] lanes at a time and finishes the ragged tail through
    /// the same kernel at one lane, so the tail cannot drift from the body —
    /// there is no second implementation for it to drift from.
    ///
    /// # Panics
    /// If `src` and `dst` have different lengths.
    #[inline]
    fn eval_slice(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len(), "eval_slice: length mismatch");
        let lanes = <Widest as Simd>::LANES;
        let mut i = 0;
        while i + lanes <= src.len() {
            let mut chunk = <Widest as Simd>::Floats::filled_default();
            chunk.as_mut_slice().copy_from_slice(&src[i..i + lanes]);
            let y = self.eval(Widest::from_array(chunk));
            dst[i..i + lanes].copy_from_slice(y.to_array().as_slice());
            i += lanes;
        }
        for j in i..src.len() {
            dst[j] = self.eval::<f64>(src[j]);
        }
    }

    /// Apply in place.
    ///
    /// # Panics
    /// Never; provided for symmetry with [`Function::eval_slice`].
    #[inline]
    fn eval_in_place(&self, buf: &mut [f64]) {
        let lanes = <Widest as Simd>::LANES;
        let mut i = 0;
        while i + lanes <= buf.len() {
            let mut chunk = <Widest as Simd>::Floats::filled_default();
            chunk.as_mut_slice().copy_from_slice(&buf[i..i + lanes]);
            let y = self.eval(Widest::from_array(chunk));
            buf[i..i + lanes].copy_from_slice(y.to_array().as_slice());
            i += lanes;
        }
        for v in &mut buf[i..] {
            *v = self.eval::<f64>(*v);
        }
    }
}

/// Generate a function object, its builder, and a default-policy free function.
macro_rules! math_fn {
    (
        $(#[$fn_doc:meta])*
        name: $Name:ident,
        builder: $Builder:ident,
        free: $free:ident,
        kernel: $kernel:ident,
        $(#[$free_doc:meta])*
    ) => {
        $(#[$fn_doc])*
        pub struct $Name<A = BitExact, D = FullRange>(core::marker::PhantomData<(A, D)>);

        impl<A, D> Clone for $Name<A, D> {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<A, D> Copy for $Name<A, D> {}
        impl<A, D> Default for $Name<A, D> {
            #[inline(always)]
            fn default() -> Self {
                Self(core::marker::PhantomData)
            }
        }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> core::fmt::Debug
            for $Name<A, D>
        {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, concat!(stringify!($Name), "<{}, {}>"), A::NAME, D::NAME)
            }
        }

        impl $Name<BitExact, FullRange> {
            /// The default configuration: bit-exact, and safe on any input.
            #[inline(always)]
            pub const fn new() -> Self {
                Self(core::marker::PhantomData)
            }

            /// Start configuring.
            #[inline(always)]
            pub const fn builder() -> $Builder<BitExact, FullRange> {
                $Builder(core::marker::PhantomData)
            }
        }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> Function for $Name<A, D> {
            #[inline(always)]
            fn eval<V: Simd>(&self, x: V) -> V {
                $crate::kernels::$kernel::eval::<V, A, D>(x)
            }
        }

        #[doc = concat!("Builder for [`", stringify!($Name), "`].")]
        ///
        /// Zero-sized; each method returns a differently-typed builder, so the
        /// configuration is resolved entirely at compile time.
        pub struct $Builder<A, D>(core::marker::PhantomData<(A, D)>);

        impl<A, D> Clone for $Builder<A, D> {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<A, D> Copy for $Builder<A, D> {}

        impl<A, D> $Builder<A, D> {
            /// Choose an accuracy: [`crate::policy::BitExact`] or
            /// [`crate::policy::Fast`].
            #[inline(always)]
            pub fn accuracy<A2>(self, _accuracy: A2) -> $Builder<A2, D> {
                $Builder(core::marker::PhantomData)
            }

            /// Choose a domain: [`crate::policy::FullRange`] or
            /// [`crate::policy::Finite`].
            #[inline(always)]
            pub fn domain<D2>(self, _domain: D2) -> $Builder<A, D2> {
                $Builder(core::marker::PhantomData)
            }

            /// Finish. Compiles to nothing.
            #[inline(always)]
            pub const fn build(self) -> $Name<A, D> {
                $Name(core::marker::PhantomData)
            }
        }

        $(#[$free_doc])*
        #[inline(always)]
        pub fn $free<V: Simd>(x: V) -> V {
            $crate::kernels::$kernel::eval::<V, BitExact, FullRange>(x)
        }
    };
}

math_fn! {
    /// `e^x`.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let f = Exp::new();
    /// assert_eq!(f.eval(0.0_f64), 1.0);
    /// ```
    name: Exp,
    builder: ExpBuilder,
    free: exp,
    kernel: exp,
    /// `e^x`, bit-exact and safe on any input.
    ///
    /// The shorthand for `Exp::new().eval(x)`.
}

math_fn! {
    /// `2^x`.
    name: Exp2,
    builder: Exp2Builder,
    free: exp2,
    kernel: exp2,
    /// `2^x`, bit-exact and safe on any input.
}

math_fn! {
    /// Natural logarithm.
    name: Ln,
    builder: LnBuilder,
    free: ln,
    kernel: ln,
    /// `ln(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Cube root.
    ///
    /// The accuracy axis is accepted but has no effect; see
    /// [`crate::kernels::cbrt`].
    name: Cbrt,
    builder: CbrtBuilder,
    free: cbrt,
    kernel: cbrt,
    /// `x^(1/3)`, bit-exact and safe on any input.
}

math_fn! {
    /// Square root. Always correctly rounded; both policy axes are no-ops.
    name: Sqrt,
    builder: SqrtBuilder,
    free: sqrt,
    kernel: sqrt,
    /// `sqrt(x)`, correctly rounded.
}
