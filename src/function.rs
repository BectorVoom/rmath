//! The builder: how a kernel becomes a configured, callable function object.
//!
//! Every function in the crate is generated from one macro invocation, which
//! is the point of the design rather than an economy. A function object is:
//!
//! * **A zero-sized type.** The configuration lives in its type parameters, so
//!   `size_of` is 0, `build()` compiles to nothing, and calling `eval` costs
//!   exactly what calling the kernel directly costs.
//! * **Generic at the call site, not at construction.** `Exp` is not tied to a
//!   lane count *or to a precision* — one built object evaluates `f64`,
//!   `f64x2`, `f64x4`, `f64x8`, `f32`, `f32x4`, `f32x8` and `f32x16`. Width and
//!   precision are properties of the data, not of the configuration, and
//!   pinning either at build time would force callers to build one object per
//!   combination for no gain.
//!
//! Adding a function is a module in [`crate::kernels`] and one `math_fn!`
//! line here.
//!
//! # Arity
//!
//! Four traits, because four shapes of signature exist and collapsing them
//! into one would mean a tuple argument that every unary call site pays for:
//!
//! | trait | shape | examples |
//! |---|---|---|
//! | [`Function`] | `f(x)` | `exp`, `ln`, `sin`, `floor` |
//! | [`Function2`] | `f(x, y)` | `pow`, `atan2`, `hypot`, `fmod` |
//! | [`FunctionPair`] | `f(x) -> (a, b)` | `sincos`, `frexp`, `modf` |
//! | [`Function2Pair`] | `f(x, y) -> (a, b)` | `remquo` |
//!
//! All four are generic over the element type, so `Pow` implements
//! `Function2<f64>` and `Function2<f32>` and one object serves both.

use crate::policy::{BitExact, FullRange};
use crate::simd::{Lanes, Real, Simd};

/// The widest `f64` vector the enabled backend provides.
///
/// What the `f64` buffer helpers step with. Without the `wide` feature this
/// is `f64`, and they degenerate to a scalar loop that still runs the same
/// kernel source.
pub type Widest = <f64 as Real>::Widest;

/// The widest `f32` vector the enabled backend provides.
pub type WidestF32 = <f32 as Real>::Widest;

/// Apply a buffer of `src` through `f`, widest vectors first, scalar tail.
///
/// Shared by every `*_slice` helper below so the chunking logic exists once.
/// The ragged tail goes through the same kernel at one lane, so it cannot
/// drift from the body — there is no second implementation for it to drift
/// from.
#[inline(always)]
fn strided<E: Real>(len: usize, mut body: impl FnMut(usize, usize)) {
    let lanes = <E::Widest as Simd>::LANES;
    let mut i = 0;
    while i + lanes <= len {
        body(i, lanes);
        i += lanes;
    }
    while i < len {
        body(i, 1);
        i += 1;
    }
}

/// Gather `n` lanes starting at `off` into a vector, padding with `src[off]`.
///
/// `n` is either `V::LANES` or 1; the padding case only arises for the tail,
/// where the padded lanes are discarded.
#[inline(always)]
fn gather<V: Simd>(src: &[V::Elem], off: usize, n: usize) -> V {
    let mut chunk = V::Floats::filled_default();
    let slot = chunk.as_mut_slice();
    for (k, s) in slot.iter_mut().enumerate() {
        *s = src[off + if k < n { k } else { 0 }];
    }
    V::from_array(chunk)
}

/// Scatter the first `n` lanes of `v` to `dst[off..]`.
#[inline(always)]
fn scatter<V: Simd>(v: V, dst: &mut [V::Elem], off: usize, n: usize) {
    let out = v.to_array();
    dst[off..off + n].copy_from_slice(&out.as_slice()[..n]);
}

/// What every one-argument function object implements.
///
/// Useful for writing code generic over *which* function, e.g. a routine that
/// applies any configured unary function to a buffer. The element type
/// defaults to `f64`, so `impl Function` still means the double-precision
/// form.
pub trait Function<E: Real = f64>: Copy {
    /// Apply to one vector of lanes.
    fn eval<V: Simd<Elem = E>>(&self, x: V) -> V;

    /// Apply to a whole buffer, `dst[i] = f(src[i])`.
    ///
    /// # Panics
    /// If `src` and `dst` have different lengths.
    #[inline]
    fn eval_slice(&self, src: &[E], dst: &mut [E]) {
        assert_eq!(src.len(), dst.len(), "eval_slice: length mismatch");
        strided::<E>(src.len(), |i, n| {
            if n == 1 {
                dst[i] = self.eval::<E>(src[i]);
            } else {
                scatter(self.eval(gather::<E::Widest>(src, i, n)), dst, i, n);
            }
        });
    }

    /// Apply in place.
    #[inline]
    fn eval_in_place(&self, buf: &mut [E]) {
        strided::<E>(buf.len(), |i, n| {
            if n == 1 {
                buf[i] = self.eval::<E>(buf[i]);
            } else {
                let v = self.eval(gather::<E::Widest>(buf, i, n));
                scatter(v, buf, i, n);
            }
        });
    }
}

/// What every two-argument function object implements.
pub trait Function2<E: Real = f64>: Copy {
    /// Apply to one vector of lanes.
    fn eval<V: Simd<Elem = E>>(&self, x: V, y: V) -> V;

    /// Apply to whole buffers, `dst[i] = f(a[i], b[i])`.
    ///
    /// # Panics
    /// If the three slices do not all have the same length.
    #[inline]
    fn eval_slice(&self, a: &[E], b: &[E], dst: &mut [E]) {
        assert_eq!(a.len(), b.len(), "eval_slice: length mismatch");
        assert_eq!(a.len(), dst.len(), "eval_slice: length mismatch");
        strided::<E>(a.len(), |i, n| {
            if n == 1 {
                dst[i] = self.eval::<E>(a[i], b[i]);
            } else {
                let v = self.eval(gather::<E::Widest>(a, i, n), gather::<E::Widest>(b, i, n));
                scatter(v, dst, i, n);
            }
        });
    }

    /// Apply with a scalar second argument, `dst[i] = f(src[i], y)`.
    ///
    /// The common case for `pow`: one exponent across a whole buffer.
    ///
    /// # Panics
    /// If `src` and `dst` have different lengths.
    #[inline]
    fn eval_slice_scalar(&self, src: &[E], y: E, dst: &mut [E]) {
        assert_eq!(src.len(), dst.len(), "eval_slice_scalar: length mismatch");
        strided::<E>(src.len(), |i, n| {
            if n == 1 {
                dst[i] = self.eval::<E>(src[i], y);
            } else {
                let v = self.eval(
                    gather::<E::Widest>(src, i, n),
                    <E::Widest as Simd>::splat(y),
                );
                scatter(v, dst, i, n);
            }
        });
    }
}

/// What every function object returning two values implements.
pub trait FunctionPair<E: Real = f64>: Copy {
    /// Apply to one vector of lanes.
    fn eval<V: Simd<Elem = E>>(&self, x: V) -> (V, V);

    /// Apply to a whole buffer, writing the two results to separate outputs.
    ///
    /// # Panics
    /// If the three slices do not all have the same length.
    #[inline]
    fn eval_slice(&self, src: &[E], first: &mut [E], second: &mut [E]) {
        assert_eq!(src.len(), first.len(), "eval_slice: length mismatch");
        assert_eq!(src.len(), second.len(), "eval_slice: length mismatch");
        strided::<E>(src.len(), |i, n| {
            if n == 1 {
                let (a, b) = self.eval::<E>(src[i]);
                first[i] = a;
                second[i] = b;
            } else {
                let (a, b) = self.eval(gather::<E::Widest>(src, i, n));
                scatter(a, first, i, n);
                scatter(b, second, i, n);
            }
        });
    }
}

/// What every two-argument function object returning two values implements.
///
/// One member — `remquo` — and it exists rather than being folded into
/// [`FunctionPair`] because the second argument is genuinely a second vector,
/// not a configuration: `remquo(x, y)` reduces `x` modulo `y` and reports the
/// low bits of the quotient, and both operands vary per lane.
pub trait Function2Pair<E: Real = f64>: Copy {
    /// Apply to one vector of lanes.
    fn eval<V: Simd<Elem = E>>(&self, x: V, y: V) -> (V, V);

    /// Apply to whole buffers, writing the two results to separate outputs.
    ///
    /// # Panics
    /// If the four slices do not all have the same length.
    #[inline]
    fn eval_slice(&self, a: &[E], b: &[E], first: &mut [E], second: &mut [E]) {
        assert_eq!(a.len(), b.len(), "eval_slice: length mismatch");
        assert_eq!(a.len(), first.len(), "eval_slice: length mismatch");
        assert_eq!(a.len(), second.len(), "eval_slice: length mismatch");
        strided::<E>(a.len(), |i, n| {
            if n == 1 {
                let (p, q) = self.eval::<E>(a[i], b[i]);
                first[i] = p;
                second[i] = q;
            } else {
                let (p, q) = self.eval(gather::<E::Widest>(a, i, n), gather::<E::Widest>(b, i, n));
                scatter(p, first, i, n);
                scatter(q, second, i, n);
            }
        });
    }
}

/// The shared body of a function object: the type, its builder, and `Debug`.
macro_rules! object {
    ($(#[$fn_doc:meta])* $Name:ident, $Builder:ident) => {
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
    };
}

/// Generate a one-argument function object, its builder, and a free function.
///
/// `k64` and `k32` are paths to the kernel modules for the two precisions.
/// They are given separately, and in full, because they are not always
/// siblings: the exact functions — rounding, sign and exponent manipulation —
/// have one precision-generic kernel that both impls point at, while a
/// transcendental has a different algorithm per precision.
macro_rules! math_fn {
    (
        $(#[$fn_doc:meta])*
        name: $Name:ident,
        builder: $Builder:ident,
        free: $free:ident,
        k64: $($k64:ident)::+,
        k32: $($k32:ident)::+,
        $(#[$free_doc:meta])*
    ) => {
        object! { $(#[$fn_doc])* $Name, $Builder }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> Function<f64>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f64>>(&self, x: V) -> V {
                $($k64)::+::eval::<V, A, D>(x)
            }
        }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> Function<f32>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f32>>(&self, x: V) -> V {
                $($k32)::+::eval::<V, A, D>(x)
            }
        }

        $(#[$free_doc])*
        #[inline(always)]
        pub fn $free<V: Simd>(x: V) -> V
        where
            $Name<BitExact, FullRange>: Function<V::Elem>,
        {
            <$Name<BitExact, FullRange> as Function<V::Elem>>::eval(&$Name::new(), x)
        }
    };
}

/// Generate a two-argument function object, its builder, and a free function.
macro_rules! math_fn2 {
    (
        $(#[$fn_doc:meta])*
        name: $Name:ident,
        builder: $Builder:ident,
        free: $free:ident,
        k64: $($k64:ident)::+,
        k32: $($k32:ident)::+,
        $(#[$free_doc:meta])*
    ) => {
        object! { $(#[$fn_doc])* $Name, $Builder }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> Function2<f64>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f64>>(&self, x: V, y: V) -> V {
                $($k64)::+::eval::<V, A, D>(x, y)
            }
        }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> Function2<f32>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f32>>(&self, x: V, y: V) -> V {
                $($k32)::+::eval::<V, A, D>(x, y)
            }
        }

        $(#[$free_doc])*
        #[inline(always)]
        pub fn $free<V: Simd>(x: V, y: V) -> V
        where
            $Name<BitExact, FullRange>: Function2<V::Elem>,
        {
            <$Name<BitExact, FullRange> as Function2<V::Elem>>::eval(&$Name::new(), x, y)
        }
    };
}

/// Generate a two-result function object, its builder, and a free function.
macro_rules! math_fn_pair {
    (
        $(#[$fn_doc:meta])*
        name: $Name:ident,
        builder: $Builder:ident,
        free: $free:ident,
        k64: $($k64:ident)::+,
        k32: $($k32:ident)::+,
        $(#[$free_doc:meta])*
    ) => {
        object! { $(#[$fn_doc])* $Name, $Builder }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> FunctionPair<f64>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f64>>(&self, x: V) -> (V, V) {
                $($k64)::+::eval::<V, A, D>(x)
            }
        }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> FunctionPair<f32>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f32>>(&self, x: V) -> (V, V) {
                $($k32)::+::eval::<V, A, D>(x)
            }
        }

        $(#[$free_doc])*
        #[inline(always)]
        pub fn $free<V: Simd>(x: V) -> (V, V)
        where
            $Name<BitExact, FullRange>: FunctionPair<V::Elem>,
        {
            <$Name<BitExact, FullRange> as FunctionPair<V::Elem>>::eval(&$Name::new(), x)
        }
    };
}

/// Generate a two-argument, two-result function object, its builder, and a
/// free function.
macro_rules! math_fn2_pair {
    (
        $(#[$fn_doc:meta])*
        name: $Name:ident,
        builder: $Builder:ident,
        free: $free:ident,
        k64: $($k64:ident)::+,
        k32: $($k32:ident)::+,
        $(#[$free_doc:meta])*
    ) => {
        object! { $(#[$fn_doc])* $Name, $Builder }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> Function2Pair<f64>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f64>>(&self, x: V, y: V) -> (V, V) {
                $($k64)::+::eval::<V, A, D>(x, y)
            }
        }

        impl<A: $crate::policy::Accuracy, D: $crate::policy::Domain> Function2Pair<f32>
            for $Name<A, D>
        {
            #[inline(always)]
            fn eval<V: Simd<Elem = f32>>(&self, x: V, y: V) -> (V, V) {
                $($k32)::+::eval::<V, A, D>(x, y)
            }
        }

        $(#[$free_doc])*
        #[inline(always)]
        pub fn $free<V: Simd>(x: V, y: V) -> (V, V)
        where
            $Name<BitExact, FullRange>: Function2Pair<V::Elem>,
        {
            <$Name<BitExact, FullRange> as Function2Pair<V::Elem>>::eval(&$Name::new(), x, y)
        }
    };
}

include!("function_defs.rs");
