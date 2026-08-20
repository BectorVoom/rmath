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

use crate::policy::{BitExact, Fast, FullRange};
use crate::simd::{Real, Simd};

/// The widest `f64` vector the enabled backend provides.
///
/// What the `f64` buffer helpers step with. Without the `wide` feature this
/// is `f64`, and they degenerate to a scalar loop that still runs the same
/// kernel source.
pub type Widest = <f64 as Real>::Widest;

/// The widest `f32` vector the enabled backend provides.
pub type WidestF32 = <f32 as Real>::Widest;

#[inline(always)]
fn eval_unary_slice<E: Real, F: Function<E>>(f: &F, src: &[E], dst: &mut [E]) {
    assert_eq!(src.len(), dst.len(), "eval_slice: length mismatch");
    let lanes = <E::Widest as Simd>::LANES;
    let rem = src.len() % lanes;
    let end = src.len() - rem;
    let (s_head, s_tail) = src.split_at(end);
    let (d_head, d_tail) = dst.split_at_mut(end);

    for (s_chunk, d_chunk) in s_head.chunks_exact(lanes).zip(d_head.chunks_exact_mut(lanes)) {
        let v = f.eval(<E::Widest as Simd>::load_slice(s_chunk));
        v.store_slice(d_chunk);
    }
    for (s, d) in s_tail.iter().zip(d_tail.iter_mut()) {
        *d = f.eval::<E>(*s);
    }
}

#[inline(always)]
fn eval_unary_in_place<E: Real, F: Function<E>>(f: &F, buf: &mut [E]) {
    let lanes = <E::Widest as Simd>::LANES;
    let rem = buf.len() % lanes;
    let end = buf.len() - rem;
    let (head, tail) = buf.split_at_mut(end);

    for chunk in head.chunks_exact_mut(lanes) {
        let v = f.eval(<E::Widest as Simd>::load_slice(chunk));
        v.store_slice(chunk);
    }
    for x in tail.iter_mut() {
        *x = f.eval::<E>(*x);
    }
}

#[inline(always)]
fn eval_binary_slice<E: Real, F: Function2<E>>(f: &F, a: &[E], b: &[E], dst: &mut [E]) {
    assert_eq!(a.len(), b.len(), "eval_slice: length mismatch");
    assert_eq!(a.len(), dst.len(), "eval_slice: length mismatch");
    let lanes = <E::Widest as Simd>::LANES;
    let rem = a.len() % lanes;
    let end = a.len() - rem;
    let (a_head, a_tail) = a.split_at(end);
    let (b_head, b_tail) = b.split_at(end);
    let (d_head, d_tail) = dst.split_at_mut(end);

    for ((a_chunk, b_chunk), d_chunk) in a_head
        .chunks_exact(lanes)
        .zip(b_head.chunks_exact(lanes))
        .zip(d_head.chunks_exact_mut(lanes))
    {
        let va = <E::Widest as Simd>::load_slice(a_chunk);
        let vb = <E::Widest as Simd>::load_slice(b_chunk);
        let v = f.eval(va, vb);
        v.store_slice(d_chunk);
    }
    for ((x, y), d) in a_tail.iter().zip(b_tail.iter()).zip(d_tail.iter_mut()) {
        *d = f.eval::<E>(*x, *y);
    }
}

#[inline(always)]
fn eval_binary_slice_scalar<E: Real, F: Function2<E>>(
    f: &F,
    src: &[E],
    y: E,
    dst: &mut [E],
) {
    assert_eq!(src.len(), dst.len(), "eval_slice_scalar: length mismatch");
    let lanes = <E::Widest as Simd>::LANES;
    let rem = src.len() % lanes;
    let end = src.len() - rem;
    let (s_head, s_tail) = src.split_at(end);
    let (d_head, d_tail) = dst.split_at_mut(end);
    let y_splat = <E::Widest as Simd>::splat(y);

    for (s_chunk, d_chunk) in s_head.chunks_exact(lanes).zip(d_head.chunks_exact_mut(lanes)) {
        let va = <E::Widest as Simd>::load_slice(s_chunk);
        let v = f.eval(va, y_splat);
        v.store_slice(d_chunk);
    }
    for (s, d) in s_tail.iter().zip(d_tail.iter_mut()) {
        *d = f.eval::<E>(*s, y);
    }
}

#[inline(always)]
fn eval_pair_slice<E: Real, F: FunctionPair<E>>(
    f: &F,
    src: &[E],
    first: &mut [E],
    second: &mut [E],
) {
    assert_eq!(src.len(), first.len(), "eval_slice: length mismatch");
    assert_eq!(src.len(), second.len(), "eval_slice: length mismatch");
    let lanes = <E::Widest as Simd>::LANES;
    let rem = src.len() % lanes;
    let end = src.len() - rem;
    let (s_head, s_tail) = src.split_at(end);
    let (f_head, f_tail) = first.split_at_mut(end);
    let (s2_head, s2_tail) = second.split_at_mut(end);

    for ((s_chunk, f_chunk), s2_chunk) in s_head
        .chunks_exact(lanes)
        .zip(f_head.chunks_exact_mut(lanes))
        .zip(s2_head.chunks_exact_mut(lanes))
    {
        let va = <E::Widest as Simd>::load_slice(s_chunk);
        let (v1, v2) = f.eval(va);
        v1.store_slice(f_chunk);
        v2.store_slice(s2_chunk);
    }
    for ((s, f1), f2) in s_tail.iter().zip(f_tail.iter_mut()).zip(s2_tail.iter_mut()) {
        let (a, b) = f.eval::<E>(*s);
        *f1 = a;
        *f2 = b;
    }
}

#[inline(always)]
fn eval_2pair_slice<E: Real, F: Function2Pair<E>>(
    f: &F,
    a: &[E],
    b: &[E],
    first: &mut [E],
    second: &mut [E],
) {
    assert_eq!(a.len(), b.len(), "eval_slice: length mismatch");
    assert_eq!(a.len(), first.len(), "eval_slice: length mismatch");
    assert_eq!(a.len(), second.len(), "eval_slice: length mismatch");
    let lanes = <E::Widest as Simd>::LANES;
    let rem = a.len() % lanes;
    let end = a.len() - rem;
    let (a_head, a_tail) = a.split_at(end);
    let (b_head, b_tail) = b.split_at(end);
    let (f_head, f_tail) = first.split_at_mut(end);
    let (s2_head, s2_tail) = second.split_at_mut(end);

    for (((a_chunk, b_chunk), f_chunk), s2_chunk) in a_head
        .chunks_exact(lanes)
        .zip(b_head.chunks_exact(lanes))
        .zip(f_head.chunks_exact_mut(lanes))
        .zip(s2_head.chunks_exact_mut(lanes))
    {
        let va = <E::Widest as Simd>::load_slice(a_chunk);
        let vb = <E::Widest as Simd>::load_slice(b_chunk);
        let (v1, v2) = f.eval(va, vb);
        v1.store_slice(f_chunk);
        v2.store_slice(s2_chunk);
    }
    for (((x, y), f1), f2) in a_tail
        .iter()
        .zip(b_tail.iter())
        .zip(f_tail.iter_mut())
        .zip(s2_tail.iter_mut())
    {
        let (p, q) = f.eval::<E>(*x, *y);
        *f1 = p;
        *f2 = q;
    }
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
        eval_unary_slice(self, src, dst);
    }

    /// Apply in place.
    #[inline]
    fn eval_in_place(&self, buf: &mut [E]) {
        eval_unary_in_place(self, buf);
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
        eval_binary_slice(self, a, b, dst);
    }

    /// Apply with a scalar second argument, `dst[i] = f(src[i], y)`.
    ///
    /// The common case for `pow`: one exponent across a whole buffer.
    ///
    /// # Panics
    /// If `src` and `dst` have different lengths.
    #[inline]
    fn eval_slice_scalar(&self, src: &[E], y: E, dst: &mut [E]) {
        eval_binary_slice_scalar(self, src, y, dst);
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
        eval_pair_slice(self, src, first, second);
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
        eval_2pair_slice(self, a, b, first, second);
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
            $Name<Fast, FullRange>: Function<V::Elem>,
        {
            <$Name<Fast, FullRange> as Function<V::Elem>>::eval(&$Name::default(), x)
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
            $Name<Fast, FullRange>: Function2<V::Elem>,
        {
            <$Name<Fast, FullRange> as Function2<V::Elem>>::eval(&$Name::default(), x, y)
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
            $Name<Fast, FullRange>: FunctionPair<V::Elem>,
        {
            <$Name<Fast, FullRange> as FunctionPair<V::Elem>>::eval(&$Name::default(), x)
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
            $Name<Fast, FullRange>: Function2Pair<V::Elem>,
        {
            <$Name<Fast, FullRange> as Function2Pair<V::Elem>>::eval(&$Name::default(), x, y)
        }
    };
}

include!("function_defs.rs");
