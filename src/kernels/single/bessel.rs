//! The Bessel functions, single precision.
//!
//! # Why `BitExact` is scalar here
//!
//! glibc's `float` Bessel routines are not a narrowed version of its `double`
//! ones — they are a different algorithm, and the difference is a *repair*.
//! Near a zero of `j0` the rational fit loses essentially all of its digits to
//! cancellation, so glibc substitutes one of 64 tabulated degree-3
//! polynomials, and beyond the 64th zero an asymptotic form with Payne-Hanek
//! reduction. Which of the three runs is decided *after* the rational fit, by
//! how small its bracket came out, and then again by whether `x` falls inside
//! a tabulated interval.
//!
//! Blending three branches — one of which is a 192-bit fixed-point argument
//! reduction — to serve a case that fires on a thin set of inputs would cost
//! every lane more than the scalar call it replaces. So `BitExact` evaluates
//! [`crate::reference::single::bessel`] one lane at a time. It is bit-exact,
//! which is the guarantee that matters; it is not faster.
//!
//! # Why `Fast` is both faster *and* more accurate
//!
//! Unusually. `Fast` widens to the double-precision vector kernel and rounds
//! once, and glibc's own bound for these routines is 9 ulps — so the widened
//! path is not a cheaper approximation of `j0f`, it is a *better* one, and it
//! vectorises. The catch is the one this crate always states: it is not the
//! same value, so a program that has to reproduce the platform's `j0f` bit for
//! bit needs the default policy.

use crate::policy::{Accuracy, Domain};
use crate::simd::{Simd, map_lanes, map_lanes2};

/// Generate a single-precision Bessel kernel of order 0 or 1.
macro_rules! bessel {
    ($(#[$doc:meta])* $name:ident, $reference:path, $($k:ident)::+) => {
        $(#[$doc])*
        pub mod $name {
            use super::*;

            #[doc = concat!("`", stringify!($name), "(x)` for a vector of `f32` lanes.")]
            #[inline(always)]
            pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
                if A::BIT_EXACT {
                    return map_lanes(x, $reference);
                }
                V::narrow($crate::kernels::$($k)::+::eval::<V::Wide, A, D>(x.widen()))
            }
        }
    };
}

bessel! { /// Order 0, first kind.
j0, crate::reference::single::j0, double::bessel::j0 }
bessel! { /// Order 1, first kind.
j1, crate::reference::single::j1, double::bessel::j1 }
bessel! { /// Order 0, second kind.
y0, crate::reference::single::y0, double::bessel::y0 }
bessel! { /// Order 1, second kind.
y1, crate::reference::single::y1, double::bessel::y1 }

/// Generate a single-precision Bessel kernel of order `n`.
///
/// Scalar under both policies: the order-`n` routines choose between three
/// algorithms on `n` against `x`, and one of them runs a continued fraction
/// whose length is decided at run time. See
/// [`crate::kernels::double::bessel`].
macro_rules! bessel_n {
    ($(#[$doc:meta])* $name:ident, $reference:path) => {
        $(#[$doc])*
        pub mod $name {
            use super::*;

            #[doc = concat!("`", stringify!($name), "(n, x)` for vectors of `f32` lanes.")]
            #[inline(always)]
            pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(n: V, x: V) -> V {
                let _ = (A::BIT_EXACT, D::CHECKED);
                map_lanes2(n, x, |n, x| $reference(n.trunc() as i32, x))
            }
        }
    };
}

bessel_n! { /// Order `n`, first kind.
jn, crate::reference::single::jn }
bessel_n! { /// Order `n`, second kind.
yn, crate::reference::single::yn }
