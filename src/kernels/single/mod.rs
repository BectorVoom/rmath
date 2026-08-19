//! Single-precision kernels.
//!
//! Two shapes:
//!
//! * **Ported** — [`exp`], [`exp2`], [`ln`], [`log2`]. These are the functions
//!   the platform does *not* compute correctly rounded, so matching them means
//!   replaying their exact schedule. That schedule is double-precision
//!   arithmetic over a small table, so the kernel widens `f32xN` to `f64xN`,
//!   runs it lane-parallel, and narrows once. Bit-exact *and* vectorised —
//!   these are the single-precision functions that beat the platform under the
//!   default policy.
//!
//! * **Delegating** — everything else. `BitExact` evaluates the platform's own
//!   `f32` routine lane by lane; `Fast` widens and takes the double-precision
//!   vector path, which is where the speed is. Three exceptions: [`cbrt`],
//!   [`tanh`] and [`trig`]'s `Fast` paths are native rather than widened —
//!   `cbrt` and `tanh` because widening measured slower than the scalar call
//!   it would replace, `trig` because the reduction and polynomials have no
//!   per-lane work to begin with, so widening would cost a round trip for
//!   nothing the native path needs.
//!
//! # Why not compute everything in double and round once
//!
//! It is tempting, and it very nearly works: most of the platform's `f32`
//! routines are correctly rounded, and evaluating in `f64` leaves an error of
//! some 2^-29 of a single-precision ulp, so rounding once "should" land on the
//! same value. That was this module's first design. Two things killed it.
//!
//! **It is not actually bit-exact.** Double rounding fails when the `f64`
//! result lands within its own error of an `f32` rounding boundary. The
//! exhaustive sweep in `tests/single.rs` found `log10f` differing on 1 input
//! of the 2^32, and `sinhf` on 2 — rates around 1 in 4e9, which no sampled
//! test would ever have caught, and which would have made the crate's headline
//! claim quietly false.
//!
//! **It is also slower.** The platform's `float` routines are much cheaper
//! than its `double` ones, so routing `f32` work through `f64` costs more than
//! the call it replaces: measured at 0.24x for `tanhf` and 0.48x for `cbrtf`.
//! Delegating is both exact and faster.
//!
//! So the wide path is kept for `Fast`, where there is no exactness claim to
//! break and the vector width is worth having, and `BitExact` delegates.

use crate::simd::{Lanes, Simd};
use crate::tables::single::exp as et;

/// Build a single-precision kernel from the double-precision one.
///
/// Widen, run the `f64` kernel at the same lane count, narrow. The policy
/// parameters pass straight through, so `Fast` and `Finite` mean here what
/// they mean there.
/// Run the double-precision kernel and round once, with no reference to fall
/// back to.
///
/// Only the Gamma functions, which have no `f32` counterpart in `std` for a
/// bit-exactness claim to refer to; see [`crate::kernels::double::gamma`].
macro_rules! via_double_raw {
    ($(#[$doc:meta])* $name:ident, $($k:ident)::+) => {
        $(#[$doc])*
        pub mod $name {
            use $crate::policy::{Accuracy, Domain};
            use $crate::simd::Simd;

            #[doc = concat!("`", stringify!($name), "(x)` for a vector of `f32` lanes.")]
            #[inline(always)]
            pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
                V::narrow($crate::kernels::$($k)::+::eval::<V::Wide, A, D>(x.widen()))
            }
        }
    };
}

/// A kernel whose bit-exact path is the scalar reference, lane by lane.
///
/// For the functions where the platform's `float` routine is neither correctly
/// rounded nor reproduced here. `Fast` still gets the double-precision vector
/// path, which is where the speed is.
macro_rules! delegating {
    ($(#[$doc:meta])* $name:ident, $reference:path, $($k:ident)::+) => {
        $(#[$doc])*
        pub mod $name {
            use $crate::policy::{Accuracy, Domain};
            use $crate::simd::{Simd, map_lanes};

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

/// [`delegating`] for a two-argument function.
macro_rules! delegating2 {
    ($(#[$doc:meta])* $name:ident, $reference:path, $($k:ident)::+) => {
        $(#[$doc])*
        pub mod $name {
            use $crate::policy::{Accuracy, Domain};
            use $crate::simd::{Simd, map_lanes2};

            #[doc = concat!("`", stringify!($name), "(x, y)` for vectors of `f32` lanes.")]
            #[inline(always)]
            pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V, y: V) -> V {
                if A::BIT_EXACT {
                    return map_lanes2(x, y, $reference);
                }
                V::narrow($crate::kernels::$($k)::+::eval::<V::Wide, A, D>(x.widen(), y.widen()))
            }
        }
    };
}

pub mod bessel;
pub mod cbrt;
pub mod erf;
pub mod erfc;
pub mod exp;
pub mod exp10;
pub mod exp2;
pub mod ln;
pub mod log2;
pub mod tanh;
pub mod trig;
delegating! { /// Arc sine.
asin, crate::reference::single::asin, double::invtrig::asin }
delegating! { /// Arc cosine.
acos, crate::reference::single::acos, double::invtrig::acos }
delegating! { /// Arc tangent.
atan, crate::reference::single::atan, double::invtrig::atan }
delegating! { /// Hyperbolic sine.
sinh, crate::reference::single::sinh, double::hyper::sinh }
delegating! { /// Hyperbolic cosine.
cosh, crate::reference::single::cosh, double::hyper::cosh }
delegating! { /// Inverse hyperbolic sine.
asinh, crate::reference::single::asinh, double::hyper::asinh }
delegating! { /// Inverse hyperbolic cosine.
acosh, crate::reference::single::acosh, double::hyper::acosh }
delegating! { /// Inverse hyperbolic tangent.
atanh, crate::reference::single::atanh, double::hyper::atanh }
delegating! { /// Base-10 logarithm.
log10, crate::reference::single::log10, double::logx::log10 }
delegating! { /// `ln(1 + x)`.
log1p, crate::reference::single::log1p, double::logx::log1p }
delegating! { /// `e^x - 1`.
expm1, crate::reference::single::expm1, double::expm1 }
/// Log-Gamma with the sign of `Gamma`.
///
/// The value takes the double-precision route and rounds once, like
/// [`lgamma`]; the sign is computed on the `f32` argument directly, because it
/// is exact at either precision and widening first would be work for nothing.
pub mod lgamma_r {
    use crate::policy::{Accuracy, Domain};
    use crate::simd::{Simd, map_lanes};

    /// `(lgamma(x), sign(Gamma(x)))` for a vector of `f32` lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> (V, V) {
        let v = V::narrow(crate::kernels::double::gamma::lgamma::eval::<V::Wide, A, D>(x.widen()));
        (
            v,
            map_lanes(x, crate::kernels::double::gamma::lgamma_r::sign_f32),
        )
    }
}

via_double_raw! { /// Log-Gamma.
lgamma, double::gamma::lgamma }
via_double_raw! { /// Gamma.
tgamma, double::gamma::tgamma }
delegating2! { /// Two-argument arc tangent.
atan2, crate::reference::single::atan2, double::invtrig::atan2 }
delegating2! { /// `sqrt(x^2 + y^2)`.
hypot, crate::reference::single::hypot, double::hypot }

delegating2! { /// `x^y`.
pow, crate::reference::single::pow, double::pow }

pub use trig::{cos, sin, sincos, tan};

/// `2^(k/32)` per lane, gathered from the shared 32-entry table.
///
/// The one part of `expf` / `exp2f` that does not vectorise — eight lanes mean
/// eight scalar loads — kept in one place because both functions index the
/// same table the same way. The exponent is folded in with an integer add
/// rather than a second multiply, which is what the pre-subtracted table
/// entries are for.
#[inline(always)]
pub(crate) fn scale32<V: Simd<Elem = f32>>(ki: &<V::Wide as Simd>::Bits) -> V::Wide {
    let mut bits = <<V::Wide as Simd>::Bits as Lanes<u64>>::filled_default();
    for i in 0..<V::Wide as Simd>::LANES {
        let k = ki.as_slice()[i];
        bits.as_mut_slice()[i] = et::TAB[(k % 32) as usize].wrapping_add(k << 47);
    }
    <V::Wide as Simd>::from_bits(bits)
}
