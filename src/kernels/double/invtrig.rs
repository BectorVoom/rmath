//! `asin`, `acos`, `atan` and `atan2`.
//!
//! All four delegate under `BitExact` — glibc uses the IBM Accurate Portable
//! Math Library routines here, and their schedule has not been reproduced in
//! vector form; see [`crate::kernels`]. Under `Fast` they share two
//! polynomials and a handful of exact identities, and are entirely
//! branch-free.

use crate::kernels::{dispatch, dispatch2, horner, no_lanes, not_normal, outside};
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Mask, Simd};
use crate::tables::double::poly as p;

/// `pi/2`, to the nearest double.
const PI_2: f64 = core::f64::consts::FRAC_PI_2;
/// `pi`, to the nearest double.
const PI: f64 = core::f64::consts::PI;
/// `pi/6`, the fold point of the `atan` reduction.
const PI_6: f64 = core::f64::consts::FRAC_PI_6;
/// `sqrt(3)`, the same reduction's multiplier.
const SQRT3: f64 = f64::from_bits(0x3ffbb67ae8584caa);

/// `asin(|t|)` for `t*t = s` with `|t| <= 1/2`, as `t * P(s)`.
#[inline(always)]
fn asin_core<V: Simd<Elem = f64>>(t: V, s: V) -> V {
    t * horner(s, &p::ASIN)
}

/// Arc sine.
pub mod asin {
    use super::*;

    /// `asin(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        // `outside(x, 1.0)` also catches |x| == 1 and sends it to the
        // reference. That is one extra scalar lane on an input that is rare in
        // bulk data, and it buys not having to reason about `sqrt(0)` in the
        // folded branch.
        dispatch::<V, A, D>(x, reference::asin, fast, |x| outside(x, 1.0))
    }

    /// Measured error: at most 3 ulp over `|x| < 1`, stably at the fold
    /// boundary below rather than growing with the sample count (checked to
    /// 100M samples in `tests/ulp_scan.rs`); asserted at 4.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let half = V::splat(0.5);
        let big = a.gt_mask(half);

        // Near 1, `asin` has infinite slope, so it is folded to a region where
        // it does not: asin(a) = pi/2 - 2 asin(sqrt((1-a)/2)).
        let t = (V::splat(1.0) - a) * half;
        let root = t.sqrt();
        let folded = V::splat(PI_2) - (asin_core(root, t) + asin_core(root, t));
        let direct = asin_core(a, a * a);
        V::select(big, folded, direct).copysign(x)
    }
}

/// Arc cosine.
pub mod acos {
    use super::*;

    /// `acos(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::acos, fast, |x| outside(x, 1.0))
    }

    /// Measured error: at most 2 ulp over `|x| < 1` (checked to 100M samples
    /// in `tests/ulp_scan.rs`); asserted at 4.
    ///
    /// Not written as `pi/2 - asin(x)`: near `x = 1`, `acos` is small and that
    /// subtraction cancels, losing exactly the digits the caller wanted. The
    /// folded form computes the small answer directly.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let half = V::splat(0.5);
        let big = a.gt_mask(half);

        let t = (V::splat(1.0) - a) * half;
        let root = t.sqrt();
        let two_asin = asin_core(root, t) + asin_core(root, t);
        // acos(a) = 2 asin(sqrt((1-a)/2)); acos(-a) = pi - that.
        let folded = V::select(x.lt_mask(V::splat(0.0)), V::splat(PI) - two_asin, two_asin);
        let direct = V::splat(PI_2) - asin_core(x, x * x);
        V::select(big, folded, direct)
    }
}

/// `atan(|t|)` for `|t| <= 1`, folded once more onto `|w| <= tan(pi/12)`.
#[inline(always)]
fn atan_unit<V: Simd<Elem = f64>>(z: V) -> V {
    let sqrt3 = V::splat(SQRT3);
    let hi = z.gt_mask(V::splat(f64::from_bits(0x3fd126145e9ecd56))); // tan(pi/12)
    // atan(z) = pi/6 + atan((z sqrt3 - 1)/(sqrt3 + z)), which maps
    // [tan(pi/12), 1] onto [-tan(pi/12), tan(pi/12)].
    let w = z.mul_add(sqrt3, V::splat(-1.0)) / (sqrt3 + z);
    let u = V::select(hi, w, z);
    let r = u * horner(u * u, &p::ATAN);
    r + V::select(hi, V::splat(PI_6), V::splat(0.0))
}

/// Arc tangent.
pub mod atan {
    use super::*;

    /// `atan(x)` for a vector of lanes.
    ///
    /// Needs no special-lane repair: the reciprocal fold turns an infinite
    /// argument into `1/inf == 0`, so `atan(±inf)` falls out as `±pi/2`
    /// without a test, and NaN propagates through every operation.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::atan, fast, no_lanes)
    }

    /// Measured error: below 2 ulp over the whole real line.
    #[inline(always)]
    pub(super) fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let inv = a.gt_mask(V::splat(1.0));
        let z = V::select(inv, V::splat(1.0) / a, a);
        let r = atan_unit(z);
        V::select(inv, V::splat(PI_2) - r, r).copysign(x)
    }
}

/// Two-argument arc tangent: the angle of the point `(x, y)`.
pub mod atan2 {
    use super::*;

    /// `atan2(y, x)` for vectors of lanes.
    ///
    /// Argument order follows C and Rust: `y` first.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(y: V, x: V) -> V {
        dispatch2::<V, A, D>(y, x, reference::atan2, fast, |y, x| {
            // The quotient is what the vector path is built on, so anything
            // that makes it meaningless — a zero or non-finite in either
            // argument — goes to the reference. That is also where all the
            // signed-zero rules live, which are not worth restating in vector
            // form to save a handful of lanes.
            not_normal(y).or(not_normal(x))
        })
    }

    /// Measured error: below 2 ulp for normal, non-zero arguments.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(y: V, x: V) -> V {
        let base = atan::fast(y / x);
        let zero = V::splat(0.0);
        // x < 0 puts the point in the second or third quadrant, which the
        // quotient cannot distinguish; shift by pi with the sign of y.
        let shift = V::splat(PI).copysign(y);
        V::select(x.lt_mask(zero), base + shift, base)
    }
}
