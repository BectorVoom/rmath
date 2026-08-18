//! Sine, cosine and tangent.
//!
//! # `BitExact`
//!
//! Delegates to [`crate::reference::double`], one lane at a time. glibc
//! implements these with the IBM Accurate Portable Math Library routines,
//! whose schedule turns on a 440-entry table and on per-expression
//! fused-multiply-add placement chosen by the compiler; that has not been
//! reproduced in vector form here, so the bit-exact path is bit-exact by
//! delegation rather than by replay. It is exactly as fast as the scalar call
//! it replaces, and no faster.
//!
//! # `Fast`
//!
//! A genuine vector path, and the reason to reach for this configuration:
//! Cody-Waite reduction against a three-part `pi/2`, one degree-7 polynomial
//! in `r^2` for each of sine and cosine, and quadrant selection done with
//! arithmetic rather than a branch. No table, no gather, no per-lane work at
//! all.
//!
//! Both polynomials are always evaluated, even when only one result is wanted.
//! That is deliberate: the quadrant decides *which* of the two is the answer,
//! so selecting between them costs one blend, whereas branching on the
//! quadrant would serialise the lanes and lose the whole point.
//!
//! `Finite` means `|x| < TRIG_LIMIT` (about 1.6e6). Past that the quadrant
//! count no longer multiplies exactly against the split `pi/2` and the reduced
//! argument loses its leading bits; `FullRange` sends those lanes to the
//! reference, `Finite` returns a wrong number.

use crate::kernels::{dispatch, horner, outside};
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Mask, Simd, map_lanes_pair};
use crate::tables::double::poly as p;

/// Lanes the Cody-Waite reduction must not be trusted with.
#[inline(always)]
fn too_large<V: Simd<Elem = f64>>(x: V) -> V::Mask {
    outside(x, p::TRIG_LIMIT)
}

/// `x` reduced to `(r, q)`: `x = r + q * pi/2` with `|r| <= pi/4` and
/// `q` the quadrant index in `0..4`, as a float.
///
/// `q` is computed as `n - floor(n/4) * 4` rather than by converting to an
/// integer and masking. Both are exact for every `n` in range, but this one is
/// three vector operations and no round trip through the lane array — and the
/// round trip is what the whole `Fast` policy exists to avoid.
#[inline(always)]
fn reduce<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let n = (x * V::splat(p::TWO_OVER_PI)).round_ties_even();
    // Three fused steps: each `n * PIO2[i]` is exact, so the only error is the
    // truncation of pi/2 itself, some 2^-150.
    let r = n.mul_add(
        V::splat(-p::PIO2[2]),
        n.mul_add(V::splat(-p::PIO2[1]), n.mul_add(V::splat(-p::PIO2[0]), x)),
    );
    let q = n - (n * V::splat(0.25)).floor() * V::splat(4.0);
    (r, q)
}

/// `(sin(r), cos(r))` on `|r| <= pi/4`.
#[inline(always)]
fn kernels<V: Simd<Elem = f64>>(r: V) -> (V, V) {
    let s = r * r;
    (r * horner(s, &p::SIN), horner(s, &p::COS))
}

/// Apply the quadrant to `(sin r, cos r)`, giving `(sin x, cos x)`.
#[inline(always)]
fn place<V: Simd<Elem = f64>>(q: V, sr: V, cr: V) -> (V, V) {
    let one = V::splat(1.0);
    let two = V::splat(2.0);
    // Odd quadrants swap the two kernels; that is the same test for both.
    let swap = q.eq_mask(one).or(q.eq_mask(V::splat(3.0)));
    // sin is negative in quadrants 2 and 3; cos in 1 and 2.
    let sin_neg = q.ge_mask(two);
    let cos_neg = q.eq_mask(one).or(q.eq_mask(two));

    let sin = V::select(swap, cr, sr);
    let cos = V::select(swap, sr, cr);
    let flip = |v: V, m| V::select(m, -v, v);
    (flip(sin, sin_neg), flip(cos, cos_neg))
}

/// `sin(x)` and `cos(x)` together, vectorised.
#[inline(always)]
fn both<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let (r, q) = reduce(x);
    let (sr, cr) = kernels(r);
    place(q, sr, cr)
}

/// Sine.
pub mod sin {
    use super::*;

    /// `sin(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::sin, |x| both(x).0, too_large)
    }
}

/// Cosine.
pub mod cos {
    use super::*;

    /// `cos(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::cos, |x| both(x).1, too_large)
    }
}

/// Sine and cosine of the same argument.
///
/// The reason this exists as its own function rather than two calls: the
/// argument reduction and both polynomials are already shared, so the pair
/// costs one blend more than either alone. Calling `sin` and `cos` separately
/// does all of that work twice.
pub mod sincos {
    use super::*;

    /// `(sin(x), cos(x))` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> (V, V) {
        if A::BIT_EXACT {
            return map_lanes_pair(x, reference::sincos);
        }
        let (s, c) = both(x);
        if !D::CHECKED {
            return (s, c);
        }
        let bad = too_large(x);
        (
            crate::simd::patch_lanes(x, s, bad, reference::sin),
            crate::simd::patch_lanes(x, c, bad, reference::cos),
        )
    }
}

/// Tangent.
///
/// Formed as a ratio of the same two kernels rather than from a tangent
/// polynomial of its own. That costs a division, and it is still the better
/// trade: a direct `tan` polynomial needs a much higher degree to hold its
/// accuracy as `r` approaches `pi/4`, where `tan` is steep, and the ratio
/// keeps the relative error bounded across the whole quadrant including near
/// the pole, which is where a `tan` caller usually is.
pub mod tan {
    use super::*;

    /// `tan(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::tan, fast, too_large)
    }

    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let (r, q) = reduce(x);
        let (sr, cr) = kernels(r);
        // Odd quadrants give the cotangent, negated.
        let odd = q.eq_mask(V::splat(1.0)).or(q.eq_mask(V::splat(3.0)));
        let num = V::select(odd, -cr, sr);
        let den = V::select(odd, sr, cr);
        num / den
    }
}
