//! Sine, cosine and tangent, single precision.
//!
//! `BitExact` delegates to the platform's routines, one lane at a time:
//! `sinf`/`cosf` are not correctly rounded on this platform, so there is no
//! stronger claim than parity to make, and widening into the double-precision
//! kernel would cost the round trip for nothing extra.
//!
//! `Fast` is native — the same shape as [`crate::kernels::double::trig`]'s,
//! at `f32` precision throughout: Cody-Waite reduction against a three-part
//! `pi/2`, one polynomial each for sine and cosine, quadrant placement by
//! arithmetic rather than branch. No widening, no gather, no per-lane work.
//!
//! `Finite` means `|x| < TRIG_LIMIT` (about 804 — see the constant's own doc
//! for why this is narrower than the naive `f64`-style derivation gives).
//! Past that the quadrant count no longer multiplies exactly against the
//! split `pi/2`; `FullRange` sends those lanes to the reference, `Finite`
//! returns a wrong number.
//!
//! # Near a zero of `sin`/`cos`, or a pole of `tan`
//!
//! The reduction's own absolute error is tiny — under `2^-32` at the domain
//! edge, fifty times below an `f32` ulp — but "ulp" is a *relative* measure,
//! and near a zero of `sin` or `cos` the true value is arbitrarily small, so
//! any nonzero absolute error is unbounded in ulp terms there. That is
//! inherent to evaluating the function at all, not a property of this
//! kernel: `tests/ulp_scan.rs` at high sample density will find inputs where
//! the reported ulp count reaches into the thousands, exactly as `lgamma`
//! does near its own zeros on the negative half-line, for the same reason.
//! `tests/accuracy.rs` measures at the sample density that matters in
//! practice, where this kernel is tight — see its own comment.

use crate::kernels::{horner, outside};
use crate::policy::{Accuracy, Domain};
use crate::reference::single as reference;
use crate::simd::{Mask, Simd, map_lanes_pair};
use crate::tables::single::poly as p;

/// Lanes the Cody-Waite reduction must not be trusted with.
#[inline(always)]
fn too_large<V: Simd<Elem = f32>>(x: V) -> V::Mask {
    outside(x, p::TRIG_LIMIT)
}

/// `x` reduced to `(r, q)`: `x = r + q * pi/2` with `|r| <= pi/4` and `q` the
/// quadrant index in `0..4`, as a float. See the `f64` kernel for why `q` is
/// computed by arithmetic rather than an integer round trip.
#[inline(always)]
fn reduce<V: Simd<Elem = f32>>(x: V) -> (V, V) {
    let n = (x * V::splat(p::TWO_OVER_PI)).round_ties_even();
    let r = n.mul_add(
        V::splat(-p::PIO2[2]),
        n.mul_add(V::splat(-p::PIO2[1]), n.mul_add(V::splat(-p::PIO2[0]), x)),
    );
    let q = n - (n * V::splat(0.25)).floor() * V::splat(4.0);
    (r, q)
}

/// `(sin(r), cos(r))` on `|r| <= pi/4`.
#[inline(always)]
fn kernels<V: Simd<Elem = f32>>(r: V) -> (V, V) {
    let s = r * r;
    (r * horner(s, &p::SIN), horner(s, &p::COS))
}

/// Apply the quadrant to `(sin r, cos r)`, giving `(sin x, cos x)`.
#[inline(always)]
fn place<V: Simd<Elem = f32>>(q: V, sr: V, cr: V) -> (V, V) {
    let one = V::splat(1.0);
    let two = V::splat(2.0);
    let swap = q.eq_mask(one).or(q.eq_mask(V::splat(3.0)));
    let sin_neg = q.ge_mask(two);
    let cos_neg = q.eq_mask(one).or(q.eq_mask(two));

    let sin = V::select(swap, cr, sr);
    let cos = V::select(swap, sr, cr);
    let flip = |v: V, m| V::select(m, -v, v);
    (flip(sin, sin_neg), flip(cos, cos_neg))
}

/// `sin(x)` and `cos(x)` together, vectorised.
#[inline(always)]
fn both<V: Simd<Elem = f32>>(x: V) -> (V, V) {
    let (r, q) = reduce(x);
    let (sr, cr) = kernels(r);
    place(q, sr, cr)
}

/// Sine.
pub mod sin {
    use super::*;

    /// `sin(x)` for a vector of `f32` lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            return crate::simd::map_lanes(x, reference::sin);
        }
        let y = both(x).0;
        if D::CHECKED {
            crate::simd::patch_lanes(x, y, too_large(x), reference::sin)
        } else {
            y
        }
    }
}

/// Cosine.
pub mod cos {
    use super::*;

    /// `cos(x)` for a vector of `f32` lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            return crate::simd::map_lanes(x, reference::cos);
        }
        let y = both(x).1;
        if D::CHECKED {
            crate::simd::patch_lanes(x, y, too_large(x), reference::cos)
        } else {
            y
        }
    }
}

/// Sine and cosine of the same argument.
pub mod sincos {
    use super::*;

    /// `(sin(x), cos(x))` for a vector of `f32` lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> (V, V) {
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
pub mod tan {
    use super::*;

    /// `tan(x)` for a vector of `f32` lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            return crate::simd::map_lanes(x, reference::tan);
        }
        let y = fast(x);
        if D::CHECKED {
            crate::simd::patch_lanes(x, y, too_large(x), reference::tan)
        } else {
            y
        }
    }

    #[inline(always)]
    fn fast<V: Simd<Elem = f32>>(x: V) -> V {
        let (r, q) = reduce(x);
        let (sr, cr) = kernels(r);
        let odd = q.eq_mask(V::splat(1.0)).or(q.eq_mask(V::splat(3.0)));
        let num = V::select(odd, -cr, sr);
        let den = V::select(odd, sr, cr);
        num / den
    }
}
