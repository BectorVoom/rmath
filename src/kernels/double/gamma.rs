//! The Gamma functions: `lgamma` and `tgamma`.
//!
//! # Why both policies run the same code here
//!
//! Rust has no `f64::tgamma` or `f64::lgamma`. There is therefore no call the
//! caller was already making for `BitExact` to be bit-exact *to* — the
//! guarantee the rest of this crate offers is "substituting rmath cannot
//! change your result", and for Gamma there is nothing to substitute for. So,
//! as with [`crate::kernels::double::cbrt`], the accuracy axis is accepted and
//! has no effect: one implementation, documented by its measured error rather
//! than by a bit-exactness claim it could not honestly make.
//!
//! `tests/accuracy.rs` measures both against the platform's `tgamma` and
//! `lgamma` through `extern "C"` and holds them to the bounds quoted below.
//!
//! # The algorithm
//!
//! Everything is reduced to `[1, 2]`, where a single minimax polynomial does
//! the work, using the recurrence `Gamma(z + 1) = z Gamma(z)` — as a sum of
//! logarithms for `lgamma`, as a running product for `tgamma`. Large
//! arguments take Stirling's asymptotic series instead, and negative ones go
//! through Euler's reflection formula first.
//!
//! Reducing to `[1, 2]` rather than up to the Stirling cutoff is what makes
//! `lgamma` usable near `x = 1` and `x = 2`. `lgamma` has zeros there, and the
//! Stirling route reaches them as a difference of two numbers near 12 — so it
//! returns something around 1e-15 where the answer is around 1e-6, which is no
//! correct digits at all. The polynomial is fitted with those zeros factored
//! out, so it reproduces them exactly.
//!
//! Every loop runs a fixed number of iterations with the still-unreduced lanes
//! selected each time. That costs what the worst lane costs, and it keeps the
//! vector intact, which is the trade this crate makes everywhere.

use crate::kernels::double::{exp, ln};
use crate::kernels::horner;
use crate::policy::{Accuracy, Domain, Fast, Finite};
use crate::simd::Simd;
use crate::tables::double::poly as p;

/// `ln(pi)`.
const LN_PI: f64 = f64::from_bits(0x3ff250d048e7a1bd);
/// `pi`.
const PI: f64 = core::f64::consts::PI;

/// Recurrence steps `lgamma` takes to reach `[1, 2]` from below its cutoff.
///
/// The cutoff is 8 and each step subtracts one, so seven steps reach `(0, 2]`
/// from anything under it.
const LG_STEPS: usize = 7;

/// Recurrence steps `tgamma` takes.
///
/// More than `lgamma` uses, because for `tgamma` a step is one multiply rather
/// than one logarithm — so extending the exactly-reduced range is nearly free,
/// and every argument it covers avoids the exponential entirely.
const TG_STEPS: usize = 16;

/// Above this, `tgamma` has to go through `exp(lgamma(x))`.
const TG_DIRECT_LIMIT: f64 = 18.0;

/// `ln(Gamma(z))` for `z` at or above the Stirling cutoff.
#[inline(always)]
fn stirling<V: Simd<Elem = f64>>(z: V) -> V {
    let inv = V::splat(1.0) / z;
    let series = inv * horner(inv * inv, &p::STIRLING);
    let lnz = ln::eval::<V, Fast, Finite>(z);
    (z - V::splat(0.5)).mul_add(lnz, V::splat(p::HALF_LN_2PI) - z) + series
}

/// `lgamma(z)` for `z` in `[1, 2]`, with its zeros reproduced exactly.
#[inline(always)]
fn lgamma_unit<V: Simd<Elem = f64>>(z: V) -> V {
    let t = z - V::splat(1.0);
    t * (t - V::splat(1.0)) * horner(t, &p::LGAMMA)
}

/// `Gamma(z)` for `z` in `[1, 2]`.
#[inline(always)]
fn gamma_unit<V: Simd<Elem = f64>>(z: V) -> V {
    horner(z - V::splat(1.0), &p::GAMMA)
}

/// `sin(pi * x)`, accurate for every `x`.
///
/// Reducing modulo 1 first is what makes this work where a plain
/// `sin(PI * x)` would not: `PI * x` rounds away the very digits the
/// reflection formula needs when `|x|` is large, and near a negative integer
/// those digits *are* the answer.
#[inline(always)]
fn sin_pi<V: Simd<Elem = f64>>(x: V) -> V {
    let n = x.round_ties_even();
    let r = x - n; // |r| <= 1/2
    let a = r.abs();
    let pi = V::splat(PI);

    // |sin(pi r)| from the sine polynomial for |r| <= 1/4, and from the cosine
    // one at pi(1/2 - |r|) for the rest — both keep the argument inside the
    // |t| <= pi/4 the polynomials were fitted on.
    let t1 = pi * a;
    let s1 = t1 * horner(t1 * t1, &p::SIN);
    let t2 = pi * (V::splat(0.5) - a);
    let s2 = horner(t2 * t2, &p::COS);
    let mag = V::select(a.le_mask(V::splat(0.25)), s1, s2);

    // sin(pi(n + r)) = (-1)^n sin(pi r).
    let even = n.eq_mask((n * V::splat(0.5)).round_ties_even() * V::splat(2.0));
    let signed = mag.copysign(r);
    V::select(even, signed, -signed)
}

/// The natural logarithm of the absolute value of Gamma.
pub mod lgamma {
    use super::*;

    /// `lgamma(x)` for a vector of lanes.
    ///
    /// Both policy axes are no-ops; `FullRange` costs nothing because the
    /// poles and the negative half-line are handled in the main path.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        fast(x)
    }

    #[inline(always)]
    pub(super) fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let zero = V::splat(0.0);
        let one = V::splat(1.0);
        let neg = x.lt_mask(zero);
        // Reflect first, so everything below only ever sees a positive
        // argument and there is one code path rather than two.
        let y = V::select(neg, one - x, x);
        let lg = positive(y);

        // lgamma(x) = ln(pi) - ln|sin(pi x)| - lgamma(1 - x).
        let reflected = V::splat(LN_PI) - ln::eval::<V, Fast, Finite>(sin_pi(x).abs()) - lg;
        V::select(neg, reflected, lg)
    }

    /// `lgamma(y)` for `y > 0`.
    #[inline(always)]
    fn positive<V: Simd<Elem = f64>>(y: V) -> V {
        let zero = V::splat(0.0);
        let one = V::splat(1.0);
        let two = V::splat(2.0);

        // Walk down to (0, 2], accumulating the logarithms stepped over.
        let mut z = y;
        let mut acc = zero;
        for _ in 0..LG_STEPS {
            let above = z.gt_mask(two);
            z = z - V::select(above, one, zero);
            acc = acc + V::select(above, ln::eval::<V, Fast, Finite>(z), zero);
        }
        // One step up for (0, 1), which the loop above cannot reach.
        let below = z.lt_mask(one);
        acc = acc - V::select(below, ln::eval::<V, Fast, Finite>(z), zero);
        z = z + V::select(below, one, zero);

        let small = lgamma_unit(z) + acc;
        V::select(y.ge_mask(V::splat(p::LGAMMA_CUTOFF)), stirling(y), small)
    }
}

/// The Gamma function itself.
pub mod tgamma {
    use super::*;

    /// `tgamma(x)` for a vector of lanes.
    ///
    /// Measured error: a few ulp for `|x| < 18`, where the recurrence reaches
    /// `[1, 2]` and no exponential is involved. Above that it is
    /// `exp(lgamma(x))`, and the error grows with the argument — an absolute
    /// error of one ulp in `lgamma(170)`, which is about 700, is a relative
    /// error near 1e-13 in the answer. That is inherent to reaching `tgamma`
    /// through a logarithm, and `tests/accuracy.rs` states the measured bound
    /// for each range rather than one number for both.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        let zero = V::splat(0.0);
        let one = V::splat(1.0);
        let neg = x.lt_mask(zero);
        let y = V::select(neg, one - x, x);
        let g = positive(y);

        // Gamma(x) = pi / (sin(pi x) Gamma(1 - x)). When Gamma(1 - x)
        // overflows the quotient is zero, which is the correct limit, so the
        // large-|x| case needs no test of its own.
        let reflected = V::splat(PI) / (sin_pi(x) * g);
        V::select(neg, reflected, g)
    }

    /// `Gamma(y)` for `y > 0`.
    #[inline(always)]
    fn positive<V: Simd<Elem = f64>>(y: V) -> V {
        let zero = V::splat(0.0);
        let one = V::splat(1.0);
        let two = V::splat(2.0);

        let mut z = y;
        let mut prod = one;
        for _ in 0..TG_STEPS {
            let above = z.gt_mask(two);
            z = z - V::select(above, one, zero);
            prod = prod * V::select(above, z, one);
        }
        let below = z.lt_mask(one);
        prod = prod / V::select(below, z, one);
        z = z + V::select(below, one, zero);

        let direct = gamma_unit(z) * prod;
        let viaexp = exp::eval::<V, Fast, Finite>(super::lgamma::fast(y));
        V::select(y.lt_mask(V::splat(TG_DIRECT_LIMIT)), direct, viaexp)
    }
}
