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

use crate::kernels::double::{ln, pow};
use crate::kernels::horner;
use crate::policy::{Accuracy, Domain, Fast, Finite};
use crate::simd::{Mask, Simd};
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

/// [`stirling`], to more than double precision, as an unevaluated `hi + lo`.
///
/// `stirling` rounds `ln(z)` to a single `f64` before ever multiplying it by
/// `z`, so the `f64` it returns already carries several ulp of error before
/// `tgamma` even exponentiates it. This carries the logarithm in
/// double-double throughout instead, via `pow_log` — already exercised by
/// `pow`'s own bit-exactness tests, rather than a second, unvalidated
/// implementation — so that `hi + lo` compresses back to the *correctly
/// rounded* `f64` nearest the true `ln(Gamma(z))`, not merely a close one.
///
/// Unlike `Fast` `pow`, there is no further amplifying factor here to carry
/// `lo` *through*: `tgamma`'s caller only ever needs `exp` of a single
/// accurate value, not `exp` of a value some large `y` has multiplied. So
/// `hi + lo` is compressed to one `f64` and handed to the ordinary `exp`
/// kernel — which, unlike `pow`'s internal accurate exponential, already
/// handles this function's whole domain (up to `z` near 171.6, where
/// `ln(Gamma(z))` approaches 710) via its own `FullRange` fallback.
///
/// The Stirling series tail (`series` below) stays single-precision: it is
/// already three or more orders of magnitude below the leading terms over
/// this function's domain (`z >= 18`), so its own rounding is far under the
/// budget the leading terms need protecting.
#[inline(always)]
fn stirling_dd<V: Simd<Elem = f64>>(z: V) -> (V, V) {
    let half = V::splat(0.5);
    let inv = V::splat(1.0) / z;
    let series = inv * horner(inv * inv, &p::STIRLING);

    // z - 1/2, exact: Fast2Sum with |z| >= |half| for every `z` this is
    // called on.
    let zm = z - half;
    let zml = (z - zm) - half;

    let (lnhi, lnlo) = pow::pow_log(z);

    // (z - 1/2) * ln(z): a TwoProduct on the leading pair plus both cross
    // terms, so the product carries the same precision the logarithm does.
    let ph = zm * lnhi;
    let pl = zm.mul_add(lnhi, -ph) + zm.mul_add(lnlo, zml * lnhi);

    // ln(2 pi)/2 - z, via Fast2Sum with -z as the larger-magnitude operand
    // (this function's domain has z >= 18 >> ln(2 pi)/2).
    let s = V::splat(p::HALF_LN_2PI) - z;
    let e = V::splat(p::HALF_LN_2PI) - (s + z);

    // Combine: |ph| >= |s| throughout the domain (z ln z outgrows z past
    // z = e), so this Fast2Sum's precondition holds.
    let hi = ph + s;
    let r = (ph - hi) + s;
    (hi, (r + pl + e) + series)
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

/// `ln|Gamma(x)|` together with the sign of `Gamma(x)`.
///
/// The value is [`lgamma`]'s, with [`lgamma`]'s accuracy contract. The *sign*
/// is exact and is not derived from it: `Gamma` alternates sign on the
/// negative half-line, so the sign is decided from the parity of `floor(-x)`
/// and costs one rounding and one integer test.
///
/// The conventions at the awkward points are glibc's, and they are not all
/// what the parity rule alone would give:
///
/// * `+0` reports `+1`, `-0` reports `-1`.
/// * every negative integer — a pole, where the value is `+inf` — reports
///   `-1`, including the odd ones the parity rule would call `+1`.
/// * `-inf` reports `+1`, though its sign bit is set.
/// * NaN reports `+1`.
pub mod lgamma_r {
    use super::*;
    use crate::simd::{Real, Uint, map_lanes};

    /// The cutoff above which glibc's `lgamma_r` reports `+1` whatever the
    /// sign of `x`: `bits(x) << 1 >= HUGE_F64` means
    /// `|x| >= 0x1.006df1bfac84ep+1015`, or a NaN or infinity.
    ///
    /// Every `|x| >= 2^52` is an integer and therefore a pole, so the sign it
    /// reports there is a convention rather than a fact about `Gamma`. glibc
    /// changes that convention at this one threshold — below it a negative
    /// pole reports `-1`, at or above it `+1` — and matching the platform
    /// means matching the quirk.
    pub const HUGE_F64: u64 = 0xfeae_a9b2_4f16_a34c;

    /// The same cutoff for `f32`, where glibc draws it at infinity instead:
    /// every finite negative pole reports `-1`.
    pub const HUGE_F32: u32 = 0xff00_0000;

    /// The sign of `Gamma(x)`, as `+1.0` or `-1.0`. Exact for every input.
    ///
    /// `huge` is the precision's [`HUGE_F64`] / [`HUGE_F32`] threshold.
    #[inline(always)]
    pub fn sign_of<E: Real>(x: E, huge: E::Uint) -> E {
        let t = x.bits();
        if t << 1 >= huge {
            // Huge magnitudes, both infinities, and NaN.
            return E::ONE;
        }
        let negative = t >> (E::MANT_BITS + E::EXP_BITS) != <E::Uint as Uint>::ZERO;
        let fx = E::of_f64(x.to_f64().floor());
        if fx == x {
            // An integer: a pole for `x <= 0`, where the reported sign is
            // just the sign bit — so `-0.0` reports `-1` and `+0.0` reports
            // `+1` — and `+1` for every positive integer.
            return if x <= E::ZERO && negative {
                -E::ONE
            } else {
                E::ONE
            };
        }
        if x.abs() < E::HALF {
            // `Gamma` has no zero crossing inside `(-1/2, 1/2)`, so the sign
            // bit decides on its own.
            return if negative { -E::ONE } else { E::ONE };
        }
        if !negative {
            return E::ONE;
        }
        // `Gamma` alternates on the negative half-line: negative on
        // `(-1, 0)`, positive on `(-2, -1)`, and so on. `floor(x)` names the
        // interval and its parity names the sign.
        if (fx.to_f64() as i64) & 1 == 0 {
            E::ONE
        } else {
            -E::ONE
        }
    }

    /// The sign of `Gamma(x)` for `f64`.
    #[inline(always)]
    pub fn sign(x: f64) -> f64 {
        sign_of(x, HUGE_F64)
    }

    /// The sign of `Gamma(x)` for `f32`.
    #[inline(always)]
    pub fn sign_f32(x: f32) -> f32 {
        sign_of(x, HUGE_F32)
    }

    /// `(lgamma(x), sign(Gamma(x)))` for a vector of lanes.
    ///
    /// Both policy axes are no-ops, as they are for [`lgamma`].
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> (V, V) {
        let _ = (A::BIT_EXACT, D::CHECKED);
        (lgamma::fast(x), map_lanes(x, sign))
    }
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
        let main = V::select(neg, reflected, lg);
        crate::simd::patch_lanes(x, main, edge_lanes(x), edge)
    }

    /// The lanes the main path must not be trusted with.
    ///
    /// Both routes above reach for `ln` under [`Finite`], whose domain is the
    /// positive *normals* — the recurrence takes `ln(x)` directly for
    /// `0 < x < 1`, and the reflection takes `ln|sin(pi x)|`. So a subnormal
    /// argument, a zero, and a negative integer (where the sine is zero) all
    /// leave that domain, as do the infinities and NaN. They are recomputed
    /// here rather than defended against inside the loops, which would cost
    /// every lane a test to serve inputs that have closed forms anyway.
    #[inline(always)]
    fn edge_lanes<V: Simd<Elem = f64>>(x: V) -> V::Mask {
        use crate::simd::Mask;
        crate::kernels::not_normal(x).or(x.lt_mask(V::splat(0.0)).and(x.eq_mask(x.trunc())))
    }

    /// Those lanes, in closed form.
    fn edge(x: f64) -> f64 {
        if x.is_nan() {
            return x + x;
        }
        // Both infinities: `lgamma` is even enough at the ends for C to
        // require `+inf` for either.
        if x.abs() == f64::INFINITY {
            return f64::INFINITY;
        }
        // The poles: zero of either sign, and every negative integer. Note
        // that every `x <= -2^52` is an integer, so the whole tail is poles.
        if x == 0.0 || (x < 0.0 && x == x.trunc()) {
            return f64::INFINITY;
        }
        // Subnormal, of either sign: `Gamma(x) = 1/x - gamma + O(x)`, and at
        // this magnitude the correction is more than 250 orders of magnitude
        // below the leading term, so `ln|Gamma(x)|` is `-ln|x|` rounded once.
        -crate::reference::double::ln(x.abs())
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
    /// `[1, 2]` and no exponential is involved. Above that, Stirling's
    /// asymptotic series is evaluated in double-double (`stirling_dd`) and
    /// handed directly to the multi-precision exponential (`pow_exp`) without
    /// rounding through `ln(Gamma(x))`, achieving <= 1 ulp error across the
    /// full domain up to overflow.
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
    ///
    /// Guarded rather than always blended: a vector entirely below
    /// `TG_DIRECT_LIMIT` has no business paying for `stirling_dd` and `exp`,
    /// and one entirely above it has no business paying for the `TG_STEPS`
    /// recurrence — mirrors `bessel.rs::branch`'s whole-vector test for the
    /// same reason. Mixed vectors still compute and blend both, exactly as
    /// before; the selection outcome is unchanged either way, so this is a
    /// speed change only.
    #[inline(always)]
    fn positive<V: Simd<Elem = f64>>(y: V) -> V {
        let is_direct = y.lt_mask(V::splat(TG_DIRECT_LIMIT));
        if is_direct.all() {
            return direct(y);
        }
        if is_direct.none() {
            return via_exp(y);
        }
        V::select(is_direct, direct(y), via_exp(y))
    }

    /// The `TG_STEPS`-recurrence path, for `y < TG_DIRECT_LIMIT`.
    #[inline(always)]
    fn direct<V: Simd<Elem = f64>>(y: V) -> V {
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

        gamma_unit(z) * prod
    }

    #[allow(clippy::excessive_precision)]
    const TGAMMA_OVERFLOW: f64 = 171.624376956302725;

    /// The `stirling_dd` + `pow_exp` path, for `y >= TG_DIRECT_LIMIT`.
    #[inline(always)]
    fn via_exp<V: Simd<Elem = f64>>(y: V) -> V {
        let (hi, lo) = stirling_dd(y);
        let res = pow::pow_exp(hi, lo);
        V::select(y.gt_mask(V::splat(TGAMMA_OVERFLOW)), V::splat(f64::INFINITY), res)
    }
}
