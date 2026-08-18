//! The functions IEEE-754 pins down exactly.
//!
//! Rounding to an integer, sign and exponent manipulation, `fmod`,
//! `remainder`, `sqrt`. What these have in common is that there is no
//! approximation anywhere in them: the mathematically correct result is always
//! representable, or (for `sqrt`) IEEE-754 requires it to be correctly
//! rounded. So there is no reference *algorithm* to reproduce — any
//! implementation that is correct is automatically bit-identical to the
//! platform's, on every input, forever. That is a much stronger footing than
//! the rest of the crate stands on, and it is why:
//!
//! * these are written once, generic over precision as well as width, rather
//!   than once per precision like the transcendentals;
//! * both policy axes are genuine no-ops — there is no cheaper approximation
//!   worth having and no special case for `FullRange` to repair, because the
//!   special cases are handled in the main path at no cost;
//! * bit-exactness here is not a claim about this platform's `libm`.
//!
//! Everything in [`floor`], [`ceil`], [`trunc`], [`round`], [`copysign`] and
//! [`sqrt`] is branch-free vector code. [`ldexp`] takes a vector main path and
//! repairs only the lanes that cross into the subnormal range.
//! [`frexp`], [`nextafter`], [`fmod`] and [`remainder`] are per-lane: they are
//! integer work on the exponent field, and writing them lane-at-a-time keeps
//! them branch-free, which is what lets LLVM vectorise them anyway.

use crate::policy::{Accuracy, Domain};
use crate::simd::{Lanes, Real, Simd, Uint, map_lanes_pair, map_lanes2};

/// `2^k` as an element, for `k` in `-(EXP_BIAS - 1) ..= EXP_BIAS`.
///
/// Built straight into the exponent field, so it is exact and needs no table.
#[inline(always)]
fn two_pow<E: Real>(k: i32) -> E {
    E::of_bits(E::Uint::from_u32((E::EXP_BIAS + k) as u32) << E::MANT_BITS)
}

/// The exponent field, shifted into place.
#[inline(always)]
fn exp_field<E: Real>() -> E::Uint {
    (E::Uint::from_u32((1u32 << E::EXP_BITS) - 1)) << E::MANT_BITS
}

/// True if `x` is NaN. Spelled from the IEEE rule rather than `is_nan`, which
/// is not available through [`Real`].
#[inline(always)]
fn is_nan<E: Real>(x: E) -> bool {
    #[allow(clippy::eq_op)] // the IEEE definition of NaN, not a mistake
    {
        x != x
    }
}

macro_rules! unary_vector {
    ($(#[$doc:meta])* $name:ident, $body:expr) => {
        $(#[$doc])*
        pub mod $name {
            use super::*;

            #[doc = concat!("`", stringify!($name), "(x)` for a vector of lanes.")]
            #[inline(always)]
            pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V {
                let _ = (A::BIT_EXACT, D::CHECKED);
                let f: fn(V) -> V = $body;
                f(x)
            }
        }
    };
}

unary_vector! {
    /// Largest integer not greater than `x`.
    floor, |x| x.floor()
}

unary_vector! {
    /// Smallest integer not less than `x`.
    ceil, |x| x.ceil()
}

unary_vector! {
    /// `x` truncated towards zero.
    trunc, |x| x.trunc()
}

unary_vector! {
    /// `sqrt(x)`, correctly rounded.
    ///
    /// Both policy axes are no-ops, and unusually that needs no apology:
    /// IEEE-754 *requires* `sqrt` to be correctly rounded, and every target
    /// this crate runs on implements it as a single instruction that already
    /// handles zero, infinity, NaN and negatives to specification. There is no
    /// reference to match — the hardware is the reference.
    sqrt, |x| x.sqrt()
}

/// `x` rounded to the nearest integer, ties away from zero.
///
/// Ties *away*, matching `f64::round` and C's `round` — not the ties-to-even
/// that hardware gives you. The difference shows up on exactly the inputs a
/// casual test never generates, so it is built explicitly rather than borrowed
/// from [`Simd::round_ties_even`].
pub mod round {
    use super::*;

    /// `round(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        let t = x.trunc();
        // `x - t` is exact (Sterbenz), and is zero for every input that is
        // already integral — which includes infinities, every |x| >= 2^52, and
        // both zeros. For NaN it is NaN, which compares false, so the step
        // below leaves NaN untouched with its payload intact.
        let frac = x - t;
        let half = V::splat(V::Elem::HALF);
        let bump = V::select(
            frac.abs().ge_mask(half),
            V::splat(V::Elem::ONE).copysign(x),
            V::splat(V::Elem::ZERO),
        );
        t + bump
    }
}

/// The magnitude of `x` with the sign of `y`.
pub mod copysign {
    use super::*;

    /// `copysign(x, y)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        x.copysign(y)
    }
}

// ---------------------------------------------------------------------------
// ldexp
// ---------------------------------------------------------------------------

/// `x * 2^n`, with `n` carried in the lanes of a float vector.
///
/// The exponent is a float rather than an integer vector because the crate has
/// exactly one lane type and threading a parallel integer vector type through
/// the whole trait to serve two functions would cost every other function
/// clarity. `n` is used as if by truncation towards zero; values outside
/// `±2^31` saturate, which is the same thing the C `int` argument does.
pub mod ldexp {
    use super::*;

    /// The scalar form, correct for every input.
    pub fn scalar<E: Real>(x: E, n: E) -> E {
        // Anything without a meaningful exponent to shift is returned as it
        // stands: zeros, infinities and NaN are all fixed points of `ldexp`.
        if x == E::ZERO || is_nan(x) || x.abs() == E::INFINITY {
            return x;
        }
        let limit = (E::EXP_BIAS + E::MANT_BITS as i32 + 2) as f64;
        let nf = n.to_f64().trunc().clamp(-limit * 4.0, limit * 4.0);
        let mut k = nf as i32;

        // Three steps rather than one, so that a shift larger than the
        // exponent range still lands on the right infinity or zero instead of
        // wrapping the exponent field. Scaling up before down also keeps a
        // result that ends up subnormal from being rounded twice.
        let up = E::EXP_BIAS;
        let down = -(E::EXP_BIAS - 1);
        let mut y = x;
        if k > up {
            y = y * two_pow::<E>(up);
            k -= up;
            if k > up {
                y = y * two_pow::<E>(up);
                k -= up;
                if k > up {
                    k = up;
                }
            }
        } else if k < down {
            // Down in two hops of `down + MANT + 1`: the extra significand
            // width means the intermediate is still normal, so the single
            // rounding happens on the last multiply.
            let step = two_pow::<E>(down) * (E::TWO_POW_MANT + E::TWO_POW_MANT);
            let back = -down - (E::MANT_BITS as i32 + 1);
            y = y * step;
            k += back;
            if k < down {
                y = y * step;
                k += back;
                if k < down {
                    k = down;
                }
            }
        }
        y * two_pow::<E>(k)
    }

    /// `ldexp(x, n)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, n: V) -> V {
        let _ = A::BIT_EXACT;
        if !D::CHECKED {
            // The caller promises a shift that keeps the result normal, which
            // is the whole of `ldexp` for a workload that is rescaling data
            // rather than probing the exponent range. One multiply.
            return x * scale_vector(n);
        }
        map_lanes2(x, n, scalar)
    }

    /// `2^n` per lane, for `n` already known to be in range.
    #[inline(always)]
    fn scale_vector<V: Simd>(n: V) -> V {
        let ns = n.to_array();
        let mut bits = V::Bits::filled_default();
        for i in 0..V::LANES {
            let k = ns.as_slice()[i].to_f64() as i32;
            bits.as_mut_slice()[i] =
                <V::Elem as Real>::Uint::from_u32((<V::Elem as Real>::EXP_BIAS + k) as u32)
                    << <V::Elem as Real>::MANT_BITS;
        }
        V::from_bits(bits)
    }
}

// ---------------------------------------------------------------------------
// frexp
// ---------------------------------------------------------------------------

/// Split `x` into a significand in `[0.5, 1)` and a power of two.
///
/// Returns `(fraction, exponent)` with `x == fraction * 2^exponent`. The
/// exponent comes back in float lanes for the same reason [`ldexp`] takes it
/// that way, and it is always an exact small integer.
pub mod frexp {
    use super::*;

    /// The scalar form, correct for every input.
    pub fn scalar<E: Real>(x: E) -> (E, E) {
        // Zero, infinity and NaN have no significand to normalise; C says the
        // exponent is unspecified for them, and every libm returns 0.
        if x == E::ZERO || is_nan(x) || x.abs() == E::INFINITY {
            return (x, E::ZERO);
        }
        let mut bits = x.bits();
        let mut e = 0i32;
        if x.abs() < E::MIN_POSITIVE {
            // Subnormal: scale into the normal range first, then correct.
            // Exact, because multiplying by a power of two only shifts.
            bits = (x * E::TWO_POW_MANT).bits();
            e -= E::MANT_BITS as i32;
        }
        let raw = ((bits >> E::MANT_BITS).as_u32() & ((1u32 << E::EXP_BITS) - 1)) as i32;
        e += raw - (E::EXP_BIAS - 1);
        // Replace the exponent field with the one for [0.5, 1).
        let frac = E::of_bits(
            (bits & !exp_field::<E>())
                | (E::Uint::from_u32((E::EXP_BIAS - 1) as u32) << E::MANT_BITS),
        );
        (frac, E::of_f64(e as f64))
    }

    /// `frexp(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> (V, V) {
        let _ = (A::BIT_EXACT, D::CHECKED);
        map_lanes_pair(x, scalar)
    }
}

// ---------------------------------------------------------------------------
// nextafter
// ---------------------------------------------------------------------------

/// The next representable value after `x` in the direction of `y`.
pub mod nextafter {
    use super::*;

    /// The scalar form, correct for every input.
    pub fn scalar<E: Real>(x: E, y: E) -> E {
        if is_nan(x) || is_nan(y) {
            return x + y; // propagate a payload, and quieten a signalling NaN
        }
        if x == y {
            // Including `nextafter(+0, -0) == -0`: the direction wins, which is
            // what C requires and what a plain `return x` would get wrong.
            return y;
        }
        if x == E::ZERO {
            // Step off zero into the smallest subnormal, signed towards `y`.
            return E::of_bits(E::Uint::ONE).copysign(y);
        }
        // Away from zero when `y` lies further out in the same direction as
        // `x`'s sign, towards zero otherwise. Magnitude and bit pattern are
        // monotone together, so this is a single integer step either way.
        let out = (x < y) == (x > E::ZERO);
        let step = if out {
            E::Uint::ONE
        } else {
            E::Uint::ZERO.wrapping_sub(E::Uint::ONE)
        };
        E::of_bits(x.bits().wrapping_add(step))
    }

    /// `nextafter(x, y)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        map_lanes2(x, y, scalar)
    }
}

// ---------------------------------------------------------------------------
// fmod and remainder
// ---------------------------------------------------------------------------

/// `x` reduced modulo `y`, with the sign of `x`.
pub mod fmod {
    use super::*;

    /// The scalar form, correct for every input.
    pub fn scalar<E: Real>(x: E, y: E) -> E {
        reduce(x, y).0
    }

    /// `fmod(x, y)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        map_lanes2(x, y, scalar)
    }
}

/// The IEEE-754 remainder: `x - y * n`, with `n` the nearest integer to `x/y`.
///
/// Differs from [`fmod`] in the rounding of the quotient — nearest, ties to
/// even, rather than towards zero — so the result can have either sign and
/// satisfies `|r| <= |y| / 2`.
pub mod remainder {
    use super::*;

    /// The scalar form, correct for every input.
    pub fn scalar<E: Real>(x: E, y: E) -> E {
        let (r, odd) = reduce(x, y);
        if is_nan(r) || y.abs() == E::INFINITY {
            return r;
        }
        let ay = y.abs();
        let ar = r.abs();
        // `ar + ar` rather than `ay * 0.5`: halving `ay` is inexact when `ay`
        // is subnormal, and the doubling only overflows when `ar` is above
        // half of MAX, in which case `2*ar > ay` is true anyway.
        let two = ar + ar;
        let over = two > ay || (two == ay && odd);
        if over {
            return (ar - ay).copysign(x);
        }
        r
    }

    /// `remainder(x, y)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        map_lanes2(x, y, scalar)
    }
}

/// `(fmod(x, y), quotient is odd)`.
///
/// The shared core of `fmod` and `remainder`. It is exact, and it is exact for
/// a reason worth stating: the divisor `d` only ever takes the values
/// `|y| * 2^j`, so doubling and halving it are exact; and a subtraction only
/// happens when `d <= r < 2 * d`, where Sterbenz's lemma makes `r - d` exact
/// too. No rounding occurs anywhere, which is why this needs no reference to
/// agree with.
#[inline(always)]
fn reduce<E: Real>(x: E, y: E) -> (E, bool) {
    let ax = x.abs();
    let ay = y.abs();
    if is_nan(x) || is_nan(y) || ax == E::INFINITY || ay == E::ZERO {
        // The invalid cases all produce a NaN. Spelled as a division so the
        // platform's own invalid-operation NaN comes back, sign included --
        // `E::NAN` is the *positive* quiet NaN, and the sign is part of the
        // contract here.
        #[allow(clippy::eq_op)]
        return ((x - x) / (y - y), false);
    }
    if ay == E::INFINITY || ax < ay {
        return (x, false);
    }
    if ax == ay {
        return (E::ZERO.copysign(x), true);
    }

    // Largest `|y| * 2^j` that does not exceed `ax`. `d + d` rather than
    // `d * 2` so that the overflow case saturates to infinity and stops the
    // loop instead of wrapping.
    let mut d = ay;
    while d + d <= ax {
        d = d + d;
    }

    let mut r = ax;
    let mut odd = false;
    loop {
        if r >= d {
            r = r - d;
            odd = d == ay;
        }
        if d == ay {
            break;
        }
        d = d * E::HALF;
    }
    (r.copysign(x), odd)
}

/// A lane-wise scalar fallback, for kernels with no vectorised schedule.
///
/// Re-exported here so the delegating kernels have one obvious place to reach
/// for, rather than each importing from [`crate::simd`].
pub use crate::simd::map_lanes as lanewise;
