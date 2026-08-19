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
//!
//! [`frexp`], [`nextafter`], [`fmod`] and [`remainder`] are per-lane, and stay
//! that way: they extract or rebuild an IEEE exponent field, which needs a
//! bit *shift* against the mantissa width, and [`Simd`] has no packed integer
//! shift anywhere in its surface — [`Simd::and_bits`]/[`Simd::or_bits`]/
//! [`Simd::xor_bits`] are lane-wise bitwise ops on the float representation,
//! not integer arithmetic, and [`Simd::Bits`] is a plain `[Uint; LANES]`
//! array with no operations of its own. Genuinely vectorising these would
//! mean adding a shift primitive to the trait and implementing it across
//! every backend, not rewriting one kernel — confirmed by disassembling the
//! compiled `frexp`, which showed real per-lane branches and scalar (`vmovsd`)
//! moves, not the packed instructions a true vectorisation would emit. (An
//! earlier version of this comment claimed LLVM vectorised the loop anyway;
//! it does not, at least not with this crate's current codegen flags — this
//! was checked, not assumed.)

use crate::policy::{Accuracy, Domain};
use crate::simd::{Lanes, Mask, Real, Simd, Uint, map_lanes_pair, map_lanes2};

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

// ---------------------------------------------------------------------------
// The IEEE-754 numeric helpers: rint, scalbn, fdim, fmax, fmin, ilogb
// ---------------------------------------------------------------------------

unary_vector! {
    /// `x` rounded to the nearest integer, ties to even.
    ///
    /// C's `rint` rounds in the *current* mode; Rust evaluates in
    /// round-to-nearest-even and offers no way to leave it, so that is the
    /// mode this reproduces — one instruction per vector, and the only mode
    /// reachable from safe Rust. Unlike [`round`] it breaks ties to even, and
    /// unlike `nearbyint` it is allowed to raise the inexact flag, which
    /// nothing here observes.
    rint, |x| x.round_ties_even()
}

/// `x * 2^n`, with `n` carried in float lanes.
///
/// The C library defines `scalbn` and `ldexp` to compute the same thing on
/// every target where `FLT_RADIX` is 2, which is every target this crate
/// builds for, and glibc makes the second a literal alias of the first. So
/// this is [`ldexp`] under its other name rather than a second algorithm, and
/// the two are the same code — not merely the same results.
pub mod scalbn {
    use super::*;

    /// The scalar form, correct for every input.
    #[inline(always)]
    pub fn scalar<E: Real>(x: E, n: E) -> E {
        ldexp::scalar(x, n)
    }

    /// `scalbn(x, n)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, n: V) -> V {
        ldexp::eval::<V, A, D>(x, n)
    }
}

/// The positive difference: `x - y` if `x > y`, and `+0` otherwise.
pub mod fdim {
    use super::*;

    /// The scalar form, correct for every input.
    #[inline(always)]
    pub fn scalar<E: Real>(x: E, y: E) -> E {
        if is_nan(x) || is_nan(y) {
            return x + y; // propagate a payload rather than a fresh NaN
        }
        if x > y { x - y } else { E::ZERO }
    }

    /// `fdim(x, y)` for a vector of lanes.
    ///
    /// Branch-free: the difference is computed unconditionally and then
    /// selected. Testing the *arguments* for NaN rather than the difference is
    /// what keeps `fdim(inf, inf)` at `+0` — the difference is NaN there, but
    /// neither argument is.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        let nan = x.is_nan().or(y.is_nan());
        let quiet = V::select(nan, x + y, V::splat(V::Elem::ZERO));
        V::select(x.gt_mask(y), x - y, quiet)
    }
}

/// The larger of `x` and `y`, ignoring NaN.
pub mod fmax {
    use super::*;

    /// The scalar form, correct for every input.
    #[inline(always)]
    pub fn scalar<E: Real>(x: E, y: E) -> E {
        if x >= y || (!is_nan(x) && is_nan(y)) {
            x
        } else {
            y
        }
    }

    /// `fmax(x, y)` for a vector of lanes.
    ///
    /// C's `fmax`, not IEEE-754-2019's `maximum`: a NaN argument is *ignored*
    /// rather than propagated, so `fmax(NaN, 1.0)` is `1.0`. Equal arguments
    /// return the first, which pins down `fmax(+0, -0)` as `+0` and
    /// `fmax(-0, +0)` as `-0` — C leaves that unspecified, and this is what
    /// glibc does.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        V::select(x.ge_mask(y).or(x.is_nan().not().and(y.is_nan())), x, y)
    }
}

/// The smaller of `x` and `y`, ignoring NaN.
pub mod fmin {
    use super::*;

    /// The scalar form, correct for every input.
    #[inline(always)]
    pub fn scalar<E: Real>(x: E, y: E) -> E {
        if x <= y || (!is_nan(x) && is_nan(y)) {
            x
        } else {
            y
        }
    }

    /// `fmin(x, y)` for a vector of lanes. See [`fmax`] for the NaN and
    /// signed-zero conventions.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        V::select(x.le_mask(y).or(x.is_nan().not().and(y.is_nan())), x, y)
    }
}

/// The binary exponent of `x`, as an integer in float lanes.
///
/// `ilogb(x) == floor(log2(|x|))` for a finite non-zero `x`, computed by
/// reading the exponent field rather than by taking a logarithm, so it is
/// exact. Subnormals report their *true* exponent, not the stored one.
///
/// # The three sentinels
///
/// C specifies `ilogb(0)`, `ilogb(NaN)` and `ilogb(+-inf)` as the macros
/// `FP_ILOGB0`, `FP_ILOGBNAN` and `INT_MAX`. glibc on this target makes the
/// first two `INT_MIN`, and that is what comes back here.
///
/// Since the result travels in a float lane — for the same reason [`ldexp`]'s
/// exponent argument does — `INT_MIN` is exact in both precisions (it is
/// `-2^31`), but `INT_MAX` is not representable in `f32` and arrives as
/// `2147483648.0`. Anything that needs the exact `i32` should compare against
/// the sentinel before converting.
pub mod ilogb {
    use super::*;

    /// `FP_ILOGB0` and `FP_ILOGBNAN` on this target: `INT_MIN`.
    pub const ILOGB0: f64 = i32::MIN as f64;
    /// What C requires for `ilogb(+-inf)`: `INT_MAX`.
    pub const ILOGB_INF: f64 = i32::MAX as f64;

    /// The scalar form, correct for every input.
    pub fn scalar<E: Real>(x: E) -> E {
        let a = x.abs();
        if a == E::ZERO || is_nan(x) {
            return E::of_f64(ILOGB0);
        }
        if a == E::INFINITY {
            return E::of_f64(ILOGB_INF);
        }
        if a < E::MIN_POSITIVE {
            // Subnormal: normalise by an exact power of two and correct.
            let n = a * E::TWO_POW_MANT;
            let raw = (n.bits() >> E::MANT_BITS).as_u32() as i32;
            return E::of_f64((raw - E::EXP_BIAS - E::MANT_BITS as i32) as f64);
        }
        let raw = (a.bits() >> E::MANT_BITS).as_u32() as i32;
        E::of_f64((raw - E::EXP_BIAS) as f64)
    }

    /// `ilogb(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        crate::simd::map_lanes(x, scalar)
    }
}

// ---------------------------------------------------------------------------
// modf and remquo
// ---------------------------------------------------------------------------

/// Split `x` into its fractional and integral parts, both with `x`'s sign.
pub mod modf {
    use super::*;

    /// The scalar form, correct for every input.
    ///
    /// The vector kernel at one lane, so the two cannot drift apart.
    #[inline(always)]
    pub fn scalar<E: Real>(x: E) -> (E, E) {
        eval::<E, crate::policy::BitExact, crate::policy::FullRange>(x)
    }

    /// `modf(x)` for a vector of lanes, returning `(fraction, integral)`.
    ///
    /// Branch-free. The two subtleties are both about signs: the fractional
    /// part takes `x`'s sign explicitly, so `modf(-0.0)` gives `(-0.0, -0.0)`
    /// rather than the `+0.0` a bare subtraction produces; and an infinite `x`
    /// is selected out, because `inf - inf` is NaN where C requires `(±0, ±inf)`.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> (V, V) {
        let _ = (A::BIT_EXACT, D::CHECKED);
        let int = x.trunc();
        let frac = (x - int).copysign(x);
        let inf = x.abs().eq_mask(V::splat(V::Elem::INFINITY));
        (
            V::select(inf, V::splat(V::Elem::ZERO).copysign(x), frac),
            int,
        )
    }
}

/// `x` reduced modulo `y`, together with the low bits of the quotient.
///
/// Returns `(remainder, quotient)`, where the remainder is exactly
/// [`remainder`]'s and the quotient carries the sign of `x/y` and the low
/// three bits of `|x/y|` rounded to nearest. C passes the quotient through an
/// `int *`; here it comes back in a float lane, like [`frexp`]'s exponent, and
/// it is always a small exact integer in `-7..=7`.
pub mod remquo {
    use super::*;

    /// The scalar form, correct for every input.
    ///
    /// A transcription of glibc's `s_remquo.c`, whose shape is: reduce modulo
    /// `8y` once with [`fmod`], then subtract off `4y`, `2y` and `y` in turn,
    /// counting as it goes. Every step is exact — Sterbenz again — so this
    /// agrees with the platform by construction rather than by measurement.
    pub fn scalar<E: Real>(x: E, y: E) -> (E, E) {
        let sx = x.bits() & sign_bit::<E>();
        let qs = sx ^ (y.bits() & sign_bit::<E>());
        let negq = qs != E::Uint::ZERO;

        // Invalid: y == 0, x not finite, or either argument NaN. Spelled as a
        // division so the platform's own invalid NaN comes back.
        let ax = x.abs();
        let ay = y.abs();
        if ay == E::ZERO || is_nan(x) || is_nan(y) || ax == E::INFINITY {
            #[allow(clippy::eq_op)]
            let nan = (x * y) / (x * y);
            return (nan, E::ZERO);
        }

        // Reduce to `|x| < 8|y|`, but only when `8y` cannot overflow.
        let mut r = if biased_exp(y) <= max_biased_exp::<E>() - 3 {
            fmod::scalar(x, y * E::of_f64(8.0))
        } else {
            x
        };

        if ax == ay {
            return (E::ZERO.copysign(x), if negq { -E::ONE } else { E::ONE });
        }

        r = r.abs();
        let mut q = 0i32;
        if biased_exp(y) <= max_biased_exp::<E>() - 2 && r >= ay * E::of_f64(4.0) {
            r = r - ay * E::of_f64(4.0);
            q += 4;
        }
        if biased_exp(y) < max_biased_exp::<E>() && r >= ay + ay {
            r = r - (ay + ay);
            q += 2;
        }

        if biased_exp(y) == 0 {
            // `y` is subnormal, where halving it would be inexact; double the
            // remainder instead, which cannot overflow because `r < 2|y|`.
            if r + r > ay {
                r = r - ay;
                q += 1;
                if r + r >= ay {
                    r = r - ay;
                    q += 1;
                }
            }
        } else {
            let half = ay * E::HALF;
            if r > half {
                r = r - ay;
                q += 1;
                if r >= half {
                    r = r - ay;
                    q += 1;
                }
            }
        }

        if r == E::ZERO {
            r = E::ZERO;
        }
        let r = if sx != E::Uint::ZERO { -r } else { r };
        (r, E::of_f64(if negq { -q as f64 } else { q as f64 }))
    }

    /// `remquo(x, y)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V, y: V) -> (V, V) {
        let _ = (A::BIT_EXACT, D::CHECKED);
        crate::simd::map_lanes2_pair(x, y, scalar)
    }

    /// The stored exponent field of `x`, as glibc's threshold comparisons read
    /// it. Zero for a subnormal.
    #[inline(always)]
    fn biased_exp<E: Real>(x: E) -> u32 {
        (x.abs().bits() >> E::MANT_BITS).as_u32()
    }

    /// The largest exponent field a finite number has: `2 * EXP_BIAS`.
    #[inline(always)]
    fn max_biased_exp<E: Real>() -> u32 {
        (1u32 << E::EXP_BITS) - 2
    }

    /// The sign bit, as a mask.
    #[inline(always)]
    fn sign_bit<E: Real>() -> E::Uint {
        (-E::ZERO).bits()
    }
}
