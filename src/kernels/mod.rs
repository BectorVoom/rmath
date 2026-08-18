//! The vector kernels.
//!
//! Every unary kernel exposes the same entry point:
//!
//! ```ignore
//! pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V
//! ```
//!
//! with `Elem = f32` for the single-precision tree, two arguments for a
//! binary function and a pair return for `sincos` / `frexp`. The builder in
//! [`crate::function`] wraps any of them without knowing what they do, so
//! adding a function means adding a module here plus one line of `math_fn!`.
//! The policy parameters are resolved at compile time, so each instantiation
//! is a straight-line specialisation with no branches on configuration.
//!
//! # Three shapes of kernel
//!
//! * **[`exact`]** — functions IEEE-754 defines exactly: rounding to integer,
//!   sign and exponent manipulation, `fmod`, `remainder`, `sqrt`. There is no
//!   reference algorithm to reproduce, because there is no approximation:
//!   every implementation that is correct is bit-identical. These are written
//!   once, generic over precision as well as width, and both policy axes are
//!   no-ops.
//!
//! * **Ported** — `exp`, `exp2`, `expm1`, `ln`, `log2`, `pow`, `cbrt`,
//!   `sinh`, `cosh`, `tanh`. The vector code
//!   replays the platform routine's operation schedule lane-parallel. The
//!   shape is the same throughout: compute the main path across all lanes,
//!   then — under [`crate::policy::FullRange`] — repair the rare lanes with
//!   the scalar reference. That keeps the hard cases (overflow, subnormals,
//!   NaN) in one place, [`crate::reference`], where they are written once
//!   against the platform routine rather than re-derived in vector form per
//!   function.
//!
//! * **Delegating** — the trigonometric and inverse-trigonometric families,
//!   the inverse hyperbolics, `log10`, `log1p` and `hypot`, under
//!   [`crate::policy::BitExact`]. Their platform schedule has not been
//!   reproduced in vector form, so the bit-exact path evaluates
//!   [`crate::reference`] one lane at a time. It is bit-exact — that is the
//!   point — but it is not faster than the scalar call it replaces.
//!   [`crate::policy::Fast`] gives those functions a genuinely vectorised
//!   path, with the error bound documented on the kernel.
//!
//! The crate-level documentation has the per-function table.

use crate::policy::{Accuracy, Domain};
use crate::simd::{Mask, Real, Simd, map_lanes, patch_lanes};

/// The shape every non-exact kernel has, in one place.
///
/// Under [`crate::policy::BitExact`] the whole vector goes to the scalar
/// reference, unless the function has a vectorised schedule of its own — those
/// call their own `bit_exact` instead of coming through here. Under
/// [`crate::policy::Fast`] the vector path runs, and `FullRange` then repairs
/// the lanes `special` selects.
///
/// Taking the three pieces as arguments rather than generating them with a
/// macro keeps each kernel's domain decision visible at its own call site,
/// which is the part most likely to be wrong.
#[inline(always)]
pub(crate) fn dispatch<V: Simd, A: Accuracy, D: Domain>(
    x: V,
    reference: impl Fn(V::Elem) -> V::Elem,
    fast: impl Fn(V) -> V,
    special: impl Fn(V) -> V::Mask,
) -> V {
    if A::BIT_EXACT {
        return map_lanes(x, reference);
    }
    let y = fast(x);
    if D::CHECKED {
        patch_lanes(x, y, special(x), reference)
    } else {
        y
    }
}

/// [`dispatch`] for a two-argument function.
#[inline(always)]
pub(crate) fn dispatch2<V: Simd, A: Accuracy, D: Domain>(
    x: V,
    y: V,
    reference: impl Fn(V::Elem, V::Elem) -> V::Elem,
    fast: impl Fn(V, V) -> V,
    special: impl Fn(V, V) -> V::Mask,
) -> V {
    if A::BIT_EXACT {
        return crate::simd::map_lanes2(x, y, reference);
    }
    let z = fast(x, y);
    if D::CHECKED {
        crate::simd::patch_lanes2(x, y, z, special(x, y), reference)
    } else {
        z
    }
}

/// Horner evaluation of `c[0] + c[1] s + c[2] s^2 + ...`.
///
/// Horner rather than Estrin, unlike the `exp` kernels: those are latency
/// bound because every lane waits on the same chain, whereas these run inside
/// a buffer loop where independent iterations already fill the pipeline, and
/// Horner's shorter code is worth more there than its longer chain costs.
#[inline(always)]
pub(crate) fn horner<V: Simd>(s: V, c: &[V::Elem]) -> V {
    let mut acc = V::splat(c[c.len() - 1]);
    for &k in c[..c.len() - 1].iter().rev() {
        acc = acc.mul_add(s, V::splat(k));
    }
    acc
}

/// A mask with no lane set.
///
/// Built from a comparison that cannot hold rather than a constructor, because
/// [`Mask`] deliberately has none — every mask in the crate comes from a real
/// comparison, and an inhabited-from-nothing mask would be the one place that
/// is not true.
#[inline(always)]
pub(crate) fn no_lanes<V: Simd>(x: V) -> V::Mask {
    x.lt_mask(x)
}

/// `2^k` per lane, for an integer-valued `k` inside the exponent range.
///
/// Built straight into the exponent field: pure integer work, no table and no
/// multiply. The per-lane loop is branch-free and fixed-length, which is the
/// shape LLVM vectorises well.
#[inline(always)]
pub(crate) fn pow2<V: Simd>(k: V) -> V {
    use crate::simd::{Lanes, Uint};
    let ks = k.to_array();
    let mut bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let e = ks.as_slice()[i].to_f64() as i32;
        bits.as_mut_slice()[i] =
            <V::Elem as Real>::Uint::from_u32((<V::Elem as Real>::EXP_BIAS + e) as u32)
                << <V::Elem as Real>::MANT_BITS;
    }
    V::from_bits(bits)
}

/// Split `x` into `(e, s)` with `x = 2^e * m`, `m` in `[1/sqrt2, sqrt2)` and
/// `s = (m - 1) / (m + 1)`.
///
/// The reduction every `Fast` logarithm shares: `ln(m) = 2 atanh(s)`, and
/// folding the significand about `sqrt2` is what keeps `|s|` below 0.1716 so
/// one short polynomial covers it. Only the exponent surgery is per-lane; the
/// division and everything after is whole-vector.
#[inline(always)]
pub(crate) fn log_split<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    use crate::simd::Lanes;
    let ix = x.to_bits();
    let mut mant = V::Bits::filled_default();
    let mut es = V::Floats::filled_default();
    for i in 0..V::LANES {
        let b = ix.as_slice()[i];
        let mut e = ((b >> 52) & 0x7ff) as i32 - 1023;
        let mut m = (b & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
        // Fold [1, 2) to [1/sqrt2, sqrt2) by borrowing one power of two.
        // `0x3ff6a09e667f3bcd` is sqrt(2); comparing bit patterns is the same
        // ordering as comparing the values, for positive normals.
        if m >= 0x3ff6_a09e_667f_3bcd {
            m -= 1u64 << 52;
            e += 1;
        }
        mant.as_mut_slice()[i] = m;
        es.as_mut_slice()[i] = e as f64;
    }
    let m = V::from_bits(mant);
    let one = V::splat(1.0);
    (V::from_array(es), (m - one) / (m + one))
}

/// `ln(m)` from the `s` of [`log_split`].
#[inline(always)]
pub(crate) fn log_poly<V: Simd<Elem = f64>>(s: V) -> V {
    let p: V = horner(s * s, &crate::tables::double::poly::ATANH);
    (s + s) * p
}

pub mod double;
pub mod exact;
pub mod single;

/// Lanes that the vector main path must not be trusted with.
///
/// `!(|x| < limit)` rather than `|x| >= limit` so that NaN — which compares
/// false against everything — is caught by the negation.
#[inline(always)]
pub(crate) fn outside<V: Simd>(x: V, limit: V::Elem) -> V::Mask {
    x.abs().lt_mask(V::splat(limit)).not()
}

/// Lanes that are not a *positive* normal number: negatives, zero,
/// subnormals, infinities and NaN.
///
/// For a function whose domain is the positive reals. Testing `x` rather than
/// `|x|` is the whole point — an earlier version used the magnitude, and every
/// negative input sailed into the main path and produced a plausible-looking
/// finite number instead of NaN.
#[inline(always)]
pub(crate) fn not_positive_normal<V: Simd>(x: V) -> V::Mask {
    x.ge_mask(V::splat(V::Elem::MIN_POSITIVE))
        .and(x.lt_mask(V::splat(V::Elem::INFINITY)))
        .not()
}

/// Lanes that are not a normal number of either sign.
///
/// For a function that accepts negatives but whose main path assumes a normal
/// magnitude — the hyperbolic family, `nextafter`'s scaling, `ldexp`.
#[allow(dead_code)] // used by the kernels added alongside it
#[inline(always)]
pub(crate) fn not_normal<V: Simd>(x: V) -> V::Mask {
    let a = x.abs();
    a.ge_mask(V::splat(V::Elem::MIN_POSITIVE))
        .and(a.lt_mask(V::splat(V::Elem::INFINITY)))
        .not()
}
