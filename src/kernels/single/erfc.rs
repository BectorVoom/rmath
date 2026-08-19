//! The complementary error function, single precision.
//!
//! `erfc` is not `1 - erf`: for `x` above about 1 that subtraction cancels
//! away every significant digit, and `erfc(10)` — which is 2e-45 — would come
//! back as zero. The platform's `erfcf` instead evaluates
//! `erfc(x) = exp(-x^2) * P(z)` with `z = (|x| - a)/(|x| + b)`, one Chebyshev
//! fit either side of `|x| = 2.08`, and gets `exp(-x^2)` from a 128-entry
//! table. Correctly rounded, and so bit-exact to any correctly rounded
//! `erfcf`; see [`crate::reference::single::erf_parts`].
//!
//! All of that is `double` arithmetic, so an `f32x8` widens to an `f64x8` and
//! the schedule replays lane-parallel. The sign of `x` never branches: it is
//! folded into the exponent field of the scale, which turns
//! `erfc(-|x|) = 2 - erfc(|x|)` into a single add.
//!
//! Both policies run the same code, unlike [`super::erf`] — a native `Fast`
//! path was tried (`erf`'s own shared `exp_neg_x2`/`far_q`, reusing its
//! two-region minimax fit) and measured, not merely assumed, before deciding
//! against it. It reached only 1.51x here (against 1.40x for the code above,
//! i.e. barely faster) at 11 ulp worst case (`tests/ulp_scan.rs::scan_erf_single_precision`),
//! well past a 4 ulp bound — a materially worse trade than `erf`'s own native
//! path, which reached 4.1x at 2 ulp on the *same* two building blocks. The
//! difference is domain, not the fit: `erfc`'s own `Finite` domain reaches
//! `0x1.41bbf8p+3` (~10.05) with no saturating shortcut anywhere in it, so
//! every lane pays the far branch's two divisions and a reflection select in
//! full, where `erf` only pays for that up to its own ~3.92 saturation point
//! and skips the rest. Kept for the record rather than deleted, so the next
//! attempt does not re-derive this from scratch: a coarser `z` reduction near
//! `|x| = 10` that avoids or cheapens the second division might change the
//! economics.
//!
//! `Finite` means `0x1.c5bf88p-26 < |x| < 0x1.41bbf8p+3` — outside that
//! `erfc` is 1, 0 or 2.

use crate::policy::{Accuracy, Domain};
use crate::reference::single::erf_parts as r;
use crate::simd::{Lanes, Mask, Simd, patch_lanes};
use crate::tables::single::erf as t;

/// `erfc(x)` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let _ = A::BIT_EXACT;
    let y = main(x);
    if D::CHECKED {
        patch_lanes(x, y, outside(x), r::erfc)
    } else {
        y
    }
}

/// The lanes the two vector branches do not cover.
///
/// Four disjoint reasons, spelled in float comparisons rather than on the bit
/// patterns the scalar routine tests, because a float compare is one vector
/// instruction and the bit test is not:
///
/// * `x < -0x1.ea8f94p+1`, where `erfc` rounds to 2;
/// * `|x| >= 0x1.41bbf8p+3` — or NaN, which the negated compare catches —
///   where `erfc(x)` is below `2^-150`;
/// * `|x| <= 0x1.c5bf88p-26`, where it rounds to 1;
/// * the single argument the Chebyshev fit does not resolve.
#[inline(always)]
fn outside<V: Simd<Elem = f32>>(x: V) -> V::Mask {
    let a = x.abs();
    crate::kernels::outside(x, f32::from_bits(r::ERFC_POS_LIMIT))
        .or(x.lt_mask(V::splat(-f32::from_bits(r::ERFC_NEG_LIMIT & 0x7fff_ffff))))
        .or(a.le_mask(V::splat(f32::from_bits(r::ERFC_UNIT))))
        .or(x.eq_mask(V::splat(f32::from_bits(r::ERFC_EXCEPTION))))
}

/// Both branches, widened, blended and narrowed once.
#[inline(always)]
fn main<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let xd = x.widen();
    let axd = xd.abs();
    let x2 = axd * axd;

    let near = near_zero::<V>(xd, x2);
    let far = asymptotic::<V>(x, axd, x2);

    let is_near = axd.le_mask(W::<V>::splat(f32::from_bits(r::ERFC_NEAR_ZERO) as f64));
    V::narrow(W::<V>::select(is_near, near, far))
}

/// `|x| <= 0x1.7p-4`: `erfc(x) = 1 - (odd series in x)`.
#[inline(always)]
fn near_zero<V: Simd<Elem = f32>>(
    xd: <V as Simd>::Wide,
    x2: <V as Simd>::Wide,
) -> <V as Simd>::Wide {
    type W<V> = <V as Simd>::Wide;
    let c = &t::CN;
    let p = W::<V>::splat(c[0])
        + x2 * (W::<V>::splat(c[1])
            + x2 * (W::<V>::splat(c[2]) + x2 * (W::<V>::splat(c[3]) + x2 * W::<V>::splat(c[4]))));
    W::<V>::splat(1.0) - xd * p
}

/// The main branch: `exp(-x^2)` from the table, times the Chebyshev fit.
#[inline(always)]
fn asymptotic<V: Simd<Elem = f32>>(
    x: V,
    axd: <V as Simd>::Wide,
    x2: <V as Simd>::Wide,
) -> <V as Simd>::Wide {
    type W<V> = <V as Simd>::Wide;

    // The scale, and the reduced argument. `j ~= -round(128 x^2 / ln 2)` is
    // read out of the significand of `x^2/ln2 - (1024 + 2^-8)`: the added
    // constant pins the binade, so the significand's top bits *are* the
    // scaled integer. This and the `E` lookup are the only per-lane work.
    let jt = (x2 * W::<V>::splat(r::ILN2) - W::<V>::splat(r::SHIFT)).to_bits();
    let sign_bits = x.to_bits();
    let mut jf = <W<V> as Simd>::Floats::filled_default();
    let mut sbits = <W<V> as Simd>::Bits::filled_default();
    let mut e0 = <W<V> as Simd>::Floats::filled_default();
    for lane in 0..<W<V> as Simd>::LANES {
        let j = ((jt.as_slice()[lane] << 12) as i64) >> 48;
        let sgn = (sign_bits.as_slice()[lane] >> 31) as i64;
        jf.as_mut_slice()[lane] = j as f64;
        // `2^(j >> 7)`, with the sign of `x` written straight into the
        // exponent field's top bit.
        sbits.as_mut_slice()[lane] = (((j >> 7) + (0x3ff | (sgn << 11))) as u64) << 52;
        e0.as_mut_slice()[lane] = t::E[(j & 127) as usize];
    }
    let j = <W<V>>::from_array(jf);
    let s = <W<V>>::from_bits(sbits);
    let e0 = <W<V>>::from_array(e0);

    let ch = &t::CH;
    let d = (x2 + W::<V>::splat(r::LN2H) * j) + W::<V>::splat(r::LN2L) * j;
    let d2 = d * d;
    let f = d + d2
        * ((W::<V>::splat(ch[0]) + d * W::<V>::splat(ch[1]))
            + d2 * (W::<V>::splat(ch[2]) + d * W::<V>::splat(ch[3])));

    // Two Chebyshev fits, selected rather than gathered: there are only two
    // rows, so a blend of sixteen splatted constants beats sixteen loads a
    // lane.
    let hi = axd.gt_mask(W::<V>::splat(f32::from_bits(r::ERFC_SPLIT) as f64));
    let pick =
        |k: usize| W::<V>::select(hi, W::<V>::splat(t::CT[1][k]), W::<V>::splat(t::CT[0][k]));

    let z = (axd - pick(0)) / (axd + pick(1));
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let c = |k: usize| pick(k + 3);
    let poly = (((c(0) + z * c(1)) + z2 * (c(2) + z * c(3)))
        + z4 * ((c(4) + z * c(5)) + z2 * (c(6) + z * c(7))))
        + z8 * (((c(8) + z * c(9)) + z2 * (c(10) + z * c(11))) + z4 * c(12));
    let poly = pick(2) + z * poly;

    let r = (s * (e0 - f * e0)) * poly;
    // `2` for a negative argument, `0` otherwise -- the reflection
    // `erfc(-|x|) = 2 - erfc(|x|)`, with the minus already in `s`.
    let two = W::<V>::select(
        x.widen().lt_mask(W::<V>::splat(0.0)),
        W::<V>::splat(2.0),
        W::<V>::splat(0.0),
    );
    two + r
}
