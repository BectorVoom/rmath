//! `e^x`.
//!
//! * [`crate::policy::BitExact`] — glibc's `__ieee754_exp_fma` schedule:
//!   argument reduction to `|r| < ln(2)/256`, a 128-entry `2^(k/128)` table,
//!   and a degree-4 correction. Bit-identical to the scalar `exp`.
//! * [`crate::policy::Fast`] — no table. Reduces to `|r| <= ln(2)/2` and
//!   takes a degree-13 series, evaluated by Estrin's scheme to keep the
//!   dependency chain short. The point is not fewer operations — it is more
//!   arithmetic and *no gather*, and the gather is the part that does not
//!   vectorise.
//!
//! `Finite` means `|x| < 512`. Outside that, results are wrong, not merely
//! imprecise; note that `exp` overflows at 709.78, so the safe range is set
//! by the reduction, not by the function's own domain.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Lanes, Simd, patch_lanes};
use crate::tables::double::exp as t;

/// Widest `|x|` the vector main path is valid for.
const MAIN_PATH_LIMIT: f64 = 512.0;

/// `e^x` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::exp)
    } else {
        y
    }
}

/// The table path, lane-for-lane identical to [`reference::exp`]'s main body.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
    let shift = V::splat(t::SHIFT);
    let kd_s = x.mul_add(V::splat(t::INVLN2N), shift);
    let ki = kd_s.to_bits();
    let kd = kd_s - shift;
    let r = kd.mul_add(
        V::splat(t::NEGLN2LON),
        kd.mul_add(V::splat(t::NEGLN2HIN), x),
    );

    // The gather. Two arrays rather than one pass returning a pair, because
    // the index arithmetic is a few cheap ALU ops and splitting the loop
    // leaves each one a clean, unconditional, fixed-length copy.
    let mut tail_bits = V::Bits::filled_default();
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let k = ki.as_slice()[i];
        let idx = ((k & 127) * 2) as usize;
        tail_bits.as_mut_slice()[i] = t::TAB[idx];
        scale_bits.as_mut_slice()[i] = t::TAB[idx + 1].wrapping_add(k << 45);
    }
    let tail = V::from_bits(tail_bits);
    let scale = V::from_bits(scale_bits);

    let p12 = r.mul_add(V::splat(t::C3), V::splat(t::C2));
    let t3 = tail + r;
    let r2 = r * r;
    let p45 = r.mul_add(V::splat(t::C5), V::splat(t::C4));
    let s1 = r2.mul_add(p12, t3);
    let r4 = r2 * r2;
    let tmp = r4.mul_add(p45, s1);
    scale.mul_add(tmp, scale)
}

/// `1/ln(2)`, for the table-free reduction.
const LOG2E: f64 = core::f64::consts::LOG2_E;
/// `ln(2)`, split so that `kd * LN2HI` is exact.
///
/// LN2HI is exactly representable in 33 bits, which makes that product exact
/// for every `kd` the reduction produces; LN2LO carries the remainder. The
/// digits are not stray precision — rounding either to the shortest
/// round-tripping form, as `clippy::excessive_precision` suggests, destroys
/// that property and costs accuracy in the reduction.
#[allow(clippy::excessive_precision)]
const LN2HI: f64 = 6.93147180369123816490e-01;

/// `ln(2)`, low part. See [`LN2HI`].
#[allow(clippy::excessive_precision)]
const LN2LO: f64 = 1.90821492927058770002e-10;

/// `1/k!` for `k` in `2..=13`. Truncating the series here leaves a relative
/// error of `r^14 / 14!`, which at the reduction's worst case
/// (`|r| = ln(2)/2`) is about `6.5e-18` — a fifth of an ulp, so the error
/// that matters is the accumulated rounding, not the truncation.
const F: [f64; 12] = [
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
];

/// The table-free path.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V) -> V {
    let shift = V::splat(t::SHIFT);
    let kd_s = x.mul_add(V::splat(LOG2E), shift);
    let kd = kd_s - shift;
    // Cody-Waite: subtract `kd * ln(2)` in two exactly-representable pieces.
    let r = kd.mul_add(V::splat(-LN2LO), kd.mul_add(V::splat(-LN2HI), x));

    // Estrin. Horner would be 13 dependent FMAs — about 52 cycles of latency
    // that no amount of lane parallelism hides, since every lane waits on the
    // same chain. This is the same operation count at depth 6.
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;

    let c01 = V::splat(1.0) + r; // c0 = c1 = 1
    let c23 = r.mul_add(V::splat(F[1]), V::splat(F[0]));
    let c45 = r.mul_add(V::splat(F[3]), V::splat(F[2]));
    let c67 = r.mul_add(V::splat(F[5]), V::splat(F[4]));
    let c89 = r.mul_add(V::splat(F[7]), V::splat(F[6]));
    let cab = r.mul_add(V::splat(F[9]), V::splat(F[8]));
    let ccd = r.mul_add(V::splat(F[11]), V::splat(F[10]));

    let lo = r2.mul_add(c23, c01);
    let mid = r2.mul_add(c67, c45);
    let hi = r2.mul_add(cab, c89);
    let lo = r4.mul_add(mid, lo);
    let hi = r4.mul_add(ccd, hi);
    let p = r8.mul_add(hi, lo);

    // 2^kd, built straight into the exponent field. Pure integer arithmetic,
    // no memory: this is what the table path cannot do.
    let ki = kd_s.to_bits();
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        // The shift constant puts `k + 2^51` in the low 52 bits.
        let k = (ki.as_slice()[i] & 0x000f_ffff_ffff_ffff).wrapping_sub(1u64 << 51);
        scale_bits.as_mut_slice()[i] = k.wrapping_add(1023) << 52;
    }
    V::from_bits(scale_bits) * p
}
