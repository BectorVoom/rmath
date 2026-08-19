//! `e^x`, single precision.
//!
//! A port of the platform's `expf`: reduce with a 32-entry `2^(i/32)` table
//! and take a degree-3 correction, all in double precision, rounding once at
//! the end. That is not this crate imposing double precision on a `float`
//! function — it is what `expf` is, and reproducing it is the only way to be
//! bit-exact. The pay-off is that the schedule is plain `f64` arithmetic, so
//! an `f32x8` widens to an `f64x8` and the whole thing runs eight lanes wide.
//!
//! `Finite` means `|x| < 88`. Outside that the results are wrong, not merely
//! imprecise.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain};
use crate::reference::single as reference;
use crate::simd::{Lanes, Simd, patch_lanes};
use crate::tables::single::exp as t;

/// Widest `|x|` the vector main path is valid for.
const MAIN_PATH_LIMIT: f32 = 88.0;

/// `e^x` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::exp)
    } else {
        y
    }
}

/// The table path, lane-for-lane identical to [`reference::exp`]'s main body.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let xd = x.widen();

    // Both uses of `INVLN2_SCALED * xd` are fused; see `reference::single::exp`
    // for why the rounded intermediate must not exist.
    let invln2 = W::<V>::splat(t::INVLN2_SCALED);
    let shift = W::<V>::splat(t::SHIFT);
    let kd_s = invln2.mul_add(xd, shift);
    let ki = kd_s.to_bits();
    let kd = kd_s - shift;
    let r = invln2.mul_add(xd, -kd);

    let s = super::scale32::<V>(&ki);

    let z = r.mul_add(W::<V>::splat(t::CS0), W::<V>::splat(t::CS1));
    let r2 = r * r;
    let y = r.mul_add(W::<V>::splat(t::CS2), W::<V>::splat(1.0));
    let y = r2.mul_add(z, y);
    V::narrow(y * s)
}

/// The table-free path.
///
/// Degree-6 series on `|r| <= ln(2)/2`, evaluated in single precision with the
/// scale built straight into the exponent field. No widening, no gather: this
/// is the configuration that exists because both of those are what stop `f32`
/// code from going as fast as its lane count says it should.
///
/// Measured error: below 2 ulp over `|x| < 88`.
///
/// The scale's exponent field is built directly (`k.wrapping_add(127) << 23`
/// below), which only produces a *normal* result — see `exp2.rs`'s `fast`
/// doc for the full account of the bug this guards and the exact-power-of-two
/// fix, shared verbatim here since the construction is identical.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    /// `log2(e)`.
    const LOG2E: f32 = core::f32::consts::LOG2_E;
    /// `ln(2)`, high part, with its low 8 significand bits cleared so that
    /// `kd * LN2HI` is exact for every `|kd| < 256` the reduction produces.
    const LN2HI: f32 = f32::from_bits(0x3f31_7200);
    /// `ln(2) - LN2HI`, to `f32`. Together they carry `ln(2)` to some 47 bits,
    /// which is what keeps the reduced argument accurate at large `|k|`.
    const LN2LO: f32 = f32::from_bits(0x35bf_be8e);
    /// `1/k!` for `k` in `2..=6`.
    const F: [f32; 5] = [0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0];
    /// The round-to-nearest-integer trick constant for `f32`, `0x1.8p23`.
    const SHIFT: f32 = f32::from_bits(0x4b40_0000);

    let kd_s = x.mul_add(V::splat(LOG2E), V::splat(SHIFT));
    let kd = kd_s - V::splat(SHIFT);
    // Cody-Waite: subtract `kd * ln(2)` in two exactly-representable pieces.
    let r = kd.mul_add(V::splat(-LN2LO), kd.mul_add(V::splat(-LN2HI), x));

    let r2 = r * r;
    let c01 = V::splat(1.0) + r;
    let c23 = r.mul_add(V::splat(F[1]), V::splat(F[0]));
    let c45 = r.mul_add(V::splat(F[3]), V::splat(F[2]));
    // 1 + r + r^2(1/2 + r/6)  +  r^4(1/24 + r/120 + r^2/720)
    let lo = r2.mul_add(c23, c01);
    let hi = r2.mul_add(V::splat(F[4]), c45);
    let p = (r2 * r2).mul_add(hi, lo);

    // 2^k, built straight into the exponent field.
    let ki = kd_s.to_bits();
    let mut bits = V::Bits::filled_default();
    let mut subnormal_adjust = V::Floats::filled_default();
    for i in 0..V::LANES {
        // The shift constant puts `k + 2^22` in the low 23 bits.
        let k = (ki.as_slice()[i] & 0x007f_ffff).wrapping_sub(1u32 << 22) as i32;
        if k <= -127 {
            bits.as_mut_slice()[i] = ((k + 25 + 127) as u32) << 23;
            subnormal_adjust.as_mut_slice()[i] = f32::from_bits(0x3300_0000); // 2^-25
        } else {
            bits.as_mut_slice()[i] = ((k + 127) as u32) << 23;
            subnormal_adjust.as_mut_slice()[i] = 1.0;
        }
    }
    (V::from_bits(bits) * V::from_array(subnormal_adjust)) * p
}
