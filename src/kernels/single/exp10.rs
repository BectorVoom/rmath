//! `10^x`, single precision.
//!
//! `exp2f` with a different reduction scale, which is exactly what the
//! platform's `exp10f` is: the same 32-entry `2^(i/32)` table, the same three
//! scaled coefficients, `N*log2(10)` in place of `N`. Double-precision
//! arithmetic rounded once, so an `f32x8` widens to an `f64x8` and the whole
//! schedule replays eight lanes wide.
//!
//! No fused multiply-adds on the bit-exact path: glibc ships no FMA variant of
//! `exp10f`, and fusing would change the last bit.
//!
//! `Finite` means `|x| < 38`. Outside that the results are wrong, not merely
//! imprecise.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain};
use crate::reference::single as reference;
use crate::simd::{Lanes, Simd, patch_lanes};
use crate::tables::single::exp as t;
use crate::tables::single::exp10 as t10;

/// Widest `|x|` the vector main path is valid for.
const MAIN_PATH_LIMIT: f32 = 38.0;

/// `10^x` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::exp10)
    } else {
        y
    }
}

/// The table path, lane-for-lane identical to [`reference::exp10`]'s main body.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let xd = x.widen();

    let shift = W::<V>::splat(t::SHIFT);
    let z = W::<V>::splat(t10::INVLN10N) * xd;
    let kd_s = z + shift;
    let ki = kd_s.to_bits();
    let kd = kd_s - shift;
    let r = z - kd;

    let s = super::scale32::<V>(&ki);

    let p = W::<V>::splat(t::CS0) * r + W::<V>::splat(t::CS1);
    let r2 = r * r;
    let y = W::<V>::splat(t::CS2) * r + W::<V>::splat(1.0);
    let y = p * r2 + y;
    V::narrow(y * s)
}

/// The table-free path: degree-6 series in `10^r` on `|r| <= log10(2)/2`,
/// evaluated in single precision with the scale built into the exponent field.
///
/// Measured error: below 2 ulp over `|x| < 38`.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    /// `(ln 10)^k / k!` for `k` in `1..=6`, the Taylor coefficients of `10^r`.
    ///
    /// Exact bit patterns rather than decimal literals: a decimal spelling of
    /// these does not round-trip, and rounding it to one that does would
    /// change the coefficient.
    const G: [f32; 6] = [
        f32::from_bits(0x40135d8e),
        f32::from_bits(0x4029a926),
        f32::from_bits(0x4002382d),
        f32::from_bits(0x3f95ebb0),
        f32::from_bits(0x3f0a1500),
        f32::from_bits(0x3e53f6b8),
    ];
    /// The round-to-nearest-integer trick constant for `f32`, `0x1.8p23`.
    const SHIFT: f32 = f32::from_bits(0x4b40_0000);

    let kd_s = x.mul_add(V::splat(t10::LOG2_10), V::splat(SHIFT));
    let kd = kd_s - V::splat(SHIFT);
    // Cody-Waite in base 10: subtract `kd * log10(2)` in two exact pieces.
    let r = kd.mul_add(
        V::splat(-t10::LOG10_2LO),
        kd.mul_add(V::splat(-t10::LOG10_2HI), x),
    );

    let r2 = r * r;
    let c01 = r.mul_add(V::splat(G[0]), V::splat(1.0));
    let c23 = r.mul_add(V::splat(G[2]), V::splat(G[1]));
    let c45 = r.mul_add(V::splat(G[4]), V::splat(G[3]));
    let lo = r2.mul_add(c23, c01);
    let hi = r2.mul_add(V::splat(G[5]), c45);
    let p = (r2 * r2).mul_add(hi, lo);

    let ki = kd_s.to_bits();
    let mut bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let k = (ki.as_slice()[i] & 0x007f_ffff).wrapping_sub(1u32 << 22);
        bits.as_mut_slice()[i] = k.wrapping_add(127) << 23;
    }
    V::from_bits(bits) * p
}
