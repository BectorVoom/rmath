//! `2^x`, single precision.
//!
//! The same shape as [`super::exp`] — 32-entry table, degree-3 correction,
//! double-precision arithmetic rounded once — but reducing against integers
//! directly, so there is no `ln(2)` to subtract and the reduced argument is
//! `N` times smaller. That is why it uses the unscaled coefficients where
//! `exp` uses the pre-divided ones.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain};
use crate::reference::single as reference;
use crate::simd::{Lanes, Simd, patch_lanes};
use crate::tables::single::exp as t;

/// Widest `|x|` the vector main path is valid for.
const MAIN_PATH_LIMIT: f32 = 128.0;

/// `2^x` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::exp2)
    } else {
        y
    }
}

/// The table path, lane-for-lane identical to [`reference::exp2`]'s main body.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let xd = x.widen();

    let shift = W::<V>::splat(t::SHIFT_SCALED);
    let kd_s = xd + shift;
    let ki = kd_s.to_bits();
    let kd = kd_s - shift; // k/N, for integer k
    let r = xd - kd;

    let s = super::scale32::<V>(&ki);

    let z = r.mul_add(W::<V>::splat(t::C0), W::<V>::splat(t::C1));
    let r2 = r * r;
    let y = r.mul_add(W::<V>::splat(t::C2), W::<V>::splat(1.0));
    let y = r2.mul_add(z, y);
    V::narrow(y * s)
}

/// The table-free path: degree-6 series in `2^r` on `|r| <= 1/2`.
///
/// Measured error: below 2 ulp over `|x| < 128`.
///
/// The scale's exponent field is built directly (`k.wrapping_add(127) << 23`
/// below), which only produces a *normal* result: for `k <= -127` — `x`
/// approaching `MAIN_PATH_LIMIT`'s lower end, where `2^x` is itself subnormal
/// or zero — that field construction wraps and returns `+-inf` or 0 instead,
/// silently, for an `x` this policy's own domain claims to handle. Lanes past
/// that threshold instead build `2^(k + 25)` (comfortably normal for every
/// `k` `Finite`'s domain admits, since the smallest representable subnormal
/// is `2^-149`) and multiply by `2^-25` separately — both factors are exact
/// powers of two, so the product is `2^k` exactly, with no rounding beyond
/// what the direct path already pays.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    /// `(ln 2)^k / k!` for `k` in `1..=6`, the Taylor coefficients of `2^r`.
    ///
    /// Exact bit patterns rather than decimal literals: a decimal spelling of
    /// these does not round-trip, and rounding it to one that does would
    /// change the coefficient.
    const G: [f32; 6] = [
        f32::from_bits(0x3f317218),
        f32::from_bits(0x3e75fdf0),
        f32::from_bits(0x3d635847),
        f32::from_bits(0x3c1d955b),
        f32::from_bits(0x3aaec3ff),
        f32::from_bits(0x39218489),
    ];
    /// The round-to-nearest-integer trick constant for `f32`, `0x1.8p23`.
    const SHIFT: f32 = f32::from_bits(0x4b40_0000);

    let kd_s = x + V::splat(SHIFT);
    let kd = kd_s - V::splat(SHIFT);
    let r = x - kd;

    let r2 = r * r;
    let c01 = r.mul_add(V::splat(G[0]), V::splat(1.0));
    let c23 = r.mul_add(V::splat(G[2]), V::splat(G[1]));
    let c45 = r.mul_add(V::splat(G[4]), V::splat(G[3]));
    // 1 + G0 r + r^2(G1 + G2 r)  +  r^4(G3 + G4 r + G5 r^2)
    let lo = r2.mul_add(c23, c01);
    let hi = r2.mul_add(V::splat(G[5]), c45);
    let p = (r2 * r2).mul_add(hi, lo);

    let ki = kd_s.to_bits();
    let mut bits = V::Bits::filled_default();
    let mut subnormal_adjust = V::Floats::filled_default();
    for i in 0..V::LANES {
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
