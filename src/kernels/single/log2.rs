//! Base-2 logarithm, single precision.
//!
//! A port of the platform's `log2f`, the same shape as [`super::ln`]: a
//! 16-subinterval table and a degree-4 correction, evaluated in double
//! precision and rounded once, so an `f32x8` widens to an `f64x8` and the
//! whole reduction runs eight lanes wide.
//!
//! `Finite` means a positive normal `x`.

use crate::kernels::not_positive_normal;
use crate::policy::{Accuracy, Domain};
use crate::reference::single as reference;
use crate::simd::{Lanes, Mask, Simd, patch_lanes};
use crate::tables::single::log2 as t;

/// The table-centring offset, `bits(0x1.66p-1)`.
const LOG_OFF: u32 = 0x3f33_0000;

/// `log2(x)` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let special = if D::CHECKED {
        not_positive_normal(x)
    } else {
        crate::kernels::no_lanes(x)
    };
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED && special.any() {
        return patch_lanes(x, y, special, reference::log2);
    }
    y
}

/// The table path, lane-for-lane identical to [`reference::log2`]'s main body.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let ix = x.to_bits();

    let mut zs = V::Floats::filled_default();
    let mut invc = <W<V> as Simd>::Floats::filled_default();
    let mut logc = <W<V> as Simd>::Floats::filled_default();
    let mut kd = <W<V> as Simd>::Floats::filled_default();
    for i in 0..V::LANES {
        let bits = ix.as_slice()[i];
        let tmp = bits.wrapping_sub(LOG_OFF);
        let idx = ((tmp >> 19) % 16) as usize;
        let top = tmp & 0xff80_0000;
        kd.as_mut_slice()[i] = ((top as i32) >> 23) as f64;
        zs.as_mut_slice()[i] = f32::from_bits(bits.wrapping_sub(top));
        invc.as_mut_slice()[i] = t::TAB[2 * idx];
        logc.as_mut_slice()[i] = t::TAB[2 * idx + 1];
    }
    let z = V::from_array(zs).widen();
    let invc = W::<V>::from_array(invc);
    let logc = W::<V>::from_array(logc);
    let kd = W::<V>::from_array(kd);

    // log2(x) = log2(z/c) + log2(c) + k
    let r = z.mul_add(invc, W::<V>::splat(-1.0));
    let y0 = logc + kd;

    let r2 = r * r;
    let y = r.mul_add(W::<V>::splat(t::A1), W::<V>::splat(t::A2));
    let p = r.mul_add(W::<V>::splat(t::A3), y0);
    let y = r2.mul_add(W::<V>::splat(t::A0), y);
    V::narrow(y.mul_add(r2, p))
}

/// The `Fast` path is the ported one.
///
/// Unusually, there is nothing to trade here. The port is already a
/// double-precision table lookup and a degree-4 polynomial over widened lanes,
/// and the table-free alternative — widening further into the double-precision
/// `Fast` logarithm — measured *slower* as well as less accurate. So `Fast`
/// runs the bit-exact schedule, and is bit-exact as a side effect.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    bit_exact(x)
}
