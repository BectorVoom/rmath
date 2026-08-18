//! Natural logarithm, single precision.
//!
//! A port of the platform's `logf`: a 16-subinterval `1/c` / `log(c)` table
//! and a degree-4 correction, evaluated in double precision and rounded once.
//! Like [`super::exp`], the double-precision schedule is what makes the vector
//! form possible at all — `f32x8` widens to `f64x8` and the whole reduction
//! runs eight lanes wide.
//!
//! `Finite` means a positive normal `x`. Zero, negatives, infinities, NaN and
//! subnormals are wrong under that policy, not merely imprecise.

use crate::kernels::not_positive_normal;
use crate::policy::{Accuracy, Domain};
use crate::reference::single as reference;
use crate::simd::{Lanes, Mask, Simd, patch_lanes};
use crate::tables::single::log as t;

/// `logf`'s table-centring offset, `bits(0x1.66p-1)`.
const LOG_OFF: u32 = 0x3f33_0000;

/// `ln(x)` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let special = if D::CHECKED {
        not_positive_normal(x)
    } else {
        // `Finite` promises there are none; build the empty mask from a
        // comparison that cannot hold rather than inventing a constructor.
        x.lt_mask(x)
    };
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED && special.any() {
        return patch_lanes(x, y, special, reference::ln);
    }
    y
}

/// The table path, lane-for-lane identical to [`reference::ln`]'s main body.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let ix = x.to_bits();

    // Decompose: `x = 2^k z` with `z` in one of 16 subintervals. Per-lane, as
    // every table gather is.
    let mut zs = V::Floats::filled_default();
    let mut invc = <W<V> as Simd>::Floats::filled_default();
    let mut logc = <W<V> as Simd>::Floats::filled_default();
    let mut kd = <W<V> as Simd>::Floats::filled_default();
    for i in 0..V::LANES {
        let bits = ix.as_slice()[i];
        let tmp = bits.wrapping_sub(LOG_OFF);
        let idx = ((tmp >> 19) % 16) as usize;
        kd.as_mut_slice()[i] = ((tmp as i32) >> 23) as f64;
        zs.as_mut_slice()[i] = f32::from_bits(bits.wrapping_sub(tmp & 0xff80_0000));
        invc.as_mut_slice()[i] = t::TAB[2 * idx];
        logc.as_mut_slice()[i] = t::TAB[2 * idx + 1];
    }
    let z = V::from_array(zs).widen();
    let invc = W::<V>::from_array(invc);
    let logc = W::<V>::from_array(logc);
    let kd = W::<V>::from_array(kd);

    // log(x) = log1p(z/c - 1) + log(c) + k*ln2
    let r = z.mul_add(invc, W::<V>::splat(-1.0));
    let y0 = kd.mul_add(W::<V>::splat(t::LN2), logc);

    let r2 = r * r;
    let y = r.mul_add(W::<V>::splat(t::A1), W::<V>::splat(t::A2));
    let y = r2.mul_add(W::<V>::splat(t::A0), y);
    V::narrow(y.mul_add(r2, y0 + r))
}

/// The table-free path.
///
/// Splits off the exponent, then takes a degree-8 series in
/// `s = (m - 1)/(m + 1)` on the reduced mantissa. All single precision, no
/// widening and no gather.
///
/// Measured error: below 2 ulp over the positive normals.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    /// `ln(2)`.
    const LN2: f32 = core::f32::consts::LN_2;
    /// `sqrt(2)`, the point the mantissa is folded about.
    const SQRT2: f32 = core::f32::consts::SQRT_2;
    /// Coefficients of `2 atanh(s) = 2s(1 + s^2/3 + s^4/5 + ...)`.
    const P: [f32; 4] = [2.0, 2.0 / 3.0, 2.0 / 5.0, 2.0 / 7.0];

    let ix = x.to_bits();
    let mut mant = V::Bits::filled_default();
    let mut es = V::Floats::filled_default();
    for i in 0..V::LANES {
        let b = ix.as_slice()[i];
        // Fold the mantissa into [sqrt(1/2), sqrt(2)) by borrowing a power of
        // two, which keeps `s` small enough for a short series.
        let e = ((b >> 23) & 0xff) as i32 - 127;
        let m = (b & 0x007f_ffff) | 0x3f80_0000;
        let (m, e) = if f32::from_bits(m) < SQRT2 {
            (m, e)
        } else {
            (m - (1 << 23), e + 1)
        };
        mant.as_mut_slice()[i] = m;
        es.as_mut_slice()[i] = e as f32;
    }
    let m = V::from_bits(mant);
    let e = V::from_array(es);

    let s = (m - V::splat(1.0)) / (m + V::splat(1.0));
    let s2 = s * s;
    let s4 = s2 * s2;
    let lo = s2.mul_add(V::splat(P[1]), V::splat(P[0]));
    let hi = s2.mul_add(V::splat(P[3]), V::splat(P[2]));
    let poly = s4.mul_add(hi, lo);
    e.mul_add(V::splat(LN2), s * poly)
}
