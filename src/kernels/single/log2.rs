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

/// The table-free path.
///
/// The same reduction as [`super::ln`]'s `Fast` path — split off the exponent,
/// fold the mantissa about `sqrt(2)`, take `s = (m - 1)/(m + 1)` — but the
/// series carries the base-2 scale inside its generated coefficients, and the
/// exponent needs no `ln 2` multiply at all: `log2(x) = e + s P(s²)`, with `e`
/// exact and the final add fused. All single precision, no widening and no
/// gather.
///
/// Measured error: at most 2 ulp, verified by an exhaustive sweep of every
/// positive normal `f32` (`tests/ulp_scan.rs::scan_single_precision_exhaustive`),
/// not a sample. Reaching that from a previously-measured 3 ulp needed
/// compensating the division that forms `s = (m - 1)/(m + 1)`: `m - 1` is
/// exact here (`m` is folded into `[sqrt(1/2), sqrt2)`), but `m + 1` rounds,
/// and that rounding was the worst case's dominant term. `dlo`/`slo` below
/// recover it with the same branch-free identity `pow.rs::log2_dd` uses at
/// `f64`, at the cost of one extra division-free fma folded into the final
/// combine.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    /// `sqrt(2)`, the point the mantissa is folded about.
    const SQRT2: f32 = core::f32::consts::SQRT_2;
    use crate::tables::single::poly::LOG2_ATANH as P;

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

    // `s = (m - 1)/(m + 1)`: `m - 1` is exact (`m` is within
    // `[sqrt(1/2), sqrt2)`), but `m + 1` rounds, and that rounding is the
    // worst case's dominant term. Recover its residue (`dlo`, the same
    // branch-free Fast2Sum `pow.rs::log2_dd` uses) and fold it into `s`'s own
    // division residue (`slo`, by the usual `s_true ~= s + slo` identity) so
    // the final combine corrects for both in one extra fma.
    let one = V::splat(1.0);
    let u = m - one;
    let d = m + one;
    let dlo = V::select(m.ge_mask(one), (m - d) + one, (one - d) + m);
    let s = u / d;
    let slo = ((-s).mul_add(d, u) - s * dlo) / d;

    let s2 = s * s;
    let s4 = s2 * s2;
    let lo = s2.mul_add(V::splat(P[1]), V::splat(P[0]));
    let hi = s2.mul_add(V::splat(P[3]), V::splat(P[2]));
    let poly = s4.mul_add(hi, lo);
    slo.mul_add(poly, s.mul_add(poly, e))
}
