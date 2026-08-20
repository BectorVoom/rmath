//! `sqrt(x^2 + y^2)`, without the intermediate overflow.
//!
//! [`BitExact`](crate::policy::BitExact) is a port of glibc's modern (2021+)
//! Borges "MyHypot3" compensated correction (`sysdeps/ieee754/dbl-64/e_hypot.c`),
//! replayed lane-parallel matching [`reference::hypot()`]. Under `Fast`, the
//! vector path uses the division-based scaling form.

use crate::kernels::{dispatch2, not_normal, outside};
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Mask, Simd, patch_lanes2};

/// Above this, `a * sqrt(1 + t^2)` could still overflow under `Fast`.
const SCALE_LIMIT: f64 = 1.0e308 / 2.0;

/// `2^-600`: the down-scale for the huge-`ax` branch.
const SCALE: f64 = f64::from_bits(0x1a70000000000000);
/// `2^600`: the up-scale for the tiny-`ay` branch (`1 / SCALE`).
const INV_SCALE: f64 = f64::from_bits(0x6570000000000000);
/// `2^511`: above this, squaring could overflow even after scaling down.
const LARGE_VAL: f64 = f64::from_bits(0x5fe0000000000000);
/// `2^-459`: below this, squaring could underflow to zero.
const TINY_VAL: f64 = f64::from_bits(0x2340000000000000);
/// `2^-54`: below this ratio, the smaller argument cannot affect the result.
const EPS: f64 = f64::from_bits(0x3c90000000000000);

/// `hypot(x, y)` for vectors of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V, y: V) -> V {
    if A::BIT_EXACT {
        let z = bit_exact(x, y);
        if D::CHECKED {
            patch_lanes2(
                x,
                y,
                z,
                outside(x, f64::INFINITY).or(outside(y, f64::INFINITY)),
                reference::hypot,
            )
        } else {
            z
        }
    } else {
        dispatch2::<V, A, D>(x, y, reference::hypot, fast, |x, y| {
            // Zeros, subnormals and non-finites all go to the reference: the
            // quotient below is `0/0` for a zero pair, and the special-value rules
            // (`hypot(inf, NaN) == inf`) are not worth restating in vector form.
            let a = x.abs();
            let b = y.abs();
            let big = V::select(a.gt_mask(b), a, b);
            not_normal(x)
                .or(not_normal(y))
                .or(big.gt_mask(V::splat(SCALE_LIMIT)))
        })
    }
}

/// The compensated correction, given `ax >= ay >= 0` scaled so that squaring
/// neither overflows nor underflows.
///
/// Every operation here is a separate rounding by design (matching glibc's
/// non-FMA build); replaying it with `mul_add` anywhere would break the EFT
/// error-extraction identities.
#[inline(always)]
fn kernel<V: Simd<Elem = f64>>(ax: V, ay: V) -> V {
    let two = V::splat(2.0);
    let mut h = (ax * ax + ay * ay).sqrt();
    let two_ay = two * ay;
    let two_ax = two * ax;
    let cond = h.le_mask(two_ay);

    // Branch 1: h <= 2.0 * ay
    let delta1 = h - ay;
    let t1_1 = ax * (two * delta1 - ax);
    let t2_1 = (delta1 - (two_ax - two_ay)) * delta1;

    // Branch 2: h > 2.0 * ay
    let delta2 = h - ax;
    let t1_2 = two * delta2 * (ax - two_ay);
    let t2_2 = (V::splat(4.0) * delta2 - ay) * ay + delta2 * delta2;

    let t1 = V::select(cond, t1_1, t1_2);
    let t2 = V::select(cond, t2_1, t2_2);

    h = h - (t1 + t2) / (two * h);
    h
}

/// The bit-exact vector path.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>>(x: V, y: V) -> V {
    let scale = V::splat(SCALE);
    let inv_scale = V::splat(INV_SCALE);
    let one = V::splat(1.0);

    let x = x.abs();
    let y = y.abs();
    let ax = V::select(x.gt_mask(y), x, y);
    let ay = V::select(x.gt_mask(y), y, x);

    let eps_mask = ay.le_mask(ax * V::splat(EPS));

    let is_large = ax.gt_mask(V::splat(LARGE_VAL));
    let is_tiny = ay.lt_mask(V::splat(TINY_VAL)).and(is_large.not());

    let in_scale = V::select(is_large, scale, V::select(is_tiny, inv_scale, one));
    let out_scale = V::select(is_large, inv_scale, V::select(is_tiny, scale, one));

    // Clamp to 1.0 when eps_mask is true to prevent 0.0/0.0 in kernel
    let kx = V::select(eps_mask, one, ax * in_scale);
    let ky = V::select(eps_mask, one, ay * in_scale);

    let kh = kernel(kx, ky) * out_scale;
    V::select(eps_mask, ax + ay, kh)
}

/// Measured error: below 2 ulp for normal, non-zero arguments.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V, y: V) -> V {
    let a = x.abs();
    let b = y.abs();
    let big = V::select(a.gt_mask(b), a, b);
    let small = V::select(a.gt_mask(b), b, a);
    // `t` is in [0, 1], so `1 + t*t` is between 1 and 2 and cannot leave the
    // exponent range whatever the inputs were.
    let t = small / big;
    big * t.mul_add(t, V::splat(1.0)).sqrt()
}
