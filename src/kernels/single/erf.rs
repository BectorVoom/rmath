//! The error function, single precision.
//!
//! Both branches of the platform's `erff` are plain `double` arithmetic over a
//! 56-interval table, so an `f32x8` widens to an `f64x8`, replays the schedule
//! lane-parallel and narrows once. Correctly rounded, and therefore bit-exact
//! to any correctly-rounded `erff` on any platform — see
//! [`crate::reference::single::erf_parts`].
//!
//! Unlike the double-precision kernel there is no rounding test and no
//! accurate path: 24 significand bits leave enough headroom that the
//! polynomial settles the rounding on its own, for every one of the 2^32
//! inputs. `tests/glibc.rs` checks all of them.
//!
//! Both policies run the same code, as they do for
//! [`crate::kernels::exact`] — there is no cheaper approximation worth having
//! when the exact one is already branch-free vector arithmetic.
//!
//! `Finite` means `|x| <= 0x1.f5a888p+1`. Above it `erf(x)` is `+-1`.

use crate::policy::{Accuracy, Domain};
use crate::reference::single::erf_parts as r;
use crate::simd::{Mask, Simd, patch_lanes};
use crate::tables::single::erf as t;

/// `erf(x)` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let _ = A::BIT_EXACT;
    let y = main(x);
    if !D::CHECKED {
        return y;
    }
    // Above 0x1.f5a888p+1 the answer is the value just below 1, signed —
    // which is two vector operations, so it is computed here rather than
    // deferred to the scalar reference. That matters more than it looks: on a
    // corpus spanning [-6, 6] roughly two lanes in five are out of range, so
    // sending them out of the vector would cost most of the speedup.
    let os = V::splat(1.0).copysign(x);
    let saturated = os - V::splat(f32::from_bits(0x3300_0000)) * os; // 0x1p-25
    let out = x.abs().gt_mask(V::splat(f32::from_bits(r::ERF_MAX_BITS)));
    let y = V::select(out, saturated, y);
    // Only the infinities and NaN are left, and they are genuinely rare.
    patch_lanes(x, y, crate::kernels::outside(x, f32::INFINITY), r::erf)
}

/// Whichever branch the lanes need, widened and narrowed once.
///
/// Guarded rather than blended. The table branch gathers eight coefficients
/// per lane, which is the expensive part of this kernel, and a vector of small
/// arguments has no business paying for it — nor a vector of large ones for
/// the series. A whole-vector test is a branch on data, which is right at this
/// granularity and wrong inside the arithmetic.
#[inline(always)]
fn main<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let s = x.widen();
    let z = s.abs();

    // `|x| < 0.4375`, below which `floor(16|x|)` drops under the table's
    // first interval.
    let is_small = z.lt_mask(W::<V>::splat(f32::from_bits(r::ERF_SMALL_BITS) as f64));
    if is_small.all() {
        return V::narrow(small_series::<V>(s));
    }
    let tab = table::<V>(z, s);
    if is_small.none() {
        return V::narrow(tab);
    }
    V::narrow(W::<V>::select(is_small, small_series::<V>(s), tab))
}

/// The `|x| < 0.4375` odd series, by Estrin.
#[inline(always)]
fn small_series<V: Simd<Elem = f32>>(s: <V as Simd>::Wide) -> <V as Simd>::Wide {
    type W<V> = <V as Simd>::Wide;
    let c = &t::C_SMALL;
    let z2 = s * s;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let c0 = W::<V>::splat(c[0]) + z2 * W::<V>::splat(c[1]);
    let c2 = W::<V>::splat(c[2]) + z2 * W::<V>::splat(c[3]);
    let c4 = W::<V>::splat(c[4]) + z2 * W::<V>::splat(c[5]);
    let c6 = W::<V>::splat(c[6]) + z2 * W::<V>::splat(c[7]);
    let c0 = c0 + z4 * c2;
    let c4 = c4 + z4 * c6;
    s * (c0 + z8 * c4)
}

/// The table branch: one degree-7 polynomial per sixteenth of `|x|`.
#[inline(always)]
fn table<V: Simd<Elem = f32>>(z: <V as Simd>::Wide, s: <V as Simd>::Wide) -> <V as Simd>::Wide {
    use crate::simd::Lanes;
    type W<V> = <V as Simd>::Wide;

    let scaled = z * W::<V>::splat(16.0);
    let v = scaled.floor();

    // The gather: eight coefficients per lane. `i` is clamped because the
    // small-`|x|` lanes index below the table and the out-of-range lanes
    // index above it, and both are discarded after the blend.
    let idx = scaled.to_array();
    let mut cols = [<W<V> as Simd>::Floats::filled_default(); 8];
    for lane in 0..<W<V> as Simd>::LANES {
        let i = (idx.as_slice()[lane] as usize).clamp(7, t::C.len() + 6);
        let row = &t::C[i - 7];
        for (k, col) in cols.iter_mut().enumerate() {
            col.as_mut_slice()[lane] = row[k];
        }
    }
    let c = cols.map(<W<V> as Simd>::from_array);

    let z = (z - W::<V>::splat(0.03125)) - W::<V>::splat(0.0625) * v;
    let z2 = z * z;
    let z4 = z2 * z2;
    let c0 = c[0] + z * c[1];
    let c2 = c[2] + z * c[3];
    let c4 = c[4] + z * c[5];
    let c6 = c[6] + z * c[7];
    let c0 = c0 + z2 * c2;
    let c4 = c4 + z2 * c6;
    (c0 + z4 * c4).copysign(s)
}
