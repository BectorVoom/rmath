//! Natural logarithm.
//!
//! * [`crate::policy::BitExact`] — glibc's `__ieee754_log_fma` schedule. Two
//!   paths: a 128-subinterval `1/c`, `log(c)` table, and a separate
//!   double-double polynomial for `0.9375 <= x < 1 + 0x1.09p-4`, where the
//!   table path would lose most of its significant digits to cancellation.
//!   **Both are evaluated for every lane and blended**, because a vector
//!   cannot take one branch for some lanes and another for the rest — except
//!   that the near-1.0 path is skipped wholesale when no lane needs it, which
//!   is the common case.
//! * [`crate::policy::Fast`] — no table. Reduces to a mantissa in
//!   `[sqrt(2)/2, sqrt(2))` and sums the `atanh` series.
//!
//! `Finite` means `x` is a positive normal number: zero, negatives,
//! subnormals, infinities and NaN all need the reference.

use crate::kernels::not_positive_normal;
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Lanes, Mask, Simd, patch_lanes};
use crate::tables::double::log as t;

/// `ln(x)` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, not_positive_normal(x), reference::ln)
    } else {
        y
    }
}

/// Low end of the near-1.0 window, `1.0 - 0x1p-4`.
const NEAR_LO: f64 = 0.9375;
/// High end, `1.0 + 0x1.09p-4`.
const NEAR_HI: f64 = f64::from_bits(0x3ff1090000000000);
/// The table-centring offset, `bits(0x1.6p-1)`.
const OFF: u64 = 0x3fe6000000000000;

/// The table path, lane-for-lane identical to `__ieee754_log_fma`.
///
/// Exposed to [`super::logx::log10`], which reduces its argument to `x`'s own
/// mantissa with a forced near-1.0 exponent field and calls straight into
/// this — `log10`'s own glibc wrapper does exactly that (confirmed by
/// disassembly: `__log10_finite` calls `__ieee754_log_fma` directly, not a
/// separate routine), so replaying it here rather than re-deriving the same
/// 128-entry table walk a second time is the schedule, not a shortcut.
#[inline(always)]
pub(super) fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
    let ix = x.to_bits();

    let mut invc_a = V::Floats::filled_default();
    let mut logc_a = V::Floats::filled_default();
    let mut z_bits = V::Bits::filled_default();
    let mut kd_a = V::Floats::filled_default();
    for i in 0..V::LANES {
        let b = ix.as_slice()[i];
        let tmp = b.wrapping_sub(OFF);
        let idx = ((tmp >> 45) & 127) as usize;
        z_bits.as_mut_slice()[i] = b.wrapping_sub(tmp & (0xfffu64 << 52));
        invc_a.as_mut_slice()[i] = t::TAB[2 * idx];
        logc_a.as_mut_slice()[i] = t::TAB[2 * idx + 1];
        kd_a.as_mut_slice()[i] = ((tmp as i64) >> 52) as f64;
    }
    let z = V::from_bits(z_bits);
    let kd = V::from_array(kd_a);

    let w = kd.mul_add(V::splat(t::LN2HI), V::from_array(logc_a));
    let r = z.mul_add(V::from_array(invc_a), V::splat(-1.0));
    let q12 = r.mul_add(V::splat(t::A2), V::splat(t::A1));
    let hi = r + w;
    let r2 = r * r;
    let t_ = (w - hi) + r;
    let lo = kd.mul_add(V::splat(t::LN2LO), t_);
    let r3 = r * r2;
    let q34 = r.mul_add(V::splat(t::A4), V::splat(t::A3));
    let s1 = r2.mul_add(V::splat(t::A0), lo);
    let q = r2.mul_add(q34, q12);
    let main = r3.mul_add(q, s1) + hi;

    let near = x
        .ge_mask(V::splat(NEAR_LO))
        .and(x.lt_mask(V::splat(NEAR_HI)));
    if near.none() {
        return main;
    }
    V::select(near, near_one(x), main)
}

/// The near-1.0 polynomial, with the Veltkamp split that keeps `r - r^2/2`
/// in double-double.
#[inline(always)]
fn near_one<V: Simd<Elem = f64>>(x: V) -> V {
    let r = x - V::splat(1.0);
    let p12 = r.mul_add(V::splat(t::B2), V::splat(t::B1));
    let p45 = r.mul_add(V::splat(t::B5), V::splat(t::B4));
    let r2 = r * r;
    let p78 = r.mul_add(V::splat(t::B8), V::splat(t::B7));
    let p123 = r2.mul_add(V::splat(t::B3), p12);
    let p456 = r2.mul_add(V::splat(t::B6), p45);
    let r3 = r * r2;
    let p789 = r2.mul_add(V::splat(t::B9), p78);
    let p78910 = r3.mul_add(V::splat(t::B10), p789);
    let pin = p78910.mul_add(r3, p456);
    let poly = pin.mul_add(r3, p123);

    const SPLIT: f64 = 134217728.0; // 0x1p27
    let split = V::splat(SPLIT);
    let rhi_t = r.mul_add(split, r);
    let rhi = (-r).mul_add(split, rhi_t);
    let rlo = r - rhi;
    let s = rhi * rhi;
    let b0 = V::splat(t::B0);
    let hi = s.mul_add(b0, r);
    let tt = r - hi;
    let rpr = r + rhi;
    let lo = s.mul_add(b0, tt);
    let lo = (b0 * rlo).mul_add(rpr, lo);

    // x == 1.0 must give exactly +0.0; the polynomial gives -0.0 there.
    let y = poly.mul_add(r3, lo) + hi;
    V::select(x.eq_mask(V::splat(1.0)), V::splat(0.0), y)
}

/// `1/(2i+1)` for `i` in `0..10`: the `atanh` series
/// `ln(m) = 2 s (1 + s²/3 + s⁴/5 + ...)`, `s = (m-1)/(m+1)`.
///
/// With `m` in `[sqrt(2)/2, sqrt(2))`, `|s| <= 0.1716` so `s² <= 0.0295`, and
/// dropping everything past `s^18` leaves a relative error near `2e-17`.
const S: [f64; 10] = [
    1.0,
    1.0 / 3.0,
    1.0 / 5.0,
    1.0 / 7.0,
    1.0 / 9.0,
    1.0 / 11.0,
    1.0 / 13.0,
    1.0 / 15.0,
    1.0 / 17.0,
    1.0 / 19.0,
];

/// `sqrt(2)`, the point at which the mantissa is folded to centre the series.
const SQRT2: f64 = core::f64::consts::SQRT_2;
/// `ln(2)`, split so that `k * LN2HI` is exact for every exponent `k`.
///
/// The digits are not stray precision — see the note in [`super::exp`].
#[allow(clippy::excessive_precision)]
const LN2HI: f64 = 6.93147180369123816490e-01;

/// `ln(2)`, low part. See [`LN2HI`].
#[allow(clippy::excessive_precision)]
const LN2LO: f64 = 1.90821492927058770002e-10;

#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V) -> V {
    let ix = x.to_bits();
    let mut k_a = V::Floats::filled_default();
    let mut m_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let b = ix.as_slice()[i];
        k_a.as_mut_slice()[i] = (((b >> 52) & 0x7ff) as i64 - 1023) as f64;
        // Mantissa in [1, 2).
        m_bits.as_mut_slice()[i] = (b & 0x000f_ffff_ffff_ffff) | (1023u64 << 52);
    }
    let m = V::from_bits(m_bits);
    let k = V::from_array(k_a);

    // Fold [sqrt(2), 2) down to [sqrt(2)/2, sqrt(2)) so the series is centred
    // and |s| is at its smallest worst case.
    let big = m.ge_mask(V::splat(SQRT2));
    let m = V::select(big, m * V::splat(0.5), m);
    let k = k + V::select(big, V::splat(1.0), V::splat(0.0));

    let s = (m - V::splat(1.0)) / (m + V::splat(1.0));
    let s2 = s * s;
    let s4 = s2 * s2;
    let s8 = s4 * s4;

    let c01 = s2.mul_add(V::splat(S[1]), V::splat(S[0]));
    let c23 = s2.mul_add(V::splat(S[3]), V::splat(S[2]));
    let c45 = s2.mul_add(V::splat(S[5]), V::splat(S[4]));
    let c67 = s2.mul_add(V::splat(S[7]), V::splat(S[6]));
    let c89 = s2.mul_add(V::splat(S[9]), V::splat(S[8]));

    let lo = s4.mul_add(c23, c01);
    let hi = s4.mul_add(c67, c45);
    let hi = s8.mul_add(c89, hi);
    let poly = s8.mul_add(hi, lo);

    // Split ln(2) so the k term stays exact for large |k|.
    let two_s = s + s;
    two_s.mul_add(poly, k.mul_add(V::splat(LN2LO), k * V::splat(LN2HI)))
}
