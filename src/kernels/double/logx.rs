//! `log2`, `log10` and `log1p`.
//!
//! [`log2`] is a port: the vector code replays glibc's `__ieee754_log2_fma`
//! schedule lane-parallel, so its `BitExact` path is both exact and fast.
//! [`log10`] and [`log1p`] delegate under `BitExact` and vectorise under
//! `Fast`; see [`crate::kernels`] for what that distinction means.

use crate::kernels::{dispatch, log_poly, log_split, not_positive_normal, outside};
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Lanes, Mask, Simd, patch_lanes};
use crate::tables::double::poly as p;

/// Base-2 logarithm.
///
/// A port of glibc's `__ieee754_log2_fma`: a 64-subinterval `1/c` / `log2(c)`
/// table, a degree-6 correction, and a separate near-1.0 polynomial where the
/// main path would lose too much to cancellation. Both paths are computed
/// across all lanes and blended, because branching between them would
/// serialise the vector.
pub mod log2 {
    use super::*;
    use crate::tables::double::log2 as t;

    /// Low end of the near-1.0 window, `1.0 - 0x1.5b51p-5`.
    const NEAR_LO: f64 = f64::from_bits(0x3fef4a4ef0000000);
    /// High end, `1.0 + 0x1.6ab2p-5`.
    const NEAR_HI: f64 = f64::from_bits(0x3ff016ab20000000);
    /// The table-centring offset, `bits(0x1.6p-1)`.
    const OFF: u64 = 0x3fe6000000000000;

    /// `log2(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
        if D::CHECKED {
            patch_lanes(x, y, not_positive_normal(x), reference::log2)
        } else {
            y
        }
    }

    /// The table path, lane-for-lane identical to the platform's.
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let ix = x.to_bits();

        let mut invc_a = V::Floats::filled_default();
        let mut logc_a = V::Floats::filled_default();
        let mut z_bits = V::Bits::filled_default();
        let mut kd_a = V::Floats::filled_default();
        for i in 0..V::LANES {
            let b = ix.as_slice()[i];
            let tmp = b.wrapping_sub(OFF);
            let idx = ((tmp >> 46) & 63) as usize;
            z_bits.as_mut_slice()[i] = b.wrapping_sub(tmp & (0xfffu64 << 52));
            invc_a.as_mut_slice()[i] = t::TAB[2 * idx];
            logc_a.as_mut_slice()[i] = t::TAB[2 * idx + 1];
            kd_a.as_mut_slice()[i] = ((tmp as i64) >> 52) as f64;
        }
        let z = V::from_bits(z_bits);
        let kd = V::from_array(kd_a);

        // r = z/c - 1, then r/ln2 carried in double-double as t1 + t2.
        let r = z.mul_add(V::from_array(invc_a), V::splat(-1.0));
        let ihi = V::splat(t::INVLN2HI);
        let t1 = r * ihi;
        let t2 = r.mul_add(V::splat(t::INVLN2LO), r.mul_add(ihi, -t1));

        let t3 = kd + V::from_array(logc_a);
        let hi = t3 + t1;
        let lo = t3 - hi + t1 + t2;

        let r2 = r * r;
        let r4 = r2 * r2;
        let poly = V::splat(t::A0)
            + r * V::splat(t::A1)
            + r2 * (V::splat(t::A2) + r * V::splat(t::A3))
            + r4 * (V::splat(t::A4) + r * V::splat(t::A5));
        let main = lo + r2 * poly + hi;

        let near = x
            .ge_mask(V::splat(NEAR_LO))
            .and(x.lt_mask(V::splat(NEAR_HI)));
        if near.none() {
            return main;
        }
        V::select(near, near_one(x), main)
    }

    /// The near-1.0 path, where `log2(x)` is small and the table path's
    /// `hi + lo` would cancel away its own accuracy.
    #[inline(always)]
    fn near_one<V: Simd<Elem = f64>>(x: V) -> V {
        let r = x - V::splat(1.0);
        let ihi = V::splat(t::INVLN2HI);
        let hi0 = r * ihi;
        let lo0 = r.mul_add(V::splat(t::INVLN2LO), r.mul_add(ihi, -hi0));

        let r2 = r * r;
        let r4 = r2 * r2;
        let pp = r2 * (V::splat(t::B0) + r * V::splat(t::B1));
        let y = hi0 + pp;
        let lo = lo0 + (hi0 - y + pp);
        let tail = r4
            * (V::splat(t::B2)
                + r * V::splat(t::B3)
                + r2 * (V::splat(t::B4) + r * V::splat(t::B5))
                + r4 * (V::splat(t::B6)
                    + r * V::splat(t::B7)
                    + r2 * (V::splat(t::B8) + r * V::splat(t::B9))));
        y + (lo + tail)
    }

    /// The table-free path: the shared significand fold, scaled to base 2.
    ///
    /// Measured error: below 2 ulp over the positive normals.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let (e, s) = log_split(x);
        log_poly(s).mul_add(V::splat(p::LOG2E), e)
    }
}

/// Base-10 logarithm.
pub mod log10 {
    use super::*;

    /// `1 / ln(10)`.
    const INV_LN10: f64 = f64::from_bits(0x3fdbcb7b1526e50e);
    /// `log10(2)`, for scaling the exponent without going through `ln`.
    const LOG10_2: f64 = f64::from_bits(0x3fd34413509f79ff);

    /// `log10(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::log10, fast, not_positive_normal)
    }

    /// Measured error: below 2 ulp over the positive normals.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        // `e * log10(2) + ln(m) / ln(10)` rather than `ln(x) / ln(10)`: the
        // exponent term is then exact to within one rounding instead of
        // inheriting the error of a large `ln`.
        let (e, s) = log_split(x);
        log_poly(s).mul_add(V::splat(INV_LN10), e * V::splat(LOG10_2))
    }
}

/// `ln(1 + x)`, accurate for small `x`.
pub mod log1p {
    use super::*;

    /// `log1p(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::log1p, fast, |x| {
            // Valid wherever `1 + x` is positive and finite. `-1` itself, and
            // anything below it, is the reference's problem.
            x.gt_mask(V::splat(-1.0))
                .and(outside(x, f64::MAX).not())
                .not()
        })
    }

    /// Measured error: below 2 ulp.
    ///
    /// One formula for the whole domain, with no branch. `u = fl(1 + x)`
    /// loses the low bits of `x` when `|x|` is small, but `c = (u - 1) - x`
    /// recovers exactly what was lost — the subtraction is exact by Sterbenz
    /// wherever it matters — and `ln(u - c) = ln(u) - c/u` to first order
    /// restores it. For tiny `x` this collapses to `0 - (-x)/1 = x`, which is
    /// the correct answer rather than a special case.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let one = V::splat(1.0);
        let u = one + x;
        let c = (u - one) - x;
        let (e, s) = log_split(u);
        let ln_u = log_poly(s).mul_add(one, e * V::splat(core::f64::consts::LN_2));
        ln_u - c / u
    }
}
