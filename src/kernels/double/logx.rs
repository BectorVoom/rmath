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
///
/// [`BitExact`](crate::policy::BitExact) is a port of glibc's
/// `__log10_finite`: disassembly (glibc 2.43, x86-64) shows it is a thin
/// wrapper, not its own table algorithm — it extracts the unbiased exponent
/// `k` and a rounding-parity bit `i`, forces `x`'s exponent field to
/// `0x3ff - i` (so the reduced argument sits within one exponent step of
/// 1.0), calls straight into `__ieee754_log_fma` (i.e. [`super::ln`]'s own
/// `bit_exact` table walk — reused here rather than re-derived), and combines
/// with three *unfused* operations, in exactly the order `e_log10.c`'s own
/// `z = y*log10_2lo + ivln10*log(x); return z + y*log10_2hi;` reads — a first
/// pass at this mapped the two spilled products to the wrong constants
/// (the compiler schedules `y*log10_2lo` before the call, `y*log10_2hi`
/// after, so reading top-to-bottom without checking which literal address
/// held which value swapped them); confirmed by reading the four constant
/// bit patterns live out of process memory rather than trusting position.
/// No `vfmadd` anywhere in `__log10_finite` itself — the final combine is
/// genuinely three separate roundings, not one fused into another.
pub mod log10 {
    use super::*;

    /// `1 / ln(10)`, bit-identical to `__log10_finite`'s `ivln10` (verified
    /// by reading the live constant out of process memory, not transcribed
    /// from the C source's decimal comment).
    const INV_LN10: f64 = f64::from_bits(0x3fdbcb7b1526e50e);
    /// `log10(2)`, for scaling the exponent without going through `ln`.
    ///
    /// Only used by [`fast`]; [`bit_exact`] needs the split `LOG10_2HI`/
    /// `LOG10_2LO` pair below instead, since a single rounding of
    /// `log10(2)` is not what the platform computes.
    const LOG10_2: f64 = f64::from_bits(0x3fd34413509f79ff);
    /// High part of `log10(2)`, `__log10_finite`'s `log10_2hi`.
    const LOG10_2HI: f64 = f64::from_bits(0x3fd34413509f6000);
    /// Low part of `log10(2)`, `__log10_finite`'s `log10_2lo`.
    const LOG10_2LO: f64 = f64::from_bits(0x3d59fef311f12b36);

    /// `log10(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
        if D::CHECKED {
            patch_lanes(x, y, not_positive_normal(x), reference::log10)
        } else {
            y
        }
    }

    /// The reduce-and-delegate-to-`ln` path, lane-for-lane identical to
    /// `__log10_finite`.
    ///
    /// Only handles the positive-normal case; zero, negative, subnormal,
    /// infinite and NaN lanes are the reference's job via [`eval`]'s
    /// `patch_lanes`, the same convention [`super::ln::bit_exact`] itself
    /// uses — `__log10_finite`'s own subnormal pre-scale and special-value
    /// branches are therefore not replayed here at all.
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let ix = x.to_bits();
        let mut y_a = V::Floats::filled_default();
        let mut r_bits = V::Bits::filled_default();
        for lane in 0..V::LANES {
            let b = ix.as_slice()[lane];
            // `x` is positive-normal here, so the raw biased exponent is
            // exactly `b >> 52`; no prescale is needed the way
            // `__log10_finite` needs one for subnormal input.
            let k = (b >> 52) as i64 - 1023;
            // `i`: 1 when `k` is negative, 0 otherwise -- `__log10_finite`'s
            // own rounding-parity bit, read via `k >> 63` as an unsigned
            // shift in the disassembly.
            let i = ((k as u64) >> 63) as i64;
            y_a.as_mut_slice()[lane] = (k + i) as f64;
            let exp_field = (0x3ffu64.wrapping_sub(i as u64)) << 52;
            r_bits.as_mut_slice()[lane] = (b & 0x000f_ffff_ffff_ffff) | exp_field;
        }
        let reduced = V::from_bits(r_bits);
        let y = V::from_array(y_a);

        let log_reduced = crate::kernels::double::ln::bit_exact(reduced);
        // Deliberately not `mul_add`: the disassembly has three separate,
        // unfused `mulsd`/`addsd` pairs here, not a fusion opportunity.
        (log_reduced * V::splat(INV_LN10) + y * V::splat(LOG10_2LO)) + y * V::splat(LOG10_2HI)
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
