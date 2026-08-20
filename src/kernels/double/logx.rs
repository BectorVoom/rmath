//! `log2`, `log10` and `log1p`.
//!
//! [`log2`], [`log10`] and [`log1p`] are ports: their vector code replays glibc's
//! schedule lane-parallel, so their `BitExact` paths are both exact and fast;
//! see [`crate::kernels`] for what that distinction means.

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
    const NEAR_LO: f64 = f64::from_bits(0x3feea4af00000000);
    /// High end, `1.0 + 0x1.6ab2p-5`.
    const NEAR_HI: f64 = f64::from_bits(0x3ff0b55900000000);
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
        let a01 = r.mul_add(V::splat(t::A1), V::splat(t::A0));
        let a23 = r.mul_add(V::splat(t::A3), V::splat(t::A2));
        let a45 = r.mul_add(V::splat(t::A5), V::splat(t::A4));
        let poly = r4.mul_add(a45, r2.mul_add(a23, a01));
        let main = hi + r2.mul_add(poly, lo);

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
    ///
    /// Replays the exact FMA schedule of glibc's `__log2_fma`.
    #[inline(always)]
    fn near_one<V: Simd<Elem = f64>>(x: V) -> V {
        let r = x - V::splat(1.0);
        let ihi = V::splat(t::INVLN2HI);
        let ilo = V::splat(t::INVLN2LO);

        let hi0 = r * ihi;
        let r2 = r * r;
        let r4 = r2 * r2;

        let lo0 = r.mul_add(ilo, r.mul_add(ihi, -hi0));

        let b01 = r.mul_add(V::splat(t::B1), V::splat(t::B0));
        let y = r2.mul_add(b01, hi0);
        let lo = lo0 + r2.mul_add(b01, hi0 - y);

        let b23 = r.mul_add(V::splat(t::B3), V::splat(t::B2));
        let b45 = r.mul_add(V::splat(t::B5), V::splat(t::B4));
        let b23_45 = r2.mul_add(b45, b23);

        let b67 = r.mul_add(V::splat(t::B7), V::splat(t::B6));
        let b89 = r.mul_add(V::splat(t::B9), V::splat(t::B8));
        let b67_89 = r2.mul_add(b89, b67);

        let tail = r4.mul_add(b67_89, b23_45);
        y + r4.mul_add(tail, lo)
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

    /// High part of `ln(2)`; `n * LN2_HI` is exact for every `|n| < 2000`.
    const LN2_HI: f64 = f64::from_bits(0x3fe62e42fee00000);
    /// Low part of `ln(2)`.
    const LN2_LO: f64 = f64::from_bits(0x3dea39ef35793c76);
    /// `Lp[1..=7]`: the odd-series minimax coefficients for `R(z)` on `s in [0, 0.1716]`.
    const LP: [f64; 7] = [
        f64::from_bits(0x3FE5555555555593),
        f64::from_bits(0x3FD999999997FA04),
        f64::from_bits(0x3FD2492494229359),
        f64::from_bits(0x3FCC71C51D8E78AF),
        f64::from_bits(0x3FC7466496CB03DE),
        f64::from_bits(0x3FC39A09D078C69F),
        f64::from_bits(0x3FC2F112DF3E5244),
    ];

    /// `log1p(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact(x);
            if D::CHECKED {
                patch_lanes(
                    x,
                    y,
                    x.gt_mask(V::splat(-1.0)).and(outside(x, f64::MAX).not()).not(),
                    reference::log1p,
                )
            } else {
                y
            }
        } else {
            dispatch::<V, A, D>(x, reference::log1p, fast, |x| {
                // Valid wherever `1 + x` is positive and finite. `-1` itself, and
                // anything below it, is the reference's problem.
                x.gt_mask(V::splat(-1.0))
                    .and(outside(x, f64::MAX).not())
                    .not()
            })
        }
    }

    /// Replays glibc's `__log1p_fma` schedule lane-parallel, matching [`reference::log1p`].
    #[allow(unused_assignments)]
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let ix = x.to_bits();
        let xs = x.to_array();

        let mut f0_a = V::Floats::filled_default();
        let mut kf_a = V::Floats::filled_default();
        let mut c_a = V::Floats::filled_default();
        let mut hu_zero_a = V::Floats::filled_default();
        let mut k_zero_a = V::Floats::filled_default();
        let mut is_tiny_a = V::Floats::filled_default();
        let mut tiny_res_a = V::Floats::filled_default();

        for i in 0..V::LANES {
            let xi = xs.as_slice()[i];
            let bits = ix.as_slice()[i];
            let hx = (bits >> 32) as i32;
            let ax = (hx & 0x7fffffff) as u32;

            let mut k = 1i32;
            let mut c = 0.0f64;
            let mut hu: u32;
            let f0: f64;

            if hx < 0x3FDA827A {
                // x < 0.41422
                if ax < 0x3e200000 {
                    // |x| < 2^-29
                    is_tiny_a.as_mut_slice()[i] = 1.0;
                    tiny_res_a.as_mut_slice()[i] = if ax < 0x3c900000 {
                        // |x| < 2^-54
                        xi
                    } else {
                        (xi * xi).mul_add(-0.5, xi)
                    };
                    f0 = 0.0;
                    k = 0;
                    hu = 1;
                } else if hx > 0 || hx <= 0xbfd2bec3u32 as i32 {
                    // -0.2929 < x < 0.41422: direct, no reduction needed.
                    k = 0;
                    hu = 1;
                    f0 = xi;
                } else {
                    // -0.41422 <= x <= -0.2929: reduce
                    let uu = 1.0 + xi;
                    let huw = (uu.to_bits() >> 32) as i32;
                    k = (huw >> 20) - 1023;
                    c = if k > 0 {
                        1.0 - (uu - xi)
                    } else {
                        xi - (uu - 1.0)
                    };
                    c /= uu;
                    let u = uu;
                    hu = (huw as u32) & 0x000fffff;
                    let low32 = u.to_bits() & 0xffff_ffff;
                    let u_norm;
                    if hu < 0x6a09e {
                        u_norm = f64::from_bits(((hu | 0x3ff00000) as u64) << 32 | low32);
                    } else {
                        k += 1;
                        u_norm = f64::from_bits(((hu | 0x3fe00000) as u64) << 32 | low32);
                        hu = (0x00100000u32.wrapping_sub(hu)) >> 2;
                    }
                    f0 = u_norm - 1.0;
                }
            } else {
                // x >= 0.41422
                let u: f64;
                if hx < 0x43400000 {
                    let uu = 1.0 + xi;
                    let huw = (uu.to_bits() >> 32) as i32;
                    k = (huw >> 20) - 1023;
                    c = if k > 0 {
                        1.0 - (uu - xi)
                    } else {
                        xi - (uu - 1.0)
                    };
                    c /= uu;
                    u = uu;
                    hu = (huw as u32) & 0x000fffff;
                } else {
                    u = xi;
                    hu = ((u.to_bits() >> 32) as u32) & 0x000fffff;
                    k = ((u.to_bits() >> 52) as i32) - 1023;
                    c = 0.0;
                }
                let low32 = u.to_bits() & 0xffff_ffff;
                let u_norm;
                if hu < 0x6a09e {
                    u_norm = f64::from_bits(((hu | 0x3ff00000) as u64) << 32 | low32);
                } else {
                    k += 1;
                    u_norm = f64::from_bits(((hu | 0x3fe00000) as u64) << 32 | low32);
                    hu = (0x00100000u32.wrapping_sub(hu)) >> 2;
                }
                f0 = u_norm - 1.0;
            }

            f0_a.as_mut_slice()[i] = f0;
            kf_a.as_mut_slice()[i] = k as f64;
            c_a.as_mut_slice()[i] = c;
            hu_zero_a.as_mut_slice()[i] = if hu == 0 { 1.0 } else { 0.0 };
            k_zero_a.as_mut_slice()[i] = if k == 0 { 1.0 } else { 0.0 };
        }

        let f = V::from_array(f0_a);
        let kf = V::from_array(kf_a);
        let c = V::from_array(c_a);
        let hu_is_zero = V::from_array(hu_zero_a).gt_mask(V::splat(0.5));
        let k_is_zero = V::from_array(k_zero_a).gt_mask(V::splat(0.5));
        let is_tiny = V::from_array(is_tiny_a).gt_mask(V::splat(0.5));
        let tiny_res = V::from_array(tiny_res_a);

        let hfsq = V::splat(0.5) * f * f;

        // hu == 0 path (|f| < 2^-20)
        let r_hu0 = f.mul_add(V::splat(-0.666_666_666_666_666_6), V::splat(1.0)) * hfsq;
        let hu0_k0 = f - r_hu0;
        let hu0_knon0 = kf * V::splat(LN2_HI) - ((r_hu0 - (kf * V::splat(LN2_LO) + c)) - f);
        let f_zero_knon0 = kf.mul_add(V::splat(LN2_HI), c + kf * V::splat(LN2_LO));
        let f_is_zero = f.eq_mask(V::splat(0.0));
        let hu0_res = V::select(
            f_is_zero,
            V::select(k_is_zero, V::splat(0.0), f_zero_knon0),
            V::select(k_is_zero, hu0_k0, hu0_knon0),
        );

        // hu != 0 path
        let s = f / (V::splat(2.0) + f);
        let z = s * s;
        let r2 = z.mul_add(V::splat(LP[2]), V::splat(LP[1]));
        let z2 = z * z;
        let r2z2 = r2 * z2;
        let z4 = z2 * z2;
        let r3 = z.mul_add(V::splat(LP[4]), V::splat(LP[3]));
        let z6 = z4 * z2;
        let r4 = z.mul_add(V::splat(LP[6]), V::splat(LP[5]));
        let acc0 = z.mul_add(V::splat(LP[0]), r2z2);
        let acc1 = z4.mul_add(r3, acc0);
        let r = z6.mul_add(r4, acc1);

        let res_k0 = f - (hfsq - s * (hfsq + r));
        let res_knon0 = kf.mul_add(
            V::splat(LN2_HI),
            -((hfsq - (s * (hfsq + r) + kf.mul_add(V::splat(LN2_LO), c))) - f),
        );
        let hu_nonzero_res = V::select(k_is_zero, res_k0, res_knon0);

        let tail_res = V::select(hu_is_zero, hu0_res, hu_nonzero_res);
        V::select(is_tiny, tiny_res, tail_res)
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
