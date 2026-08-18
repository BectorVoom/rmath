//! `2^x`.
//!
//! Shares [`crate::tables::exp`]'s 128-entry table with [`super::exp`] — the
//! reduction differs (`r = x - k/128`, with no `ln 2` to subtract), but the
//! `2^(k/128)` lookup is the same one, which is why adding this function cost
//! a polynomial and nothing else.
//!
//! **This kernel uses no fused multiply-adds in its bit-exact path**, and that
//! is deliberate. glibc ships no `_fma` ifunc variant of `exp2`, so what
//! actually runs on x86-64 is the baseline SSE2 build: separate `mulsd` and
//! `addsd`, two roundings where [`super::exp`] has one. Fusing here would be
//! *more* accurate and would stop matching. See [`crate::reference::exp2`].
//!
//! `Finite` means `|x| < 512`.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain};
use crate::reference;
use crate::simd::{Lanes, Simd, patch_lanes};
use crate::tables::exp as t;

const MAIN_PATH_LIMIT: f64 = 512.0;

/// `2^x` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::exp2)
    } else {
        y
    }
}

/// The table path. Every `*` and `+` here is a separate rounding on purpose.
#[inline(always)]
fn bit_exact<V: Simd>(x: V) -> V {
    let shift = V::splat(t::EXP2_SHIFT);
    let kd_s = x + shift;
    let ki = kd_s.to_bits();
    let kd = kd_s - shift;
    let r = x - kd;

    let mut tail_bits = V::Bits::filled_default();
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let k = ki.as_slice()[i];
        let idx = ((k & 127) * 2) as usize;
        tail_bits.as_mut_slice()[i] = t::TAB[idx];
        scale_bits.as_mut_slice()[i] = t::TAB[idx + 1].wrapping_add(k << 45);
    }
    let tail = V::from_bits(tail_bits);
    let scale = V::from_bits(scale_bits);

    let r2 = r * r;
    let tmp = tail
        + r * V::splat(t::EXP2_C1)
        + r2 * (V::splat(t::EXP2_C2) + r * V::splat(t::EXP2_C3))
        + r2 * r2 * (V::splat(t::EXP2_C4) + r * V::splat(t::EXP2_C5));
    scale + scale * tmp
}

/// `ln(2)^k / k!` for `k` in `2..=13`: the series for `2^r = e^(r ln 2)`.
///
/// With `|r| <= 1/2` the effective argument `r ln 2` stays under `ln(2)/2`,
/// the same worst case [`super::exp`]'s series is cut for, leaving the same
/// sub-ulp truncation error.
const G: [f64; 12] = [
    0.2402265069591007,     // ln(2)^2/2!
    0.055504108664821576,   // ln(2)^3/3!
    0.009618129107628477,   // ln(2)^4/4!
    0.0013333558146428441,  // ln(2)^5/5!
    0.00015403530393381606, // ln(2)^6/6!
    1.5252733804059838e-05, // ln(2)^7/7!
    1.3215486790144305e-06, // ln(2)^8/8!
    1.0178086009239696e-07, // ln(2)^9/9!
    7.054911620801121e-09,  // ln(2)^10/10!
    4.44553827187081e-10,   // ln(2)^11/11!
    2.5678435993488196e-11, // ln(2)^12/12!
    1.3691488853904124e-12, // ln(2)^13/13!
];

/// `ln(2)`, the `k = 1` coefficient.
const LN2: f64 = core::f64::consts::LN_2;

/// The table-free path: reduce to `|r| <= 1/2`, then one Estrin-evaluated
/// series, with `2^k` written straight into the exponent field.
#[inline(always)]
fn fast<V: Simd>(x: V) -> V {
    let shift = V::splat(t::SHIFT);
    let kd_s = x + shift;
    let kd = kd_s - shift;
    let r = x - kd;

    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;

    let c01 = r.mul_add(V::splat(LN2), V::splat(1.0));
    let c23 = r.mul_add(V::splat(G[1]), V::splat(G[0]));
    let c45 = r.mul_add(V::splat(G[3]), V::splat(G[2]));
    let c67 = r.mul_add(V::splat(G[5]), V::splat(G[4]));
    let c89 = r.mul_add(V::splat(G[7]), V::splat(G[6]));
    let cab = r.mul_add(V::splat(G[9]), V::splat(G[8]));
    let ccd = r.mul_add(V::splat(G[11]), V::splat(G[10]));

    let lo = r2.mul_add(c23, c01);
    let mid = r2.mul_add(c67, c45);
    let hi = r2.mul_add(cab, c89);
    let lo = r4.mul_add(mid, lo);
    let hi = r4.mul_add(ccd, hi);
    let p = r8.mul_add(hi, lo);

    let ki = kd_s.to_bits();
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let k = (ki.as_slice()[i] & 0x000f_ffff_ffff_ffff).wrapping_sub(1u64 << 51);
        scale_bits.as_mut_slice()[i] = k.wrapping_add(1023) << 52;
    }
    V::from_bits(scale_bits) * p
}
