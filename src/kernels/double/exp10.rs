//! `10^x`.
//!
//! * [`crate::policy::BitExact`] — glibc's `__exp10` schedule: reduction
//!   against `log10(2)/128`, the 128-entry `2^(k/128)` table `exp` and `exp2`
//!   already share, and a degree-4 correction. Bit-identical to the scalar
//!   `exp10`, and — unlike the scalar routine — eight lanes at a time.
//! * [`crate::policy::Fast`] — no table. Reduces to `|r| <= log10(2)/2` and
//!   takes the degree-13 Taylor series of `10^r`, by Estrin's scheme. Measured
//!   error below 2 ulp over `|x| < 256`.
//!
//! Every operation on the bit-exact path is a separate multiply and add:
//! glibc ships no FMA variant of `exp10`, so fusing anything here would be
//! more accurate and would stop the results matching.
//!
//! `Finite` means `|x| < 256`. Outside it the table path is wrong rather than
//! imprecise, and note that `exp10` itself only overflows at 308.25 — the safe
//! range is set by the reduction, not by the function's domain.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Lanes, Simd, patch_lanes};
use crate::tables::double::exp as et;
use crate::tables::double::exp10 as t;

/// Widest `|x|` the vector main path is valid for: `top12(0x1p8)`.
const MAIN_PATH_LIMIT: f64 = 256.0;

/// `10^x` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::exp10)
    } else {
        y
    }
}

/// The table path, lane-for-lane identical to [`reference::exp10`]'s main body.
///
/// The tiny-argument branch of the scalar routine needs no counterpart here:
/// for `|x| < 2^-57` the reduction produces `k = 0` and `r = x`, and the main
/// path evaluates to `1 + x*ln(10)`, which rounds to the same `1 + x` the
/// scalar routine returns early with.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
    let shift = V::splat(et::SHIFT);
    let z = V::splat(t::INVLOG10_2N) * x;
    let kd_s = z + shift;
    let ki = kd_s.to_bits();
    let kd = kd_s - shift;

    let r = V::splat(t::NEGLOG10_2HIN) * kd + x;
    let r = V::splat(t::NEGLOG10_2LON) * kd + r;

    let mut tail_bits = V::Bits::filled_default();
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let k = ki.as_slice()[i];
        let idx = ((k & 127) * 2) as usize;
        tail_bits.as_mut_slice()[i] = et::TAB[idx];
        scale_bits.as_mut_slice()[i] = et::TAB[idx + 1].wrapping_add(k << 45);
    }
    let tail = V::from_bits(tail_bits);
    let scale = V::from_bits(scale_bits);

    let r2 = r * r;
    let p = V::splat(t::C0) + r * V::splat(t::C1);
    let y = V::splat(t::C2) + r * V::splat(t::C3);
    let y = y + r2 * V::splat(t::C4);
    let y = p + r2 * y;
    let tmp = tail + y * r;
    scale * tmp + scale
}

/// The table-free path.
///
/// The same shape as [`crate::kernels::double::exp`]'s: reduce to an integer
/// power of two plus a small remainder, then a degree-13 series by Estrin, so
/// the dependency chain is six levels deep rather than thirteen.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V) -> V {
    let shift = V::splat(et::SHIFT);
    let kd_s = x.mul_add(V::splat(t::LOG2_10), shift);
    let kd = kd_s - shift;
    // Cody-Waite in base 10: subtract `kd * log10(2)` in two exact pieces.
    let r = kd.mul_add(
        V::splat(-t::LOG10_2LO),
        kd.mul_add(V::splat(-t::LOG10_2HI), x),
    );

    let g = &t::G;
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;

    let c01 = r.mul_add(V::splat(g[0]), V::splat(1.0));
    let c23 = r.mul_add(V::splat(g[2]), V::splat(g[1]));
    let c45 = r.mul_add(V::splat(g[4]), V::splat(g[3]));
    let c67 = r.mul_add(V::splat(g[6]), V::splat(g[5]));
    let c89 = r.mul_add(V::splat(g[8]), V::splat(g[7]));
    let cab = r.mul_add(V::splat(g[10]), V::splat(g[9]));
    let ccd = r.mul_add(V::splat(g[12]), V::splat(g[11]));

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
