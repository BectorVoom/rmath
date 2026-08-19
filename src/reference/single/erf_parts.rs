//! `erff` and `erfcf`.
//!
//! Both are correctly rounded on this platform — glibc takes them from
//! CORE-MATH — so, as with [`crate::reference::double::erf_parts`],
//! reproducing them is not a claim about *this* `libm`: a correctly rounded
//! result is the same result everywhere.
//!
//! Neither needs the double-double machinery the `f64` versions do. A `float`
//! has 24 significand bits and a `double` polynomial evaluated over a table of
//! 56 intervals is comfortably inside half an ulp of one, so both routines are
//! a table lookup and a short `double` polynomial, rounded once. That is the
//! best possible shape for this crate: straight-line `f64` arithmetic, which
//! [`crate::kernels::single`] replays eight lanes at a time.

use crate::tables::single::erf as t;

/// `bits(0x1.f5a888p+1)`: the largest `f32` whose `erf` is not 1.
pub(crate) const ERF_MAX_BITS: u32 = 0x407a_d444;
/// `bits(0.4375f)`: below this the `i < 7` series runs instead of the table.
pub(crate) const ERF_SMALL_BITS: u32 = 0x3ee0_0000;

/// `erf(x)`, correctly rounded — and so bit-identical to glibc's `erff`.
pub fn erf(x: f32) -> f32 {
    let ax = x.abs();
    let ux = ax.to_bits();
    let s = x as f64;
    let z = ax as f64;

    if ux > ERF_MAX_BITS {
        let os = 1.0f32.copysign(x);
        if ux > 0xff << 23 {
            return x + x; // NaN
        }
        if ux == 0xff << 23 {
            return os;
        }
        return os - f32::from_bits(0x3300_0000) * os; // 0x1p-25
    }

    if ux < ERF_SMALL_BITS {
        return erf_small(s) as f32;
    }
    erf_table(z, s) as f32
}

/// The `|x| < 0.4375` branch: a degree-15 odd series, by Estrin.
#[inline(always)]
pub(crate) fn erf_small(s: f64) -> f64 {
    let c = &t::C_SMALL;
    let z2 = s * s;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let c0 = c[0] + z2 * c[1];
    let c2 = c[2] + z2 * c[3];
    let c4 = c[4] + z2 * c[5];
    let c6 = c[6] + z2 * c[7];
    let c0 = c0 + z4 * c2;
    let c4 = c4 + z4 * c6;
    let c0 = c0 + z8 * c4;
    s * c0
}

/// The table branch: a degree-7 polynomial per sixteenth of `|x|`.
#[inline(always)]
pub(crate) fn erf_table(z: f64, s: f64) -> f64 {
    let v = (16.0 * z).floor();
    let i = (16.0 * z) as usize;
    let z = (z - 0.03125) - 0.0625 * v;
    let c = &t::C[i - 7];

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

// ---------------------------------------------------------------------------
// erfcf
// ---------------------------------------------------------------------------

/// `bits(-0x1.ea8f94p+1)`: below this `erfc` rounds to 2.
pub(crate) const ERFC_NEG_LIMIT: u32 = 0xc075_47ca;
/// `bits(0x1.41bbf8p+3)`: at or above this `erfc(x) < 2^-150`.
pub(crate) const ERFC_POS_LIMIT: u32 = 0x4120_ddfc;
/// `bits(0x1.7p-4)`: at or below this the near-zero series runs.
pub(crate) const ERFC_NEAR_ZERO: u32 = 0x3db8_0000;
/// `bits(0x1.c5bf88p-26)`: at or below this `erfc(x)` rounds to 1.
pub(crate) const ERFC_UNIT: u32 = 0x32e2_dfc4;
/// The one argument the Chebyshev fit does not resolve.
pub(crate) const ERFC_EXCEPTION: u32 = 0xb76c_9f62;
/// `|x| > 0x1.0a2p+1` selects the second Chebyshev fit.
pub(crate) const ERFC_SPLIT: u32 = 0x4005_1000;

/// `1/ln(2)`.
pub(crate) const ILN2: f64 = f64::from_bits(0x3ff71547652b82fe);
/// `ln(2)/128`, high part — exactly representable, so `j * LN2H` is exact.
pub(crate) const LN2H: f64 = f64::from_bits(0x3f762e42fefa0000);
/// `ln(2)/128`, low part.
pub(crate) const LN2L: f64 = f64::from_bits(0x3d0cf79abd6f5dc8);
/// The rounding-trick constant: `1024 + 0x1p-8`.
pub(crate) const SHIFT: f64 = 1024.0 + f64::from_bits(0x3f70000000000000);

/// `erfc(x)`, correctly rounded — and so bit-identical to glibc's `erfcf`.
pub fn erfc(x: f32) -> f32 {
    let t_bits = x.to_bits();
    let at = t_bits & 0x7fff_ffff;
    let sgn = (t_bits >> 31) as usize;

    if t_bits > ERFC_NEG_LIMIT {
        // x < -0x1.ea8f94p+1, or x is a negative NaN.
        if t_bits >= 0xff80_0000 {
            if t_bits == 0xff80_0000 {
                return 2.0;
            }
            return x + x; // NaN
        }
        return 2.0 - f32::from_bits(0x3300_0000); // rounds to 2 or below it
    }
    if at >= ERFC_POS_LIMIT {
        if at >= 0x7f80_0000 {
            if at == 0x7f80_0000 {
                return 0.0;
            }
            return x + x; // NaN
        }
        // erfc(x) < 2^-150: zero, or the smallest subnormal under a directed
        // rounding mode.
        return f32::from_bits(1) * 0.25;
    }
    if at <= ERFC_NEAR_ZERO {
        if t_bits == ERFC_EXCEPTION {
            return f32::from_bits(0x3f800085) + f32::from_bits(0x3300_0000);
        }
        if at <= ERFC_UNIT {
            if at == 0 {
                return 1.0;
            }
            // 1 - erf(x) rounds to the value either side of 1, by the sign.
            const D: [f32; 2] = [f32::from_bits(0xb280_0000), f32::from_bits(0x3300_0000)];
            return 1.0 + D[sgn];
        }
        return erfc_near_zero(x as f64, (x as f64) * (x as f64)) as f32;
    }
    let axd = x.abs() as f64;
    erfc_main(axd, axd * axd, sgn, at > ERFC_SPLIT) as f32
}

/// The `|x| <= 0x1.7p-4` branch: `erfc(x) = 1 - (odd series)`.
#[inline(always)]
pub(crate) fn erfc_near_zero(xd: f64, x2: f64) -> f64 {
    let c = &t::CN;
    1.0 - xd * (c[0] + x2 * (c[1] + x2 * (c[2] + x2 * (c[3] + x2 * c[4]))))
}

/// The main branch: `exp(-x^2)` from a 128-entry table, times a Chebyshev fit
/// of `erfc(x) exp(x^2)` in `z = (|x| - a) / (|x| + b)`.
#[inline(always)]
pub(crate) fn erfc_main(axd: f64, x2: f64, sgn: usize, split: bool) -> f64 {
    // `j ~= -round(128 * x^2 / ln 2)`, read straight out of the significand
    // of `x^2/ln2 - (1024 + 2^-8)`: the added constant fixes the binade, so
    // the significand's top bits *are* the scaled integer.
    let jt = (x2 * ILN2 + -SHIFT).to_bits();
    let j = ((jt << 12) as i64) >> 48;

    // `2^(j/128)` split as `2^(j >> 7) * 2^((j & 127)/128)`, with the sign of
    // `x` folded into the exponent field of the first factor — which is what
    // turns `erfc(-|x|) = 2 - erfc(|x|)` into one add at the end.
    let s = f64::from_bits((((j >> 7) + (0x3ff | ((sgn as i64) << 11))) as u64) << 52);

    let ch = &t::CH;
    let d = (x2 + LN2H * j as f64) + LN2L * j as f64;
    let d2 = d * d;
    let e0 = t::E[(j & 127) as usize];
    let f = d + d2 * ((ch[0] + d * ch[1]) + d2 * (ch[2] + d * ch[3]));

    let ct = &t::CT[split as usize];
    let z = (axd - ct[0]) / (axd + ct[1]);
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let c = &ct[3..];
    let poly = (((c[0] + z * c[1]) + z2 * (c[2] + z * c[3]))
        + z4 * ((c[4] + z * c[5]) + z2 * (c[6] + z * c[7])))
        + z8 * (((c[8] + z * c[9]) + z2 * (c[10] + z * c[11])) + z4 * c[12]);
    let poly = ct[2] + z * poly;

    let r = (s * (e0 - f * e0)) * poly;
    if sgn == 1 { 2.0 + r } else { r }
}
