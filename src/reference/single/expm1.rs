//! `expm1f` and `cbrtf`.
//!
//! Both CORE-MATH, both **correctly rounded**, and neither built with
//! `-mfma` — glibc's `sysdeps/x86_64/fpu/multiarch/Makefile` has no
//! `s_expm1f-fma.c` or `s_cbrtf-fma.c`, and neither `__expm1f` nor `__cbrtf`
//! in `libm.so.6` contains a `vfmadd`. So the arithmetic below is unfused
//! except at the two places the C calls `fmaf` outright.
//!
//! Both take CORE-MATH's two-step shape: a fast form with a proven error
//! bound, a bracket test asking whether that bound settles the last bit, and
//! a slower refinement for the arguments where it does not — a
//! Newton-corrected cube root in `cbrt`'s case, a longer exponential
//! polynomial in `expm1`'s.
//!
//! Verified by exhaustive sweep: all 2^32 `f32` bit patterns.

/// `expm1`'s degree-3 `2^z` polynomial for the fast path.
const C: [f64; 4] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3f962e42fef4c4e7),
    f64::from_bits(0x3f2ebfd1b232f475),
    f64::from_bits(0x3ebc6b19384ecd93),
];
/// The accurate path's degree-6 replacement, missing the leading `1`
/// because that path factors `s - 1` out by hand.
const CH: [f64; 6] = [
    f64::from_bits(0x3f962e42fefa39ef),
    f64::from_bits(0x3f2ebfbdff82c58f),
    f64::from_bits(0x3ebc6b08d702e0ed),
    f64::from_bits(0x3e43b2ab6fb92e5e),
    f64::from_bits(0x3dc5d886e6d54203),
    f64::from_bits(0x3d4430976b8ce6ef),
];
/// The `|x| < 0.15625` branch's own polynomial: `expm1(z) = z + z^2 * B(z)`,
/// which keeps the cancellation `exp(z) - 1` would suffer out of the answer.
const B: [f64; 8] = [
    f64::from_bits(0x3fdfffffffffffc2),
    f64::from_bits(0x3fc55555555555fe),
    f64::from_bits(0x3fa555555559767f),
    f64::from_bits(0x3f81111111098dc1),
    f64::from_bits(0x3f56c16bca988aa9),
    f64::from_bits(0x3f2a01a07658483f),
    f64::from_bits(0x3efa05b04d2c3503),
    f64::from_bits(0x3ec71de3a960b5e3),
];
/// `TD[k] = 2^(k/32)`, exactly. Held as bit patterns because the exponent is
/// added straight into them rather than applied by a multiply.
const TD: [u64; 32] = [
    0x3ff0000000000000,
    0x3ff059b0d3158574,
    0x3ff0b5586cf9890f,
    0x3ff11301d0125b51,
    0x3ff172b83c7d517b,
    0x3ff1d4873168b9aa,
    0x3ff2387a6e756238,
    0x3ff29e9df51fdee1,
    0x3ff306fe0a31b715,
    0x3ff371a7373aa9cb,
    0x3ff3dea64c123422,
    0x3ff44e086061892d,
    0x3ff4bfdad5362a27,
    0x3ff5342b569d4f82,
    0x3ff5ab07dd485429,
    0x3ff6247eb03a5585,
    0x3ff6a09e667f3bcd,
    0x3ff71f75e8ec5f74,
    0x3ff7a11473eb0187,
    0x3ff82589994cce13,
    0x3ff8ace5422aa0db,
    0x3ff93737b0cdc5e5,
    0x3ff9c49182a3f090,
    0x3ffa5503b23e255d,
    0x3ffae89f995ad3ad,
    0x3ffb7f76f2fb5e47,
    0x3ffc199bdd85529c,
    0x3ffcb720dcef9069,
    0x3ffd5818dcfba487,
    0x3ffdfc97337b9b5f,
    0x3ffea4afa2a490da,
    0x3fff50765b6e4540,
];
/// `32 / ln(2)`: the scale that puts the table index in the low bits.
const ILN2: f64 = f64::from_bits(0x40471547652b82fe);
/// `1.5 * 2^52`, the round-to-integer constant.
const BIG: f64 = f64::from_bits(0x4338000000000000);
/// The fast path's error bound, as a multiple of the table entry.
const EPS: f64 = f64::from_bits(0x3de3b30000000000);
/// High part of `32 / ln(2)`, for the accurate path's re-reduction.
const ILN2H: f64 = f64::from_bits(0x4047154765000000);
/// Low part.
const ILN2L: f64 = f64::from_bits(0x3e55c17f0bbbe880);

/// `exp(x) - 1`, correctly rounded.
pub fn expm1(x: f32) -> f32 {
    let z = f64::from(x);
    let ux = x.to_bits();
    let ax = ux << 1;

    if ax < 0x7c40_0000 {
        // |x| < 0.15625
        if ax < 0x676a_09e8 {
            // |x| < 0x1.6a09e8p-24: `expm1 x` is `x` to within the rounding.
            if ax == 0 {
                return x;
            }
            return x.abs().mul_add(f32::from_bits(0x33000000), x);
        }
        let z2 = z * z;
        let z4 = z2 * z2;
        return (z + z2
            * ((B[0] + z * B[1])
                + z2 * (B[2] + z * B[3])
                + z4 * ((B[4] + z * B[5]) + z2 * (B[6] + z * B[7])))) as f32;
    }
    if ax >= 0x8562_e430 {
        // |x| > 88.72
        if ax > 0xffu32 << 24 {
            return x + x; // NaN
        }
        if ux >> 31 != 0 {
            // `expm1(-inf)` is exactly `-1`; below the underflow threshold
            // the answer is `-1` rounded *up*, which the sum spells.
            if ax == 0xffu32 << 24 {
                return -1.0;
            }
            return -1.0 + f32::from_bits(0x32800000);
        }
        if ax == 0xffu32 << 24 {
            return f32::INFINITY;
        }
        // `__math_oflowf (0)`: `0x1p97f * 0x1p97f`, i.e. `+inf` with the
        // overflow flag raised.
        return f32::INFINITY;
    }

    let a = ILN2 * z;
    let ia = a.round_ties_even();
    let h = a - ia;
    let h2 = h * h;
    let u = (ia + BIG).to_bits();
    let c2 = C[2] + h * C[3];
    let c0 = C[0] + h * C[1];
    // `2^(k/32) * 2^(e)`: the table entry with the exponent added into its
    // bits, which is exact and needs no multiply.
    let sv = f64::from_bits(TD[(u & 0x1f) as usize].wrapping_add((u >> 5) << 52));
    let r = (c0 + h2 * c2) * sv - 1.0;
    let ub = r as f32;
    let lb = (r - sv * EPS) as f32;
    if ub == lb {
        return ub;
    }

    // Below -17.32 every result rounds to the same value just above `-1`.
    if ux > 0xc18a_a123 {
        return -1.0 + f32::from_bits(0x32800000);
    }
    // Re-reduce with `32/ln2` split in two, so `h` is accurate past the
    // point where the single-rounded product lost it.
    let s = sv;
    let h = (ILN2H * z - ia) + ILN2L * z;
    let h2 = h * h;
    let w = s * h;
    ((s - 1.0) + w * ((CH[0] + h * CH[1]) + h2 * ((CH[2] + h * CH[3]) + h2 * (CH[4] + h * CH[5]))))
        as f32
}

/// `2^(i/3)` for `i` in `0..3`, the scale that puts back the exponent
/// remainder integer division dropped.
const ESCALE: [u64; 3] = [0x3ff0000000000000, 0x3ff428a2f98d728b, 0x3ff965fea53d6e3d];
/// The cube root's degree-7 polynomial on `[1, 2)`.
const CBRT_C: [f64; 8] = [
    f64::from_bits(0x3fe2319d352ea5d5),
    f64::from_bits(0x3fe67ad8ee258d1a),
    f64::from_bits(0xbfd9342edf9cbad9),
    f64::from_bits(0x3fcb6388fc510a75),
    f64::from_bits(0xbfb6002455599e2f),
    f64::from_bits(0x3f97b096936192c4),
    f64::from_bits(0xbf6e5577187e8bf8),
    f64::from_bits(0x3f3169ef81d6c34e),
];
/// The polynomial's `1/z` term, carried separately because a rational tail
/// buys more accuracy per coefficient than two more polynomial terms would.
const CBRT_R0: f64 = f64::from_bits(0xbf99931c6c2d19d1);
/// The fast path's relative error bound.
const CBRT_EPS: f64 = f64::from_bits(0x3e185d4f3f7556ec);
/// The Newton step's scale, `-3 / CBRT_R0` folded into one constant.
const CBRT_U0: f64 = f64::from_bits(0xc02ab16ec65d138f);

/// `x^(1/3)`, correctly rounded.
pub fn cbrt(x: f32) -> f32 {
    let u = x.to_bits();
    let mut au = u << 1;
    let sgn = u >> 31;
    let mut e = au >> 24;

    // `x` is subnormal, zero, infinite or a NaN — the C spells this as two
    // comparisons; the range form is the same test.
    if !(1u32 << 24..0xffu32 << 24).contains(&au) {
        if au >= 0xffu32 << 24 {
            return x + x; // inf, NaN
        }
        if au == 0 {
            return x; // +-0
        }
        // Subnormal: normalise, and pay for it in the exponent.
        let nz = au.leading_zeros() - 7;
        au <<= nz;
        e = e.wrapping_sub(nz.wrapping_sub(1));
    }

    let mant = au & 0xff_ffff;
    let e = e.wrapping_add(899);
    let (et, it) = (e / 3, e % 3);
    // The exponent splits three ways: `et` goes into the scale's exponent
    // field, and the remainder picks one of the three `2^(i/3)` factors.
    let mut isc = ESCALE[it as usize];
    isc = isc.wrapping_add((u64::from(et).wrapping_sub(342)) << 52);
    isc |= u64::from(sgn) << 63;
    let cvt2 = f64::from_bits(isc);

    let z = f64::from_bits((u64::from(mant) << 28) | (0x3ffu64 << 52));
    let r0 = CBRT_R0 / z;
    let z2 = z * z;
    let z4 = z2 * z2;
    let mut f = ((CBRT_C[0] + z * CBRT_C[1]) + z2 * (CBRT_C[2] + z * CBRT_C[3]))
        + z4 * ((CBRT_C[4] + z * CBRT_C[5]) + z2 * (CBRT_C[6] + z * CBRT_C[7]))
        + r0;
    let mut r = f * cvt2;
    let ub = r as f32;
    let lb = (r - cvt2 * CBRT_EPS) as f32;
    if ub == lb {
        return ub;
    }

    // One Newton step on `f^3 - z`, which is enough to place every remaining
    // argument — except that the corrected value can itself land exactly on
    // a rounding boundary, which the mantissa test below detects and nudges.
    let h = f * f * f - z;
    f -= (f * r0 * CBRT_U0) * h;
    r = f * cvt2;
    let mut cvt1 = r.to_bits();
    let mut ub = r as f32;
    let m0 = (cvt1 << 19) as i64;
    let m1 = m0 >> 63;
    if ((m0 ^ m1) as u64) < (1u64 << 31) {
        cvt1 = (cvt1.wrapping_add(1u64 << 31)) & 0xffff_ffff_0000_0000;
        ub = f64::from_bits(cvt1) as f32;
    }
    ub
}

/// `sqrt(x^2 + y^2)` without spurious overflow.
///
/// A port of glibc's `__hypotf` (`sysdeps/ieee754/flt-32/e_hypotf.c`), which
/// is not CORE-MATH and needs no polynomial at all: every `f32` square fits
/// a `double` exactly, so one `double` `sqrt` of the exact sum is correctly
/// rounded by construction.
pub fn hypot(x: f32, y: f32) -> f32 {
    if !x.is_finite() || !y.is_finite() {
        // An infinity wins over a quiet NaN, but not over a signalling one:
        // `hypot(inf, sNaN)` must still raise, so it falls through to the
        // addition that quietens it.
        if (x.is_infinite() || y.is_infinite()) && !is_signaling(x) && !is_signaling(y) {
            return f32::INFINITY;
        }
        return x + y;
    }
    let (xd, yd) = (f64::from(x), f64::from(y));
    (xd * xd + yd * yd).sqrt() as f32
}

/// True for a signalling NaN: a NaN whose quiet bit is clear.
#[inline(always)]
fn is_signaling(x: f32) -> bool {
    let b = x.to_bits();
    b & 0x7f80_0000 == 0x7f80_0000 && b & 0x007f_ffff != 0 && b & 0x0040_0000 == 0
}
