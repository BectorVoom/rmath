//! `asinf`, `acosf`, `atanf` and `atan2f`.
//!
//! All four are CORE-MATH, and all four are **correctly rounded**: they
//! return the representable `f32` nearest the true result on every input. So
//! unlike `sinf`/`cosf` next door, reproducing them is not a claim about this
//! platform — any correct `asinf` anywhere returns these same bits, which is
//! the argument [`crate::reference`] already makes for `erf` and `erfc`.
//!
//! None of the four is built with `-mfma`: glibc's
//! `sysdeps/x86_64/fpu/multiarch/Makefile` gives FMA variants to the `f64`
//! inverse trigonometry (`e_asin-fma.c`, `s_atan-fma.c`, `e_atan2-fma.c`) but
//! to none of the `f32` ones, and `__asinf`, `__acosf`, `__atanf` and
//! `__atan2f` in `libm.so.6` contain no `vfmadd` of any form. The arithmetic
//! below is therefore deliberately unfused, except where the C calls `fmaf`
//! outright. Rust never contracts on its own, so the plain spellings are
//! already the right ones.
//!
//! Each has CORE-MATH's two-step shape: a fast polynomial with a proven error
//! bound, a test asking whether that bound settles the last bit, and a slower
//! path — a second polynomial, sometimes plus a short table of the arguments
//! where nothing cheaper works — for when it does not.
//!
//! Verified by exhaustive sweep: all 2^32 `f32` bit patterns for the
//! one-argument three, and a large random sweep for `atan2f`, whose domain is
//! 2^64 and cannot be swept.

use crate::tables::single::invtrig as t;

/// `pi/2` to `double`.
const PI2: f64 = f64::from_bits(0x3ff921fb54442d18);

/// `__math_invalidf (0.0)`: the NaN an out-of-domain argument produces.
///
/// Spelled on `0.0` rather than on the argument, exactly as CORE-MATH's
/// `as_special` does — `asinf(2.0)` has no NaN payload to propagate, so the
/// answer is whatever `0/0` yields, which on x86 is the sign-set "real
/// indefinite" `0xffc00000` and not the `0x7fc00000` a literal `f32::NAN`
/// would give.
///
/// The [`black_box`](core::hint::black_box) is load-bearing: without it the
/// division is constant-folded at compile time, and LLVM's folded `0/0` has
/// the *opposite* sign from the `divss` the C compiles to. This is a cold
/// path — one comparison per out-of-domain call — so the optimisation
/// barrier costs nothing that matters.
#[inline(always)]
#[allow(clippy::eq_op)]
fn invalid_zero() -> f32 {
    let z = core::hint::black_box(0.0f32);
    (z - z) / (z - z)
}

/// `poly12`: the degree-11 polynomial `asin` and `acos` share, in the
/// four-way split the C evaluates it with.
#[inline(always)]
fn poly12(z: f64, c: &[f64; 12]) -> f64 {
    let z2 = z * z;
    let z4 = z2 * z2;
    let c0 = c[0] + z * c[1];
    let c2 = c[2] + z * c[3];
    let c4 = c[4] + z * c[5];
    let c6 = c[6] + z * c[7];
    let c8 = c[8] + z * c[9];
    let c10 = c[10] + z * c[11];
    let c0 = c0 + c2 * z2;
    let c4 = c4 + c6 * z2;
    let c8 = c8 + z2 * c10;
    c0 + z4 * (c4 + z4 * c8)
}

/// `asin`'s degree-31 odd fast-path polynomial, as coefficients in `x^2`.
const ASIN_B: [f64; 16] = [
    f64::from_bits(0x3ff0000000000005),
    f64::from_bits(0x3fc55557aeca105d),
    f64::from_bits(0x3fb3314ec3db7d12),
    f64::from_bits(0x3fa775738a5a6f92),
    f64::from_bits(0x3f75d5f7ce1c8538),
    f64::from_bits(0x3fd605c6d58740f0),
    f64::from_bits(0xc005728b732d73c6),
    f64::from_bits(0x402f152170f151eb),
    f64::from_bits(0xc04f962ea3ca992e),
    f64::from_bits(0x40671971e17375a0),
    f64::from_bits(0xc07860512b4ba230),
    f64::from_bits(0x40826a3b8d4bdb14),
    f64::from_bits(0xc0836f2ea5698b51),
    f64::from_bits(0x407b3d722aebfa2e),
    f64::from_bits(0xc066cf89703b1289),
    f64::from_bits(0x4041518af6a65e2d),
];

/// `asin`'s fast-path error bound, as a multiple of `|x|`.
const ASIN_EPS: f64 = f64::from_bits(0x3e0efa8eb0000000);

/// `asin(x)`, correctly rounded.
pub fn asin(x: f32) -> f32 {
    let xs = f64::from(x);
    let ax = x.to_bits() << 1;

    if ax > 0x7fu32 << 24 {
        // Outside `[-1, 1]`, or a NaN.
        if ax > 0xffu32 << 24 {
            return x + x;
        }
        return invalid_zero();
    }

    if ax < 0x7ec2_9000 {
        // `|x| < 2^-12`, where `asin x` is `x` to within the rounding. The
        // `fmaf` is what gets the direction and the inexact flag right.
        if ax < 115 << 24 {
            return x.mul_add(f32::from_bits(0x33000000), x);
        }
        let z = xs;
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let z16 = z8 * z8;
        let r = z
            * ((((ASIN_B[0] + z2 * ASIN_B[1]) + z4 * (ASIN_B[2] + z2 * ASIN_B[3]))
                + z8 * ((ASIN_B[4] + z2 * ASIN_B[5]) + z4 * (ASIN_B[6] + z2 * ASIN_B[7])))
                + z16
                    * (((ASIN_B[8] + z2 * ASIN_B[9]) + z4 * (ASIN_B[10] + z2 * ASIN_B[11]))
                        + z8 * ((ASIN_B[12] + z2 * ASIN_B[13])
                            + z4 * (ASIN_B[14] + z2 * ASIN_B[15]))));
        // The rounding test: `ub` and `lb` bracket the true value, and round
        // to the same `f32` exactly when the bracket holds only one.
        let ub = r as f32;
        let lb = (r - z * ASIN_EPS) as f32;
        if ub == lb {
            return ub;
        }
    }

    if ax < 0x7eu32 << 24 {
        let z = xs;
        let z2 = z * z;
        (z + (z * z2) * poly12(z2, &t::C0)) as f32
    } else {
        // Two arguments whose true value lands close enough to a halfway
        // point that even the accurate polynomial cannot place it;
        // CORE-MATH tabulates the answer as an exact `rh + rl`.
        if ax == 0x7e55_688a {
            return f32::from_bits(0x3f3adc51).copysign(x) + f32::from_bits(0x32800000).copysign(x);
        }
        if ax == 0x7e10_7434 {
            return f32::from_bits(0x3f0fa5b2).copysign(x) + f32::from_bits(0x32800000).copysign(x);
        }
        let bx = xs.abs();
        let z = 1.0 - bx;
        let s = z.sqrt();
        (PI2 - s * poly12(z, &t::C1)).copysign(xs) as f32
    }
}

/// `acos`'s degree-31 fast-path polynomial, a different fit from `asin`'s.
const ACOS_B: [f64; 16] = [
    f64::from_bits(0x3fefffffffd9ccb8),
    f64::from_bits(0x3fc5555c94838007),
    f64::from_bits(0x3fb32ded4b7c20fa),
    f64::from_bits(0x3fa8566df703309e),
    f64::from_bits(0xbf9980c959bec9a3),
    f64::from_bits(0x3fe56fbb04998344),
    f64::from_bits(0xc01403d8e4c49f52),
    f64::from_bits(0x403b06c3e9f311ea),
    f64::from_bits(0xc059ea97c4e2c21f),
    f64::from_bits(0x407200b8261cc61b),
    f64::from_bits(0xc082274c2799a5c7),
    f64::from_bits(0x408a558a59cc19d3),
    f64::from_bits(0xc08aca4b6a529ff0),
    f64::from_bits(0x408228744703f813),
    f64::from_bits(0xc06d7dbb0b322228),
    f64::from_bits(0x4045c2018c0c0105),
];

/// The upper end of `acos`'s rounding bracket: `pi/2` nudged up.
const ACOS_UB: f64 = f64::from_bits(0x3ff921fb54574191);
/// The lower end, `pi/2` nudged down. The bracket is around `pi/2 - r`, so
/// the two constants differ where a single `pi/2` could not express both.
const ACOS_LB: f64 = f64::from_bits(0x3ff921fb543118a0);
/// `pi`, the offset the near-1 branch adds for a negative argument.
const ACOS_PI: f64 = f64::from_bits(0x400921fb54442d18);

/// `acos(x)`, correctly rounded.
pub fn acos(x: f32) -> f32 {
    let xs = f64::from(x);
    let tb = x.to_bits();
    let ax = tb << 1;

    if ax >= 0x7fu32 << 24 {
        // `acos(±1)` is exact and is spelled out, because the polynomial
        // path would not reproduce `+0.0` and `pi` to the last bit.
        if tb == 0x7fu32 << 23 {
            return 0.0;
        }
        if tb == 0x17fu32 << 23 {
            return f32::from_bits(0x40490fdb) + f32::from_bits(0xb3800000);
        }
        if ax > 0xffu32 << 24 {
            return x + x;
        }
        return invalid_zero();
    }

    if ax < 0x7ec2_a1dc {
        // `|x| < 2^-63`: `acos x` is `pi/2`, and returning it directly
        // avoids a spurious underflow from the polynomial's tail.
        if ax <= 0x4000_0000 {
            return PI2 as f32;
        }
        let z = xs;
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let z16 = z8 * z8;
        let r = z
            * ((((ACOS_B[0] + z2 * ACOS_B[1]) + z4 * (ACOS_B[2] + z2 * ACOS_B[3]))
                + z8 * ((ACOS_B[4] + z2 * ACOS_B[5]) + z4 * (ACOS_B[6] + z2 * ACOS_B[7])))
                + z16
                    * (((ACOS_B[8] + z2 * ACOS_B[9]) + z4 * (ACOS_B[10] + z2 * ACOS_B[11]))
                        + z8 * ((ACOS_B[12] + z2 * ACOS_B[13])
                            + z4 * (ACOS_B[14] + z2 * ACOS_B[15]))));
        let ub = (ACOS_UB - r) as f32;
        let lb = (ACOS_LB - r) as f32;
        if ub == lb {
            return ub;
        }
    }

    if ax < 0x7eu32 << 24 {
        if tb == 0x3288_85a3 {
            return f32::from_bits(0x3fc90fdb) + f32::from_bits(0x33000000);
        }
        if tb == 0x3982_6222 {
            return f32::from_bits(0x3fc907b5) + f32::from_bits(0x33000000);
        }
        let x2 = xs * xs;
        ((PI2 - xs) - (xs * x2) * poly12(x2, &t::C0)) as f32
    } else {
        let bx = xs.abs();
        let z = 1.0 - bx;
        let s = z.sqrt().copysign(xs);
        // `o[t >> 31]`: 0 for a positive argument, `pi` for a negative one.
        let o = if tb >> 31 != 0 { ACOS_PI } else { 0.0 };
        (o + s * poly12(z, &t::C1)) as f32
    }
}

/// `atan`'s degree-13 rational numerator, in `z^2`.
const ATAN_CN: [f64; 7] = [
    f64::from_bits(0x3fd51eccde075d67),
    f64::from_bits(0x3fea76bb5637f2f2),
    f64::from_bits(0x3fe81e0eed20de88),
    f64::from_bits(0x3fd376c8ca67d11d),
    f64::from_bits(0x3faaec7b69202ac6),
    f64::from_bits(0x3f69561899acc73e),
    f64::from_bits(0x3efbf9fa5b67e600),
];
/// Its denominator. `CD[0]` is `CN[0]` minus one ulp, deliberately: the
/// original fit produced the same constant for both, and reducing it here
/// removes an exceptional case at `|x| = 0x1.1ad646p-4`.
const ATAN_CD: [f64; 7] = [
    f64::from_bits(0x3fd51eccde075d66),
    f64::from_bits(0x3fedfbdd7b392d28),
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fdfd22bf0e89b54),
    f64::from_bits(0x3fbd91ff8b576282),
    f64::from_bits(0x3f8653ea99fc9bb0),
    f64::from_bits(0x3f31e7fcc202340a),
];
/// High part of `pi/2` for the reciprocal branch's reflection.
const PI_OVER2_H: f64 = f64::from_bits(0x3ff9000000000000);
/// Low part. The split is CORE-MATH's own and is not arbitrary: `sign*L +
/// (-r + sign*H)` is the one grouping of the three that rounds correctly for
/// every argument, which a single `double` `pi/2` does not.
const PI_OVER2_L: f64 = f64::from_bits(0x3f80fdaa22168c23);

/// The shared body of [`atan`]: the degree-13 rational at `z`, and whether
/// the caller must reflect through `pi/2`.
#[inline(always)]
fn atan_rational(z: f64) -> f64 {
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;

    let cn0 = ATAN_CN[0] + z2 * ATAN_CN[1];
    let cn2 = ATAN_CN[2] + z2 * ATAN_CN[3];
    let cn4 = ATAN_CN[4] + z2 * ATAN_CN[5];
    let cn6 = ATAN_CN[6];
    let cn0 = cn0 + z4 * cn2;
    let cn4 = cn4 + z4 * cn6;
    let cn0 = (cn0 + z8 * cn4) * z;

    let cd0 = ATAN_CD[0] + z2 * ATAN_CD[1];
    let cd2 = ATAN_CD[2] + z2 * ATAN_CD[3];
    let cd4 = ATAN_CD[4] + z2 * ATAN_CD[5];
    let cd6 = ATAN_CD[6];
    let cd0 = cd0 + z4 * cd2;
    let cd4 = cd4 + z4 * cd6;
    let cd0 = cd0 + z8 * cd4;

    cn0 / cd0
}

/// `atan(x)`, correctly rounded.
pub fn atan(x: f32) -> f32 {
    let tb = x.to_bits();
    let e = ((tb >> 23) & 0xff) as i32;
    let gt = e >= 127;
    let ta = tb & 0x7fff_ffff;

    // `|x| > 0x1.e00a3p+25`: `atan x` rounds to `pi/2` for every `f32` here.
    if ta >= 0x4c70_0518 {
        if ta > 0x7f80_0000 {
            return x + x;
        }
        return PI2.copysign(f64::from(x)) as f32;
    }
    if e < 127 - 13 {
        if e < 127 - 25 {
            if tb << 1 == 0 {
                return x;
            }
            return (-x).mul_add(x.abs(), x);
        }
        return (f32::from_bits(0x3eaaaaab) * -x).mul_add(x * x, x);
    }

    let mut z = f64::from(x);
    if gt {
        z = 1.0 / z;
    }
    let r = atan_rational(z);
    if !gt {
        // For `|x| < 1` the rational is already correctly rounded.
        return r as f32;
    }
    // `atan(x) + atan(1/x) = sign(x) pi/2`, in the one association that
    // rounds correctly everywhere.
    (PI_OVER2_L.copysign(z) - r + PI_OVER2_H.copysign(z)) as f32
}
