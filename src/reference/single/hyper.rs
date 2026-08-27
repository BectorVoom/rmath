//! The `f32` hyperbolic family: `sinhf`, `coshf`, `tanhf`, `asinhf`,
//! `acoshf` and `atanhf`.
//!
//! All six are CORE-MATH and all six are **correctly rounded**, so the bits
//! below are not a claim about this platform: any correct `sinhf` anywhere
//! returns them. None is built with `-mfma` — glibc's
//! `sysdeps/x86_64/fpu/multiarch/Makefile` gives FMA variants to the `f64`
//! hyperbolics (`e_sinh-fma.c`, `s_tanh-fma.c`, `e_atanh-fma.c`) but to none
//! of these, and none of `__sinhf`, `__coshf`, `__tanhf`, `__asinhf`,
//! `__acoshf`, `__atanhf` contains a `vfmadd`. The arithmetic is therefore
//! unfused except where the C calls `fmaf` outright.
//!
//! Three shapes here, not one:
//!
//! * `sinh` and `cosh` share an exponential — one table of `2^(k/32)`, one
//!   short polynomial, evaluated at `+h` and `-h` and then added or
//!   subtracted. That shared work is why they also share
//!   [`crate::tables::single::sincosh`].
//! * `asinh` and `acosh` share a *logarithm*, of `x + sqrt(x^2 ± 1)`, over
//!   [`crate::tables::single::asincosh`]'s 129-entry table.
//! * `tanh` is a single degree-15 rational with no second path at all.
//!
//! `atanh` is the exception to all of this: Rust's `f32::atanh` is not a libm
//! call, so what this module reproduces is Rust's own composition rather than
//! glibc's `atanhf`. See its own documentation.
//!
//! Verified by exhaustive sweep: all 2^32 `f32` bit patterns.

use crate::tables::single::asincosh as at;
use crate::tables::single::sincosh as st;

/// `32 / ln(2)`: the scale that puts `sinh`/`cosh`'s table index in the low
/// bits of the rounded product.
const ILN2: f64 = f64::from_bits(0x40471547652b82fe);
/// High part of `32 / ln(2)`, for the accurate path's re-reduction.
const ILN2H: f64 = f64::from_bits(0x4047154765000000);
/// Low part.
const ILN2L: f64 = f64::from_bits(0x3e55c17f0bbbe880);
/// `1.5 * 2^52`, the round-to-integer constant.
const BIG: f64 = f64::from_bits(0x4338000000000000);
/// High part of `ln(2)`, for the `asinh`/`acosh`/`atanh` accurate paths.
const LN2H: f64 = f64::from_bits(0x3fe62e4000000000);
/// Low part.
const LN2L: f64 = f64::from_bits(0x3eb7f7d1cf79abca);

/// `__math_invalidf (x)`: the NaN an out-of-domain argument produces.
///
/// The [`black_box`](core::hint::black_box) stops Rust folding `0/0` at
/// compile time, which would give the opposite sign from the `divss` the C
/// compiles to. See [`super::invtrig`] for the same note.
#[inline(always)]
#[allow(clippy::eq_op)]
fn invalid(x: f32) -> f32 {
    let x = core::hint::black_box(x);
    (x - x) / (x - x)
}

/// `2^(k/32) * 2^e` for a rounded index carried in `j`'s low bits.
///
/// The exponent is *added into* the table entry's bit pattern rather than
/// applied by a multiply, which is exact and costs nothing.
#[inline(always)]
fn scale(j: i64) -> f64 {
    f64::from_bits(st::TB[(j & 31) as usize].wrapping_add(((j >> 5) as u64) << 52))
}

/// `sinh`'s `|x| < 0.125` odd series.
const SINH_CP: [f64; 4] = [
    f64::from_bits(0x3fc5555555555555),
    f64::from_bits(0x3f811111111146e1),
    f64::from_bits(0x3f2a01a00930dda6),
    f64::from_bits(0x3ec71f92198aa6e9),
];

/// `cosh`'s `|x| < 0.125` even series.
const COSH_CP: [f64; 4] = [
    f64::from_bits(0x3fdfffffffffffe3),
    f64::from_bits(0x3fa55555555723cf),
    f64::from_bits(0x3f56c16bee4a5986),
    f64::from_bits(0x3efa0483fc0328f7),
];

/// `tanh`'s degree-15 rational numerator, in `x^2`.
const TANH_CN: [f64; 8] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fc30877b8b72d33),
    f64::from_bits(0x3f7694aa09ae9e5e),
    f64::from_bits(0x3f14101377abb729),
    f64::from_bits(0x3e9e0392b1db0018),
    f64::from_bits(0x3e12533756e546f7),
    f64::from_bits(0x3d6d62e5abe6ae8a),
    f64::from_bits(0x3c9b06be534182de),
];

/// Its denominator.
const TANH_CD: [f64; 8] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fded99131b0ebea),
    f64::from_bits(0x3fa0d27ed6c95a69),
    f64::from_bits(0x3f47cbdaca0e9fcc),
    f64::from_bits(0x3edb4e60b892578e),
    f64::from_bits(0x3e5a6f707c5c71ab),
    f64::from_bits(0x3dc35a8b6e2cd94c),
    f64::from_bits(0x3d0ca8230677aa01),
];

/// `asinh`'s `|x| <= 0x1.02accep-2` odd series.
const ASINH_C: [f64; 8] = [
    f64::from_bits(0x3fc5555555555553),
    f64::from_bits(0xbfb3333333330e9d),
    f64::from_bits(0x3fa6db6db67cb37a),
    f64::from_bits(0xbf9f1c71699375d0),
    f64::from_bits(0x3f96e8a374c39ff9),
    f64::from_bits(0xbf91c1e98f9d01e1),
    f64::from_bits(0x3f8c277e96d84026),
    f64::from_bits(0xbf8329ff5faf02ab),
];

/// `acosh`'s near-1 series, in `x - 1`; the `sqrt(2z)` prefactor is
/// applied at the call site.
const ACOSH_C: [f64; 8] = [
    f64::from_bits(0xbfb555555555491e),
    f64::from_bits(0x3f9333333319c200),
    f64::from_bits(0xbf76db6da26e5e0e),
    f64::from_bits(0x3f5f1c698b7100ce),
    f64::from_bits(0xbf46e7ca9e152dc2),
    f64::from_bits(0x3f31b4cf187158bd),
    f64::from_bits(0xbf1b5137f11a88e5),
    f64::from_bits(0x3f0056c6c57152b2),
];

/// `sinh(x)`, correctly rounded.
pub fn sinh(x: f32) -> f32 {
    let z = f64::from(x);
    let ux = x.to_bits() << 1;

    if ux > 0x8565_a9f8 {
        // |x| >~ 89.4
        if ux >= 0xff00_0000 {
            return x + x; // ±inf, NaN
        }
        // `__math_oflowf (x < 0)`: `±0x1p97f * 0x1p97f`.
        return if x < 0.0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    if ux < 0x7c00_0000 {
        // |x| < 0.125
        if ux <= 0x7425_0bfe {
            if ux < 0x6600_0000 {
                // |x| < 2^-24
                return x.mul_add(x.abs(), x);
            }
            // The single argument where the cubic below lands on a halfway
            // point; CORE-MATH tabulates it.
            if ux == 0x7425_0bfe {
                let sgn = 1.0f32.copysign(x);
                return sgn * f32::from_bits(0x3a1285ff) + sgn * f32::from_bits(0x2d800000);
            }
            return (x * f32::from_bits(0x3e2aaaab)) * (x * x) + x;
        }
        let z2 = z * z;
        let z4 = z2 * z2;
        return (z
            + (z2 * z) * ((SINH_CP[0] + z2 * SINH_CP[1]) + z4 * (SINH_CP[2] + z2 * SINH_CP[3])))
            as f32;
    }

    // `sinh x = (e^x - e^-x)/2`, with both exponentials read off one table.
    let a = ILN2 * z;
    let ia = a.round_ties_even();
    let h = a - ia;
    let h2 = h * h;
    let jp = (ia + BIG).to_bits() as i64;
    let jm = jp.wrapping_neg();
    let sp = scale(jp);
    let sm = scale(jm);
    let te = st::C[0] + h2 * st::C[2];
    let to = st::C[1] + h2 * st::C[3];
    let r = sp * (te + h * to) - sm * (te - h * to);
    let ub = r as f32;
    let lb = (r - f64::from_bits(0x3de4e406496685f4) * r) as f32;
    if ub == lb {
        return ub;
    }

    let h = (ILN2H * z - ia) + ILN2L * z;
    let h2 = h * h;
    let te = st::CH[0] + h2 * st::CH[2] + (h2 * h2) * (st::CH[4] + h2 * st::CH[6]);
    let to = st::CH[1] + h2 * (st::CH[3] + h2 * st::CH[5]);
    (sp * (te + h * to) - sm * (te - h * to)) as f32
}

/// `cosh(x)`, correctly rounded.
pub fn cosh(x: f32) -> f32 {
    let z = f64::from(x);
    let ax = x.to_bits() << 1;

    if ax > 0x8565_a9f8 {
        if ax >= 0xff00_0000 {
            if ax << 8 != 0 {
                return x + x; // NaN
            }
            return f32::INFINITY; // cosh(±inf)
        }
        return f32::INFINITY; // `__math_oflowf (0)`
    }
    if ax < 0x7c00_0000 {
        if ax < 0x7400_0000 {
            // |x| < 2^-11
            if ax < 0x6600_0000 {
                // |x| < 2^-24: `cosh x` is `1` rounded up.
                return x.abs().mul_add(f32::from_bits(0x33000000), 1.0);
            }
            return (0.5 * x) * x + 1.0;
        }
        let z2 = z * z;
        let z4 = z2 * z2;
        return (1.0 + z2 * ((COSH_CP[0] + z2 * COSH_CP[1]) + z4 * (COSH_CP[2] + z2 * COSH_CP[3])))
            as f32;
    }

    let a = ILN2 * z;
    let ia = a.round_ties_even();
    let h = a - ia;
    let h2 = h * h;
    let jp = (ia + BIG).to_bits() as i64;
    let jm = jp.wrapping_neg();
    let sp = scale(jp);
    let sm = scale(jm);
    let te = st::C[0] + h2 * st::C[2];
    let to = st::C[1] + h2 * st::C[3];
    let r = sp * (te + h * to) + sm * (te - h * to);
    let ub = r as f32;
    let lb = (r - f64::from_bits(0x3de3edbbe4560327) * r) as f32;
    if ub == lb {
        return ub;
    }

    let h = (ILN2H * z - ia) + ILN2L * z;
    let h2 = h * h;
    let te = st::CH[0] + h2 * st::CH[2] + (h2 * h2) * (st::CH[4] + h2 * st::CH[6]);
    let to = st::CH[1] + h2 * (st::CH[3] + h2 * st::CH[5]);
    (sp * (te + h * to) + sm * (te - h * to)) as f32
}

/// `tanh(x)`, correctly rounded.
///
/// One degree-15 rational and no second path: the fit is good enough over
/// the whole of `[2^-13, 8.5]` that no argument needs refining, which is why
/// this is the shortest routine in the module.
pub fn tanh(x: f32) -> f32 {
    let z = f64::from(x);
    let ux = x.to_bits();
    let e = ((ux >> 23) & 0xff) as i32;

    if e == 0xff {
        if ux << 9 != 0 {
            return x + x; // NaN
        }
        return if ux >> 31 != 0 { -1.0 } else { 1.0 };
    }
    if e < 115 {
        if e < 102 {
            if ux << 1 == 0 {
                return x;
            }
            return (-x).mul_add(x.abs(), x);
        }
        let x2 = x * x;
        return x.mul_add(-f32::from_bits(0x3eaaaaab) * x2, x);
    }
    // Past `0x1.2059 66p+3` every result rounds to `±1` minus one ulp.
    if (ux << 1) > (0x4110_2cb3u32 << 1) {
        return 1.0f32.copysign(x) - f32::from_bits(0x33000000).copysign(x);
    }

    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;

    let n0 = TANH_CN[0] + z2 * TANH_CN[1];
    let n2 = TANH_CN[2] + z2 * TANH_CN[3];
    let n4 = TANH_CN[4] + z2 * TANH_CN[5];
    let n6 = TANH_CN[6] + z2 * TANH_CN[7];
    let n0 = n0 + z4 * n2;
    let n4 = n4 + z4 * n6;
    let n0 = n0 + z8 * n4;

    let d0 = TANH_CD[0] + z2 * TANH_CD[1];
    let d2 = TANH_CD[2] + z2 * TANH_CD[3];
    let d4 = TANH_CD[4] + z2 * TANH_CD[5];
    let d6 = TANH_CD[6] + z2 * TANH_CD[7];
    let d0 = d0 + z4 * d2;
    let d4 = d4 + z4 * d6;
    let d0 = d0 + z8 * d4;

    (z * n0 / d0) as f32
}

/// The shared logarithm of `asinh` and `acosh`: `log(w * 2^e)` for a `w` in
/// `[1, 2)`, over [`at`]'s 129-entry table.
///
/// Returns `(r, j, z)` — the fast-path result, the table index and the
/// reduced argument — because the caller needs the latter two if the
/// bracket test sends it to the accurate path.
#[inline(always)]
fn asincosh_log(tp: u64) -> (f64, usize, f64, i32) {
    let m = tp & (!0u64 >> 12);
    let j = ((m + (1u64 << (52 - 8))) >> (52 - 7)) as usize;
    let e = ((tp >> 52) as i64 - 0x3ff) as i32;
    let w = f64::from_bits(m | (0x3ffu64 << 52));
    let z = w * at::IX[j] - 1.0;
    let z2 = z * z;
    let r = ((at::LIX[128] * f64::from(e) + at::LIX[j]) + z * at::C[0])
        + z2 * (at::C[1] + z * at::C[2]);
    (r, j, z, e)
}

/// The accurate replacement for [`asincosh_log`]'s result.
#[inline(always)]
fn asincosh_log_accurate(j: usize, z: f64, e: i32) -> f64 {
    let z2 = z * z;
    let c0 = at::CP[0] + z * at::CP[1];
    let c2 = at::CP[2] + z * at::CP[3];
    let c4 = at::CP[4] + z * at::CP[5];
    let c0 = c0 + z2 * (c2 + z2 * c4);
    let lh = LN2H * f64::from(e);
    let ll = LN2L * f64::from(e);
    let r = (z * c0 + (ll + at::LIX[j])) + lh;
    if r.to_bits() & 0xfff_ffff == 0 {
        // On an `f32` rounding boundary: recover the part the sum dropped
        // and scale it up so it decides the direction.
        let h = (z * c0 + (ll + at::LIX[j])) + (lh - r);
        return r + 64.0 * h;
    }
    r
}

/// `asinh(x)`, correctly rounded.
pub fn asinh(x: f32) -> f32 {
    let t = x.to_bits() & (!0u32 >> 1);
    let xs = f64::from(x);

    if t <= 0x3e81_5667 {
        if t <= 0x39dd_b3d7 {
            if t == 0 {
                return x;
            }
            // `asinh x` is `x` rounded towards zero.
            return x.mul_add(-f32::from_bits(0x33000000), x);
        }
        let x2 = xs * xs;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        let f = x2
            * (((ASINH_C[0] + x2 * ASINH_C[1]) + x4 * (ASINH_C[2] + x2 * ASINH_C[3]))
                + x8 * ((ASINH_C[4] + x2 * ASINH_C[5]) + x4 * (ASINH_C[6] + x2 * ASINH_C[7])));
        return (xs - xs * f) as f32;
    }

    if t >= 0x7f80_0000 {
        return x + x; // ±inf, NaN
    }
    // `asinh x = log(x + sqrt(x^2 + 1))`, with the logarithm tabulated.
    let xd = xs.abs();
    let x2 = xd * xd;
    let tp = (xd + (x2 + 1.0).sqrt()).to_bits();
    let (mut r, j, z, e) = asincosh_log(tp);
    // The bracket test is spelled on the result's own low bits: a mantissa
    // ending near a multiple of 2^28 means the sum lost the bits that decide
    // the `f32` rounding.
    if (r.to_bits().wrapping_add(259_000) & 0xfff_ffff) < 260_000 {
        r = asincosh_log_accurate(j, z, e);
    }
    (r as f32).copysign(x)
}

/// `acosh`'s out-of-band arguments: exact at 1, `+inf` passing through, a
/// NaN quietened, and everything else — which, `t` being unsigned, means
/// every argument below 1 including all the negatives — invalid.
#[inline(always)]
fn acosh_special(x: f32) -> f32 {
    let t = x.to_bits();
    if t == 0x3f80_0000 {
        return 0.0;
    }
    if (t << 1) > 0xff00_0000 {
        return x + x; // NaN
    }
    if t == 0x7f80_0000 {
        return x; // +inf
    }
    invalid(x)
}

/// `acosh(x)`, correctly rounded.
pub fn acosh(x: f32) -> f32 {
    let t = x.to_bits();

    if t <= 0x3f80_0000 {
        return acosh_special(x);
    }
    if t <= 0x3f99_db23 {
        // Just above 1, where `sqrt(x^2 - 1)` cancels: factor out the
        // `sqrt(2(x-1))` that carries the singularity and fit the rest.
        let zf = x - 1.0;
        let z = f64::from(zf);
        let a = (2.0 * z).sqrt();
        let z2 = z * z;
        let z4 = z2 * z2;
        let f = ((ACOSH_C[0] + z * ACOSH_C[1]) + z2 * (ACOSH_C[2] + z * ACOSH_C[3]))
            + z4 * ((ACOSH_C[4] + z * ACOSH_C[5]) + z2 * (ACOSH_C[6] + z * ACOSH_C[7]));
        return (a + (a * z) * f) as f32;
    }
    if t < 0x7f80_0000 {
        let xd = f64::from(x);
        let x2 = xd * xd;
        let tp = (xd + (x2 - 1.0).sqrt()).to_bits();
        let (mut r, j, z, e) = asincosh_log(tp);
        if (r.to_bits().wrapping_add(259_000) & 0xfff_ffff) < 260_000 {
            r = asincosh_log_accurate(j, z, e);
        }
        return r as f32;
    }
    acosh_special(x)
}

/// `atanh(x)`, matching Rust's `f32::atanh`.
///
/// The one function in this module that is deliberately **not** a port of the
/// platform's routine. Rust does not forward `f32::atanh` to `libm` at all —
/// it evaluates `0.5 * ln_1p(2x / (1 - x))` itself, and that disagrees with
/// glibc's correctly-rounded `atanhf` on roughly one input in ten. rmath's
/// promise is that substituting it for the call you were already making
/// cannot change a result, and the call a Rust caller was already making is
/// `f32::atanh`. So this reproduces the composition, over [`super::logx`]'s
/// own `log1p` rather than the platform's `log1pf` — which is bit-exact to
/// it, so the result is `f32::atanh`'s to the last bit without a libm call
/// anywhere underneath. [`crate::reference::double::atanh`] makes the same
/// choice for the same reason.
///
/// Verified by exhaustive sweep against `f32::atanh`: all 2^32 bit patterns.
pub fn atanh(x: f32) -> f32 {
    0.5 * super::logx::log1p((2.0 * x) / (1.0 - x))
}
