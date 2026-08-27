//! `powf` and `atan2f` — the two-argument members of the `f32` family.
//!
//! Different upstreams, and different verification stories.
//!
//! `powf` is ARM optimized-routines (glibc's
//! `sysdeps/ieee754/flt-32/e_powf.c`) and is *not* correctly rounded — the
//! documented worst case is 0.82 ULP — so bit-exactness here is a claim about
//! a schedule. Unlike every other `f32` routine in this crate, `powf` **is**
//! built with `-mfma` (`CFLAGS-e_powf-fma.c = -mfma -mavx2`), and the fusion
//! below was read out of the compiled `__powf_fma` (`objdump -d`, glibc 2.43,
//! x86-64): every multiply feeding an add inside `log2_inline` and
//! `exp2_inline` fuses, and the one multiply that does not is `y * log2(x)`
//! between them, which is a bare `vmulsd`.
//!
//! `atan2f` is CORE-MATH and *is* correctly rounded, so its bits are not a
//! platform claim at all — any correct `atan2f` anywhere returns them. It has
//! no FMA variant and `__atan2f` contains no `vfmadd`, so its arithmetic is
//! unfused except where the C calls `fma` outright, in the double-double
//! helpers, where the fusion *is* the algorithm.
//!
//! Neither can be swept exhaustively: the domain is 2^64. Both are checked
//! against the platform over a large structured random corpus instead.

use crate::tables::single::{exp as et, pow as pt};

/// `powf`'s table-centring offset. Not `logf`'s or `log2f`'s: each of the
/// three 16-subinterval tables is centred for its own polynomial.
const POWF_OFF: u32 = 0x3f33_0000;
/// `1 << (EXP2F_TABLE_BITS + 11)`: the bit that negates `exp2_inline`'s
/// result, by landing in the sign of the scale it assembles.
const SIGN_BIAS: u32 = 1 << (5 + 11);

/// `log2(x)` to double precision, for a positive-normal `x` given as bits.
#[inline(always)]
fn log2_inline(ix: u32) -> f64 {
    let tmp = ix.wrapping_sub(POWF_OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let top = tmp & 0xff80_0000;
    let iz = ix.wrapping_sub(top);
    // `POWF_SCALE_BITS` is 0 on x86-64 — `TOINT_INTRINSICS` is not defined
    // there — so this is a plain arithmetic shift and nothing is rescaled.
    let k = f64::from((top as i32) >> 23);
    let invc = pt::TAB[2 * i];
    let logc = pt::TAB[2 * i + 1];
    let z = f64::from(f32::from_bits(iz));

    let r = z.mul_add(invc, -1.0);
    let y0 = logc + k;

    let r2 = r * r;
    let y = pt::A0.mul_add(r, pt::A1);
    let p = pt::A2.mul_add(r, pt::A3);
    let r4 = r2 * r2;
    let q = r.mul_add(pt::A4, y0);
    let q = r2.mul_add(p, q);
    y.mul_add(r4, q)
}

/// `2^xd`, with `sign_bias` folding a negated result in for free.
#[inline(always)]
fn exp2_inline(xd: f64, sign_bias: u32) -> f64 {
    // `SHIFT_SCALED`, not `SHIFT`: the `TOINT_INTRINSICS == 0` arm of
    // `e_powf.c` reduces to `k/N + r` with `r` in `[-1/2N, 1/2N]`, so the
    // round-to-integer constant is `1.5 * 2^52 / N` and the table index falls
    // out of `kd`'s low bits directly.
    let kd = xd + et::SHIFT_SCALED;
    let ki = kd.to_bits();
    let kd = kd - et::SHIFT_SCALED;
    let r = xd - kd;

    let t = et::TAB[(ki % 32) as usize];
    let ski = ki.wrapping_add(u64::from(sign_bias));
    let t = t.wrapping_add(ski << (52 - 5));
    let s = f64::from_bits(t);

    let z = et::C0.mul_add(r, et::C1);
    let r2 = r * r;
    let y = et::C2.mul_add(r, 1.0);
    let y = z.mul_add(r2, y);
    y * s
}

/// `0` if `y` is not an integer, `1` if it is odd, `2` if it is even.
#[inline(always)]
fn checkint(iy: u32) -> i32 {
    let e = (iy >> 23 & 0xff) as i32;
    if e < 0x7f {
        return 0;
    }
    if e > 0x7f + 23 {
        return 2;
    }
    if iy & ((1u32 << (0x7f + 23 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1u32 << (0x7f + 23 - e)) != 0 {
        1
    } else {
        2
    }
}

/// True for a zero, infinity or NaN bit pattern.
#[inline(always)]
fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 0x7f80_0000u32.wrapping_mul(2).wrapping_sub(1)
}

/// True for a signalling NaN.
#[inline(always)]
fn is_signaling(x: f32) -> bool {
    let b = x.to_bits();
    b & 0x7f80_0000 == 0x7f80_0000 && b & 0x007f_ffff != 0 && b & 0x0040_0000 == 0
}

/// `__math_invalidf (x)`.
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

/// `x^y`, bit-identical to the platform's.
pub fn pow(x: f32, y: f32) -> f32 {
    let mut sign_bias: u32 = 0;
    let mut ix = x.to_bits();
    let iy = y.to_bits();

    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        // Either `x` is subnormal / infinite / NaN, or `y` is zero /
        // infinite / NaN. Everything unusual is decided here.
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if is_signaling(x) { x + y } else { 1.0 };
            }
            if ix == 0x3f80_0000 {
                return if is_signaling(y) { x + y } else { 1.0 };
            }
            if ix.wrapping_mul(2) > 0x7f80_0000u32.wrapping_mul(2)
                || iy.wrapping_mul(2) > 0x7f80_0000u32.wrapping_mul(2)
            {
                return x + y; // NaN
            }
            if ix.wrapping_mul(2) == 2 * 0x3f80_0000 {
                return 1.0; // (-1)^(±inf)
            }
            // `|x| < 1` with `y = +inf`, or `|x| > 1` with `y = -inf`.
            if (ix.wrapping_mul(2) < 2 * 0x3f80_0000) == (iy & 0x8000_0000 == 0) {
                return 0.0;
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix & 0x8000_0000 != 0 && checkint(iy) == 1 {
                x2 = -x2;
                sign_bias = 1;
            }
            if ix.wrapping_mul(2) == 0 && iy & 0x8000_0000 != 0 {
                // `__math_divzerof (sign_bias)`.
                return if sign_bias != 0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
            }
            return if iy & 0x8000_0000 != 0 { 1.0 / x2 } else { x2 };
        }
        // `x` and `y` are both non-zero and finite from here.
        if ix & 0x8000_0000 != 0 {
            match checkint(iy) {
                0 => return invalid(x), // negative base, fractional exponent
                1 => sign_bias = SIGN_BIAS,
                _ => {}
            }
            ix &= 0x7fff_ffff;
        }
        if ix < 0x0080_0000 {
            // Normalise a subnormal base so its exponent goes negative.
            ix = (x * f32::from_bits(0x4b00_0000)).to_bits();
            ix &= 0x7fff_ffff;
            ix -= 23 << 23;
        }
    }

    // `y` is single precision, so this product cannot overflow.
    let ylogx = f64::from(y) * log2_inline(ix);

    // `|y log2 x| >= 126`: the only band where the result can over- or
    // underflow, tested on the exponent bits alone.
    if (ylogx.to_bits() >> 47 & 0xffff) >= (126.0f64.to_bits() >> 47) {
        if ylogx <= -150.0 {
            // `__math_uflowf (sign_bias)`: a signed zero.
            return if sign_bias != 0 { -0.0 } else { 0.0 };
        }
        if ylogx > f64::from_bits(0x405fffffffa3aae2) {
            // Above `0x1.ffffffp127` the true result rounds to infinity in
            // every mode.
            if ylogx > f64::from_bits(0x405fffffffd1d571) {
                return if sign_bias != 0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
            }
            // Between the two thresholds `|x^y| > 0x1.fffffep127`, and
            // whether that overflows depends on the rounding mode. The C
            // decides by probing it — `1.0f + 0x1p-25f != 1.0f` — and under
            // round-to-nearest that probe is *false*: `2^-25` is below half
            // an ulp of 1, so the sum is exactly 1 and no overflow is
            // signalled. Round-to-nearest is the only mode this crate
            // models, so the band saturates rather than overflowing, and the
            // `ylogx != 0x1.fffffffa3aae3p+6` guard the C wraps the probe in
            // changes nothing here.
            return if sign_bias != 0 { -f32::MAX } else { f32::MAX };
        }
    }
    exp2_inline(ylogx, sign_bias) as f32
}

// ---------------------------------------------------------------------------
// atan2
// ---------------------------------------------------------------------------

/// `pi`.
const PI: f64 = f64::from_bits(0x400921fb54442d18);
/// `pi/2`.
const PI2: f64 = f64::from_bits(0x3ff921fb54442d18);
/// The part of `pi/2` a single `double` cannot hold.
const PI2L: f64 = f64::from_bits(0x3c91a62633145c07);

/// The quadrant offset, indexed by `sign(y)`, `sign(x)` and whether
/// `|y| > |x|` — three bits, eight entries.
const AT2_OFF: [f64; 8] = [0.0, PI2, PI, PI2, -0.0, -PI2, -PI, -PI2];
/// The low halves of [`AT2_OFF`], for the accurate path.
const AT2_OFFL: [f64; 8] = [0.0, PI2L, 2.0 * PI2L, PI2L, -0.0, -PI2L, -2.0 * PI2L, -PI2L];

/// `atan2`'s degree-13 rational numerator, in `z^2`.
const AT2_CN: [f64; 7] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x40040e0698f94c35),
    f64::from_bits(0x400248c5da347f0d),
    f64::from_bits(0x3fed873386572976),
    f64::from_bits(0x3fc46fa40b20f1d0),
    f64::from_bits(0x3f833f5e041eed0f),
    f64::from_bits(0x3f1546bbf28667c5),
];

/// Its denominator.
const AT2_CD: [f64; 7] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x4006b8b143a3f6da),
    f64::from_bits(0x4008421201d18ed5),
    f64::from_bits(0x3ff8221d086914eb),
    f64::from_bits(0x3fd670657e3a07ba),
    f64::from_bits(0x3fa0f4951fd1e72d),
    f64::from_bits(0x3f4b3874b8798286),
];

/// The accurate path's degree-31 `atan` series, each coefficient as an
/// unevaluated double-double pair. This is what makes `atan2f`
/// correctly rounded rather than merely accurate: the fast rational
/// above cannot place the last bit near a quadrant boundary, where the
/// offset it is added to is itself inexact.
static AT2_C: [[f64; 2]; 32] = [
    [
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xba88c1dac5492248),
    ],
    [
        f64::from_bits(0xbfd5555555555555),
        f64::from_bits(0xbc755553bf3a2abe),
    ],
    [
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbc699deed1ec9071),
    ],
    [
        f64::from_bits(0xbfc2492492492492),
        f64::from_bits(0xbc5fd99c8d18269a),
    ],
    [
        f64::from_bits(0x3fbc71c71c71c717),
        f64::from_bits(0xbc2651eee4c4d9d0),
    ],
    [
        f64::from_bits(0xbfb745d1745d1649),
        f64::from_bits(0xbc5632683d6c44a6),
    ],
    [
        f64::from_bits(0x3fb3b13b13b11c63),
        f64::from_bits(0x3c5bf69c1f8af41d),
    ],
    [
        f64::from_bits(0xbfb11111110e6338),
        f64::from_bits(0x3c23c3e431e8bb68),
    ],
    [
        f64::from_bits(0x3fae1e1e1dc45c4a),
        f64::from_bits(0xbc4be2db05c77bbf),
    ],
    [
        f64::from_bits(0xbfaaf286b8164b4f),
        f64::from_bits(0x3c2a4673491f0942),
    ],
    [
        f64::from_bits(0x3fa86185e9ad4846),
        f64::from_bits(0x3c4e12e32d79fcee),
    ],
    [
        f64::from_bits(0xbfa642c6d5161fae),
        f64::from_bits(0x3c43ce76c1ca03f0),
    ],
    [
        f64::from_bits(0x3fa47ad6f277e5bf),
        f64::from_bits(0xbc3abd8d85bdb714),
    ],
    [
        f64::from_bits(0xbfa2f64a2ee8896d),
        f64::from_bits(0x3c2ef87d4b615323),
    ],
    [
        f64::from_bits(0x3fa1a6a2b31741b5),
        f64::from_bits(0x3c1a5d9d973547ee),
    ],
    [
        f64::from_bits(0xbfa07fbdad65e0a6),
        f64::from_bits(0xbc265ac07f5d35f4),
    ],
    [
        f64::from_bits(0x3f9ee9932a9a5f8b),
        f64::from_bits(0x3c2f8b9623f6f55a),
    ],
    [
        f64::from_bits(0xbf9ce8b5b9584dc6),
        f64::from_bits(0x3c2fe5af96e8ea2d),
    ],
    [
        f64::from_bits(0x3f9ac9cb288087b7),
        f64::from_bits(0xbc3450cdfceaf5ca),
    ],
    [
        f64::from_bits(0xbf984b025351f3e6),
        f64::from_bits(0x3c2579561b0d73da),
    ],
    [
        f64::from_bits(0x3f952f5b8ecdd52b),
        f64::from_bits(0x3c3036bd2c6fba47),
    ],
    [
        f64::from_bits(0xbf9163a8c44909dc),
        f64::from_bits(0x3c318f735ffb9f16),
    ],
    [
        f64::from_bits(0x3f8a400dce3eea6f),
        f64::from_bits(0xbc2c90569c0c1b5c),
    ],
    [
        f64::from_bits(0xbf81caa78ae6db3a),
        f64::from_bits(0xbc24c60f8161ea09),
    ],
    [
        f64::from_bits(0x3f752672453c0731),
        f64::from_bits(0x3c1834efb598c338),
    ],
    [
        f64::from_bits(0xbf65850c5be137cf),
        f64::from_bits(0xbc0445fc150ca7f5),
    ],
    [
        f64::from_bits(0x3f523eb98d22e1ca),
        f64::from_bits(0xbbf388fbaf1d7830),
    ],
    [
        f64::from_bits(0xbf38f4e974a40741),
        f64::from_bits(0x3bd271198a97da34),
    ],
    [
        f64::from_bits(0x3f1a5cf2e9cf76e5),
        f64::from_bits(0xbbb887eb4a63b665),
    ],
    [
        f64::from_bits(0xbef420c270719e32),
        f64::from_bits(0x3b8efd595b27888b),
    ],
    [
        f64::from_bits(0x3ec3ba2d69b51677),
        f64::from_bits(0xbb64fb06829cdfc7),
    ],
    [
        f64::from_bits(0xbe829b7e6f676385),
        f64::from_bits(0xbb2a783b6de718fb),
    ],
];

/// `muldd`: the double-double product `(xh + xl) * (ch + cl)`.
///
/// The `fma` is the algorithm, not an optimisation: `fma(ch, xh, -ahhh)` is
/// the exact rounding error of `ch * xh`, which is the whole point of the
/// representation and cannot be spelled without a fused multiply-add.
#[inline(always)]
fn muldd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = ch.mul_add(xh, -ahhh);
    ahhl += alhh + ahlh;
    let ch = ahhh + ahhl;
    (ch, (ahhh - ch) + ahhl)
}

/// `polydd`: Horner in double-double, over a table of double-double
/// coefficients.
#[inline(always)]
fn polydd(xh: f64, xl: f64, c: &[[f64; 2]]) -> (f64, f64) {
    let mut i = c.len() - 1;
    let mut ch = c[i][0];
    let mut cl = c[i][1];
    while i > 0 {
        i -= 1;
        let (h, l) = muldd(xh, xl, ch, cl);
        ch = h;
        cl = l;
        let th = ch + c[i][0];
        let tl = (c[i][0] - th) + ch;
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}

/// `atan2(y, x)` where `|y|` is so much smaller than `|x|` that the rational
/// path would round twice — once into `double`, once into `f32`.
fn atan2_tiny(y: f32, x: f32) -> f32 {
    let dy = f64::from(y);
    let dx = f64::from(x);
    let z = dy / dx;
    // `z * x + e = y`, so `y/x = z + e/x` exactly.
    let e = (-z).mul_add(f64::from(x), dy);
    let zz = z * z;
    let cz = f64::from_bits(0xbfd5555555555555) * z;
    let e = e / dx + cz * zz;
    let mut t = z.to_bits();
    if t & 0xfff_ffff == 0 {
        // Sitting on an `f32` rounding boundary: nudge the significand
        // towards the sign of the correction so the narrowing rounds the way
        // the true value does.
        if z * e > 0.0 {
            t += 1;
        } else {
            t -= 1;
        }
    }
    let rf = f64::from_bits(t) as f32;
    if rf == 0.0 {
        // `__math_uflowf (t >> 63)`: a signed zero, with underflow.
        return if t >> 63 != 0 { -0.0 } else { 0.0 };
    }
    rf
}

/// `atan2(y, x)`, correctly rounded.
pub fn atan2(y: f32, x: f32) -> f32 {
    const SGN: [f64; 2] = [1.0, -1.0];
    const M: [f64; 2] = [0.0, 1.0];

    let ux = x.to_bits();
    let uy = y.to_bits();
    let ax = ux & (!0u32 >> 1);
    let ay = uy & (!0u32 >> 1);

    if ay >= (0xff << 23) || ax >= (0xff << 23) {
        // The two NaN tests come first and separately, so that a quiet and a
        // signalling NaN together still produce the right payload.
        if ay > (0xff << 23) {
            return x + y;
        }
        if ax > (0xff << 23) {
            return x + y;
        }
        let yinf = ay == (0xff << 23);
        let xinf = ax == (0xff << 23);
        if yinf && xinf {
            return if ux >> 31 != 0 {
                (f64::from_bits(0x4002d97c7f3321d2) * SGN[(uy >> 31) as usize]) as f32
            } else {
                (f64::from_bits(0x3fe921fb54442d18) * SGN[(uy >> 31) as usize]) as f32
            };
        }
        if xinf {
            return if ux >> 31 != 0 {
                (PI * SGN[(uy >> 31) as usize]) as f32
            } else {
                (0.0 * SGN[(uy >> 31) as usize]) as f32
            };
        }
        if yinf {
            return (PI2 * SGN[(uy >> 31) as usize]) as f32;
        }
    }
    if ay == 0 {
        if ax == 0 {
            let i = ((uy >> 31) * 4 + (ux >> 31) * 2) as usize;
            return if ux >> 31 != 0 {
                (AT2_OFF[i] + AT2_OFFL[i]) as f32
            } else {
                AT2_OFF[i] as f32
            };
        }
        if ux >> 31 == 0 {
            return (0.0 * SGN[(uy >> 31) as usize]) as f32;
        }
    }

    let gt = u32::from(ay > ax) as usize;
    let i = ((uy >> 31) * 4 + (ux >> 31) * 2) as usize + gt;

    let zx = f64::from(x);
    let zy = f64::from(y);
    // `z = x/y` when `|y| > |x|`, and `z = y/x` otherwise, so `|z| <= 1`.
    let z = (M[gt] * zx + M[1 - gt] * zy) / (M[gt] * zy + M[1 - gt] * zx);

    let d = ax as i32 - ay as i32;
    let mut r = if d < (27 << 23) && d > -(27 << 23) {
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let cn0 = AT2_CN[0] + z2 * AT2_CN[1];
        let cn2 = AT2_CN[2] + z2 * AT2_CN[3];
        let cn4 = AT2_CN[4] + z2 * AT2_CN[5];
        let cn6 = AT2_CN[6];
        let cn0 = cn0 + z4 * cn2;
        let cn4 = cn4 + z4 * cn6;
        let cn0 = cn0 + z8 * cn4;
        let cd0 = AT2_CD[0] + z2 * AT2_CD[1];
        let cd2 = AT2_CD[2] + z2 * AT2_CD[3];
        let cd4 = AT2_CD[4] + z2 * AT2_CD[5];
        let cd6 = AT2_CD[6];
        let cd0 = cd0 + z4 * cd2;
        let cd4 = cd4 + z4 * cd6;
        let cd0 = cd0 + z8 * cd4;
        cn0 / cd0
    } else {
        // The ratio is so extreme that `atan z` is `z`; the rational's `z4`
        // and `z8` would only underflow.
        1.0
    };

    let z = z * SGN[gt];
    r = z * r + AT2_OFF[i];

    if ((r.to_bits().wrapping_add(8)) & 0xfff_ffff) <= 16 {
        if ay < ax && ((ax - ay) >> 23) >= 25 {
            return atan2_tiny(y, x);
        }
        // Redo the division in double-double, then the whole series.
        let (zh, zl) = if gt == 0 {
            let zh = zy / zx;
            (zh, zh.mul_add(-zx, zy) / zx)
        } else {
            let zh = zx / zy;
            (zh, zh.mul_add(-zy, zx) / zy)
        };
        let (z2h, z2l) = muldd(zh, zl, zh, zl);
        let (ph, pl) = polydd(z2h, z2l, &AT2_C);
        let zh = zh * SGN[gt];
        let zl = zl * SGN[gt];
        let (ph, pl) = muldd(zh, zl, ph, pl);
        let sh = ph + AT2_OFF[i];
        let sl = ((AT2_OFF[i] - sh) + ph) + pl + AT2_OFFL[i];
        let rf = sh as f32;
        let th = f64::from(rf);
        let dh = sh - th;
        let mut tm = dh + sl;
        // `th ± th * 2^-60` collapsing means `th` is exactly representable
        // and `tm` decides the rounding; scale `tm` away from the halfway
        // point so the narrowing below cannot round the wrong way.
        if th + th * f64::from_bits(0x3c30000000000000)
            == th - th * f64::from_bits(0x3c30000000000000)
        {
            let mut tth = th.to_bits();
            tth &= 0x7ffu64 << 52;
            tth -= 24u64 << 52;
            if tm.abs() > f64::from_bits(tth) {
                tm *= 1.25;
            } else {
                tm *= 0.75;
            }
        }
        r = th + tm;
    }

    let rf = r as f32;
    if rf == 0.0 {
        return if rf.to_bits() >> 31 != 0 { -0.0 } else { 0.0 };
    }
    rf
}
