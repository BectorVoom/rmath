//! Scalar reference implementations: what [`crate::policy::BitExact`] means.
//!
//! Each function here is a faithful port of the routine the platform's C
//! library actually runs, reproducing its *operation schedule* — which table,
//! which polynomial, which association, and critically where a fused
//! multiply-add is used and where two separate roundings are.
//!
//! That last point is not pedantry, and it is not guessable from the C source.
//! Whether `scale + scale * tmp` compiles to one `vfmadd` or to `vmulsd` +
//! `vaddsd` is the compiler's choice, it changes the last bit, and it is not
//! even consistent within one function: in glibc's `exp`, the `k > 0` arm of
//! the special-case handler is fused and the `k < 0` arm is not. Every
//! placement below was read out of a disassembly of the compiled library
//! (glibc 2.43, x86-64), not inferred from the source.
//!
//! Two consequences worth being explicit about:
//!
//! * `exp` and `log` are ported from their `_fma` ifunc variants, which is
//!   what any FMA-capable x86-64 machine selects at load time. `exp2` has no
//!   `_fma` variant, so it is ported from the baseline SSE2 build and uses
//!   *no* fused operations at all.
//! * Bit-exactness is therefore a claim about a platform, not a universal
//!   truth. It is not assumed: `tests/bit_exact.rs` checks it against the
//!   host's libm, and fails loudly on a platform whose libm differs.
//!
//! These are also the crate's fallback path. A vector kernel handles rare
//! lanes — overflow, subnormals, NaN — by calling the function here for those
//! lanes only, so correctness at the edges never depends on the vector code
//! getting the hard cases right.

use crate::tables::{exp as et, log as lt};

/// The sign and exponent bits, as glibc's `top12`.
#[inline(always)]
fn top12(x: f64) -> u32 {
    (x.to_bits() >> 52) as u32
}

/// `0x1p-54`, the cutoff below which `exp(x)` is `1 + x`.
const TINY: f64 = f64::from_bits(0x3c90000000000000);
/// `0x1p1009`, the rescale in `exp`'s overflow-side special case.
const P1009: f64 = f64::from_bits(0x7f00000000000000);
/// `0x1p-1022`, the smallest normal, used to scale into the subnormal range.
const P_M1022: f64 = f64::from_bits(0x0010000000000000);
/// `0x1p52`, used to normalise a subnormal before `log`'s table lookup.
const P52: f64 = f64::from_bits(0x4330000000000000);

// ---------------------------------------------------------------------------
// exp
// ---------------------------------------------------------------------------

/// `e^x`, bit-identical to glibc's `__ieee754_exp_fma`.
pub fn exp(x: f64) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    if abstop.wrapping_sub(top12(TINY)) >= top12(512.0).wrapping_sub(top12(TINY)) {
        if abstop.wrapping_sub(top12(TINY)) >= 0x8000_0000 {
            // |x| < 0x1p-54: exp(x) rounds to 1 + x. Also the x == 0 case.
            return 1.0 + x;
        }
        if abstop >= top12(1024.0) {
            if x.to_bits() == f64::NEG_INFINITY.to_bits() {
                return 0.0;
            }
            if abstop >= top12(f64::INFINITY) {
                return 1.0 + x; // +inf, or NaN (propagating the payload)
            }
            // Genuine overflow / underflow; glibc raises the flag and
            // returns these values.
            return if x.to_bits() >> 63 != 0 {
                0.0
            } else {
                f64::INFINITY
            };
        }
        // Large but representable: fall through, then take `specialcase`.
        abstop = 0;
    }

    let (tmp, sbits, ki) = exp_core(x);
    if abstop == 0 {
        return exp_specialcase(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    // Fused: `vfmadd132sd %xmm1,%xmm0,%xmm0` at the tail of __ieee754_exp_fma.
    scale.mul_add(tmp, scale)
}

/// The shared main path: returns `(tmp, sbits, ki)` so the caller can decide
/// between the fast tail and `specialcase`.
#[inline(always)]
fn exp_core(x: f64) -> (f64, u64, u64) {
    let kd_s = x.mul_add(et::INVLN2N, et::SHIFT);
    let ki = kd_s.to_bits();
    let kd = kd_s - et::SHIFT;
    let r = kd.mul_add(et::NEGLN2LON, kd.mul_add(et::NEGLN2HIN, x));

    let idx = ((ki & 127) * 2) as usize;
    let tail = f64::from_bits(et::TAB[idx]);
    let sbits = et::TAB[idx + 1].wrapping_add(ki << 45);

    let p12 = r.mul_add(et::C3, et::C2);
    let t3 = tail + r;
    let r2 = r * r;
    let p45 = r.mul_add(et::C5, et::C4);
    let s1 = r2.mul_add(p12, t3);
    let r4 = r2 * r2;
    let tmp = r4.mul_add(p45, s1);
    (tmp, sbits, ki)
}

/// glibc `e_exp.c: specialcase`. Note the asymmetry between the two arms:
/// the `k > 0` side was compiled fused, the `k < 0` side was not.
fn exp_specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        // k > 0: scale's exponent may have overflowed by up to 460.
        let scale = f64::from_bits(sbits.wrapping_sub(1009u64 << 52));
        return P1009 * scale.mul_add(tmp, scale);
    }
    // k < 0: the result may be subnormal, where a second rounding would cost
    // half an ulp, so the sum is renormalised around 1.0 before scaling down.
    let scale = f64::from_bits(sbits.wrapping_add(1022u64 << 52));
    let st = scale * tmp;
    let mut y = scale + st;
    if y < 1.0 {
        let lo = scale - y + st;
        let hi = 1.0 + y;
        let lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
    }
    P_M1022 * y
}

// ---------------------------------------------------------------------------
// exp2
// ---------------------------------------------------------------------------

/// `2^x`, bit-identical to glibc's `__ieee754_exp2`.
///
/// Uses **no** fused multiply-adds. glibc ships no `_fma` variant of `exp2`,
/// so what runs is the baseline SSE2 build: `mulsd` and `addsd` throughout,
/// two roundings where `exp` has one. Replacing any of the arithmetic below
/// with `mul_add` would be more accurate and would break bit-exactness.
pub fn exp2(x: f64) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    if abstop.wrapping_sub(top12(TINY)) >= top12(512.0).wrapping_sub(top12(TINY)) {
        if abstop.wrapping_sub(top12(TINY)) >= 0x8000_0000 {
            return 1.0 + x;
        }
        if abstop >= top12(1024.0) {
            if x.to_bits() == f64::NEG_INFINITY.to_bits() {
                return 0.0;
            }
            if abstop >= top12(f64::INFINITY) {
                return 1.0 + x;
            }
            if x.to_bits() >> 63 == 0 {
                return f64::INFINITY;
            } else if x.to_bits() >= (-1075.0f64).to_bits() {
                return 0.0;
            }
        }
        if 2u64.wrapping_mul(x.to_bits()) > 2u64.wrapping_mul(928.0f64.to_bits()) {
            abstop = 0;
        }
    }

    let (tmp, sbits, ki) = exp2_core(x);
    if abstop == 0 {
        return exp2_specialcase(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    scale + scale * tmp
}

#[inline(always)]
fn exp2_core(x: f64) -> (f64, u64, u64) {
    let kd_s = x + et::EXP2_SHIFT;
    let ki = kd_s.to_bits();
    let kd = kd_s - et::EXP2_SHIFT;
    let r = x - kd;

    let idx = ((ki & 127) * 2) as usize;
    let tail = f64::from_bits(et::TAB[idx]);
    let sbits = et::TAB[idx + 1].wrapping_add(ki << 45);

    let r2 = r * r;
    let tmp = tail
        + r * et::EXP2_C1
        + r2 * (et::EXP2_C2 + r * et::EXP2_C3)
        + r2 * r2 * (et::EXP2_C4 + r * et::EXP2_C5);
    (tmp, sbits, ki)
}

/// glibc `e_exp2.c: specialcase`. Differs from `exp`'s: the overflow arm
/// backs the exponent off by 1 and doubles, rather than by 1009.
fn exp2_specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        let scale = f64::from_bits(sbits.wrapping_sub(1u64 << 52));
        return 2.0 * (scale + scale * tmp);
    }
    let scale = f64::from_bits(sbits.wrapping_add(1022u64 << 52));
    let st = scale * tmp;
    let mut y = scale + st;
    if y < 1.0 {
        let lo = scale - y + st;
        let hi = 1.0 + y;
        let lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
    }
    P_M1022 * y
}

// ---------------------------------------------------------------------------
// ln
// ---------------------------------------------------------------------------

/// Low end of `log`'s near-1.0 window, `bits(1.0 - 0x1p-4)`.
const LOG_NEAR_LO: u64 = 0x3fee000000000000;
/// High end, `bits(1.0 + 0x1.09p-4)`.
const LOG_NEAR_HI: u64 = 0x3ff1090000000000;
/// `log`'s table-centring offset, `bits(0x1.6p-1)`.
const LOG_OFF: u64 = 0x3fe6000000000000;

/// `ln(x)`, bit-identical to glibc's `__ieee754_log_fma`.
pub fn ln(x: f64) -> f64 {
    let mut ix = x.to_bits();
    let top = (ix >> 48) as u32;

    if ix.wrapping_sub(LOG_NEAR_LO) < LOG_NEAR_HI - LOG_NEAR_LO {
        if ix == 1.0f64.to_bits() {
            return 0.0;
        }
        return ln_near_one(x);
    }
    if top.wrapping_sub(0x0010) >= 0x7ff0 - 0x0010 {
        if ix.wrapping_mul(2) == 0 {
            return f64::NEG_INFINITY; // log(+-0)
        }
        if ix == f64::INFINITY.to_bits() {
            return x;
        }
        if (top & 0x8000) != 0 || (top & 0x7ff0) == 0x7ff0 {
            // glibc's `__math_invalid(x)`, spelled exactly as it spells it.
            #[allow(clippy::eq_op)]
            // Not `f64::NAN`: that is the *positive* quiet NaN, while
            // `0.0 / 0.0` on x86 yields the negative one, and this is a
            // bit-exactness contract — the sign of a NaN counts.
            return (x - x) / (x - x);
        }
        // Subnormal: scale into the normal range and correct the exponent.
        ix = (x * P52).to_bits();
        ix -= 52u64 << 52;
    }
    ln_main(ix)
}

/// The table path, taking already-normalised bits.
#[inline(always)]
fn ln_main(ix: u64) -> f64 {
    let tmp = ix.wrapping_sub(LOG_OFF);
    let i = ((tmp >> 45) & 127) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));

    let invc = lt::TAB[2 * i];
    let logc = lt::TAB[2 * i + 1];
    let z = f64::from_bits(iz);
    let kd = k as f64;

    let w = kd.mul_add(lt::LN2HI, logc);
    let r = z.mul_add(invc, -1.0);
    let q12 = r.mul_add(lt::A2, lt::A1);
    let hi = r + w;
    let r2 = r * r;
    let t = (w - hi) + r;
    let lo = kd.mul_add(lt::LN2LO, t);
    let r3 = r * r2;
    let q34 = r.mul_add(lt::A4, lt::A3);
    let s1 = r2.mul_add(lt::A0, lo);
    let q = r2.mul_add(q34, q12);
    r3.mul_add(q, s1) + hi
}

/// The `0.9375 <= x < 1 + 0x1.09p-4` path, where the table path would lose
/// too much to cancellation. Carries a Veltkamp split so `r - r^2/2` is
/// evaluated in double-double.
#[inline(always)]
fn ln_near_one(x: f64) -> f64 {
    let r = x - 1.0;
    let p12 = r.mul_add(lt::B2, lt::B1);
    let p45 = r.mul_add(lt::B5, lt::B4);
    let r2 = r * r;
    let p78 = r.mul_add(lt::B8, lt::B7);
    let p123 = r2.mul_add(lt::B3, p12);
    let p456 = r2.mul_add(lt::B6, p45);
    let r3 = r * r2;
    let p789 = r2.mul_add(lt::B9, p78);
    let p78910 = r3.mul_add(lt::B10, p789);
    let pin = p78910.mul_add(r3, p456);
    let poly = pin.mul_add(r3, p123);

    const SPLIT: f64 = 134217728.0; // 0x1p27
    let rhi_t = r.mul_add(SPLIT, r);
    let rhi = (-r).mul_add(SPLIT, rhi_t);
    let rlo = r - rhi;
    let s = rhi * rhi;
    let hi = s.mul_add(lt::B0, r);
    let t = r - hi;
    let rpr = r + rhi;
    let lo = s.mul_add(lt::B0, t);
    let lo = (lt::B0 * rlo).mul_add(rpr, lo);
    poly.mul_add(r3, lo) + hi
}

// ---------------------------------------------------------------------------
// cbrt
// ---------------------------------------------------------------------------

/// `2^(i/3)` for `i` in `0..3`.
pub(crate) const ESCALE: [f64; 3] = [
    1.0,
    f64::from_bits(0x3ff428a2f98d728b), // 0x1.428a2f98d728bp+0
    f64::from_bits(0x3ff965fea53d6e3d), // 0x1.965fea53d6e3dp+0
];

/// Degree-3 seed for `z^(1/3)` on `[1, 2]`; maximum error below 9.2e-5.
pub(crate) const CB: [f64; 4] = [
    f64::from_bits(0x3fe1b0babccfef9c), // 0x1.1b0babccfef9cp-1
    f64::from_bits(0x3fe2c9a3e94d1da5), // 0x1.2c9a3e94d1da5p-1
    f64::from_bits(0xbfc4dc30b1a1ddba), // -0x1.4dc30b1a1ddbap-3
    f64::from_bits(0x3f97a8d3e4ec9b07), // 0x1.7a8d3e4ec9b07p-6
];

pub(crate) const U0: f64 = f64::from_bits(0x3fd5555555555555); // 1/3, to working precision
pub(crate) const U1: f64 = f64::from_bits(0x3fcc71c71c71c71c); // 2/9

/// `+-2^-k` selectors, indexed by `(it << 1) | sign`.
pub(crate) const RSC: [f64; 6] = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25];

pub(crate) const OFF_NEAREST: f64 = f64::from_bits(0x3ca0000000000000); // 0x1p-53
pub(crate) const P_M52: f64 = f64::from_bits(0x3cb0000000000000);
pub(crate) const P_M75: f64 = f64::from_bits(0x3b40000000000000);
pub(crate) const P_M98: f64 = f64::from_bits(0x39d0000000000000);
pub(crate) const P_M60: f64 = f64::from_bits(0x3c30000000000000);

/// `x^(1/3)`, bit-identical to Rust's `f64::cbrt`.
///
/// **The reference here is Rust's, not the C library's**, and for this one
/// function those differ. Rust's `std` does not forward `cbrt` to the platform
/// `libm`; it uses its own port of the correctly-rounded core-math routine.
/// glibc's `cbrt` is a much cruder frexp-plus-one-Halley-step algorithm,
/// accurate to about 1 ulp, and the two disagree on a large fraction of
/// inputs — measured here at roughly half of a random sweep, by one ulp.
///
/// Matching Rust is the right choice for a Rust crate: the whole point of
/// [`crate::policy::BitExact`] is that substituting `rmath` for the call you
/// were already making cannot change a result, and the call you were already
/// making is `f64::cbrt`. If you need to reproduce a C program's `cbrt`
/// instead, this is not the function for it.
///
/// Ported from `libm` 0.2 (rust-lang/compiler-builtins), itself a port of
/// core-math's `cbrt.c`, Copyright (c) 2021-2022 Alexei Sibidanov, MIT.
/// The directed-rounding exception table is omitted: Rust evaluates in
/// round-to-nearest, which is the only mode reachable here.
pub fn cbrt(x: f64) -> f64 {
    let hx = x.to_bits();
    let mut mant = hx & 0x000f_ffff_ffff_ffff;
    let sign = hx >> 63;
    let mut e = ((hx >> 52) as u32) & 0x7ff;

    if ((e + 1) & 0x7ff) < 2 {
        let ix = hx & 0x7fff_ffff_ffff_ffff;
        // 0, inf, NaN. `x + x` rather than `x` so a signalling NaN raises.
        if e == 0x7ff || ix == 0 {
            return x + x;
        }
        // Subnormal: shift the leading bit up into place.
        let nz = ix.leading_zeros() - 11;
        mant <<= nz;
        mant &= 0x000f_ffff_ffff_ffff;
        e = e.wrapping_sub(nz - 1);
    }

    e = e.wrapping_add(3072);
    let cvt1 = mant | (0x3ffu64 << 52);
    let et = e / 3;
    let it = e % 3;

    // 2^(3k+it) <= x < 2^(3k+it+1), with zz in [1, 8).
    let cvt5 = (cvt1 + ((it as u64) << 52)) | (sign << 63);
    let zz = f64::from_bits(cvt5);
    let cvt2 = ESCALE[it as usize].to_bits() | (sign << 63);
    let z = f64::from_bits(cvt1);

    let r = 1.0 / z;
    let rr = r * RSC[((it as usize) << 1) | sign as usize];
    let z2 = z * z;
    let c0 = CB[0] + z * CB[1];
    let c2 = CB[2] + z * CB[3];
    let mut y = c0 + z2 * c2;
    let mut y2 = y * y;

    // Cubic Newton on f(y) = 1 - z/y^3.
    let mut h = y2 * (y * r) - 1.0;
    y -= (h * y) * (U0 - U1 * h);
    y *= f64::from_bits(cvt2);

    // One more Newton step, this time carrying y^2 and y^3 in double-double
    // so the residual survives the cancellation in `y3 - zz`.
    y2 = y * y;
    let mut y2l = y.mul_add(y, -y2);
    let mut y3 = y2 * y;
    let mut y3l = y.mul_add(y2, -y3) + y * y2l;
    h = ((y3 - zz) + y3l) * rr;
    let mut dy = h * (y * U0);
    let mut y1 = y - dy;
    dy = (y - y1) - dy;

    let mut ady = dy.abs();
    let mut ady0 = (ady - OFF_NEAREST).abs();
    let mut ady1 = (ady - (P_M52 + OFF_NEAREST)).abs();

    // Too close to a rounding boundary to call: refine once more.
    if ady0 < P_M75 || ady1 < P_M75 {
        y2 = y1 * y1;
        y2l = y1.mul_add(y1, -y2);
        y3 = y2 * y1;
        y3l = y1.mul_add(y2, -y3) + y1 * y2l;
        h = ((y3 - zz) + y3l) * rr;
        dy = h * (y1 * U0);
        y = y1 - dy;
        dy = (y1 - y) - dy;
        y1 = y;
        ady = dy.abs();
        ady0 = (ady - OFF_NEAREST).abs();
        ady1 = (ady - (P_M52 + OFF_NEAREST)).abs();

        // Still undecidable: two inputs are hard enough to be tabulated.
        if ady0 < P_M98 || ady1 < P_M98 {
            let azz = zz.abs();
            if azz == f64::from_bits(0x4009b78223aa307c) {
                y1 = copysign(f64::from_bits(0x3ff79d15d0e8d59c), zz);
            }
            if azz == f64::from_bits(0x401a202bfc89ddff) {
                y1 = copysign(f64::from_bits(0x3ffde87aa837820f), zz);
            }
        }
    }

    let mut cvt3 = y1
        .to_bits()
        .wrapping_add((et.wrapping_sub(342).wrapping_sub(1023) as u64) << 52);
    let m0 = cvt3 << 30;
    let m1 = ((m0 as i64) >> 63) as u64;

    // The result sits near the middle of a rounding interval; snap it if the
    // remaining error says it should be.
    if (m0 ^ m1) <= (1u64 << 30) {
        let cvt4 = (y1.to_bits() + (164 << 15)) & 0xffff_ffff_ffff_0000;
        if ((f64::from_bits(cvt4) - y1) - dy).abs() < P_M60 || zz.abs() == 1.0 {
            cvt3 = (cvt3 + (1u64 << 15)) & 0xffff_ffff_ffff_0000;
        }
    }
    f64::from_bits(cvt3)
}

#[inline(always)]
fn copysign(mag: f64, sign: f64) -> f64 {
    f64::from_bits(
        (mag.to_bits() & 0x7fff_ffff_ffff_ffff) | (sign.to_bits() & 0x8000_0000_0000_0000),
    )
}

// ---------------------------------------------------------------------------
// Small helpers, written against bits so they cost nothing and need no std.
// ---------------------------------------------------------------------------
