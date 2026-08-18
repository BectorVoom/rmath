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

use crate::tables::double::{exp as et, log as lt, pow as pt};

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
        ix = ix.wrapping_sub(52u64 << 52);
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

// ---------------------------------------------------------------------------
// Delegating references
//
// These do not reproduce a schedule; they *are* the platform routine, reached
// through Rust's own wrapper. That is a deliberate choice and worth stating
// plainly, because it is the one place this crate calls the library it exists
// to replace.
//
// glibc implements the trigonometric and inverse-trigonometric families with
// the IBM Accurate Portable Math Library routines. Their operation schedule
// depends on tables of several hundred entries and on per-expression
// fused-multiply-add placement that the compiler chooses and the C source does
// not show; reproducing it is a larger job than everything else in this crate
// put together, and getting it subtly wrong would be worse than not doing it,
// because `BitExact` would then be a false claim rather than a slow one.
//
// So `BitExact` stays true for these functions by delegation: substituting
// rmath for the call you were already making cannot change a result, which is
// the whole promise. What it does not do is make them faster --
// `crate::policy::Fast` is where the vector path lives. The crate-level
// documentation says which functions this covers.
//
// Using `f64`'s own methods rather than `extern "C"` keeps this portable: on
// every target Rust's `sin` *is* the platform's `sin`, so the delegation is
// exact wherever the crate builds, not only where a C `libm` is linkable.
// ---------------------------------------------------------------------------

/// Generate a reference that forwards to the platform routine.
macro_rules! delegate {
    ($(#[$doc:meta])* $name:ident => $method:ident) => {
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(x: f64) -> f64 {
            f64::$method(x)
        }
    };
    ($(#[$doc:meta])* $name:ident => $method:ident, 2) => {
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(x: f64, y: f64) -> f64 {
            f64::$method(x, y)
        }
    };
}

delegate! { /// `sin(x)`, the platform's. `x` in radians.
sin => sin }
delegate! { /// `cos(x)`, the platform's. `x` in radians.
cos => cos }
delegate! { /// `tan(x)`, the platform's. `x` in radians.
tan => tan }
delegate! { /// `asin(x)`, the platform's.
asin => asin }
delegate! { /// `acos(x)`, the platform's.
acos => acos }
delegate! { /// `atan(x)`, the platform's.
atan => atan }

delegate! { /// `asinh(x)`, the platform's.
asinh => asinh }
delegate! { /// `acosh(x)`, the platform's.
acosh => acosh }
delegate! { /// `log10(x)`, the platform's.
log10 => log10 }
delegate! { /// `log1p(x)`, the platform's `log1p`.
log1p => ln_1p }

delegate! { /// `log2(x)`, the platform's.
log2 => log2 }
delegate! { /// `atan2(y, x)`, the platform's.
atan2 => atan2, 2 }
delegate! { /// `hypot(x, y)`, the platform's.
hypot => hypot, 2 }

/// `atanh(x)`, matching Rust's `f64::atanh`.
///
/// **Rust's, not the C library's**, and for this one function of the inverse
/// hyperbolic family they differ. Rust does not forward `atanh` to `libm`; it
/// evaluates `0.5 * ln_1p(2x / (1 - x))` itself, which disagrees with glibc's
/// `atanh` on roughly one input in ten. Matching Rust is the right choice here
/// for the same reason it is for [`cbrt`]: the call the caller was already
/// making is `f64::atanh`.
#[inline(always)]
pub fn atanh(x: f64) -> f64 {
    f64::atanh(x)
}

/// `(sin(x), cos(x))`, the platform's.
#[inline(always)]
pub fn sincos(x: f64) -> (f64, f64) {
    (f64::sin(x), f64::cos(x))
}

// ---------------------------------------------------------------------------
// pow
// ---------------------------------------------------------------------------

/// The table-centring offset for `pow`'s logarithm.
///
/// Not `log`'s `0x1.6p-1`. `pow` has its own table, centred differently, and
/// borrowing the other constant costs about four digits of the result — which
/// looks like an accuracy bug rather than an indexing one.
pub(crate) const POW_OFF: u64 = 0x3fe6955500000000;

/// What `exp_inline` adds to the exponent field to negate its result.
///
/// `0x800 << EXP_TABLE_BITS`: it lands in the sign bit once the table index is
/// shifted into place, so a negative base with an odd integer exponent costs
/// no branch at the end.
const SIGN_BIAS: u32 = 0x800 << 7;

/// True for a zero, infinity or NaN bit pattern.
#[inline(always)]
fn zeroinfnan(i: u64) -> bool {
    i.wrapping_mul(2).wrapping_sub(1) >= f64::INFINITY.to_bits().wrapping_mul(2).wrapping_sub(1)
}

/// `0` if `y` is not an integer, `1` if it is odd, `2` if it is even.
///
/// Decided from the exponent and the trailing significand bits rather than by
/// comparing against `trunc`, so it stays exact for every magnitude.
fn checkint(iy: u64) -> u32 {
    let e = ((iy >> 52) & 0x7ff) as i32;
    if e < 0x3ff {
        return 0; // |y| < 1
    }
    if e > 0x3ff + 52 {
        return 2; // too large to have a fractional part; always even
    }
    let shift = (0x3ff + 52 - e) as u32;
    if iy & ((1u64 << shift) - 1) != 0 {
        return 0; // fractional bits set
    }
    if iy & (1u64 << shift) != 0 { 1 } else { 2 }
}

/// True for a signalling NaN.
///
/// `pow(sNaN, 0)` and `pow(1, sNaN)` are the two inputs where the answer is
/// the NaN rather than 1.0, so the distinction has to be made.
#[inline(always)]
fn is_signaling(x: f64) -> bool {
    let b = x.to_bits();
    b & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000
        && b & 0x000f_ffff_ffff_ffff != 0
        && b & 0x0008_0000_0000_0000 == 0
}

/// `log(x)` to more than double precision, as an unevaluated `hi + lo`.
///
/// This is the part that makes `pow` hard, and the reason it needs a table of
/// its own rather than reusing `log`'s: `y` multiplies the error in the
/// logarithm as well as its value, so a single-double logarithm gives up a bit
/// of the answer for every power of two in `y`. `logc` is therefore carried as
/// an unevaluated pair, and every cancellation below is tracked into `lo`.
#[inline(always)]
fn pow_log(ix: u64) -> (f64, f64) {
    let tmp = ix.wrapping_sub(POW_OFF);
    let i = ((tmp >> 45) & 127) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let z = f64::from_bits(iz);
    let kd = k as f64;

    let invc = pt::TAB[3 * i];
    let logc = pt::TAB[3 * i + 1];
    let logctail = pt::TAB[3 * i + 2];

    // |z/c - 1| < 1/N, and the fused form makes r exactly representable.
    let r = z.mul_add(invc, -1.0);

    // k*ln2 + log(c) + r, with the low half kept.
    let t1 = kd * pt::LN2HI + logc;
    let t2 = t1 + r;
    let lo1 = kd * pt::LN2LO + logctail;
    let lo2 = t1 - t2 + r;

    let ar = pt::A0 * r; // A0 = -0.5
    let ar2 = r * ar;
    let ar3 = r * ar2;
    let hi = t2 + ar2;
    let lo3 = ar.mul_add(r, -ar2);
    let lo4 = t2 - hi + ar2;
    let p = ar3 * (pt::A1 + r * pt::A2 + ar2 * (pt::A3 + r * pt::A4 + ar2 * (pt::A5 + r * pt::A6)));
    let lo = lo1 + lo2 + lo3 + lo4 + p;
    let y = hi + lo;
    (y, hi - y + lo)
}

/// glibc `e_pow.c: exp_inline`, for `sign * exp(x + xtail)`.
///
/// Not the same routine as [`exp`]: it takes the argument as an unevaluated
/// pair, and it folds the result's sign into the exponent field through
/// `sign_bias` rather than negating at the end.
fn pow_exp(x: f64, xtail: f64, sign_bias: u32) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    if abstop.wrapping_sub(top12(TINY)) >= top12(512.0).wrapping_sub(top12(TINY)) {
        if abstop.wrapping_sub(top12(TINY)) >= 0x8000_0000 {
            // |x| < 0x1p-54: the result is 1, signed.
            let one = 1.0 + x;
            return if sign_bias != 0 { -one } else { one };
        }
        if abstop >= top12(1024.0) {
            return if x.to_bits() >> 63 != 0 {
                if sign_bias != 0 { -0.0 } else { 0.0 }
            } else if sign_bias != 0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        // Large but representable: fall through, then take `specialcase`.
        abstop = 0;
    }

    // `InvLn2N * x` is fused straight into the rounding add, so it never
    // exists as a rounded double — the same placement `expf` uses, and the
    // same one the C source hides.
    let kd_s = x.mul_add(et::INVLN2N, et::SHIFT);
    let ki = kd_s.to_bits();
    let kd = kd_s - et::SHIFT;
    let r = kd.mul_add(et::NEGLN2LON, kd.mul_add(et::NEGLN2HIN, x)) + xtail;

    let idx = (2 * (ki % 128)) as usize;
    let top = (ki.wrapping_add(sign_bias as u64)) << 45;
    let tail = f64::from_bits(et::TAB[idx]);
    let sbits = et::TAB[idx + 1].wrapping_add(top);

    // The polynomial, in the association `__pow_fma` compiles to: two
    // independent coefficient pairs, then two fused accumulations.
    let r2 = r * r;
    let p23 = et::C3.mul_add(r, et::C2);
    let s = tail + r;
    let p45 = r.mul_add(et::C5, et::C4);
    let t = p23.mul_add(r2, s);
    let tmp = p45.mul_add(r2 * r2, t);
    if abstop == 0 {
        return pow_exp_specialcase(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    scale.mul_add(tmp, scale)
}

/// glibc `e_pow.c: specialcase`.
///
/// Differs from [`exp`]'s in that `sbits` carries a sign, so the subnormal arm
/// has to renormalise around `±1` rather than `+1`.
fn pow_exp_specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        // k > 0: scale's exponent may have overflowed by up to 460. Fused
        // here and *not* in the arm below — the same asymmetry `exp`'s
        // special-case handler has, and for the same reason: it is what the
        // compiler chose, and it changes the last bit.
        let scale = f64::from_bits(sbits.wrapping_sub(1009u64 << 52));
        return P1009 * scale.mul_add(tmp, scale);
    }
    // k < 0: the result may be subnormal, where a second rounding would cost
    // half an ulp, so the sum is renormalised around ±1 before scaling down.
    let sbits = sbits.wrapping_add(1022u64 << 52);
    let scale = f64::from_bits(sbits);
    let mut y = scale + scale * tmp;
    if y.abs() < 1.0 {
        let one = if y < 0.0 { -1.0 } else { 1.0 };
        let lo = scale - y + scale * tmp;
        let hi = one + y;
        let lo = one - hi + y + lo;
        y = (hi + lo) - one;
        // Fix the sign of a zero the renormalisation produced.
        if y == 0.0 {
            y = f64::from_bits(sbits & 0x8000_0000_0000_0000);
        }
    }
    P_M1022 * y
}

/// `x^y`, bit-identical to glibc's `__ieee754_pow_fma`.
///
/// The special-case tree is most of the function, and it is not decoration:
/// `pow` has more of them than anything else in a `libm` — signed zeros,
/// negative bases with integer exponents, exponents so small the result is
/// `1 ± y`, and the parity of `y` deciding the sign of the result.
pub fn pow(x: f64, y: f64) -> f64 {
    let mut sign_bias: u32 = 0;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = top12(x);
    let topy = top12(y);
    let one = 1.0f64.to_bits();
    let inf2 = f64::INFINITY.to_bits().wrapping_mul(2);

    // x is not a positive normal, or |y| is outside [2^-65, 2^63).
    if topx.wrapping_sub(1) >= 0x7ff - 1 || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if is_signaling(x) { x + y } else { 1.0 };
            }
            if ix == one {
                return if is_signaling(y) { x + y } else { 1.0 };
            }
            if ix.wrapping_mul(2) > inf2 || iy.wrapping_mul(2) > inf2 {
                return x + y; // a NaN in either argument
            }
            if ix.wrapping_mul(2) == one.wrapping_mul(2) {
                return 1.0; // (-1)^inf
            }
            if (ix.wrapping_mul(2) < one.wrapping_mul(2)) == (iy >> 63 == 0) {
                return 0.0; // |x|<1 with y=+inf, or |x|>1 with y=-inf
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix >> 63 != 0 && checkint(iy) == 1 {
                x2 = -x2;
            }
            // The division is what produces ±inf for a zero base with a
            // negative exponent, sign included.
            return if iy >> 63 != 0 { 1.0 / x2 } else { x2 };
        }
        if ix >> 63 != 0 {
            // Finite x < 0: only an integer exponent is defined.
            match checkint(iy) {
                0 => {
                    #[allow(clippy::eq_op)]
                    return (x - x) / (x - x); // __math_invalid
                }
                1 => sign_bias = SIGN_BIAS,
                _ => {}
            }
            ix &= 0x7fff_ffff_ffff_ffff;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
            // sign_bias is 0 here, because such a y cannot be odd.
            if ix == one {
                return 1.0;
            }
            if (topy & 0x7ff) < 0x3be {
                // |y| < 2^-65: x^y is 1 + y*log(x) to within a rounding.
                return if ix > one { 1.0 + y } else { 1.0 - y };
            }
            return if (ix > one) == (topy < 0x800) {
                f64::INFINITY
            } else {
                0.0
            };
        }
        if topx == 0 {
            // Subnormal x: normalise so the exponent becomes negative.
            ix = (x * P52).to_bits();
            ix &= 0x7fff_ffff_ffff_ffff;
            ix -= 52u64 << 52;
        }
    }

    let (hi, lo) = pow_log(ix);
    // `y * (hi + lo)` as an unevaluated pair, both products fused.
    let ehi = y * hi;
    let elo = y.mul_add(lo, y.mul_add(hi, -ehi));
    pow_exp(ehi, elo, sign_bias)
}

// ---------------------------------------------------------------------------
// expm1, and the hyperbolic family built on it
//
// These four are ports rather than delegations, and the second three come
// almost free once the first exists: glibc's `sinh`, `cosh` and `tanh` are the
// fdlibm compositions on `exp` and `expm1`, exactly, with no fused-multiply-add
// subtleties of their own. Verified at 300000 inputs each on first attempt --
// which is worth stating, because nothing else in this crate went that way.
// ---------------------------------------------------------------------------

/// Overflow threshold for `expm1`, `ln(DBL_MAX)`.
const EXPM1_OTHRESHOLD: f64 = f64::from_bits(0x40862e42fefa39ef);
/// `ln(2)`, high part, exact in 33 bits.
pub(crate) const EXPM1_LN2HI: f64 = f64::from_bits(0x3fe62e42fee00000);
/// `ln(2)`, low part.
pub(crate) const EXPM1_LN2LO: f64 = f64::from_bits(0x3dea39ef35793c76);
/// `1 / ln(2)`.
pub(crate) const EXPM1_INVLN2: f64 = f64::from_bits(0x3ff71547652b82fe);
/// Scaled minimax coefficients for the rational core, `Q1..Q5`.
pub(crate) const EXPM1_Q: [f64; 5] = [
    f64::from_bits(0xbfa11111111110f4),
    f64::from_bits(0x3f5a01a019fe5585),
    f64::from_bits(0xbf14ce199eaadbb7),
    f64::from_bits(0x3ed0cfca86e65239),
    f64::from_bits(0xbe8afdb76e09c32d),
];

/// `e^x - 1`, bit-identical to glibc's `__expm1_fma`.
///
/// The polynomial is evaluated by Estrin's scheme, not Horner — that is what
/// the compiled library does, and the two round differently even though they
/// compute the same polynomial. Three operations are fused: `3 - r1*hfx`,
/// `6 - x*t`, and the reconstruction `x*(e - c) - c`.
pub fn expm1(mut x: f64) -> f64 {
    let hx = ((x.to_bits() >> 32) & 0x7fff_ffff) as u32;
    let sign = x.to_bits() >> 63 != 0;

    // Huge and non-finite arguments.
    if hx >= 0x4043_687A {
        // |x| >= 56*ln2
        if x.is_nan() {
            return x + x;
        }
        if sign {
            return -1.0;
        }
        if x > EXPM1_OTHRESHOLD {
            return x * f64::from_bits(0x7fe0000000000000);
        }
    }

    let hi: f64;
    let lo: f64;
    let k: i32;
    let c: f64;

    if hx > 0x3fd6_2e42 {
        // |x| > 0.5 ln2: reduce to x = k ln2 + r.
        if hx < 0x3FF0_A2B2 {
            // |x| < 1.5 ln2, so k is +-1 and the multiply can be skipped.
            if !sign {
                hi = x - EXPM1_LN2HI;
                lo = EXPM1_LN2LO;
                k = 1;
            } else {
                hi = x + EXPM1_LN2HI;
                lo = -EXPM1_LN2LO;
                k = -1;
            }
        } else {
            k = (EXPM1_INVLN2 * x + if sign { -0.5 } else { 0.5 }) as i32;
            let t = k as f64;
            hi = x - t * EXPM1_LN2HI; // exact
            lo = t * EXPM1_LN2LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3c90_0000 {
        // |x| < 2^-54: the result is x.
        return x;
    } else {
        c = 0.0;
        k = 0;
    }

    // Primary range.
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let hxs2 = hxs * hxs;
    let c01 = hxs.mul_add(EXPM1_Q[0], 1.0);
    let c23 = EXPM1_Q[2].mul_add(hxs, EXPM1_Q[1]);
    let c45 = EXPM1_Q[4].mul_add(hxs, EXPM1_Q[3]);
    let r1 = (hxs2 * hxs2).mul_add(c45, hxs2.mul_add(c23, c01));
    let t = (-r1).mul_add(hfx, 3.0);
    let mut e = hxs * ((r1 - t) / (-t).mul_add(x, 6.0));

    if k == 0 {
        return x - e.mul_add(x, -hxs);
    }
    e = (e - c).mul_add(x, -c) - hxs;

    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return 1.0 + 2.0 * (x - e);
    }
    let twopk = f64::from_bits(((0x3ff + k) as u64) << 52);
    if !(0..=56).contains(&k) {
        let y = x - e + 1.0;
        let y = if k == 1024 {
            y * 2.0 * f64::from_bits(0x7fe0000000000000)
        } else {
            y * twopk
        };
        return y - 1.0;
    }
    let uf = f64::from_bits(((0x3ff - k) as u64) << 52); // 2^-k
    if k < 20 {
        (x - e + (1.0 - uf)) * twopk
    } else {
        (x - (e + uf) + 1.0) * twopk
    }
}

/// Hyperbolic cosine, bit-identical to the platform's.
pub fn cosh(x: f64) -> f64 {
    let ix = ((x.to_bits() >> 32) & 0x7fff_ffff) as u32;
    if ix >= 0x7ff0_0000 {
        return x * x; // infinity, or a NaN to quieten
    }
    let a = x.abs();
    if ix < 0x3fd6_2e43 {
        // |x| < 0.5 ln2: 1 + t^2 / (2(1+t)), which does not cancel.
        let t = expm1(a);
        let w = 1.0 + t;
        if ix < 0x3c80_0000 {
            return w;
        }
        return 1.0 + (t * t) / (w + w);
    }
    if ix < 0x4036_0000 {
        // |x| < 22
        let t = exp(a);
        return 0.5 * t + 0.5 / t;
    }
    if ix < 0x4086_2E42 {
        // |x| < ln(DBL_MAX): the reciprocal term has underflowed away.
        return 0.5 * exp(a);
    }
    if ix <= 0x4086_33CE {
        // Up to the overflow threshold, in two halves so neither overflows.
        let w = exp(0.5 * a);
        let t = 0.5 * w;
        return t * w;
    }
    f64::MAX * f64::MAX
}

/// Hyperbolic sine, bit-identical to the platform's.
pub fn sinh(x: f64) -> f64 {
    let jx = (x.to_bits() >> 32) as u32;
    let ix = jx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        return x + x;
    }
    let h = if jx >> 31 != 0 { -0.5 } else { 0.5 };
    let a = x.abs();
    if ix < 0x4036_0000 {
        // |x| < 22
        if ix < 0x3e30_0000 {
            return x; // |x| < 2^-28
        }
        let t = expm1(a);
        if ix < 0x3ff0_0000 {
            return h * (2.0 * t - t * t / (t + 1.0));
        }
        return h * (t + t / (t + 1.0));
    }
    if ix < 0x4086_2E42 {
        return h * exp(a);
    }
    if ix <= 0x4086_33CE {
        let w = exp(0.5 * a);
        let t = h * w;
        return t * w;
    }
    x * f64::MAX
}

/// Hyperbolic tangent, bit-identical to the platform's.
pub fn tanh(x: f64) -> f64 {
    let jx = (x.to_bits() >> 32) as u32;
    let ix = jx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        return if jx >> 31 == 0 {
            1.0 / x + 1.0
        } else {
            1.0 / x - 1.0
        };
    }
    let z;
    if ix < 0x4036_0000 {
        // |x| < 22
        // 2^-55, not fdlibm's 2^-28. Between the two thresholds the platform
        // takes the general path, and the division there rounds the result one
        // ulp away from `x` — so the wider fdlibm shortcut is not what runs.
        if ix < 0x3c80_0000 {
            return x;
        }
        if ix >= 0x3ff0_0000 {
            let t = expm1(2.0 * x.abs());
            z = 1.0 - 2.0 / (t + 2.0);
        } else {
            let t = expm1(-2.0 * x.abs());
            z = -t / (t + 2.0);
        }
    } else {
        z = 1.0 - f64::MIN_POSITIVE;
    }
    if jx >> 31 == 0 { z } else { -z }
}
