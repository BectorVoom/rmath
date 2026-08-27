//! `sinf`, `cosf`, `sincosf` and `tanf`.
//!
//! Two different upstreams, and the split matters for how each is verified.
//!
//! `sinf`/`cosf`/`sincosf` are ARM optimized-routines (glibc's
//! `sysdeps/ieee754/flt-32/s_sinf.c`, `s_cosf.c`, `s_sincosf.c` and the
//! shared `s_sincosf.h` / `sincosf_poly.h`). They are *not* correctly
//! rounded — the documented worst case is 0.5607 ULP — so bit-exactness here
//! is a claim about a schedule, and the fused-multiply-add placement below
//! was read out of the compiled `__sinf_fma` / `__cosf_fma` / `__sincosf_fma`
//! (`objdump -d`, glibc 2.43, x86-64), which is the ifunc variant any
//! FMA-capable machine selects. Every multiply that feeds an add fuses,
//! including `reduce_fast`'s `x - n * hpi` (`vfnmadd132sd`).
//!
//! `tanf` is CORE-MATH, and is *correctly rounded*: it returns the nearest
//! representable value on every input, so reproducing it is not a claim about
//! this platform at all — any correct `tanf` anywhere returns these bits. It
//! is also built without `-mfma` (glibc's
//! `sysdeps/x86_64/fpu/multiarch/Makefile` gives `-mfma -mavx2` to
//! `s_sinf-fma.c` and `s_sincosf-fma.c` but has no `s_tanf-fma.c` at all, and
//! `__tanf` in `libm.so.6` contains no `vfmadd` of any form), so its
//! arithmetic below is deliberately unfused except where the C calls `fmaf`
//! outright.
//!
//! Verified by exhaustive sweep: all 2^32 `f32` bit patterns, not a sample.

use crate::tables::single::trig as t;

/// The top 12 bits of a `float`, sign cleared — `s_sincosf.h`'s `abstop12`.
///
/// Every band test in the sine family is one comparison on this, which is why
/// the thresholds below are spelled as `abstop12` of a float rather than as
/// the float itself.
#[inline(always)]
const fn abstop12(x: f32) -> u32 {
    (x.to_bits() >> 20) & 0x7ff
}

/// `pi/4`, the top of the band where no reduction is needed.
const PIO4: f32 = f32::from_bits(0x3f490fdb);
/// `2 pi * 2^-64`, the scale that turns [`reduce_large`]'s fixed-point
/// remainder into radians.
const PI63: f64 = f64::from_bits(0x3c1921fb54442d18);

/// `__math_invalidf (x)`: the NaN an infinite or NaN argument produces.
///
/// `(x - x) / (x - x)` is glibc's own spelling, not a stand-in for a
/// literal NaN — for a NaN argument it propagates that NaN's payload
/// quietened, which a fresh `f32::NAN` would not.
#[inline(always)]
#[allow(clippy::eq_op)]
fn invalid(x: f32) -> f32 {
    (x - x) / (x - x)
}

/// `sinf_poly`: the sine (`n` even) or cosine (`n` odd) polynomial of the
/// reduced argument, from the record `p`.
#[inline(always)]
fn sinf_poly(x: f64, x2: f64, p: usize, n: i32) -> f32 {
    let c = &t::POLY[p];
    if n & 1 == 0 {
        let x3 = x * x2;
        let s1 = x2.mul_add(c[t::S3], c[t::S2]);
        let x7 = x3 * x2;
        let s = x3.mul_add(c[t::S1], x);
        x7.mul_add(s1, s) as f32
    } else {
        let x4 = x2 * x2;
        let c2 = x2.mul_add(c[t::C4], c[t::C3]);
        let c1 = x2.mul_add(c[t::C1], c[t::C0]);
        let x6 = x4 * x2;
        let cc = x4.mul_add(c[t::C2], c1);
        x6.mul_add(c2, cc) as f32
    }
}

/// `sincosf_poly`: both polynomials at once, already swapped for odd `n`.
///
/// Not `sinf_poly` called twice: the two share `x4`, and the C evaluates
/// them interleaved. The results are the same values either way, but this
/// module reproduces control flow rather than paraphrasing it.
#[inline(always)]
fn sincosf_poly(x: f64, x2: f64, p: usize, n: i32) -> (f32, f32) {
    let c = &t::POLY[p];
    let x4 = x2 * x2;
    let x3 = x2 * x;
    let c2 = x2.mul_add(c[t::C4], c[t::C3]);
    let s1 = x2.mul_add(c[t::S3], c[t::S2]);
    let c1 = x2.mul_add(c[t::C1], c[t::C0]);
    let x5 = x3 * x2;
    let x6 = x4 * x2;
    let s = x3.mul_add(c[t::S1], x);
    let cc = x4.mul_add(c[t::C2], c1);

    let sin = x5.mul_add(s1, s) as f32;
    let cos = x6.mul_add(c2, cc) as f32;
    // The C swaps the two output pointers for odd `n` rather than the values.
    if n & 1 != 0 { (cos, sin) } else { (sin, cos) }
}

/// `reduce_fast`: `x` modulo `pi/2` by one multiply-subtract, valid to
/// `|x| <= 120`.
///
/// `hpi_inv` is prescaled by `2^24` so that the quadrant falls out of the
/// truncating conversion's high bits — which is what makes this correct for
/// negative arguments, where a plain truncation would round the wrong way.
#[inline(always)]
fn reduce_fast(x: f64) -> (f64, i32) {
    let r = x * t::HPI_INV;
    let n = ((r as i32).wrapping_add(0x800000)) >> 24;
    (f64::from(n).mul_add(-t::HPI, x), n)
}

/// `reduce_large`: `x` modulo `pi/2` by a 32x96-to-128 bit multiply against
/// `4/pi` held to 192 bits.
///
/// Past `|x| = 120` a `double` `pi/2` no longer holds enough bits: the
/// reduced argument of a `float` near `2^127` depends on bits of `4/pi` far
/// below anything a `double` carries, and only the exact table recovers them.
#[inline(always)]
fn reduce_large(xi: u32) -> (f64, i32) {
    let arr = &t::INV_PIO4[((xi >> 26) & 15) as usize..];
    let shift = (xi >> 23) & 7;

    let xi = ((xi & 0x00ff_ffff) | 0x0080_0000) << shift;
    let res0 = (xi as u64).wrapping_mul(arr[0] as u64);
    let res1 = (xi as u64).wrapping_mul(arr[4] as u64);
    let res2 = (xi as u64).wrapping_mul(arr[8] as u64);
    let res0 = (res2 >> 32) | (res0 << 32);
    let mut res0 = res0.wrapping_add(res1);

    let n = (res0.wrapping_add(1u64 << 61)) >> 62;
    res0 = res0.wrapping_sub(n << 62);
    ((res0 as i64) as f64 * PI63, n as i32)
}

/// `sin(x)`. `x` in radians.
pub fn sin(y: f32) -> f32 {
    let x = f64::from(y);

    if abstop12(y) < abstop12(PIO4) {
        let x2 = x * x;
        // Below `2^-12` the polynomial's first correction is already below
        // half an ulp of `y`. The C forces the underflow flag for subnormal
        // arguments here; nothing in this crate observes flags.
        if abstop12(y) < abstop12(f32::from_bits(0x39800000)) {
            return y;
        }
        return sinf_poly(x, x2, 0, 0);
    }
    if abstop12(y) < abstop12(120.0) {
        let (x, n) = reduce_fast(x);
        let s = t::SIGN[(n & 3) as usize];
        let p = usize::from(n & 2 != 0);
        return sinf_poly(x * s, x * x, p, n);
    }
    if abstop12(y) < abstop12(f32::INFINITY) {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i32;
        let (x, n) = reduce_large(xi);
        // The argument's own sign is folded into the quadrant here rather
        // than applied to the result: `reduce_large` ignores it.
        let s = t::SIGN[((n + sign) & 3) as usize];
        let p = usize::from((n + sign) & 2 != 0);
        return sinf_poly(x * s, x * x, p, n);
    }
    invalid(y)
}

/// `cos(x)`. `x` in radians.
pub fn cos(y: f32) -> f32 {
    let x = f64::from(y);

    if abstop12(y) < abstop12(PIO4) {
        let x2 = x * x;
        if abstop12(y) < abstop12(f32::from_bits(0x39800000)) {
            return 1.0;
        }
        return sinf_poly(x, x2, 0, 1);
    }
    if abstop12(y) < abstop12(120.0) {
        let (x, n) = reduce_fast(x);
        let s = t::SIGN[(n & 3) as usize];
        let p = usize::from(n & 2 != 0);
        return sinf_poly(x * s, x * x, p, n ^ 1);
    }
    if abstop12(y) < abstop12(f32::INFINITY) {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i32;
        let (x, n) = reduce_large(xi);
        let s = t::SIGN[((n + sign) & 3) as usize];
        let p = usize::from((n + sign) & 2 != 0);
        return sinf_poly(x * s, x * x, p, n ^ 1);
    }
    invalid(y)
}

/// `(sin(x), cos(x))`. `x` in radians.
pub fn sincos(y: f32) -> (f32, f32) {
    let x = f64::from(y);

    if abstop12(y) < abstop12(PIO4) {
        let x2 = x * x;
        if abstop12(y) < abstop12(f32::from_bits(0x39800000)) {
            return (y, 1.0);
        }
        return sincosf_poly(x, x2, 0, 0);
    }
    if abstop12(y) < abstop12(120.0) {
        let (x, n) = reduce_fast(x);
        let s = t::SIGN[(n & 3) as usize];
        let p = usize::from(n & 2 != 0);
        return sincosf_poly(x * s, x * x, p, n);
    }
    if abstop12(y) < abstop12(f32::INFINITY) {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i32;
        let (x, n) = reduce_large(xi);
        let s = t::SIGN[((n + sign) & 3) as usize];
        let p = usize::from((n + sign) & 2 != 0);
        return sincosf_poly(x * s, x * x, p, n);
    }
    // Both halves get `y - y`, and the `__math_invalidf` the C also calls
    // here exists only to set `errno`.
    #[allow(clippy::eq_op)]
    {
        (y - y, y - y)
    }
}

// ---------------------------------------------------------------------------
// tan
// ---------------------------------------------------------------------------

/// `tan`'s numerator, a degree-3 polynomial in `z^2`.
const TAN_CN: [f64; 4] = [
    f64::from_bits(0x3ff921fb54442d18),
    f64::from_bits(0xbfdfd226e573289f),
    f64::from_bits(0x3f9b7a60c8dac9f6),
    f64::from_bits(0xbf2725beb40f33e5),
];
/// `tan`'s denominator, likewise.
const TAN_CD: [f64; 4] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0xbff2395347fb829d),
    f64::from_bits(0x3fc2313660f29c36),
    f64::from_bits(0xbf69a707ab98d1c1),
];
/// `2/pi` to 256 bits, `rbig`'s exact multiplier — a different spelling of
/// the same constant as [`t::INV_PIO4`], in 64-bit limbs rather than
/// overlapping 32-bit windows, because `rbig` does its multiply in `u128`.
const TAN_IPI: [u64; 4] = [
    0xfe51_63ab_debb_c562,
    0xdb62_9599_3c43_9041,
    0xfc27_57d1_f534_ddc0,
    0xa2f9_836e_4e44_1529,
];

/// The eight arguments where the rounding test below cannot settle the last
/// bit, with the correctly-rounded answer as an unevaluated `rh + rl`.
///
/// CORE-MATH's own table. `tanf` is correctly rounded, and these are the
/// inputs whose true value sits close enough to a halfway point that the
/// degree-3 rational's error bound does not decide which way it falls; the
/// pair is exact enough that `rh + rl` rounds correctly on its own.
const TAN_EXCEPTIONS: [(u32, f32, f32); 8] = [
    (
        0x3f8a1f62,
        f32::from_bits(0x3feefcfb),
        f32::from_bits(0xa5c48e92),
    ),
    (
        0x4d56d355,
        f32::from_bits(0x3e740182),
        f32::from_bits(0x22a0cfa3),
    ),
    (
        0x57d7b0ed,
        f32::from_bits(0x3eb068e4),
        f32::from_bits(0xa416b61d),
    ),
    (
        0x5980445e,
        f32::from_bits(0x3fe50f68),
        f32::from_bits(0x257b0298),
    ),
    (
        0x63fc86fe,
        f32::from_bits(0x3f2cbfce),
        f32::from_bits(0x25492cbc),
    ),
    (
        0x6a662711,
        f32::from_bits(0xc0799ac2),
        f32::from_bits(0x266b92a5),
    ),
    (
        0x6ad36709,
        f32::from_bits(0xbf62b097),
        f32::from_bits(0xa513619f),
    ),
    (
        0x72b505bb,
        f32::from_bits(0xbff2150f),
        f32::from_bits(0xa58ee483),
    ),
];

/// `rltl`: reduce `|x| < 2^28` to `2/pi * x = q + r`, returning `r`.
#[inline(always)]
fn rltl(z: f32) -> (f64, i32) {
    let x = f64::from(z);
    let idl = f64::from_bits(0xbdfb1bbead603d8b) * x;
    let idh = f64::from_bits(0x3fe45f306e000000) * x;
    let id = idh.round_ties_even();
    ((idh - id) + idl, id as i64 as i32)
}

/// `rbig`: the same reduction for `|x| >= 2^28`, exactly, via a 24x256-bit
/// multiply against `2/pi`.
fn rbig(u: u32) -> (f64, i32) {
    let e = ((u >> 23) & 0xff) as i32;
    let m = ((u & (!0u32 >> 9)) | 1 << 23) as u128;

    let p0 = m * TAN_IPI[0] as u128;
    let p1 = m * TAN_IPI[1] as u128 + (p0 >> 64);
    let p2 = m * TAN_IPI[2] as u128 + (p1 >> 64);
    let p3 = m * TAN_IPI[3] as u128 + (p2 >> 64);

    let p3h = (p3 >> 64) as u64;
    let p3l = p3 as u64;
    let p2l = p2 as u64;
    let p1l = p1 as u64;

    let k = e - 127;
    let s = k - 23;
    // `155 <= e <= 254` here, so `5 <= s <= 104`: every shift below is in
    // range and none of the three arms is dead.
    let (i, a): (i32, i64) = if s < 64 {
        let sh = s as u32;
        (
            ((p3h << sh) | (p3l >> (64 - sh))) as i32,
            ((p3l << sh) | (p2l >> (64 - sh))) as i64,
        )
    } else if s == 64 {
        (p3l as i32, p2l as i64)
    } else {
        let sh = (s - 64) as u32;
        (
            ((p3l << sh) | (p2l >> (128 - s as u32))) as i32,
            ((p2l << sh) | (p1l >> (128 - s as u32))) as i64,
        )
    };

    let sgn = (u as i32) >> 31;
    let sm = a >> 63;
    let i = i.wrapping_sub(sm as i32);
    let z = ((a ^ i64::from(sgn)) as f64) * f64::from_bits(0x3bf0000000000000);
    ((z), (i ^ sgn).wrapping_sub(sgn))
}

/// `tan(x)`. `x` in radians.
///
/// Correctly rounded, so this is bit-exact against any correct `tanf`, not
/// only against this platform's.
pub fn tan(x: f32) -> f32 {
    let t_bits = x.to_bits();
    let e = ((t_bits >> 23) & 0xff) as i32;

    let (z, i) = if e < 127 + 28 {
        if e < 115 {
            // `|x| < 2^-12`: `tan x` is `x` to within the rounding, but the
            // `fmaf` keeps the inexact flag and the correct direction.
            if e < 102 {
                return x.mul_add(x.abs(), x);
            }
            let x2 = x * x;
            return x.mul_add(f32::from_bits(0x3eaaaaab) * x2, x);
        }
        rltl(x)
    } else if e < 0xff {
        rbig(t_bits)
    } else {
        if t_bits << 9 != 0 {
            return x + x; // NaN
        }
        return invalid(x);
    };

    let z2 = z * z;
    let z4 = z2 * z2;
    let n = TAN_CN[0] + z2 * TAN_CN[1];
    let n2 = TAN_CN[2] + z2 * TAN_CN[3];
    let n = n + z4 * n2;
    let d = TAN_CD[0] + z2 * TAN_CD[1];
    let d2 = TAN_CD[2] + z2 * TAN_CD[3];
    let d = d + z4 * d2;
    let n = n * z;

    // `tan` for even quadrants, `-cot` for odd, selected by multiplying
    // rather than branching — `s0`/`s1` are 0 and 1 either way round.
    let s0 = f64::from(i & 1);
    let s1 = f64::from(1 - (i & 1));
    let r1 = (n * s1 - d * s0) / (n * s0 + d * s1);

    // The correct-rounding test: if the result's low bits sit within the
    // rational's error bound of a halfway point, the polynomial cannot
    // decide the last bit and the answer comes from the exception table.
    let tail = (r1.to_bits().wrapping_add(7)) & (!0u64 >> 35);
    if tail <= 14 {
        let ax = t_bits & (!0u32 >> 1);
        let sgn = t_bits >> 31;
        for (arg, rh, rl) in TAN_EXCEPTIONS {
            if arg == ax {
                return if sgn != 0 { -rh - rl } else { rh + rl };
            }
        }
    }
    r1 as f32
}
