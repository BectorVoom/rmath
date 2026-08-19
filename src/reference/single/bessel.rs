//! The Bessel functions in single precision: `j0f`, `j1f`, `y0f`, `y1f`,
//! `jnf`, `ynf`.
//!
//! Same family as [`crate::reference::double::bessel`], different algorithm.
//! glibc's `float` versions are fdlibm's rational fits *plus* a repair that
//! the `double` ones do not have, and the repair is the interesting part.
//!
//! # Why the rational fit alone is not enough
//!
//! `j0` is computed as `sqrt(2/(pi x)) (P cos(x0) - Q sin(x0))`. Near a zero
//! of `j0` the bracket is the difference of two nearly equal numbers, so it
//! loses digits — not a few, but most of them, because the zeros of `j0` are
//! not representable and the cancellation is essentially total. glibc's answer
//! is to fit a degree-3 polynomial to each of the first 64 zeros and use it
//! whenever the bracket comes out small, and beyond the 64th zero to fall back
//! on an asymptotic form with Payne-Hanek argument reduction.
//!
//! So each of the four order-0 and order-1 routines has three paths — the
//! rational fit, the near-a-zero polynomial, and the asymptotic form — and
//! which one runs is decided *after* the rational fit, by the size of the
//! bracket. That is a data-dependent branch, and it is why these are the one
//! family in this crate whose kernels do not vectorise; see
//! [`crate::kernels::single::bessel`].
//!
//! These are not correctly rounded — glibc states a bound of 9 ulps — so
//! matching them means reproducing the schedule exactly, as everywhere else in
//! this crate.

use crate::tables::single::bessel as t;

/// `pi` in `f32`, glibc's `M_PIf`.
const PI_F32: f32 = f32::from_bits(0x40490fdb);
/// `sqrt(2/pi)` rounded to `f32`, the asymptotic branches' scale.
const SQRT_2_OVER_PI: f32 = f32::from_bits(0x3f4c422a);
/// How many zeros the near-a-zero tables cover.
const SMALL_SIZE: usize = 64;

/// `|x|`'s bit pattern.
#[inline(always)]
fn ix(x: f32) -> u32 {
    x.to_bits() & 0x7fff_ffff
}

/// Which of the four rational fits covers `|x|`.
#[inline(always)]
fn interval(i: u32) -> usize {
    if i >= 0x4100_0000 {
        0
    } else if i >= 0x40f7_1c58 {
        1
    } else if i >= 0x4036_db68 {
        2
    } else {
        3
    }
}

// ---------------------------------------------------------------------------
// Payne-Hanek reduction, for the asymptotic branches
// ---------------------------------------------------------------------------

/// `2 pi * 2^-64`, the scale that turns the fixed-point remainder into radians.
const PI63: f64 = f64::from_bits(0x3c1921fb54442d18);
/// `pi/2` to `double`.
const PI_OVER_2: f64 = f64::from_bits(0x3ff921fb54442d18);

/// Reduce `|x|` modulo `pi/2` exactly, returning `(h, n)` with
/// `|x| = h + n pi/2` and `|h| <= pi/4`.
///
/// A 32x96-to-128 bit multiply against `4/pi` held to 192 bits. The
/// alternative — subtracting a `double` approximation of `pi/2` — cannot
/// work here: for `x` near `2^127` the reduced argument depends on bits of
/// `4/pi` far below anything a `double` holds. glibc's `reduce_large`.
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

/// `(h, n)` with `x - pi/4 - alpha = h + n pi/2` modulo `2 pi`.
///
/// glibc's `reduce_aux`. The `alpha` argument is the asymptotic phase
/// correction, folded into the reduction rather than added afterwards so that
/// it does not reintroduce the cancellation the reduction just removed.
fn reduce_aux(x: f32, alpha: f64) -> (f64, i32) {
    let (mut h, mut n) = reduce_large(x.to_bits());
    if x < 0.0 {
        h = -h;
        n = -n;
    }
    if h >= 0.0 {
        h -= PI_OVER_2 / 2.0;
    } else {
        h += PI_OVER_2 / 2.0;
        n -= 1;
    }
    h -= alpha;
    if h > PI_OVER_2 {
        h -= PI_OVER_2;
        n += 1;
    } else if h < -PI_OVER_2 {
        h += PI_OVER_2;
        n -= 1;
    }
    (h, n)
}

/// `t * cos(xr + n pi/2)`, the tail every asymptotic branch ends with.
#[inline(always)]
fn quadrant_cos(t: f32, xr: f32, n: i32) -> f32 {
    match n & 3 {
        0 => t * xr.cos(),
        2 => -t * xr.cos(),
        1 => -t * xr.sin(),
        _ => t * xr.sin(),
    }
}

/// `t * sin(xr + n pi/2)`.
#[inline(always)]
fn quadrant_sin(t: f32, xr: f32, n: i32) -> f32 {
    match n & 3 {
        0 => t * xr.sin(),
        2 => -t * xr.sin(),
        1 => t * xr.cos(),
        _ => -t * xr.cos(),
    }
}

/// `(beta0, alpha0)`: the order-0 asymptotic amplitude and phase, from
/// Harrison's expansion — `beta0 = 1 - 1/(16x^2) + 53/(512x^4)`,
/// `alpha0 = 1/(8x) - 25/(384x^3)`.
#[inline(always)]
fn ab0(x: f32) -> (f64, f64) {
    let y = 1.0 / x as f64;
    let y2 = y * y;
    let beta = 1.0 + y2 * (-0.0625 + f64::from_bits(0x3fba800000000000) * y2);
    let alpha = y * (0.125 - f64::from_bits(0x3fb0aaaaa0000000) * y2);
    (beta, alpha)
}

/// `(beta1, alpha1)`: the order-1 amplitude and phase —
/// `beta1 = 1 + 3/(16x^2) - 99/(512x^4)`,
/// `alpha1 = -3/(8x) + 21/(128x^3) - 1899/(5120x^5)`.
#[inline(always)]
fn ab1(x: f32) -> (f64, f64) {
    let y = 1.0 / x as f64;
    let y2 = y * y;
    let beta = 1.0 + y2 * (0.1875 - f64::from_bits(0x3fc8c00000000000) * y2);
    let alpha = y * (-0.375 + y2 * (0.1640625 - f64::from_bits(0x3fd7bccccccccccd) * y2));
    (beta, alpha)
}

// ---------------------------------------------------------------------------
// The asymptotic branches
// ---------------------------------------------------------------------------

/// `j0(x)` for `x` beyond the near-a-zero tables.
fn j0_asympt(x: f32) -> f32 {
    // Two arguments the expansion misses by more than 9 ulps, tabulated.
    if x == f32::from_bits(0x4ba332e9) {
        return f32::from_bits(0x27250206);
    }
    if x == f32::from_bits(0x4354d7ef) {
        return f32::from_bits(0x33747039);
    }
    let (beta, alpha) = ab0(x);
    let (h, n) = reduce_aux(x, alpha);
    let t = SQRT_2_OVER_PI / x.sqrt() * beta as f32;
    quadrant_cos(t, h as f32, n)
}

/// `y0(x)` for `x` beyond the near-a-zero tables.
fn y0_asympt(x: f32) -> f32 {
    if x == f32::from_bits(0x435fd6cb) {
        return f32::from_bits(0xb0fe657a);
    }
    if x == f32::from_bits(0x48171521) {
        return f32::from_bits(0x2bd244ba);
    }
    let (beta, alpha) = ab0(x);
    let (h, n) = reduce_aux(x, alpha);
    let t = SQRT_2_OVER_PI / x.sqrt() * beta as f32;
    quadrant_sin(t, h as f32, n)
}

/// `j1(x)` for `x` beyond the near-a-zero tables. Odd, so the sign is carried
/// on the scale rather than applied afterwards.
fn j1_asympt(x: f32) -> f32 {
    let (x, cst) = if x < 0.0 {
        (-x, -SQRT_2_OVER_PI)
    } else {
        (x, SQRT_2_OVER_PI)
    };
    let (beta, alpha) = ab1(x);
    let (h, n) = reduce_aux(x, alpha);
    let t = cst / x.sqrt() * beta as f32;
    quadrant_cos(t, h as f32, n - 1)
}

/// `y1(x)` for `x` beyond the near-a-zero tables.
fn y1_asympt(x: f32) -> f32 {
    let (beta, alpha) = ab1(x);
    let (h, n) = reduce_aux(x, alpha);
    let t = SQRT_2_OVER_PI / x.sqrt() * beta as f32;
    quadrant_sin(t, h as f32, n - 1)
}

// ---------------------------------------------------------------------------
// The near-a-zero repairs
// ---------------------------------------------------------------------------

/// Which of the 64 tabulated zeros `x` is nearest, or `None` if it is past
/// them.
///
/// Clamped at zero: glibc reads the table with whatever `roundf` produced, on
/// the reasoning that the caller only reaches here when `x` really is near a
/// zero. Clamping makes that reasoning unnecessary — a negative index would
/// panic in Rust, and the interval test below rejects the lane anyway.
#[inline(always)]
fn zero_index(x: f32, first: f32) -> Option<usize> {
    let i = ((x - first) / PI_F32).round();
    if i >= SMALL_SIZE as f32 {
        return None;
    }
    Some(if i < 0.0 { 0 } else { i as usize })
}

/// `j0` near one of its zeros; `z` is what the rational fit produced.
fn j0_near_root(x: f32, z: f32) -> f32 {
    let Some(index) = zero_index(x, f32::from_bits(0x4019e8a9)) else {
        return j0_asympt(x);
    };
    let p = &t::J0_ZEROS[index];
    if !(p[0] <= x && x <= p[2]) {
        return z;
    }
    let y = x - p[1];
    p[3] + y * (p[4] + y * (p[5] + y * p[6]))
}

/// `y0` near one of its zeros. The first zero needs two extra degrees, which
/// glibc hard-codes rather than widening the whole table for one row.
fn y0_near_root(x: f32, z: f32) -> f32 {
    let Some(index) = zero_index(x, f32::from_bits(0x3f64c166)) else {
        return y0_asympt(x);
    };
    let p = &t::Y0_ZEROS[index];
    if !(p[0] <= x && x <= p[2]) {
        return z;
    }
    let y = x - p[1];
    let p6 = if index > 0 {
        p[6]
    } else {
        p[6] + y * (f32::from_bits(0xbe691b24) + y * f32::from_bits(0x3e5cd51e))
    };
    p[3] + y * (p[4] + y * (p[5] + y * p6))
}

/// `j1` near one of its zeros.
fn j1_near_root(x: f32, z: f32) -> f32 {
    let (x, sign) = if x < 0.0 { (-x, -1.0f32) } else { (x, 1.0f32) };
    let Some(index) = zero_index(x, f32::from_bits(0x40753aac)) else {
        return sign * j1_asympt(x);
    };
    let p = &t::J1_ZEROS[index];
    if !(p[0] <= x && x <= p[2]) {
        return z;
    }
    let y = x - p[1];
    let p6 = if index > 0 {
        p[6]
    } else {
        p[6] + y * f32::from_bits(0xbb9f28d5)
    };
    sign * (p[3] + y * (p[4] + y * (p[5] + y * p6)))
}

/// `y1` near one of its zeros.
fn y1_near_root(x: f32, z: f32) -> f32 {
    let Some(index) = zero_index(x, f32::from_bits(0x400c9df7)) else {
        return y1_asympt(x);
    };
    let p = &t::Y1_ZEROS[index];
    if !(p[0] <= x && x <= p[2]) {
        return z;
    }
    let y = x - p[1];
    let p6 = match index {
        0 => p[6] + y * (f32::from_bits(0xbb940218) + y * f32::from_bits(0x3c143a0c)),
        1 => p[6] + y * f32::from_bits(0xbb7ff6b8),
        _ => p[6],
    };
    p[3] + y * (p[4] + y * (p[5] + y * p6))
}

// ---------------------------------------------------------------------------
// The rational fits for |x| >= 2
// ---------------------------------------------------------------------------

/// `1 + R(s)/S(s)` with `s = 1/x^2`.
#[inline(always)]
fn rational_p(x: f32, p: &[f32; 6], q: &[f32; 5]) -> f32 {
    let z = 1.0 / (x * x);
    let r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    let s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * q[4]))));
    1.0 + r / s
}

/// `R(s)/S(s)` with `s = 1/x^2`. One term longer in the denominator.
#[inline(always)]
fn rational_q(x: f32, p: &[f32; 6], q: &[f32; 6]) -> f32 {
    let z = 1.0 / (x * x);
    let r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    let s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * (q[4] + z * q[5])))));
    r / s
}

fn pzero(x: f32) -> f32 {
    rational_p(x, &t::P0R[interval(ix(x))], &t::P0S[interval(ix(x))])
}
fn qzero(x: f32) -> f32 {
    (-0.125 + rational_q(x, &t::Q0R[interval(ix(x))], &t::Q0S[interval(ix(x))])) / x
}
fn pone(x: f32) -> f32 {
    rational_p(x, &t::P1R[interval(ix(x))], &t::P1S[interval(ix(x))])
}
fn qone(x: f32) -> f32 {
    (0.375 + rational_q(x, &t::Q1R[interval(ix(x))], &t::Q1S[interval(ix(x))])) / x
}

// ---------------------------------------------------------------------------
// The drivers
// ---------------------------------------------------------------------------

/// The Bessel function of the first kind, order 0. Bit-identical to glibc's
/// `j0f`.
pub fn j0(x: f32) -> f32 {
    let i = ix(x);
    if i >= 0x7f80_0000 {
        return 1.0 / (x * x);
    }
    let x = x.abs();
    if i >= 0x4000_0000 {
        let s = x.sin();
        let c = x.cos();
        let mut ss = s - c;
        let mut cc = s + c;
        if i >= 0x7f00_0000 {
            return j0_asympt(x); // x >= 2^127, where x + x would overflow
        }
        let z = -(x + x).cos();
        if s * c < 0.0 {
            cc = z / ss;
        } else {
            ss = z / cc;
        }
        if i <= 0x5c00_0000 {
            cc = pzero(x) * cc - qzero(x) * ss;
        }
        let z = (t::INVSQRTPI * cc) / x.sqrt();
        // A small bracket means the subtraction above cancelled, and the
        // result has lost more digits than the 9-ulp claim allows.
        return if cc.abs() > f32::from_bits(0x3dabcd93) {
            z
        } else {
            j0_near_root(x, z)
        };
    }
    if i < 0x3900_0000 {
        // |x| < 2^-13.
        if i < 0x3200_0000 {
            return 1.0; // |x| < 2^-27
        }
        return 1.0 - 0.25 * x * x;
    }
    let z = x * x;
    let r = &t::J0_R;
    let s = &t::J0_S;
    let rn = z * (r[0] + z * (r[1] + z * (r[2] + z * r[3])));
    let sd = 1.0 + z * (s[0] + z * (s[1] + z * (s[2] + z * s[3])));
    if i < 0x3f80_0000 {
        1.0 + z * (-0.25 + (rn / sd))
    } else {
        let u = 0.5 * x;
        (1.0 + u) * (1.0 - u) + z * (rn / sd)
    }
}

/// The Bessel function of the second kind, order 0. Bit-identical to glibc's
/// `y0f`.
pub fn y0(x: f32) -> f32 {
    let hx = x.to_bits();
    let i = ix(x);
    if i >= 0x7f80_0000 {
        if hx == 0xff80_0000 {
            return f32::NAN; // y0(-inf) is NaN
        }
        return 1.0 / (x + x * x);
    }
    if i == 0 {
        return -1.0 / 0.0;
    }
    if hx >> 31 != 0 {
        return f32::NAN;
    }
    // The second window is around y0's first zero, where the rational fit
    // below is useless even though |x| < 2.
    if i >= 0x4000_0000 || (0x3f53_40ed..=0x3f77_b5e5).contains(&i) {
        let s = x.sin();
        let c = x.cos();
        let mut ss = s - c;
        let mut cc = s + c;
        if i >= 0x7f00_0000 {
            return y0_asympt(x);
        }
        let z = -(x + x).cos();
        if s * c < 0.0 {
            cc = z / ss;
        } else {
            ss = z / cc;
        }
        if i <= 0x5c00_0000 {
            ss = pzero(x) * ss + qzero(x) * cc;
        }
        let z = (t::INVSQRTPI * ss) / x.sqrt();
        return if ss.abs() > f32::from_bits(0x3ddf2c2d) {
            z
        } else {
            y0_near_root(x, z)
        };
    }
    if i <= 0x3980_0000 {
        return t::Y0_U[0] + t::TPI * super::ln(x);
    }
    let z = x * x;
    let u = &t::Y0_U;
    let v = &t::Y0_V;
    let un = u[0] + z * (u[1] + z * (u[2] + z * (u[3] + z * (u[4] + z * (u[5] + z * u[6])))));
    let vd = 1.0 + z * (v[0] + z * (v[1] + z * (v[2] + z * v[3])));
    un / vd + t::TPI * (j0(x) * super::ln(x))
}

/// The Bessel function of the first kind, order 1. Bit-identical to glibc's
/// `j1f`.
pub fn j1(x: f32) -> f32 {
    let hx = x.to_bits();
    let i = ix(x);
    if i >= 0x7f80_0000 {
        return 1.0 / x;
    }
    let y = x.abs();
    if i >= 0x4000_0000 {
        let s = y.sin();
        let c = y.cos();
        let mut ss = -s - c;
        let mut cc = s - c;
        if i >= 0x7f00_0000 {
            return j1_asympt(x);
        }
        let z = (y + y).cos();
        if s * c > 0.0 {
            cc = z / ss;
        } else {
            ss = z / cc;
        }
        if i <= 0x5c00_0000 {
            cc = pone(y) * cc - qone(y) * ss;
        }
        let z = (t::INVSQRTPI * cc) / y.sqrt();
        let z = if hx >> 31 != 0 { -z } else { z };
        return if cc.abs() > f32::from_bits(0x3ddbcb1c) {
            z
        } else {
            j1_near_root(x, z)
        };
    }
    if i < 0x3200_0000 {
        return 0.5 * x; // |x| < 2^-27
    }
    let z = x * x;
    let r = &t::J1_R;
    let s = &t::J1_S;
    let rn = z * (r[0] + z * (r[1] + z * (r[2] + z * r[3])));
    let sd = 1.0 + z * (s[0] + z * (s[1] + z * (s[2] + z * (s[3] + z * s[4]))));
    x * 0.5 + rn * x / sd
}

/// The Bessel function of the second kind, order 1. Bit-identical to glibc's
/// `y1f`.
pub fn y1(x: f32) -> f32 {
    let hx = x.to_bits();
    let i = ix(x);
    if i >= 0x7f80_0000 {
        if hx == 0xff80_0000 {
            return f32::NAN;
        }
        return 1.0 / (x + x * x);
    }
    if i == 0 {
        return -1.0 / 0.0;
    }
    if hx >> 31 != 0 {
        return f32::NAN;
    }
    // The crossover is below 2 here — `y1`'s first zero is at 2.197, and the
    // rational fit has already lost its digits by 1.757.
    if i >= 0x3fe0_dfbc {
        let s = x.sin();
        let c = x.cos();
        let mut ss = -s - c;
        let mut cc = s - c;
        if i >= 0x7f00_0000 {
            return y1_asympt(x);
        }
        let z = (x + x).cos();
        if s * c > 0.0 {
            cc = z / ss;
        } else {
            ss = z / cc;
        }
        if i <= 0x5c00_0000 {
            ss = pone(x) * ss + qone(x) * cc;
        }
        let z = (t::INVSQRTPI * ss) / x.sqrt();
        return if ss.abs() > f32::from_bits(0x3e9f00a6) {
            z
        } else {
            y1_near_root(x, z)
        };
    }
    if i <= 0x3300_0000 {
        return -t::TPI / x; // x < 2^-25
    }
    let z = x * x;
    let u = &t::Y1_U;
    let v = &t::Y1_V;
    let un = u[0] + z * (u[1] + z * (u[2] + z * (u[3] + z * u[4])));
    let vd = 1.0 + z * (v[0] + z * (v[1] + z * (v[2] + z * (v[3] + z * v[4]))));
    x * (un / vd) + t::TPI * (j1(x) * super::ln(x) - 1.0 / x)
}

// ---------------------------------------------------------------------------
// jnf and ynf
// ---------------------------------------------------------------------------

/// The Bessel function of the first kind, order `n`. Bit-identical to glibc's
/// `jnf`.
///
/// The same three regimes as the double-precision [`super::super::double::bessel::jn`],
/// with one difference worth noting: the forward recurrence is evaluated in
/// `double` and rounded back to `float` each step. That is glibc's, and it is
/// what stops the recurrence underflowing to zero long before the answer does.
pub fn jn(n: i32, x: f32) -> f32 {
    let i0 = ix(x);
    if i0 > 0x7f80_0000 {
        return x + x; // NaN
    }
    let (n, x) = if n < 0 { (-n, -x) } else { (n, x) };
    let negative = x.to_bits() >> 31 != 0;
    if n == 0 {
        return j0(x);
    }
    if n == 1 {
        return j1(x);
    }
    let sgn = (n & 1) != 0 && negative;
    let x = x.abs();

    let b = if i0 == 0 || i0 >= 0x7f80_0000 {
        return if sgn { -0.0 } else { 0.0 };
    } else if (n as f32) <= x {
        let mut a = j0(x);
        let mut b = j1(x);
        for i in 1..n {
            let temp = b;
            b = (b as f64 * ((i + i) as f64 / x as f64) - a as f64) as f32;
            a = temp;
        }
        b
    } else if i0 < 0x3080_0000 {
        // x < 2^-29: J(n,x) = (x/2)^n / n!.
        if n > 33 {
            0.0
        } else {
            let temp = x * 0.5;
            let mut b = temp;
            let mut a = 1.0f32;
            for i in 2..=n {
                a *= i as f32;
                b *= temp;
            }
            b / a
        }
    } else {
        let w = (n + n) as f32 / x;
        let h = 2.0f32 / x;
        let mut q0 = w;
        let mut z = w + h;
        let mut q1 = w * z - 1.0;
        let mut k = 1i32;
        while q1 < 1.0e9 {
            k += 1;
            z += h;
            let tmp = z * q1 - q0;
            q0 = q1;
            q1 = tmp;
        }
        let m = n + n;
        let mut tt = 0.0f32;
        let mut i = 2 * (n + k);
        while i >= m {
            tt = 1.0 / (i as f32 / x - tt);
            i -= 2;
        }
        let mut a = tt;
        let mut b = 1.0f32;

        let v = 2.0f32 / x;
        let tmp = (n as f32) * super::ln((v * n as f32).abs());
        // `ln(FLT_MAX)`, as fdlibm spells it; see the double-precision port
        // for why the digits are not trimmed.
        #[allow(clippy::excessive_precision)]
        const LN_MAX: f32 = 8.8721679688e+01;
        if tmp < LN_MAX {
            let mut di = ((n - 1) + (n - 1)) as f32;
            for _ in (1..n).rev() {
                let temp = b;
                b *= di;
                b = b / x - a;
                a = temp;
                di -= 2.0;
            }
        } else {
            let mut di = ((n - 1) + (n - 1)) as f32;
            for _ in (1..n).rev() {
                let temp = b;
                b *= di;
                b = b / x - a;
                a = temp;
                di -= 2.0;
                if b > 1e10 {
                    a /= b;
                    tt /= b;
                    b = 1.0;
                }
            }
        }
        let z = j0(x);
        let w = j1(x);
        if z.abs() >= w.abs() {
            tt * z / b
        } else {
            tt * w / a
        }
    };

    if sgn { -b } else { b }
}

/// The Bessel function of the second kind, order `n`. Bit-identical to glibc's
/// `ynf`.
pub fn yn(n: i32, x: f32) -> f32 {
    let hx = x.to_bits();
    let i0 = ix(x);
    if i0 >= 0x7f80_0000 {
        if hx == 0xff80_0000 {
            return f32::NAN;
        }
        return 1.0 / (x + x * x);
    }
    let (n, sign) = if n < 0 {
        let m = -n;
        (m, 1 - ((m & 1) << 1))
    } else {
        (n, 1)
    };
    if n == 0 {
        return y0(x);
    }
    if i0 == 0 {
        return if sign == 1 { -1.0 / 0.0 } else { 1.0 / 0.0 };
    }
    if hx >> 31 != 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sign as f32 * y1(x);
    }
    if i0 == 0x7f80_0000 {
        return 0.0;
    }

    let mut a = y0(x);
    let mut b = y1(x);
    let mut ib = b.to_bits();
    let mut i = 1;
    while i < n && ib != 0xff80_0000 {
        let temp = b;
        // Note the operand order: `ynf` multiplies the ratio by `b`, while
        // `jnf` multiplies `b` by the ratio. In `double` the two round the
        // same way, but this is transcribed rather than tidied.
        b = (((i + i) as f64 / x as f64) * b as f64 - a as f64) as f32;
        ib = b.to_bits();
        a = temp;
        i += 1;
    }
    if sign > 0 { b } else { -b }
}
