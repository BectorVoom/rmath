//! The Bessel functions of the first and second kind: `j0`, `j1`, `y0`, `y1`.
//!
//! Ports of what glibc actually runs, which for this family is still Sun's
//! fdlibm — verified by finding fdlibm's own constants in the installed
//! `libm.so`, where `erf` and `lgamma` no longer have theirs. The schedule is
//! plain `double` arithmetic with no fused multiply-adds, which is what the
//! compiled library contains: `mulsd`, `addsd`, `divsd` throughout, and no
//! `vfmadd` anywhere in `__j0_finite`.
//!
//! # The two regimes
//!
//! Below `|x| = 2` each function is a rational approximation in `x^2`, with
//! `y0` and `y1` carrying an explicit `log` term because they are singular at
//! the origin.
//!
//! At or above 2 all four use the asymptotic form
//! `j0(x) = sqrt(2/(pi x)) (P(x) cos(x0) - Q(x) sin(x0))` with `x0 = x - pi/4`,
//! and the interesting part is how `cos(x0)` and `sin(x0)` are obtained:
//! writing them as `(cos x +- sin x)/sqrt 2` would lose every significant
//! digit whenever the two nearly cancel, so fdlibm computes the *larger* of
//! the two that way and recovers the smaller from
//! `sin x +- cos x = -cos(2x) / (sin x -+ cos x)`, which has no cancellation
//! at all. That is why these routines call `cos(x + x)` as well as `sincos`.
//!
//! # What the trigonometry costs
//!
//! `sin`, `cos` and `log` here are the platform's, for the reason set out in
//! [`crate::reference`]: glibc computes them with the IBM Accurate Portable
//! Math routines, whose schedule is not reproduced in this crate. So the
//! Bessel functions inherit that — bit-exact by delegation for the
//! trigonometric part, ported for everything else.

use super::ln;
use crate::tables::double::bessel as t;

/// `1e300`, fdlibm's `huge`. Only ever used to raise the inexact flag, which
/// nothing here observes, but it is what decides the tiny-argument branches.
const HUGE: f64 = 1e300;

/// The high word of `|x|`, as fdlibm's `GET_HIGH_WORD` plus a mask.
#[inline(always)]
fn hi(x: f64) -> u32 {
    ((x.to_bits() >> 32) & 0x7fff_ffff) as u32
}

/// Which of the four rational fits covers `|x|`, as an index into the
/// `P0R`-style tables. fdlibm's `if` ladder, spelled as a lookup.
#[inline(always)]
fn interval(ix: u32) -> usize {
    if ix >= 0x4020_0000 {
        0 // [8, inf)
    } else if ix >= 0x4012_2E8B {
        1 // [4.5454, 8)
    } else if ix >= 0x4006_DB6D {
        2 // [2.8571, 4.547)
    } else {
        3 // [2, 2.8570)
    }
}

/// `(sin x - cos x, sin x + cos x)`, each computed the way that does not
/// cancel.
///
/// The shared preamble of all four asymptotic branches. `ss` is
/// `sqrt(2) sin(x - pi/4)` and `cc` is `sqrt(2) cos(x - pi/4)`; whichever of
/// the two is small is recovered from `-cos(2x)` divided by the other, since
/// `(sin x - cos x)(sin x + cos x) = -cos(2x)` exactly.
#[inline(always)]
fn ss_cc(x: f64, ix: u32) -> (f64, f64) {
    let s = x.sin();
    let c = x.cos();
    let mut ss = s - c;
    let mut cc = s + c;
    if ix < 0x7fe0_0000 {
        // Guarded so that `x + x` cannot overflow to infinity.
        let z = -(x + x).cos();
        if s * c < 0.0 {
            cc = z / ss;
        } else {
            ss = z / cc;
        }
    }
    (ss, cc)
}

/// The Bessel function of the first kind, order 0. Bit-identical to glibc's
/// `j0`.
pub fn j0(x: f64) -> f64 {
    let ix = hi(x);
    if ix >= 0x7ff0_0000 {
        return 1.0 / (x * x);
    }
    let x = x.abs();
    if ix >= 0x4000_0000 {
        let (ss, cc) = ss_cc(x, ix);
        return if ix > 0x4800_0000 {
            // Beyond 2^129 the rational fits are indistinguishable from 1 and
            // 0, so they are dropped entirely.
            (t::INVSQRTPI * cc) / x.sqrt()
        } else {
            t::INVSQRTPI * (pzero(x) * cc - qzero(x) * ss) / x.sqrt()
        };
    }
    if ix < 0x3f20_0000 {
        // |x| < 2^-13.
        if HUGE + x > 1.0 {
            if ix < 0x3e40_0000 {
                return 1.0; // |x| < 2^-27
            }
            return 1.0 - 0.25 * x * x;
        }
    }
    let z = x * x;
    let r = j0_num(z);
    let s = j0_den(z);
    if ix < 0x3ff0_0000 {
        1.0 + z * (-0.25 + (r / s))
    } else {
        let u = 0.5 * x;
        (1.0 + u) * (1.0 - u) + z * (r / s)
    }
}

/// `j0`'s numerator on `[0, 2]`, in fdlibm's exact association.
#[inline(always)]
fn j0_num(z: f64) -> f64 {
    let r = &t::J0_R;
    let r1 = z * r[2];
    let z2 = z * z;
    let r2 = r[3] + z * r[4];
    let z4 = z2 * z2;
    r1 + z2 * r2 + z4 * r[5]
}

/// `j0`'s denominator on `[0, 2]`.
#[inline(always)]
fn j0_den(z: f64) -> f64 {
    let s = &t::J0_S;
    let s1 = 1.0 + z * s[1];
    let z2 = z * z;
    let s2 = s[2] + z * s[3];
    let z4 = z2 * z2;
    s1 + z2 * s2 + z4 * s[4]
}

/// The Bessel function of the second kind, order 0. Bit-identical to glibc's
/// `y0`.
pub fn y0(x: f64) -> f64 {
    let ix = hi(x);
    let lx = x.to_bits() as u32;
    if ix >= 0x7ff0_0000 {
        return 1.0 / (x + x * x);
    }
    if (ix | lx) == 0 {
        return -1.0 / 0.0; // -inf, and the divide-by-zero flag
    }
    if x.to_bits() >> 63 != 0 {
        return 0.0 / (0.0 * x); // NaN for a negative argument
    }
    if ix >= 0x4000_0000 {
        let (ss, cc) = ss_cc(x, ix);
        return if ix > 0x4800_0000 {
            (t::INVSQRTPI * ss) / x.sqrt()
        } else {
            t::INVSQRTPI * (pzero(x) * ss + qzero(x) * cc) / x.sqrt()
        };
    }
    if ix <= 0x3e40_0000 {
        // x < 2^-27: U/V is U[0] and j0(x) is 1.
        return t::Y0_U[0] + t::TPI * ln(x);
    }
    let z = x * x;
    let u = &t::Y0_U;
    let v = &t::Y0_V;
    let u1 = u[0] + z * u[1];
    let z2 = z * z;
    let u2 = u[2] + z * u[3];
    let z4 = z2 * z2;
    let u3 = u[4] + z * u[5];
    let z6 = z4 * z2;
    let un = u1 + z2 * u2 + z4 * u3 + z6 * u[6];
    let v1 = 1.0 + z * v[0];
    let v2 = v[1] + z * v[2];
    let vd = v1 + z2 * v2 + z4 * v[3];
    un / vd + t::TPI * (j0(x) * ln(x))
}

/// `pzero`: the amplitude factor of the order-0 asymptotic form.
fn pzero(x: f64) -> f64 {
    let ix = hi(x);
    if ix >= 0x41b0_0000 {
        return 1.0;
    }
    let i = interval(ix);
    rational_p(x, &t::P0R[i], &t::P0S[i])
}

/// `qzero`: the phase factor of the order-0 asymptotic form.
fn qzero(x: f64) -> f64 {
    let ix = hi(x);
    if ix >= 0x41b0_0000 {
        return -0.125 / x;
    }
    let i = interval(ix);
    (-0.125 + rational_q(x, &t::Q0R[i], &t::Q0S[i])) / x
}

/// `pone`: the amplitude factor of the order-1 asymptotic form.
fn pone(x: f64) -> f64 {
    let ix = hi(x);
    if ix >= 0x41b0_0000 {
        return 1.0;
    }
    let i = interval(ix);
    rational_p(x, &t::P1R[i], &t::P1S[i])
}

/// `qone`: the phase factor of the order-1 asymptotic form.
fn qone(x: f64) -> f64 {
    let ix = hi(x);
    if ix >= 0x41b0_0000 {
        return 0.375 / x;
    }
    let i = interval(ix);
    (0.375 + rational_q(x, &t::Q1R[i], &t::Q1S[i])) / x
}

/// `1 + R(s)/S(s)` with `s = 1/x^2`, the shape both `p` factors share.
#[inline(always)]
fn rational_p(x: f64, p: &[f64; 6], q: &[f64; 5]) -> f64 {
    let z = 1.0 / (x * x);
    let r1 = p[0] + z * p[1];
    let z2 = z * z;
    let r2 = p[2] + z * p[3];
    let z4 = z2 * z2;
    let r3 = p[4] + z * p[5];
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = 1.0 + z * q[0];
    let s2 = q[1] + z * q[2];
    let s3 = q[3] + z * q[4];
    let s = s1 + z2 * s2 + z4 * s3;
    1.0 + r / s
}

/// `R(s)/S(s)` with `s = 1/x^2`, the shape both `q` factors share. One term
/// longer in the denominator than [`rational_p`].
#[inline(always)]
fn rational_q(x: f64, p: &[f64; 6], q: &[f64; 6]) -> f64 {
    let z = 1.0 / (x * x);
    let r1 = p[0] + z * p[1];
    let z2 = z * z;
    let r2 = p[2] + z * p[3];
    let z4 = z2 * z2;
    let r3 = p[4] + z * p[5];
    let z6 = z4 * z2;
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = 1.0 + z * q[0];
    let s2 = q[1] + z * q[2];
    let s3 = q[3] + z * q[4];
    let s = s1 + z2 * s2 + z4 * s3 + z6 * q[5];
    r / s
}

/// The Bessel function of the first kind, order 1. Bit-identical to glibc's
/// `j1`.
pub fn j1(x: f64) -> f64 {
    let ix = hi(x);
    if ix >= 0x7ff0_0000 {
        return 1.0 / x;
    }
    let y = x.abs();
    if ix >= 0x4000_0000 {
        // The order-1 combination is the order-0 one with the roles and signs
        // of `ss` and `cc` exchanged, because `x0` is `x - 3pi/4` rather than
        // `x - pi/4`.
        let s = y.sin();
        let c = y.cos();
        let mut ss = -s - c;
        let mut cc = s - c;
        if ix < 0x7fe0_0000 {
            let z = (y + y).cos();
            if s * c > 0.0 {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }
        let z = if ix > 0x4800_0000 {
            (t::INVSQRTPI * cc) / y.sqrt()
        } else {
            t::INVSQRTPI * (pone(y) * cc - qone(y) * ss) / y.sqrt()
        };
        return if x.to_bits() >> 63 != 0 { -z } else { z };
    }
    if ix < 0x3e40_0000 {
        // |x| < 2^-27: j1(x) is x/2, and the multiply is what raises
        // underflow when it should.
        if HUGE + x > 1.0 {
            return 0.5 * x;
        }
    }
    let z = x * x;
    let r = &t::J1_R;
    let s = &t::J1_S;
    let r1 = z * r[0];
    let z2 = z * z;
    let r2 = r[1] + z * r[2];
    let z4 = z2 * z2;
    let rn = (r1 + z2 * r2 + z4 * r[3]) * x;
    let s1 = 1.0 + z * s[1];
    let s2 = s[2] + z * s[3];
    let s3 = s[4] + z * s[5];
    let sd = s1 + z2 * s2 + z4 * s3;
    x * 0.5 + rn / sd
}

/// The Bessel function of the second kind, order 1. Bit-identical to glibc's
/// `y1`.
pub fn y1(x: f64) -> f64 {
    let ix = hi(x);
    let lx = x.to_bits() as u32;
    if ix >= 0x7ff0_0000 {
        return 1.0 / (x + x * x);
    }
    if (ix | lx) == 0 {
        return -1.0 / 0.0;
    }
    if x.to_bits() >> 63 != 0 {
        return 0.0 / (0.0 * x);
    }
    if ix >= 0x4000_0000 {
        let s = x.sin();
        let c = x.cos();
        let mut ss = -s - c;
        let mut cc = s - c;
        if ix < 0x7fe0_0000 {
            let z = (x + x).cos();
            if s * c > 0.0 {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }
        return if ix > 0x4800_0000 {
            (t::INVSQRTPI * ss) / x.sqrt()
        } else {
            t::INVSQRTPI * (pone(x) * ss + qone(x) * cc) / x.sqrt()
        };
    }
    if ix <= 0x3c90_0000 {
        // x < 2^-54, where y1(x) is -2/(pi x) and nothing else survives.
        return -t::TPI / x;
    }
    let z = x * x;
    let u = &t::Y1_U;
    let v = &t::Y1_V;
    let u1 = u[0] + z * u[1];
    let z2 = z * z;
    let u2 = u[2] + z * u[3];
    let z4 = z2 * z2;
    let un = u1 + z2 * u2 + z4 * u[4];
    let v1 = 1.0 + z * v[0];
    let v2 = v[1] + z * v[2];
    let v3 = v[3] + z * v[4];
    let vd = v1 + z2 * v2 + z4 * v3;
    x * (un / vd) + t::TPI * (j1(x) * ln(x) - 1.0 / x)
}

// ---------------------------------------------------------------------------
// jn and yn
// ---------------------------------------------------------------------------

/// The Bessel function of the first kind, order `n`. Bit-identical to glibc's
/// `jn`.
///
/// Three regimes, and which one runs depends on `n` against `x` rather than on
/// `x` alone:
///
/// * `n <= x` — forward recurrence from `j0` and `j1`, which is stable in that
///   direction.
/// * `n > x`, `x` not tiny — *backward* recurrence, because forward recurrence
///   is violently unstable there. The starting ratio comes from a continued
///   fraction whose length is decided at run time by iterating until a
///   convergent exceeds `1e9`, and the whole thing is renormalised at the end
///   against whichever of `j0(x)` and `j1(x)` is further from zero.
/// * `x < 2^-29` — the leading term of the Taylor series, `(x/2)^n / n!`.
pub fn jn(n: i32, x: f64) -> f64 {
    let ix = hi(x);
    let lx = x.to_bits() as u32;
    if (ix | ((lx | lx.wrapping_neg()) >> 31)) > 0x7ff0_0000 {
        return x + x; // NaN
    }
    // J(-n, x) = J(n, -x), so the negative orders fold away immediately.
    let (n, mut x) = if n < 0 { (-n, -x) } else { (n, x) };
    let negative = x.to_bits() >> 63 != 0;
    if n == 0 {
        return j0(x);
    }
    if n == 1 {
        return j1(x);
    }
    // Even `n` is even in `x`; odd `n` is odd.
    let sgn = (n & 1) != 0 && negative;
    x = x.abs();

    let b = if (ix | lx) == 0 || ix >= 0x7ff0_0000 {
        return if sgn { -0.0 } else { 0.0 };
    } else if (n as f64) <= x {
        if ix >= 0x52d0_0000 {
            // x > 2^302, where x dwarfs n^2 and the asymptotic form is exact
            // to the last bit: Jn(x) = cos(x - (2n+1)pi/4) sqrt(2/(pi x)).
            let s = x.sin();
            let c = x.cos();
            let temp = match n & 3 {
                0 => c + s,
                1 => -c + s,
                2 => -c - s,
                _ => c - s,
            };
            t::INVSQRTPI * temp / x.sqrt()
        } else {
            let mut a = j0(x);
            let mut b = j1(x);
            for i in 1..n {
                let temp = b;
                // The division rather than a multiply by `1/x` is fdlibm's,
                // and it is what keeps the recurrence from underflowing.
                b = b * ((i + i) as f64 / x) - a;
                a = temp;
            }
            b
        }
    } else if ix < 0x3e10_0000 {
        // x < 2^-29: J(n,x) = (x/2)^n / n!, and above n = 33 that underflows.
        if n > 33 {
            0.0
        } else {
            let temp = x * 0.5;
            let mut b = temp;
            let mut a = 1.0;
            for i in 2..=n {
                a *= i as f64;
                b *= temp;
            }
            b / a
        }
    } else {
        // Backward recurrence. The continued fraction for J(n,x)/J(n-1,x) is
        // run until a convergent exceeds 1e9, which is fdlibm's stopping rule
        // for double precision.
        let w = (n + n) as f64 / x;
        let h = 2.0 / x;
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
        let mut tt = 0.0f64;
        let mut i = 2 * (n + k);
        while i >= m {
            tt = 1.0 / (i as f64 / x - tt);
            i -= 2;
        }
        let mut a = tt;
        let mut b = 1.0f64;

        // If `n ln(2n/x)` exceeds ln(DBL_MAX) the recurrence overflows on the
        // way down, so the second loop renormalises whenever it gets large.
        // Both loops exist because the test is once, not per iteration.
        let v = 2.0 / x;
        let tmp = (n as f64) * ln((v * n as f64).abs());
        // `ln(DBL_MAX)`, spelled as fdlibm spells it. Rounding it to the
        // shortest round-tripping decimal is what clippy asks for and would
        // move the branch boundary, so the digits stay.
        #[allow(clippy::excessive_precision)]
        const LN_MAX: f64 = 7.09782712893383973096e+02;
        if tmp < LN_MAX {
            let mut di = ((n - 1) + (n - 1)) as f64;
            for _ in (1..n).rev() {
                let temp = b;
                b *= di;
                b = b / x - a;
                a = temp;
                di -= 2.0;
            }
        } else {
            let mut di = ((n - 1) + (n - 1)) as f64;
            for _ in (1..n).rev() {
                let temp = b;
                b *= di;
                b = b / x - a;
                a = temp;
                di -= 2.0;
                if b > 1e100 {
                    a /= b;
                    tt /= b;
                    b = 1.0;
                }
            }
        }
        // Normalise against j0 or j1 — whichever is further from zero, since
        // their zeros never coincide.
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
/// `yn`.
///
/// Forward recurrence from `y0` and `y1` throughout, which for `y` is the
/// *stable* direction — the opposite of `j`. It stops early if the recurrence
/// reaches `-inf`, which it does for moderate `n` at small `x`.
pub fn yn(n: i32, x: f64) -> f64 {
    let ix = hi(x);
    let lx = x.to_bits() as u32;
    if (ix | ((lx | lx.wrapping_neg()) >> 31)) > 0x7ff0_0000 {
        return x + x; // NaN
    }
    // Y(-n, x) = (-1)^n Y(n, x).
    let (n, sign) = if n < 0 {
        let m = -n;
        (m, 1 - ((m & 1) << 1))
    } else {
        (n, 1)
    };
    if n == 0 {
        return y0(x);
    }
    if (ix | lx) == 0 {
        return -(sign as f64) / 0.0;
    }
    if x.to_bits() >> 63 != 0 {
        return 0.0 / (0.0 * x);
    }

    if n == 1 {
        return sign as f64 * y1(x);
    }
    if ix == 0x7ff0_0000 {
        return 0.0;
    }
    {
        let b = if ix >= 0x52d0_0000 {
            let s = x.sin();
            let c = x.cos();
            let temp = match n & 3 {
                0 => s - c,
                1 => -s - c,
                2 => -s + c,
                _ => s + c,
            };
            t::INVSQRTPI * temp / x.sqrt()
        } else {
            let mut a = y0(x);
            let mut b = y1(x);
            let mut high = (b.to_bits() >> 32) as u32;
            let mut i = 1;
            while i < n && high != 0xfff0_0000 {
                let temp = b;
                b = ((i + i) as f64 / x) * b - a;
                high = (b.to_bits() >> 32) as u32;
                a = temp;
                i += 1;
            }
            b
        };
        if sign > 0 { b } else { -b }
    }
}
