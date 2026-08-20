//! `asin`, `acos`, `atan` and `atan2`: a port of glibc's `__ieee754_asin`,
//! `__ieee754_acos`, `__atan` and `__ieee754_atan2`
//! (`sysdeps/ieee754/dbl-64/e_asin.c`, `s_atan.c`, `e_atan2.c` — the IBM
//! Accurate Mathematical Library), not a delegation.
//!
//! FMA placement was read out of a disassembly of the compiled `_fma`
//! entry points (`__ieee754_asin_fma`, `__ieee754_acos_fma`, `__atan_fma`,
//! `__ieee754_atan2_fma`; glibc 2.43, x86-64), the same discipline
//! [`crate::reference::double::trig`] used. Two patterns recur throughout,
//! confirmed independently in every band of every function below:
//!
//! * Every band's final "polynomial times something, plus a leading term"
//!   step is one fused multiply-add, even where the C source spells it as a
//!   separate multiply and add (`t *= z; y = base + t;` compiles to
//!   `y = z.mul_add(poly, base)` type code, not two roundings).
//! * `atan`'s reciprocal bands use `dla.h`'s `EMULV`/`ESUB` macros, which
//!   `#ifdef __FP_FAST_FMA` (true here) collapse `EMULV` to the ordinary
//!   2Product (`z = x*y; zz = x.mul_add(y, -z)`) — exactly this crate's
//!   `crate::kernels::double::dd::a_mul` shape, reused below rather than
//!   reinvented. `ESUB` has no FMA-specialised form in `dla.h`, but the one
//!   call site that needs it (`ESUB(HPI, w, ..)` in the `D <= u < E` band)
//!   is only ever reached with `w = 1/u <= 1/16 < HPI`, and the compiled code
//!   confirms the compiler proved that range and folded the macro's
//!   magnitude-comparison branch away at compile time — so the port does the
//!   same rather than reproducing a branch that never executes.

use crate::kernels::double::dd::{a_mul, two_sum};
use crate::tables::double::asincos_data as ac;
use crate::tables::double::atan_data as at;
use crate::tables::double::atan2_data as at2;

/// `2^52`: added to round a small positive double to the nearest integer in
/// round-to-nearest mode, then subtracted back off. Exact throughout: the
/// multiply that feeds it (`256.0 * w`, a power of two) never rounds either,
/// so this trick is insensitive to fusing that multiply into the add.
const TWO52: f64 = 4503599627370496.0;

// ---------------------------------------------------------------------------
// atan
// ---------------------------------------------------------------------------

/// `cij` row `i`'s 7 fields, unpacked from the flat table.
#[inline(always)]
fn cij_row(i: i64) -> (f64, f64, f64, f64, f64, f64, f64) {
    let row = (i as usize) * 7;
    (
        f64::from_bits(at::CIJ[row]),
        f64::from_bits(at::CIJ[row + 1]),
        f64::from_bits(at::CIJ[row + 2]),
        f64::from_bits(at::CIJ[row + 3]),
        f64::from_bits(at::CIJ[row + 4]),
        f64::from_bits(at::CIJ[row + 5]),
        f64::from_bits(at::CIJ[row + 6]),
    )
}

/// The round-to-nearest-256th-of-a-unit table index, `((2^52 + 256w) - 2^52) - 16`.
#[inline(always)]
fn table_index(w: f64) -> i64 {
    ((TWO52 + 256.0 * w) - TWO52) as i64 - 16
}

/// `atan(|x|)` for `A <= u < B`: direct Taylor series in `v = x*x`.
#[inline(always)]
fn atan_taylor(x: f64, v: f64) -> f64 {
    let yy = v.mul_add(at::D13, at::D11);
    let yy = v.mul_add(yy, at::D9);
    let yy = v.mul_add(yy, at::D7);
    let yy = v.mul_add(yy, at::D5);
    let yy = v.mul_add(yy, at::D3);
    (x * v).mul_add(yy, x)
}

/// `atan(u)` for `B <= u < C`: direct table band, `z = u - x0`.
#[inline(always)]
fn atan_table(u: f64) -> f64 {
    let i = table_index(u);
    let (x0, t1, c2, c3, c4, c5, c6) = cij_row(i);
    let z = u - x0;
    let yy = z.mul_add(c6, c5);
    let yy = z.mul_add(yy, c4);
    let yy = z.mul_add(yy, c3);
    let yy = z.mul_add(yy, c2);
    z.mul_add(yy, t1)
}

/// `atan(u)` for `C <= u < D`: reciprocal fold plus table, `w = 1/u`.
#[inline(always)]
fn atan_recip_table(u: f64) -> f64 {
    let w = 1.0 / u;
    let (t1p, t2p) = a_mul(w, u);
    let s = (1.0 - t1p) - t2p;
    let i = table_index(w);
    let (x0, t1, c2, c3, c4, c5, c6) = cij_row(i);
    let z = s.mul_add(w, w - x0);
    let yy = z.mul_add(c6, c5);
    let yy = z.mul_add(yy, c4);
    let yy = z.mul_add(yy, c3);
    let yy = z.mul_add(yy, c2);
    let yy = (-z).mul_add(yy, at::HPI1);
    let t1_final = at::HPI - t1;
    t1_final + yy
}

/// `atan(u)` for `D <= u < E`: reciprocal fold plus Taylor series.
#[inline(always)]
fn atan_recip_taylor(u: f64) -> f64 {
    let w = 1.0 / u;
    let v = w * w;
    let (t1p, t2p) = a_mul(w, u);
    let yy = v.mul_add(at::D13, at::D11);
    let yy = v.mul_add(yy, at::D9);
    let yy = v.mul_add(yy, at::D7);
    let yy = v.mul_add(yy, at::D5);
    let yy = v.mul_add(yy, at::D3);
    // `ESUB(HPI, w, t3, cor)`: this band only ever has `w <= 1/16 < HPI`, so
    // the magnitude-ordered branch always takes the `fabs(x) > fabs(y)` arm.
    let t3 = at::HPI - w;
    let cor = (at::HPI - t3) - w;
    let s = (1.0 - t1p) - t2p;
    let hpi1_cor = at::HPI1 + cor;
    // Both fused (`vfnmadd132sd` twice): `s*w` and `wv*yy` are each never
    // separately rounded before being subtracted.
    let inner = (-s).mul_add(w, hpi1_cor);
    let wv = w * v;
    let combined = (-wv).mul_add(yy, inner);
    t3 + combined
}

/// `atan(x)`, the platform's — a genuine port, not a delegation.
///
/// Needs no `math_check_force_underflow`-equivalent: that glibc call only
/// raises the underflow exception, which this crate does not model (see
/// [`crate::reference::double`]'s module doc on what a reference reproduces).
pub fn atan(x: f64) -> f64 {
    if x.is_nan() {
        return x + x;
    }
    let u = x.abs();
    if u < at::A {
        x
    } else if u < at::B {
        atan_taylor(x, x * x)
    } else if u < at::C {
        atan_table(u).copysign(x)
    } else if u < at::D {
        atan_recip_table(u).copysign(x)
    } else if u < at::E {
        atan_recip_taylor(u).copysign(x)
    } else if x > 0.0 {
        at::HPI
    } else {
        at::MHPI
    }
}

// ---------------------------------------------------------------------------
// atan2
// ---------------------------------------------------------------------------
//
// `__ieee754_atan2` does not call into `atan`'s own code (confirmed: its
// disassembly has no `call` into `__atan_fma`) — it reimplements the same
// two-shapes-per-band structure (direct Taylor below `1/16`, `cij`-table
// above) inline, once per quadrant, each with its own additive constant
// (`0`, `HPI`, `OPI`) and an extra compensated term (`du`, the division's
// own rounding residual) that `atan`'s own bands do not carry, because here
// the argument is a ratio `y/x` rather than `x` itself.

/// `zz = du + u*v*poly(d3..d13)` (case i's Taylor band) or `zz = u*v*poly`
/// (cases ii/iii/iv's Taylor bands, no `du` folded in) — the shared
/// degree-13 Horner chain both need.
#[inline(always)]
fn atan2_taylor_poly(v: f64) -> f64 {
    let yy = v.mul_add(at::D13, at::D11);
    let yy = v.mul_add(yy, at::D9);
    let yy = v.mul_add(yy, at::D7);
    let yy = v.mul_add(yy, at::D5);
    v.mul_add(yy, at::D3)
}

/// Case (i): `x > 0`, `ay < ax`, `u < 1/16`. `zz = du + (u*v)*poly; z = u+zz`.
#[inline(always)]
fn atan2_i_taylor(u: f64, du: f64) -> f64 {
    let v = u * u;
    let poly = atan2_taylor_poly(v);
    let zz = (u * v).mul_add(poly, du);
    u + zz
}

/// Case (i): `x > 0`, `ay < ax`, `u >= 1/16`. `cij` row is `[x0, t1, t2, c3..c6]`
/// here (index `2` is its own leading coefficient, not folded into the poly),
/// unlike `atan`'s own `[x0, t1, c2..c6]` layout.
#[inline(always)]
fn atan2_i_table(u: f64, du: f64) -> f64 {
    let i = table_index(u);
    let row = (i as usize) * 7;
    let x0 = f64::from_bits(at::CIJ[row]);
    let t1 = f64::from_bits(at::CIJ[row + 1]);
    let t2 = f64::from_bits(at::CIJ[row + 2]);
    let c3 = f64::from_bits(at::CIJ[row + 3]);
    let c4 = f64::from_bits(at::CIJ[row + 4]);
    let c5 = f64::from_bits(at::CIJ[row + 5]);
    let c6 = f64::from_bits(at::CIJ[row + 6]);
    let t3 = u - x0;
    let (v, dv) = two_sum::<f64>(t3, du);
    let poly = v.mul_add(c6, c5);
    let poly = v.mul_add(poly, c4);
    let poly = v.mul_add(poly, c3);
    let inner = (v * v).mul_add(poly, dv * t2);
    let zz = v.mul_add(t2, inner);
    t1 + zz
}

/// Cases (ii)/(iii)/(iv)'s table bands share one shape — `atan`'s own
/// `atan_recip_table` shape exactly, just with a caller-supplied base
/// (`HPI`/`OPI`), sign, and a `du`-compensated `v` in place of a plain
/// subtraction.
#[inline(always)]
fn atan2_table_shared(u: f64, du: f64, base: f64, base1: f64, add: bool) -> f64 {
    let i = table_index(u);
    let (x0, t1c, c2, c3, c4, c5, c6) = cij_row(i);
    let v = (u - x0) + du;
    let poly = v.mul_add(c6, c5);
    let poly = v.mul_add(poly, c4);
    let poly = v.mul_add(poly, c3);
    let poly = v.mul_add(poly, c2);
    let zz = if add {
        v.mul_add(poly, base1)
    } else {
        (-v).mul_add(poly, base1)
    };
    let t1 = if add { base + t1c } else { base - t1c };
    t1 + zz
}

/// Case (ii): `x > 0`, `ay >= ax`, `u < 1/16`. `pi/2 - atan(u)`.
#[inline(always)]
fn atan2_ii_taylor(u: f64, du: f64) -> f64 {
    let v = u * u;
    let poly = atan2_taylor_poly(v);
    let zz = (u * v) * poly;
    let t2 = at::HPI - u;
    let cor = (at::HPI - t2) - u; // ESUB, branch folded: u < 1/16 < HPI always
    let t3 = ((at::HPI1 + cor) - du) - zz;
    t2 + t3
}

/// Case (iii): `x < 0`, `ax < ay`, `u < 1/16`. `pi/2 + atan(u)`.
#[inline(always)]
fn atan2_iii_taylor(u: f64, du: f64) -> f64 {
    let v = u * u;
    let poly = atan2_taylor_poly(v);
    let zz = (u * v) * poly;
    let t2 = at::HPI + u;
    let cor = (at::HPI - t2) + u; // EADD, branch folded: u < 1/16 < HPI always
    let t3 = ((at::HPI1 + cor) + du) + zz;
    t2 + t3
}

/// Case (iv): `x < 0`, `ax >= ay`, `u < 1/16`. `pi - atan(u)`.
#[inline(always)]
fn atan2_iv_taylor(u: f64, du: f64) -> f64 {
    let v = u * u;
    let poly = atan2_taylor_poly(v);
    let zz = (u * v) * poly;
    let t2 = at2::OPI - u;
    let cor = (at2::OPI - t2) - u; // ESUB, branch folded: u <= 1 < OPI always
    let t3 = ((at2::OPI1 + cor) - du) - zz;
    t2 + t3
}

/// `atan2(y, x)`, the platform's — a genuine port, not a delegation.
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() {
        return x + y;
    }
    if y.is_nan() {
        return y + y;
    }

    if y == 0.0 {
        return if y.is_sign_positive() {
            if x.is_sign_positive() { 0.0 } else { at2::OPI }
        } else if x.is_sign_positive() {
            -0.0
        } else {
            at2::MOPI
        };
    }

    if x == 0.0 {
        return if y.is_sign_positive() {
            at::HPI
        } else {
            at::MHPI
        };
    }

    if x.is_infinite() {
        return if x > 0.0 {
            if y.is_infinite() {
                if y > 0.0 { at2::QPI } else { at2::MQPI }
            } else if y.is_sign_positive() {
                0.0
            } else {
                -0.0
            }
        } else if y.is_infinite() {
            if y > 0.0 { at2::TQPI } else { at2::MTQPI }
        } else if y.is_sign_positive() {
            at2::OPI
        } else {
            at2::MOPI
        };
    }

    if y.is_infinite() {
        return if y > 0.0 { at::HPI } else { at::MHPI };
    }

    let mut ax = x.abs();
    let mut ay = y.abs();
    const EP: i64 = 59768832;
    let de =
        (((y.to_bits() >> 32) & 0x7ff00000) as i64) - (((x.to_bits() >> 32) & 0x7ff00000) as i64);
    if de >= EP {
        return if y > 0.0 { at::HPI } else { at::MHPI };
    } else if de <= -EP {
        return if x > 0.0 {
            (ay / ax).copysign(y)
        } else if y > 0.0 {
            at2::OPI
        } else {
            at2::MOPI
        };
    }

    if ax < at2::TWOM500 || ay < at2::TWOM500 {
        ax *= at2::TWO500;
        ay *= at2::TWO500;
    }
    if ax > at2::TWO500 || ay > at2::TWO500 {
        ax *= at2::TWOM500;
        ay *= at2::TWOM500;
    }

    let (u, du) = if ay < ax {
        let u = ay / ax;
        let (v, vv) = a_mul::<f64>(ax, u);
        let du = ((ay - v) - vv) / ax;
        (u, du)
    } else {
        let u = ax / ay;
        let (v, vv) = a_mul::<f64>(ay, u);
        let du = ((ax - v) - vv) / ay;
        (u, du)
    };

    let z = if x > 0.0 {
        if ay < ax {
            if u < at::B {
                atan2_i_taylor(u, du)
            } else {
                atan2_i_table(u, du)
            }
        } else if u < at::B {
            atan2_ii_taylor(u, du)
        } else {
            atan2_table_shared(u, du, at::HPI, at::HPI1, false)
        }
    } else if ax < ay {
        if u < at::B {
            atan2_iii_taylor(u, du)
        } else {
            atan2_table_shared(u, du, at::HPI, at::HPI1, true)
        }
    } else if u < at::B {
        atan2_iv_taylor(u, du)
    } else {
        atan2_table_shared(u, du, at2::OPI, at2::OPI1, false)
    };
    z.copysign(y)
}

// ---------------------------------------------------------------------------
// asin / acos
// ---------------------------------------------------------------------------
//
// Both delegate to `__ieee754_asin_fma`/`__ieee754_acos_fma`, which share
// the same six-band `asncs.x` table lookup (bands differ only in the linear
// combination applied to the table poly's result) plus a shared near-1 band
// built on `inroot`/`powtwo`/`rt0..rt3` (`root.tbl`/`powtwo.tbl`). Every
// table band was confirmed against the disassembly directly; two fusions
// recur beyond what the C source shows: the inner Horner chain's last step
// folds the `xx*xx` scale in (`p = xx².mul_add(poly, outer)`), and `t=..;
// t+=p;` collapses into one more FMA (`t_plus_p = xx.mul_add(t1, p)`) — the
// final `+asncs[..]` stays a separate, unfused add in every band checked.

/// `1/sqrt(z)`'s seed via a bit-pattern lookup, refined by a degree-3
/// Newton-style polynomial in `r = 1 - t²z`.
///
/// `e_asin.c`'s `t=inroot[..]*powtwo[..]; r=1-t*t*z;
/// t=t*(rt0+r*(rt1+r*(rt2+r*rt3)));` — `r`'s own `t*t*z` is two roundings,
/// not three: the disassembly fuses the final `-(t*t)*z` into the `1 -`,
/// but leaves `t*t` itself a separate, earlier rounding.
#[inline(always)]
fn near_one_root(z: f64) -> f64 {
    let k = z.to_bits() >> 32;
    let seed =
        ac::INROOT[((k & 0x001fffff) >> 14) as usize] * ac::POWTWO[(511 - (k >> 21)) as usize];
    let tt = seed * seed;
    let r = (-tt).mul_add(z, 1.0);
    let poly = r.mul_add(ac::RT3, ac::RT2);
    let poly = r.mul_add(poly, ac::RT1);
    let poly = r.mul_add(poly, ac::RT0);
    seed * poly
}

/// One `asncs.x` table band's polynomial, shared by every band from `0.125`
/// up to `0.96875`: `row = [x0, t1, c_first..c_last, outer, final]`.
///
/// `p = xx² * (c_first + xx*(.. + xx*c_last)) + outer; t = t1*xx; t += p;`
/// `res = final + t` — the last two lines each fuse into one FMA (see the
/// module doc), the very last add does not.
#[inline(always)]
fn asncs_band(xx: f64, row: &[f64]) -> f64 {
    let degree = row.len() - 4;
    let mut val = row[2 + degree - 1];
    for &c in row[2..2 + degree - 1].iter().rev() {
        val = xx.mul_add(val, c);
    }
    let p = (xx * xx).mul_add(val, row[2 + degree]);
    let t_plus_p = xx.mul_add(row[1], p);
    row[3 + degree] + t_plus_p
}

/// `asncs.x[n..n+len]`, unpacked from the `u64`-bit-pattern table.
#[inline(always)]
fn asncs_row(n: usize, len: usize) -> [f64; 13] {
    let mut out = [0.0; 13];
    for (dst, &bits) in out.iter_mut().zip(&ac::ASNCS[n..n + len]) {
        *dst = f64::from_bits(bits);
    }
    out
}

/// The six band-index formulas from `e_asin.c`, keyed by `k = high32(|x|)`.
/// Returns `(n, degree)` — `degree` is the inner polynomial's term count.
#[inline(always)]
fn asncs_band_index(k: u32) -> (usize, usize) {
    if k < 0x3fe00000 {
        let n = if k < 0x3fd00000 {
            11 * ((k & 0x000fffff) >> 15)
        } else {
            11 * ((k & 0x000fffff) >> 14) + 352
        };
        (n as usize, 5)
    } else if k < 0x3fe80000 {
        (1056 + (((k & 0x000fe000) >> 11) * 3) as usize, 6)
    } else if k < 0x3fed8000 {
        (992 + (((k & 0x000fe000) >> 13) * 13) as usize, 7)
    } else if k < 0x3fee8000 {
        (884 + (((k & 0x000fe000) >> 13) * 14) as usize, 8)
    } else {
        (768 + (((k & 0x000fe000) >> 13) * 15) as usize, 9)
    }
}

/// `asin(|t|)` for `t*t = s`, `|t| < 0.125`: `res = x + (x2*x)*poly(x2)`,
/// the same fused shape `atan`'s own direct-Taylor band uses.
#[inline(always)]
fn asin_taylor(x: f64, x2: f64) -> f64 {
    let t = x2.mul_add(ac::F6, ac::F5);
    let t = x2.mul_add(t, ac::F4);
    let t = x2.mul_add(t, ac::F3);
    let t = x2.mul_add(t, ac::F2);
    let t = x2.mul_add(t, ac::F1);
    (x2 * x).mul_add(t, x)
}

/// `asin(x)`, the platform's — a genuine port, not a delegation.
pub fn asin(x: f64) -> f64 {
    // A NaN's `k` never matches a valid band below (its exponent field is
    // `0x7ff`), so an explicit early return is not just documentation, it is
    // load-bearing: it keeps the payload/sign quiet-NaN propagation (`x+x`)
    // separate from the *canonical* NaN the true `|x| > 1` domain error
    // below returns. Matches `atan`'s identical wrapper-boundary reasoning.
    if x.is_nan() {
        return x + x;
    }
    let k = (x.abs().to_bits() >> 32) as u32;
    if k < 0x3e500000 {
        return x;
    }
    if k < 0x3fc00000 {
        return asin_taylor(x, x * x);
    }
    if k < 0x3fef0000 {
        let (n, degree) = asncs_band_index(k);
        let row = asncs_row(n, degree + 4);
        let xx = x.abs() - row[0];
        let res = asncs_band(xx, &row[..degree + 4]);
        return if x > 0.0 { res } else { -res };
    }
    if k < 0x3ff00000 {
        let a = x.abs();
        let z = 0.5 * (1.0 - a);
        let t = near_one_root(z);
        let c = t * z;
        let half_t = t * 0.5;
        let inner = half_t.mul_add(-c, 1.5);
        let y = (c + ac::T24) - ac::T24;
        let t_plus_y = inner.mul_add(c, y);
        let z_minus_y2 = y.mul_add(-y, z);
        let cc = z_minus_y2 / t_plus_y;
        let p = z.mul_add(ac::F6, ac::F5);
        let p = z.mul_add(p, ac::F4);
        let p = z.mul_add(p, ac::F3);
        let p = z.mul_add(p, ac::F2);
        let p = z.mul_add(p, ac::F1);
        let p = p * z;
        let hp1_minus_2cc = cc.mul_add(-2.0, ac::HP1);
        let res1 = y.mul_add(-2.0, ac::HP0);
        let y_plus_cc_x2 = (y + cc) + (y + cc);
        let cor = y_plus_cc_x2.mul_add(-p, hp1_minus_2cc);
        let res = res1 + cor;
        return if x > 0.0 { res } else { -res };
    }
    if k == 0x3ff00000 && x.to_bits() as u32 == 0 {
        return ac::HP0.copysign(x);
    }
    // `__ieee754_asin`'s own `(x-x)/(x-x)` here is dead code on this
    // platform: the exported `asin` wrapper checks the domain itself
    // (confirmed by disassembly) and returns this canonical NaN through
    // its own error path before ever calling `__ieee754_asin_fma`.
    f64::NAN
}

/// `acos(x)`, the platform's — a genuine port, not a delegation.
pub fn acos(x: f64) -> f64 {
    // See `asin`'s identical early return: load-bearing, not just style.
    if x.is_nan() {
        return x + x;
    }
    let k = (x.abs().to_bits() >> 32) as u32;
    if k < 0x3c880000 {
        return ac::HP0;
    }
    if k < 0x3fc00000 {
        let x2 = x * x;
        let t = x2.mul_add(ac::F6, ac::F5);
        let t = x2.mul_add(t, ac::F4);
        let t = x2.mul_add(t, ac::F3);
        let t = x2.mul_add(t, ac::F2);
        let t = x2.mul_add(t, ac::F1);
        let r = ac::HP0 - x;
        let inner = ((ac::HP0 - r) - x) + ac::HP1;
        // Fused: `cor = inner - (x2*x)*t`, `(x2*x)*t` never separately rounded.
        let cor = (x2 * x).mul_add(-t, inner);
        return r + cor;
    }
    if k < 0x3fef0000 {
        let (n, degree) = asncs_band_index(k);
        let row = asncs_row(n, degree + 4);
        let xx = x.abs() - row[0];
        let t = acos_table_t(xx, &row[..degree + 4]);
        let y = if x > 0.0 {
            ac::HP0 - row[3 + degree]
        } else {
            ac::HP0 + row[3 + degree]
        };
        let t = if x > 0.0 { ac::HP1 - t } else { ac::HP1 + t };
        return y + t;
    }
    if k < 0x3ff00000 {
        let a = x.abs();
        let z = 0.5 * (1.0 - a);
        let t = near_one_root(z);
        let c = t * z;
        let half_t = t * 0.5;
        let inner = half_t.mul_add(-c, 1.5);
        let p = z.mul_add(ac::F6, ac::F5);
        let p = z.mul_add(p, ac::F4);
        let p = z.mul_add(p, ac::F3);
        let p = z.mul_add(p, ac::F2);
        let p = z.mul_add(p, ac::F1);
        let p = p * z;
        if x < 0.0 {
            let y = (ac::T27.mul_add(c, c)) - ac::T27 * c;
            let t_plus_y = inner.mul_add(c, y);
            let z_minus_y2 = y.mul_add(-y, z);
            let cc = z_minus_y2 / t_plus_y;
            let cor = (ac::HP1 - cc) - (y + cc) * p;
            let res1 = ac::HP0 - y;
            let res = res1 + cor;
            return res + res;
        } else {
            // `y = (t27*c+c)-t27*c` in `e_asin.c` — the *same* Dekker split
            // the negative arm uses, not `asin`'s `t24` shape. The first
            // product is one fused FMA (`dla.h`'s `CN` split at `2^27`), the
            // subtraction is separate. The initial port transcribed `asin`'s
            // `(c + t24) - t24` here by mistake; the C source never uses
            // `t24` in `acos` at all. This was the 8-in-3M 1-ulp gap in the
            // near-1 positive-`x` arm (see `ROADMAP.md`'s A4 entry).
            let y = ac::T27.mul_add(c, c) - ac::T27 * c;
            let t_plus_y = inner.mul_add(c, y);
            let z_minus_y2 = y.mul_add(-y, z);
            let cc = z_minus_y2 / t_plus_y;
            let cor = cc + p * (y + cc);
            let res = y + cor;
            return res + res;
        }
    }
    if k == 0x3ff00000 && x.to_bits() as u32 == 0 {
        return if x > 0.0 { 0.0 } else { 2.0 * ac::HP0 };
    }
    // See `asin`'s equivalent comment above: dead code on this platform.
    f64::NAN
}

/// `acos`'s table-band polynomial, without the outer `HP0 +- ..` combine
/// (`asin`'s table bands are `res = final + (t + p)`; `acos`'s callers apply
/// their own `hp0`/`hp1` combination on top, so this stops one step short —
/// see `e_asin.c`'s `acos` bands, which never add `row[3+degree]` at all).
#[inline(always)]
fn acos_table_t(xx: f64, row: &[f64]) -> f64 {
    let degree = row.len() - 4;
    let mut val = row[2 + degree - 1];
    for &c in row[2..2 + degree - 1].iter().rev() {
        val = xx.mul_add(val, c);
    }
    let p = (xx * xx).mul_add(val, row[2 + degree]);
    xx.mul_add(row[1], p)
}
