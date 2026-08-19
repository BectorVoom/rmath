//! `sin`, `cos` and `sincos`: a port of glibc's `__sin`/`__cos`/`__sincos`
//! (`sysdeps/ieee754/dbl-64/s_sin.c`, `s_sincos.c` — the IBM Accurate
//! Mathematical Library), not a delegation.
//!
//! Ported rather than left as `f64::sin`/`f64::cos` because
//! `src/kernels/double/trig.rs`'s vector `BitExact` path needs a schedule to
//! replay lane-parallel, and the platform call is opaque to that. The huge-
//! argument band (`|x| >= 105414350`, where the source reaches for
//! `__branred`'s Payne-Hanek reduction) is deliberately **not** ported: it
//! still calls straight through to `f64::sin`/`f64::cos`, which is bit-exact
//! by construction and correct for every input — a `patch_lanes` repair the
//! vector kernel reaches for, not a port with a gap in it. `__branred` itself
//! may be ported later if profiling shows that band matters; nothing here
//! depends on it not being.
//!
//! Three entry points, not one shared by projection: `__sin`'s and
//! `__sincos`'s mid-band (`0.855469 <= |x| < 2.426265`) compute the
//! complementary angle differently — `__sin` calls `do_cos(y, hp1)` with the
//! low part passed straight through, `__sincos` first forms a compensated
//! sum `a = fl(y + hp1)`, `da = (y - a) + hp1` and calls `do_cos(a, da)`
//! instead. Those are not the same computation: `do_cos`'s own reduction
//! rounds through its `x = fabs(x) - (u.x - big) + dx` step using whichever
//! of `y` or `a` is passed as `x`, and `a != y` in general. So each of `sin`,
//! `cos` and `sincos` below reproduces its own function's control flow from
//! the C source, not a shared helper's.

use crate::tables::double::trig as t;

/// `SINCOS_TABLE_LOOKUP`: the four table entries `[sn, ssn, cs, ccs]` for the
/// rounded index carried in `u`'s low 32 bits.
///
/// `u` is the bit pattern of `t::BIG + fabs(x)` (or the reduced argument's
/// own rounded form in the table band) — the classic round-to-N-bits trick,
/// where the low bits of the *result* are the lookup index because adding
/// `BIG` rounds away everything above the table's resolution.
#[inline(always)]
fn sincos_table_lookup(u_bits: u64) -> (f64, f64, f64, f64) {
    let low32 = u_bits as u32 as i32;
    let k = ((low32 << 2) as isize) as usize;
    (
        f64::from_bits(t::TAB[k]),
        f64::from_bits(t::TAB[k + 1]),
        f64::from_bits(t::TAB[k + 2]),
        f64::from_bits(t::TAB[k + 3]),
    )
}

/// `TAYLOR_SIN`: `sin(x + dx)` via the degree-11 Taylor series, for
/// `|x| < 0.126` where `do_sin`'s table-based form is not needed.
///
/// Every step here fuses (read from the compiled `__sin_fma`, `objdump -d`,
/// glibc 2.43) except the leading `dx * 0.5` and the final `x + t_val`,
/// which are separate `mulsd`/`addsd`.
#[inline(always)]
fn taylor_sin(xx: f64, x: f64, dx: f64) -> f64 {
    let poly = xx.mul_add(t::S5, t::S4);
    let poly = xx.mul_add(poly, t::S3);
    let poly = xx.mul_add(poly, t::S2);
    let poly = xx.mul_add(poly, t::S1);
    let half_dx = dx * 0.5;
    let inner = poly.mul_add(x, -half_dx);
    let t_val = xx.mul_add(inner, dx);
    x + t_val
}

/// `cs2 + xx*(cs4 + xx*cs6)`, do_cos/do_sin's shared cosine-part inner
/// polynomial, fused following the C source's own grouping.
#[inline(always)]
fn cos_inner(xx: f64) -> f64 {
    let inner = xx.mul_add(t::CS6, t::CS4);
    xx.mul_add(inner, t::CS2)
}

/// `sn3 + xx*sn5`, the shared sine-part inner polynomial.
#[inline(always)]
fn sin_inner(xx: f64) -> f64 {
    xx.mul_add(t::SN5, t::SN3)
}

/// `do_cos`: cosine of `x + dx`, `|x + dx|` already folded into the table's
/// domain by the caller.
#[inline(always)]
fn do_cos(x: f64, dx: f64) -> f64 {
    let dx = if x < 0.0 { -dx } else { dx };
    let u = t::BIG + x.abs();
    let x = x.abs() - (u - t::BIG) + dx;

    let xx = x * x;
    let s = (x * xx).mul_add(sin_inner(xx), x);
    let c = xx * cos_inner(xx);
    let (sn, ssn, cs, ccs) = sincos_table_lookup(u.to_bits());
    // `cor = (ccs - s*ssn - cs*c) - sn*s`, as three separate FMAs (read from
    // the compiled `__cos_fma`, `objdump -d`, glibc 2.43) — not the plain
    // arithmetic the C source's grouping alone would suggest.
    let step1 = s.mul_add(-ssn, ccs);
    let step2 = c.mul_add(-cs, step1);
    let cor = s.mul_add(-sn, step2);
    cs + cor
}

/// `do_sin`: sine of `x + dx`.
#[inline(always)]
fn do_sin(x: f64, dx: f64) -> f64 {
    let xold = x;
    if x.abs() < 0.126 {
        return taylor_sin(x * x, x, dx);
    }

    let dx = if x <= 0.0 { -dx } else { dx };
    let u = t::BIG + x.abs();
    let x = x.abs() - (u - t::BIG);

    let xx = x * x;
    let s = x + (x * xx).mul_add(sin_inner(xx), dx);
    // `c = x*dx + xx*cos_inner(xx)`: `xx*cos_inner` rounds separately
    // (plain `mulsd`) and *then* `x*dx` fuses into the add — the opposite
    // pairing from what the C source's left-to-right reading would suggest,
    // and confirmed by trace, not assumed.
    let c = x.mul_add(dx, xx * cos_inner(xx));
    let (sn, ssn, cs, ccs) = sincos_table_lookup(u.to_bits());
    // `cor = (ssn + s*ccs - sn*c) + cs*s`, as three FMAs mirroring `do_cos`'s
    // schedule (same compiled-code source; see `do_cos`'s doc).
    let step1 = s.mul_add(ccs, ssn);
    let step2 = c.mul_add(-sn, step1);
    let cor = s.mul_add(cs, step2);
    (sn + cor).copysign(xold)
}

/// `reduce_sincos`: reduce `x` to `(a, da)` with `|a + da| <= pi/4` and
/// return the quadrant `n` (`x = n*pi/2 + a + da`), for `|x| < 105414350`.
///
/// FMA placement here was read out of the compiled `__sin_fma`/`__cos_fma`
/// (`objdump -d`, glibc 2.43, x86-64), not inferred from the C source's
/// arithmetic grouping: every multiply that feeds directly into the next
/// add/subtract fuses, *including* the two places (`t2`/`db` from `pp3`,
/// then `b`/`da`'s tail from `pp4`) where the compiler re-issues the same
/// product (`xn * pp3`, `xn * pp4`) a second time inside another FMA rather
/// than reusing a once-computed value — because reusing it would mean
/// keeping an unfused intermediate around, and the schedule below is
/// provably what is actually compiled, not merely plausible.
#[inline(always)]
fn reduce_sincos(x: f64) -> (f64, f64, i32) {
    let t_val = x.mul_add(t::HPINV, t::TOINT);
    let xn = t_val - t::TOINT;
    let v_bits = t_val.to_bits();
    let y = xn.mul_add(-t::MP1, x);
    let y = xn.mul_add(-t::MP2, y);
    let n = (v_bits as u32 as i32) & 3;

    let t2 = xn.mul_add(-t::PP3, y);
    let w1 = y - t2;
    let db = xn.mul_add(-t::PP3, w1);

    let b = xn.mul_add(-t::PP4, t2);
    let w2 = t2 - b;
    let tail = xn.mul_add(-t::PP4, w2);
    let da = tail + db;

    (b, da, n)
}

/// `do_sincos`: `sin`/`cos` of `a + da` in quadrant `n`, as one value —
/// `__sin`/`__cos`'s own shape (not `__sincos`'s, which inlines the
/// equivalent choice differently; see the module doc).
#[inline(always)]
fn do_sincos_one(a: f64, da: f64, n: i32) -> f64 {
    let retval = if n & 1 != 0 { do_cos(a, da) } else { do_sin(a, da) };
    if n & 2 != 0 { -retval } else { retval }
}

/// `sin(x)`.
pub fn sin(x: f64) -> f64 {
    let m = x.to_bits();
    let k = (0x7fffffff_u32 & (m >> 32) as u32) as i32;

    if k < 0x3e500000 {
        return x;
    }
    if k < 0x3feb6000 {
        return do_sin(x, 0.0);
    }
    if k < 0x400368fd {
        let t = t::HP0 - x.abs();
        return do_cos(t, t::HP1).copysign(x);
    }
    if k < 0x419921FB {
        let (a, da, n) = reduce_sincos(x);
        return do_sincos_one(a, da, n);
    }
    // |x| >= 105414350: the `__branred` band, deliberately not ported.
    f64::sin(x)
}

/// `cos(x)`.
pub fn cos(x: f64) -> f64 {
    let m = x.to_bits();
    let k = (0x7fffffff_u32 & (m >> 32) as u32) as i32;

    if k < 0x3e400000 {
        return 1.0;
    }
    if k < 0x3feb6000 {
        return do_cos(x, 0.0);
    }
    if k < 0x400368fd {
        let y = t::HP0 - x.abs();
        let a = y + t::HP1;
        let da = (y - a) + t::HP1;
        return do_sin(a, da);
    }
    if k < 0x419921FB {
        let (a, da, n) = reduce_sincos(x);
        return do_sincos_one(a, da, n + 1);
    }
    f64::cos(x)
}

/// `(sin(x), cos(x))`.
pub fn sincos(x: f64) -> (f64, f64) {
    let m = x.to_bits();
    let k = (m >> 32) as u32 & 0x7fffffff;

    if k < 0x400368fd {
        if k < 0x3e400000 {
            return (x, 1.0);
        }
        if k < 0x3feb6000 {
            return (do_sin(x, 0.0), do_cos(x, 0.0));
        }
        // |x| < 2.426265.
        let y = t::HP0 - x.abs();
        let a = y + t::HP1;
        let da = (y - a) + t::HP1;
        return (do_cos(a, da).copysign(x), do_sin(a, da));
    }
    if k < 0x7ff00000 {
        if k < 0x419921FB {
            let (mut a, mut da, n) = reduce_sincos(x);
            let n = n & 3;
            if n == 1 || n == 2 {
                a = -a;
                da = -da;
            }
            let (sinx, cosx) = (do_sin(a, da), do_cos(a, da));
            return if n & 1 != 0 {
                (if n & 2 != 0 { -cosx } else { cosx }, sinx)
            } else {
                (sinx, if n & 2 != 0 { -cosx } else { cosx })
            };
        }
        // |x| >= 105414350: not ported, see the module doc.
        return (f64::sin(x), f64::cos(x));
    }
    (f64::sin(x), f64::cos(x))
}
