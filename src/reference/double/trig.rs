//! `sin`, `cos`, `sincos` and `tan`: a port of glibc's
//! `__sin`/`__cos`/`__sincos`/`__tan` (`sysdeps/ieee754/dbl-64/s_sin.c`,
//! `s_sincos.c`, `s_tan.c` — the IBM Accurate Mathematical Library), not a
//! delegation.
//!
//! Ported rather than left as `f64::sin`/`f64::cos`/`f64::tan` because
//! `src/kernels/double/trig.rs`'s vector `BitExact` path needs a schedule to
//! replay lane-parallel, and the platform call is opaque to that. The huge-
//! argument bands (`|x| >= 105414350` for the sine family, `|x| > 1e8` for
//! `tan`, where the source reaches for `__branred`'s Payne-Hanek reduction)
//! are deliberately **not** ported: they still call straight through to the
//! platform function, which is bit-exact by construction and correct for
//! every input — a `patch_lanes` repair the vector kernel reaches for, not a
//! port with a gap in it. `__branred` itself may be ported later if
//! profiling shows that band matters; nothing here depends on it not being.
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
    let retval = if n & 1 != 0 {
        do_cos(a, da)
    } else {
        do_sin(a, da)
    };
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

/// `tan(x)`: a port of glibc's `__tan` (`sysdeps/ieee754/dbl-64/s_tan.c` —
/// the IBM Accurate Mathematical Library), not a delegation.
///
/// Ported rather than left as `f64::tan` for the same reason `sin`/`cos`
/// are: `src/kernels/double/trig.rs`'s vector `BitExact` path needs a
/// schedule to replay lane-parallel, and the platform call is opaque to
/// that. The huge-argument band (`|x| > 1e8`, where the source reaches for
/// `__branred`'s Payne-Hanek reduction) is deliberately **not** ported: it
/// still calls straight through to `f64::tan`, which is bit-exact by
/// construction and correct for every input — a `patch_lanes` repair the
/// vector kernel reaches for, not a port with a gap in it.
///
/// The reduction is shared with `sin`/`cos` (`HPINV`, `TOINT`, `MP1`/`MP2`,
/// `PP3`/`PP4` are bit-identical between `usncs.h` and `utan.h`, asserted by
/// the table generator), but the reduction *constants differ*: `__tan` adds
/// the extra third (`MP3`) / fourth (`PP4`) `pi/2` residue terms that
/// `reduce_sincos` does not use, because `tan` must keep the reduced angle
/// accurate to the very last bit to reach its 0.62-ULP bound.
///
/// Every `mul_add` here fuses (read from the compiled `__tan_fma`,
/// `objdump -d`, glibc 2.43): the band-II polynomial's last step and final
/// product, the table bands' index `256*w/ya - 15.5`, the `e`/`pz`
/// interpolations, the reductions' `t = x*HPINV + TOINT` and their `a`/`da`
/// steps, and `DIV2`'s `uu = c*b - u` and `cc = t3 - db*c`. The `+ 0.0` in
/// `DIV2` matters: it is the `xx` term of the dividend `1.0 + 0.0`, and is
/// kept exactly as the compiled code writes it.
#[inline(always)]
pub fn tan(x: f64) -> f64 {
    let m = x.to_bits();
    let k = (0x7fffffff_u32 & (m >> 32) as u32) as i32;

    // Specials: `|x|` is `inf` or NaN — `retval = x - x` (NaN). The `EDOM`
    // errno `__tan` raises for `inf` is not Rust's to set. The `x - x` is
    // glibc's own idiom, not a typo for `0.0`: it is the exact instruction
    // `vsubsd %xmm0, %xmm0, %xmm0`, whose NaN carries `x`'s own payload.
    #[allow(clippy::eq_op)]
    if k >= 0x7ff00000 {
        return x - x;
    }

    let w = x.abs();

    // (I): `|x| < 1.259e-8`. The underflow-force `w * w` for subnormal `w`
    // changes flags only, never `retval`, so it is dropped.
    if w <= t::G1 {
        return x;
    }
    // (II): `|x| < 0.0608` — polynomial I, the direct Taylor series in `x`.
    if w <= t::G2 {
        let x2 = x * x;
        let t2 = x2.mul_add(t::D11, t::D9);
        let t2 = x2.mul_add(t2, t::D7);
        let t2 = x2.mul_add(t2, t::D5);
        let t2 = x2.mul_add(t2, t::D3);
        return (x * x2).mul_add(t2, x);
    }
    // (III): `|x| < 0.787` — the `w`-indexed table, `s = (x < 0) ? -1 : 1`
    // on the *signed* argument (the disassembly tests `x`, not `w`).
    if w <= t::G3 {
        let i = (w.mul_add(256.0, t::MFFTNHF)) as i32;
        let xfg = f64::from_bits(t::XFG[i as usize * 3]);
        let fi = f64::from_bits(t::XFG[i as usize * 3 + 1]);
        let gi = f64::from_bits(t::XFG[i as usize * 3 + 2]);
        let z = w - xfg;
        let z2 = z * z;
        let sy = if x < 0.0 { -1.0 } else { 1.0 };
        let pz = (z * z2).mul_add(z2.mul_add(t::E1, t::E0), z);
        let t2 = pz * (gi + fi) / (gi - pz);
        return sy * (fi + t2);
    }
    // (IV): `|x| < 25.0` — reduction by algorithm i (three-part `mp`).
    if w <= t::G4 {
        let t = x.mul_add(t::HPINV, t::TOINT);
        let xn = t - t::TOINT;
        let t1 = xn.mul_add(-t::MP1, x);
        let t1 = xn.mul_add(-t::MP2, t1);
        let a = xn.mul_add(-t::MP3, t1);
        let da = xn.mul_add(-t::MP3, t1 - a);
        return tan_sub(t, a, da);
    }
    // (V): `|x| <= 1e8` — reduction by algorithm ii (four-part `pp`).
    if w <= t::G5 {
        let t = x.mul_add(t::HPINV, t::TOINT);
        let xn = t - t::TOINT;
        let t1 = xn.mul_add(-t::MP1, x);
        let t1 = xn.mul_add(-t::MP2, t1);
        let a = xn.mul_add(-t::PP3, t1);
        let da = xn.mul_add(-t::PP3, t1 - a);
        let b = xn.mul_add(-t::PP4, a);
        let db = xn.mul_add(-t::PP4, a - b);
        let da = db + da;
        // EADD(a, da, sum, cc) — the four-part reduction's compensation.
        let sum = b + da;
        let cc = if b.abs() > da.abs() { (b - sum) + da } else { (da - sum) + b };
        return tan_sub(t, sum, cc);
    }
    // (VI): `|x| > 1e8` — `__branred`'s Payne-Hanek reduction, not ported,
    // see the module doc.
    f64::tan(x)
}

/// The `(IV)`/`(V)` common tail: `tan` (or `-cot`) of the reduced argument
/// `a + da`, with quadrant parity `n`.
///
/// Bands (VI)/(VIII)/(X) evaluate the *signed* `a`/`da` — the polynomial is
/// odd, and `-cot` is its own odd evaluation, so the sign is carried by the
/// values, not by `sy`; only the table bands multiply `sy`, with `-cot` as
/// `-sy * y`. The disassembly shows the `sy` load (`+-1.0`) and the negated
/// `a`/`da` prepared together, before the `ya <= gy2` branch.
#[inline(always)]
fn tan_sub(t: f64, a: f64, da: f64) -> f64 {
    let n = (t.to_bits() & 1) as u32;
    let (ya, yya, sy) = if a < 0.0 { (-a, -da, -1.0) } else { (a, da, 1.0) };

    // (VI)/(VIII)/(X): `0 < |y| <= 0.0608` — the odd polynomial, on the
    // signed values.
    if ya <= t::GY2 {
        let a2 = a * a;
        let t2 = a2.mul_add(t::D11, t::D9);
        let t2 = a2.mul_add(t2, t::D7);
        let t2 = a2.mul_add(t2, t::D5);
        let t2 = a2.mul_add(t2, t::D3);
        let t2 = (a * a2).mul_add(t2, da);
        let y = a + t2;
        if n & 1 != 0 {
            // EADD(a, t2, b, db); DIV2(1.0, 0.0, b, db, ...) — `-cot`.
            let b = y;
            let db = if a.abs() > t2.abs() { (a - b) + t2 } else { (t2 - b) + a };
            let c = 1.0 / b;
            let u = c * b;
            let uu = c.mul_add(b, -u);
            let t3 = ((1.0 - u) - uu) + 0.0;
            let cc = (-db).mul_add(c, t3) / b;
            let z = c + cc;
            let zz = (c - z) + cc;
            -(z + zz)
        } else {
            y
        }
    } else {
        // (VII)/(IX)/(XI): `0.0608 < |y| <= 0.787` — the `ya`-indexed table.
        let i = (ya.mul_add(256.0, t::MFFTNHF)) as i32;
        let xfg = f64::from_bits(t::XFG[i as usize * 3]);
        let fi = f64::from_bits(t::XFG[i as usize * 3 + 1]);
        let gi = f64::from_bits(t::XFG[i as usize * 3 + 2]);
        let z = (ya - xfg) + yya;
        let z2 = z * z;
        let pz = (z * z2).mul_add(z2.mul_add(t::E1, t::E0), z);
        if n & 1 != 0 {
            let t2 = pz * (fi + gi) / (fi + pz);
            -sy * (gi - t2)
        } else {
            let t2 = pz * (gi + fi) / (gi - pz);
            sy * (fi + t2)
        }
    }
}
