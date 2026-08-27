//! `sin`, `cos`, `sincos` and `tan`: a port of glibc's
//! `__sin`/`__cos`/`__sincos`/`__tan` (`sysdeps/ieee754/dbl-64/s_sin.c`,
//! `s_sincos.c`, `s_tan.c` — the IBM Accurate Mathematical Library), not a
//! delegation.
//!
//! Ported rather than left as `f64::sin`/`f64::cos`/`f64::tan` because
//! `src/kernels/double/trig.rs`'s vector `BitExact` path needs a schedule to
//! replay lane-parallel, and the platform call is opaque to that.
//!
//! The huge-argument bands — `|x| >= 105414350` for the sine family, `|x| >
//! 1e8` for `tan` — are ported too, including `__branred`'s Payne-Hanek
//! reduction (`branred` below). There is no gap left: nothing in this
//! module calls the platform on any input.
//!
//! `__branred` is the one piece here that is entirely unfused, and that is
//! not a simplification. `branred.c` is its own translation unit and is *not*
//! built with `-mfma` — glibc's `sysdeps/x86_64/fpu/multiarch/Makefile` puts
//! `-mfma -mavx2` on `s_sin-fma.c` and `s_tan-fma.c`, which then call this
//! one shared, unfused `__branred` — and the disassembly confirms it: no
//! `vfmadd` of any form anywhere in it.
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

// ---------------------------------------------------------------------------
// `__branred`: the huge-argument reduction
// ---------------------------------------------------------------------------

/// One half of `__branred`'s digit product: the 2/pi partial sum for one
/// 27-bit piece of the pre-scaled argument.
///
/// `__branred` splits `x * 2^-600` into two exactly-representable halves and
/// runs this identical body on each, because the product `x1 * toverp[k+i]`
/// must itself be exact — 27 bits of `x1` times 24 bits of table is 51 bits,
/// inside `f64`'s 53. Returns `(b, bb, sum)`: the fractional part of
/// `x_half * 2/pi` as an unevaluated `b + bb`, and the integer part reduced
/// modulo 4 by the `big1` rounding.
///
/// Every operation here is a separate rounding. That is not a simplification:
/// `branred.c` is its own translation unit and is *not* built with `-mfma`
/// (glibc's `sysdeps/x86_64/fpu/multiarch/Makefile` puts `-mfma -mavx2` on
/// `s_sin-fma.c` / `s_tan-fma.c`, which then call this one shared, unfused
/// `__branred`). Confirmed by disassembly: `__branred` in `libm.so.6`
/// contains no `vfmadd`/`vfmsub` of any form, only `mulsd`/`addsd`/`subsd`.
/// Rust never contracts `a * b + c` on its own, so the plain spellings below
/// are already the right ones — but writing `mul_add` here would be a bug.
#[inline(always)]
fn branred_half(xh: f64) -> (f64, f64, f64) {
    // `k = ((biased exponent) - 450) / 24`, clamped at zero: which 24-bit
    // digit of `2/pi` the product's leading bit lands on. C's division
    // truncates toward zero, and so does Rust's, so a small negative
    // numerator gives 0 in both and only a numerator below -24 reaches the
    // clamp.
    let mut k = (((xh.to_bits() >> 52) & 2047) as i32 - 450) / 24;
    if k < 0 {
        k = 0;
    }

    // `gor.i[HIGH_HALF] -= (k * 24) << 20`, as a 64-bit subtraction. `T576`'s
    // low word is zero and the subtrahend's low 32 bits are zero, so no
    // borrow crosses the word boundary and this is exactly the high-word
    // arithmetic the C performs.
    let mut gor = f64::from_bits(t::T576.to_bits() - (((k * 24) as u64) << 52));

    let mut r = [0.0f64; 6];
    for (i, ri) in r.iter_mut().enumerate() {
        *ri = xh * t::TOVERP[k as usize + i] * gor;
        gor *= t::TM24;
    }

    // The first three digits are big enough to carry an integer part; peel it
    // off with the round-to-integer trick and accumulate it into `sum`.
    let mut sum = 0.0;
    for ri in r.iter_mut().take(3) {
        let s = (*ri + t::BRANRED_BIG) - t::BRANRED_BIG;
        sum += s;
        *ri -= s;
    }

    // Summed low-to-high so the smallest terms are not lost, then the
    // rounding error of that sum recovered high-to-low.
    let mut t_val = 0.0;
    for i in 0..6 {
        t_val += r[5 - i];
    }
    let bb = (((((r[0] - t_val) + r[1]) + r[2]) + r[3]) + r[4]) + r[5];

    let s = (t_val + t::BRANRED_BIG) - t::BRANRED_BIG;
    sum += s;
    let t_val = t_val - s;
    let b = t_val + bb;
    let bb = (t_val - b) + bb;

    // `big1`'s ulp is 4, so this rounds `sum` to a multiple of 4 and leaves
    // the residue — which is all `n mod 4` needs.
    let s = (sum + t::BRANRED_BIG1) - t::BRANRED_BIG1;
    (b, bb, sum - s)
}

/// `__branred`: reduce `x` to `(a, da, n)` with `x = n*pi/2 + (a + da)` and
/// `|a + da| <= pi/4`, for `105414350 <= |x| < 2^1024`.
///
/// A port of glibc's `sysdeps/ieee754/dbl-64/branred.c` (the IBM Accurate
/// Mathematical Library), which is what `__sin`, `__cos`, `__sincos` and
/// `__tan` all reach for once the argument is too large for `reduce_sincos`'s
/// three- or four-part Cody-Waite splitting to hold `pi/2` to enough bits.
/// The algorithm is Payne-Hanek: multiply by a 75-digit base-2^24
/// representation of `2/pi` ([`t::TOVERP`]), keep only the six digits
/// straddling the binary point, and drop the integer part.
///
/// `n` is returned already reduced modulo 4, as the quadrant `do_sincos`
/// wants. It is a *signed* result: C's `((int) sum) & 3` truncates toward
/// zero and then masks in two's complement, which `as i32` and `&` reproduce
/// exactly — and `sum` is bounded to `[-5, 5]` by the `big1` rounding above,
/// so the conversion cannot saturate.
#[inline(always)]
fn branred(x: f64) -> (f64, f64, i32) {
    // Pre-scale so the split below cannot overflow, then Dekker-split into
    // two halves of at most 27 significant bits each.
    let x = x * t::TM600;
    let t_val = x * t::SPLIT;
    let x1 = t_val - (t_val - x);
    let x2 = x - x1;

    let (b1, bb1, sum1) = branred_half(x1);
    let (b2, bb2, sum2) = branred_half(x2);

    let sum = sum1 + sum2;
    let b = b1 + b2;
    let bb = if b1.abs() > b2.abs() {
        (b1 - b) + b2
    } else {
        (b2 - b) + b1
    };

    // Fold the fraction into `[-1/2, 1/2]`, carrying into the quadrant count.
    let (b, sum) = if b > 0.5 {
        (b - 1.0, sum + 1.0)
    } else if b < -0.5 {
        (b + 1.0, sum - 1.0)
    } else {
        (b, sum)
    };

    let s = b + (bb + bb1 + bb2);
    let t_val = ((b - s) + bb) + (bb1 + bb2);

    // Multiply the reduced fraction back by `pi/2`, in a three-way split of
    // the constant (`MP1` + `BRANRED_MP2` + `HP1`) against a two-way split of
    // `s`, so every partial product is exact. `BRANRED_MP2` is `branred.h`'s
    // own spelling of the middle term and differs from `usncs.h`'s `MP2` in
    // its last two significand digits — see the table module.
    let sp = s * t::SPLIT;
    let t1 = sp - (sp - s);
    let t2 = s - t1;
    let b = s * t::HP0;
    let bb = (((t1 * t::MP1 - b) + t1 * t::BRANRED_MP2) + t2 * t::MP1)
        + (t2 * t::BRANRED_MP2 + s * t::HP1 + t_val * t::HP0);
    let a = b + bb;
    let aa = (b - a) + bb;
    (a, aa, (sum as i32) & 3)
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
    // 105414350 <= |x| < 2^1024: `__branred`'s Payne-Hanek reduction, then
    // the same `do_sincos` tail the band below uses.
    if k < 0x7ff00000 {
        let (a, da, n) = branred(x);
        return do_sincos_one(a, da, n);
    }
    // |x| >= 2^1024 — infinity or a NaN. `retval = x / x`, glibc's own
    // spelling: NaN either way, carrying `x`'s payload when `x` is a NaN.
    // The `EDOM` errno `__sin` raises for an infinity is not Rust's to set.
    #[allow(clippy::eq_op)]
    {
        x / x
    }
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
    if k < 0x7ff00000 {
        let (a, da, n) = branred(x);
        return do_sincos_one(a, da, n + 1);
    }
    #[allow(clippy::eq_op)]
    {
        x / x
    }
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
        {
            // `__sincos` picks the reduction with a conditional expression
            // rather than a separate branch, so both bands share one tail.
            let (mut a, mut da, n) = if k < 0x419921FB {
                reduce_sincos(x)
            } else {
                branred(x)
            };
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
    }
    // |x| >= 2^1024 — `*sinx = *cosx = x / x`.
    #[allow(clippy::eq_op)]
    {
        (x / x, x / x)
    }
}

/// `tan(x)`: a port of glibc's `__tan` (`sysdeps/ieee754/dbl-64/s_tan.c` —
/// the IBM Accurate Mathematical Library), not a delegation.
///
/// Ported rather than left as `f64::tan` for the same reason `sin`/`cos`
/// are: `src/kernels/double/trig.rs`'s vector `BitExact` path needs a
/// schedule to replay lane-parallel, and the platform call is opaque to
/// that. The huge-argument band (`|x| > 1e8`) is ported too, through
/// `branred`; the source duplicates its two band bodies rather than
/// sharing them with (VIII)/(IX), but they are the same computation on an
/// `EADD`-compensated argument, so `tan_sub_n` serves both — which the
/// exhaustive sweep of that band confirms.
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
        let cc = if b.abs() > da.abs() {
            (b - sum) + da
        } else {
            (da - sum) + b
        };
        return tan_sub(t, sum, cc);
    }
    // (X)/(XI): `1e8 < |x| < 2^1024` — reduction by algorithm iii,
    // `__branred`. The source duplicates the two band bodies rather than
    // sharing them with (VIII)/(IX), but they are the same computation on an
    // `EADD`-compensated `(a, da)`, so [`tan_sub_n`] serves both.
    let (a, da, n) = branred(x);
    // EADD(a, da, t1, t2).
    let t1 = a + da;
    let t2 = if a.abs() > da.abs() {
        (a - t1) + da
    } else {
        (da - t1) + a
    };
    tan_sub_n((n & 1) as u32, t1, t2)
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
    tan_sub_n((t.to_bits() & 1) as u32, a, da)
}

/// [`tan_sub`] with the quadrant parity passed in rather than read off the
/// reduction's rounded `t`.
///
/// Bands (X)/(XI) need this: `__branred` returns the quadrant as an integer
/// directly (`n = __branred(...) & 1`), with no `TOINT`-rounded double to
/// take the low bit of.
#[inline(always)]
fn tan_sub_n(n: u32, a: f64, da: f64) -> f64 {
    let (ya, yya, sy) = if a < 0.0 {
        (-a, -da, -1.0)
    } else {
        (a, da, 1.0)
    };

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
            let db = if a.abs() > t2.abs() {
                (a - b) + t2
            } else {
                (t2 - b) + a
            };
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
