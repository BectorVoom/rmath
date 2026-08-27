//! `asinh` and `acosh`.
//!
//! The last two functions in this crate that used to reach for the platform.
//! Both are CORE-MATH (glibc's `sysdeps/ieee754/dbl-64/s_asinh.c` and
//! `e_acosh.c`) and both are **correctly rounded**, so reproducing them is
//! not a claim about this platform: any correct `asinh` anywhere returns
//! these bits, the same argument [`crate::reference`] makes for `erf` and
//! `erfc`. `sinh`, `cosh` and `tanh` are elsewhere — they are compositions on
//! `exp`/`expm1` and live in [`super`] beside them; only the inverse pair
//! needs the machinery below.
//!
//! # Why there is so much of it
//!
//! Both reduce to one logarithm — of `x + sqrt(x^2 + 1)` for `asinh`, of
//! `x + sqrt(x^2 - 1)` for `acosh` — and both have a *zero* inside their
//! domain, `asinh` at 0 and `acosh` at 1. Near a zero the leading term
//! cancels against everything the table contributes, so a double-precision
//! result cannot settle the last bit and no amount of extra polynomial terms
//! helps. CORE-MATH answers that with three tiers:
//!
//! 1. A fast path with a proven error bound, and a bracket test (`lb == ub`)
//!    asking whether that bound leaves only one representable `f64` in
//!    range. Almost every argument stops here — it is also what
//!    [`crate::kernels::double::hyper`]'s vector code replays lane-parallel.
//! 2. A refinement in double-double and triple-double arithmetic over a
//!    four-stage logarithm table ([`t::LL`]), reached when the bracket fails.
//! 3. A short sorted list of the arguments where even *that* lands on a
//!    halfway point, searched by bisection.
//!
//! The double-word helpers below are CORE-MATH's `ddcoremath.h`. Their
//! `fma`s are the algorithm, not an optimisation: `fma(ch, xh, -ahhh)` is the
//! exact rounding error of `ch * xh`, which is the whole point of carrying an
//! unevaluated pair, and cannot be spelled without a fused multiply-add.
//! Nothing else here fuses — `s_asinh.c` and `e_acosh.c` have no `-mfma`
//! variant in `sysdeps/x86_64/fpu/multiarch/Makefile`, and `__asinh` and
//! `__acosh` in `libm.so.6` contain no `vfmadd` beyond the ones the source
//! calls for by name.

use crate::tables::double::asincosh as ft;
use crate::tables::double::asincosh_acc as t;

// ---------------------------------------------------------------------------
// Double-word arithmetic (CORE-MATH's `ddcoremath.h`)
// ---------------------------------------------------------------------------

/// `fasttwosum`: the exact sum `x + y` as an unevaluated pair, valid when
/// `|x| >= |y|`.
#[inline(always)]
fn fasttwosum(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    let z = s - x;
    (s, y - z)
}

/// `adddd`: the double-word sum, with no ordering precondition.
#[inline(always)]
fn adddd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let s = xh + ch;
    let d = s - xh;
    (s, ((ch - d) + (xh + (d - s))) + (xl + cl))
}

/// `muldd_acc`: the double-word product, with the low word recovered as
/// `(ahhh - ch) + ahhl`.
#[inline(always)]
fn muldd_acc(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = ch.mul_add(xh, -ahhh);
    ahhl += alhh + ahlh;
    let ch = ahhh + ahhl;
    (ch, (ahhh - ch) + ahhl)
}

/// `muldd_acc2`: the same product, but recombined with [`fasttwosum`].
///
/// Not interchangeable with [`muldd_acc`]: the two differ exactly when
/// `|ahhh| < |ahhl|`, which is why upstream keeps both and why `acosh`'s
/// refinement uses this one where `asinh`'s uses the other.
#[inline(always)]
fn muldd_acc2(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = ch.mul_add(xh, -ahhh);
    ahhl += alhh + ahlh;
    fasttwosum(ahhh, ahhl)
}

/// `mulddd`: a double-word times a plain `double`.
#[inline(always)]
fn mulddd(xh: f64, xl: f64, ch: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let ahhh = ch * xh;
    let mut ahhl = ch.mul_add(xh, -ahhh);
    ahhl += ahlh;
    let ch = ahhh + ahhl;
    (ch, (ahhh - ch) + ahhl)
}

/// `polydd`: Horner over double-word coefficients, seeded with an incoming
/// low word `l` that carries the single-double tail of the fit.
#[inline(always)]
fn polydd(xh: f64, xl: f64, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let mut i = c.len() - 1;
    let (mut ch, mut cl) = fasttwosum(c[i][0], l);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        let (h, lo) = muldd_acc(xh, xl, ch, cl);
        ch = h;
        cl = lo;
        let th = ch + c[i][0];
        let tl = (c[i][0] - th) + ch;
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}

/// `polydd3`: the same recurrence built on [`muldd_acc2`] and
/// [`fasttwosum`] instead.
#[inline(always)]
fn polydd3(xh: f64, xl: f64, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let mut i = c.len() - 1;
    let (mut ch, mut cl) = fasttwosum(c[i][0], l);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        let (h, lo) = muldd_acc2(xh, xl, ch, cl);
        ch = h;
        cl = lo;
        let (th, tl) = fasttwosum(c[i][0], ch);
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}

/// `__math_invalid (x)`: `(x - x) / (x - x)`.
///
/// The [`black_box`](core::hint::black_box) stops Rust folding `0/0` at
/// compile time, which produces the opposite sign from the `divsd` the C
/// compiles to.
#[inline(always)]
#[allow(clippy::eq_op)]
fn invalid(x: f64) -> f64 {
    let x = core::hint::black_box(x);
    (x - x) / (x - x)
}

/// Bisect a `(argument, high, low)` exception table, returning `high + low`
/// on an exact hit and `f` otherwise.
#[inline(always)]
fn database(db: &[[f64; 3]], key: f64, sgn: f64, f: f64) -> f64 {
    let (mut a, mut b) = (0isize, db.len() as isize - 1);
    let mut m = (a + b) / 2;
    while a <= b {
        let row = &db[m as usize];
        if row[0] < key {
            a = m + 1;
        } else if row[0] == key {
            return sgn * row[1] + sgn * row[2];
        } else {
            b = m - 1;
        }
        m = (a + b) / 2;
    }
    f
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// High part of `ln(2)`, as the fast paths split it.
const L2H: f64 = f64::from_bits(0x3fe62e42fefa3800);
/// Low part.
const L2L: f64 = f64::from_bits(0x3d2ef35793c76730);
/// `1 / ln(2)`: converts the fast path's result into the exponent index the
/// refinement's four-stage table wants.
const INV_LN2: f64 = f64::from_bits(0x3ff71547652b82fe);
/// The refinement's `ln(2)` in three limbs, high.
const L20: f64 = f64::from_bits(0x3fd62e42fefa3800);
/// Middle.
const L21: f64 = f64::from_bits(0x3d1ef35793c76800);
/// Low.
const L22: f64 = f64::from_bits(0xba49ff0342542fc3);
/// `1 + 2^-16`: the offset that turns the fast path's logarithm into the
/// refinement's 16-bit table index without a branch on the sign.
const IDX_BIAS: f64 = f64::from_bits(0x3ff0000800000000);

/// The four-stage table walk both refinements share.
///
/// Given the reduced significand bits `t_bits`, the exponent `e` and the
/// fast path's estimate `a` of `log2` of the argument, returns the residual
/// `(xh, xl)` *before* the second table stage is folded in, that stage's own
/// `(sh, sl)`, the stages' product `th`, the summed triple-double logarithm
/// `L`, and the exponent's two lower limbs.
///
/// The pieces are handed back separately rather than combined because the
/// caller has to fold the argument's own low word in between them.
#[inline(always)]
#[allow(clippy::type_complexity)]
fn refine_reduce(t_bits: u64, e: i32, a: f64) -> (f64, f64, f64, f64, f64, [f64; 3], f64, f64) {
    let ed = f64::from(e);
    let v = (a - ed + IDX_BIAS).to_bits();
    let i = (v.wrapping_sub(0x3ffu64 << 52)) >> (52 - 16);
    let i1 = ((i >> 12) & 0x1f) as usize;
    let i2 = ((i >> 8) & 0xf) as usize;
    let i3 = ((i >> 4) & 0xf) as usize;
    let i4 = (i & 0xf) as usize;

    let el2 = L22 * ed;
    let el1 = L21 * ed;
    let el0 = L20 * ed;

    let mut l = [0.0f64; 3];
    for (k, lk) in l.iter_mut().enumerate() {
        *lk = t::LL[0][i1][k] + t::LL[1][i2][k] + (t::LL[2][i3][k] + t::LL[3][i4][k]);
    }
    l[0] += el0;

    let t12 = t::T1[i1] * t::T2[i2];
    let t34 = t::T3[i3] * t::T4[i4];
    let th = t12 * t34;
    let tl = t12.mul_add(t34, -th);
    let tf = f64::from_bits(t_bits);
    let dh = th * tf;
    let dl = th.mul_add(tf, -dh);
    let sh = tl * tf;
    let sl = tl.mul_add(tf, -sh);
    let (xh, xl) = fasttwosum(dh - 1.0, dl);
    (xh, xl, sh, sl, th, l, el1, el2)
}

/// The fast paths' shared table walk over [`ft`]'s two stages.
///
/// Returns `(e, i1, i2, dx, f)`: the unbiased exponent, the two table
/// indices, the reduced residual and its polynomial correction. The *sum* is
/// deliberately not assembled here — `asinh` and `acosh` combine these same
/// five values in different associations, and floating-point addition is not
/// associative, so sharing the combination would cost the last bit. Compare
/// `s_asinh.c`'s `l2h * ed + (l1 + l2)` against `e_acosh.c`'s
/// `(l1 + l2) + l2h * ed`.
#[inline(always)]
fn fast_log_parts(ah: f64, off: i32) -> (i32, usize, usize, f64, f64) {
    let mut t_bits = ah.to_bits();
    let e = (t_bits >> 52) as i32 - off;
    t_bits &= !0u64 >> 12;
    let i = (t_bits >> (52 - 5)) as usize;
    let d = (t_bits & (!0u64 >> 17)) as i64;
    let j = ((t_bits as i64) + ((ft::B[i].0 as i64) << 33) + ((ft::B[i].1 as i64) * (d >> 16)))
        as u64
        >> (52 - 10);
    t_bits |= 0x3ffu64 << 52;
    let i1 = (j >> 5) as usize;
    let i2 = (j & 0x1f) as usize;
    let r = ft::R1[i1] * ft::R2[i2];
    let dx = r.mul_add(f64::from_bits(t_bits), -1.0);
    let dx2 = dx * dx;
    let f =
        dx2 * ((ft::C[0] + dx * ft::C[1]) + dx2 * ((ft::C[2] + dx * ft::C[3]) + dx2 * ft::C[4]));
    (e, i1, i2, dx, f)
}

// ---------------------------------------------------------------------------
// asinh
// ---------------------------------------------------------------------------

/// `asinh`'s `|x| < 0x1.ap-26` cubic.
const ASINH_S1: [f64; 1] = [f64::from_bits(0xbfc5555555555555)];
/// Its `|x| < 0x1p-12` quintic.
const ASINH_S2: [f64; 2] = [
    f64::from_bits(0xbfc5555555555555),
    f64::from_bits(0x3fb3333327c57c60),
];
/// Its `|x| < 0x1.3p-6` degree-9 fit.
const ASINH_S3: [f64; 4] = [
    f64::from_bits(0xbfc5555555555555),
    f64::from_bits(0x3fb333333332f2ff),
    f64::from_bits(0xbfa6db6d9a665159),
    f64::from_bits(0x3f9f186866d775f0),
];
/// Its `|x| < 0x1.bp-4` degree-15 fit.
const ASINH_S4: [f64; 7] = [
    f64::from_bits(0xbfc5555555555555),
    f64::from_bits(0x3fb3333333333310),
    f64::from_bits(0xbfa6db6db6da466c),
    f64::from_bits(0x3f9f1c71c2ea7be4),
    f64::from_bits(0xbf96e8b651b09d72),
    f64::from_bits(0x3f91c309fc0e69c2),
    f64::from_bits(0xbf8bab7833c1e000),
];

/// `as_asinh_zero`: the double-word series for arguments small enough that
/// the logarithm has nothing left to contribute.
fn asinh_zero(x: f64, x2h: f64, x2l: f64) -> f64 {
    let y2 = x2h
        * (t::ASINH_ZERO_CL[0]
            + x2h
                * (t::ASINH_ZERO_CL[1]
                    + x2h
                        * (t::ASINH_ZERO_CL[2]
                            + x2h * (t::ASINH_ZERO_CL[3] + x2h * t::ASINH_ZERO_CL[4]))));
    let (y1, y2) = polydd(x2h, x2l, &t::ASINH_ZERO_CH, y2);
    let (y1, y2) = muldd_acc(y1, y2, x2h, x2l);
    let (y1, y2) = mulddd(y1, y2, x);
    let (y0, y1) = fasttwosum(x, y1);
    let (y1, y2) = fasttwosum(y1, y2);

    let mut bits = y1.to_bits();
    // A `y1` whose significand is exactly zero leaves the sum on a rounding
    // boundary; nudge it in the direction `y2` points.
    if bits & (!0u64 >> 12) == 0 {
        if (y2.to_bits() ^ bits) >> 63 != 0 {
            bits -= 1;
        } else {
            bits += 1;
        }
        return y0 + f64::from_bits(bits);
    }
    y0 + y1
}

/// `as_asinh_refine`: the triple-double refinement.
fn asinh_refine(x: f64, zh: f64, zl: f64, a: f64) -> f64 {
    let mut t_bits = zh.to_bits();
    let e = (t_bits >> 52) as i32 - 0x3ff + i32::from(zl == 0.0);
    t_bits &= !0u64 >> 12;
    t_bits |= 0x3ffu64 << 52;

    let (xh, mut xl, sh0, sl0, th, l, el1, el2) = refine_reduce(t_bits, e, a);
    if zl != 0.0 {
        // Fold the argument's own low word back in at the reduced scale, by
        // shifting its exponent rather than dividing.
        let b = zl.to_bits().wrapping_sub((e as i64 as u64) << 52);
        xl += th * f64::from_bits(b);
    }
    let (xh, xl) = adddd(xh, xl, sh0, sl0);

    let sl =
        xh * (t::ASINH_REFINE_CL[0] + xh * (t::ASINH_REFINE_CL[1] + xh * t::ASINH_REFINE_CL[2]));
    let (sh, sl) = polydd(xh, xl, &t::ASINH_REFINE_CH, sl);
    let (sh, sl) = muldd_acc(xh, xl, sh, sl);
    let (sh, sl) = adddd(sh, sl, el1, el2);
    let (sh, sl) = adddd(sh, sl, l[1], l[2]);

    let (v0, v2) = fasttwosum(l[0], sh);
    let (v1, v2) = fasttwosum(v2, sl);
    let (v0, v1) = fasttwosum(v0, v1);
    let (v1, v2) = fasttwosum(v1, v2);
    let s = 2.0f64.copysign(x);
    let (v0, mut v1, v2) = (v0 * s, v1 * s, v2 * s);

    let mut bits = v1.to_bits();
    // A `v1` whose significand is exactly zero leaves the sum on a rounding
    // boundary; nudge it the way `v2` points.
    if bits & (!0u64 >> 12) == 0 {
        if (v2.to_bits() ^ bits) >> 63 != 0 {
            bits -= 1;
        } else {
            bits += 1;
        }
        v1 = f64::from_bits(bits);
    }
    let t0 = v0.to_bits();
    let er = bits.wrapping_add(41) & (!0u64 >> 12);
    let de = ((t0 >> 52) & 0x7ff).wrapping_sub((bits >> 52) & 0x7ff);
    let res = v0 + v1;
    if de > 99 || er < 80 {
        return database(&t::ASINH_DB, x.abs(), 1.0f64.copysign(x), res);
    }
    res
}

/// `asinh(x)`, correctly rounded.
pub fn asinh(x: f64) -> f64 {
    let ax = x.abs();
    let u = ax.to_bits();

    if u < 0x3fbb_0000_0000_0000 {
        // |x| < 0x1.bp-4
        if u < 0x3e57_1374_4912_3ef7 {
            // Below `0x1.7137449123ef7p-26`, `asinh x` rounds to `x`; the
            // `fma` keeps the direction and the inexact flag.
            if u == 0 {
                return x;
            }
            return (-f64::from_bits(0x3c30000000000000)).mul_add(x, x);
        }
        let x2h = x * x;
        let x2l = x.mul_add(x, -x2h);
        let x3h = x2h * x;
        let sl = if u < 0x3f93_0000_0000_0000 {
            if u < 0x3f30_0000_0000_0000 {
                if u < 0x3e5a_0000_0000_0000 {
                    x3h * ASINH_S1[0]
                } else {
                    x3h * (ASINH_S2[0] + x2h * ASINH_S2[1])
                }
            } else {
                x3h * (ASINH_S3[0] + x2h * (ASINH_S3[1] + x2h * (ASINH_S3[2] + x2h * ASINH_S3[3])))
            }
        } else {
            let c1 = ASINH_S4[1] + x2h * ASINH_S4[2];
            let c3 = ASINH_S4[3] + x2h * ASINH_S4[4];
            let c5 = ASINH_S4[5] + x2h * ASINH_S4[6];
            let x4 = x2h * x2h;
            x3h * (ASINH_S4[0] + x2h * (c1 + x4 * (c3 + x4 * c5)))
        };
        let eps = f64::from_bits(0x3ca7900000000000) * x3h;
        let lb = x + (sl - eps);
        let ub = x + (sl + eps);
        if lb == ub {
            return lb;
        }
        return asinh_zero(x, x2h, x2l);
    }

    // |x| >= 0x1.bp-4: one logarithm of `|x| + sqrt(x^2 + 1)`.
    let mut x2h = 0.0;
    let mut x2l = 0.0;
    let mut off = 0x3ff;
    let (ah, al);
    if u < 0x4190_0000_0000_0000 {
        // |x| < 2^26, where the square root still needs its low word.
        x2h = x * x;
        x2l = x.mul_add(x, -x2h);
        let (th, mut tl) = if u < 0x3ff0_0000_0000_0000 {
            fasttwosum(1.0, x2h)
        } else {
            fasttwosum(x2h, 1.0)
        };
        tl += x2l;
        let s = th.sqrt();
        let rs = 0.5 / th;
        let sl = (tl - s.mul_add(s, -th)) * (rs * s);
        let (h, t2) = fasttwosum(s, ax);
        ah = h;
        al = sl + t2;
    } else if u < 0x4330_0000_0000_0000 {
        // 2^26 <= |x| < 2^52: `sqrt(x^2 + 1)` is `|x|` to a `double`, so the
        // sum is `2|x|` with `1/(2|x|)` as its whole low word.
        ah = 2.0 * ax;
        al = 0.5 / ax;
    } else {
        if u >= 0x7ff0_0000_0000_0000 {
            return x + x; // ±inf or NaN
        }
        // Past 2^52 the `+1` is gone entirely and the doubling is folded
        // into the exponent offset instead.
        off = 0x3fe;
        ah = ax;
        al = 0.0;
    }

    let (e, i1, i2, dx, f) = fast_log_parts(ah, off);
    let ed = f64::from(e);
    let sgn = 1.0f64.copysign(x);
    let lh = (L2H * ed + (ft::L1[i1][1] + ft::L2[i2][1])) * sgn;
    let ll = (L2L * ed + ft::L1[i1][0] + ft::L2[i2][0] + al / ah + f + dx) * sgn;

    let eps = f64::from_bits(0x3c080df744ce99c7);
    let lb = lh + (ll - eps);
    let ub = lh + (ll + eps);
    if lb == ub {
        return lb;
    }
    // Below 1/4 the logarithm has cancelled so badly that the refinement
    // would be refining nothing; the small-argument series is the accurate
    // path here instead.
    if ax < 0.25 {
        return asinh_zero(x, x2h, x2l);
    }
    asinh_refine(x, ah, al, INV_LN2 * lb.abs())
}

// ---------------------------------------------------------------------------
// acosh
// ---------------------------------------------------------------------------

/// `acosh`'s near-1 fit, in `x - 1`, with `sqrt(2(x-1))` factored out.
const ACOSH_S1: [f64; 9] = [
    f64::from_bits(0xbfb5555555555555),
    f64::from_bits(0x3f93333333332f95),
    f64::from_bits(0xbf76db6db6d5534c),
    f64::from_bits(0x3f5f1c71c1e04356),
    f64::from_bits(0xbf46e8b8e3e40d58),
    f64::from_bits(0x3f31c4ba825ac4fe),
    f64::from_bits(0xbf1c9045534e6d9e),
    f64::from_bits(0x3f071fedae26a76b),
    f64::from_bits(0xbeef1f4f8cc65342),
];
/// The `0x1.bfp+6 <= x < 0x1.71p+9` asymptotic correction, in `1/x^2`.
const ACOSH_S2: [f64; 4] = [
    f64::from_bits(0x3bd5c4b6148816e2),
    f64::from_bits(0xbfd000000000005c),
    f64::from_bits(0xbfb7fffffebf3e6c),
    f64::from_bits(0xbfaaab6691f2bae7),
];
/// The `0x1.71p+9 <= x < 0x1.01p+15` correction.
const ACOSH_S3: [f64; 3] = [
    f64::from_bits(0xbbc7f77c8429c6c6),
    f64::from_bits(0xbfcffffffffff214),
    f64::from_bits(0xbfb8000268641bfe),
];
/// The `0x1.01p+15 <= x < 0x1.ap+31` correction. Past `0x1.ap+31` there is
/// none: `sqrt(x^2 - 1)` is `x` to a `double`.
const ACOSH_S4: [f64; 2] = [
    f64::from_bits(0x3bc7a0ed2effdd10),
    f64::from_bits(0xbfd000000017d048),
];

/// `as_acosh_one`: the double-word series just above 1.
fn acosh_one(x: f64, sh: f64, sl: f64) -> f64 {
    let y2 = x
        * (t::ACOSH_ONE_CL[0]
            + x * (t::ACOSH_ONE_CL[1]
                + x * (t::ACOSH_ONE_CL[2]
                    + x * (t::ACOSH_ONE_CL[3]
                        + x * (t::ACOSH_ONE_CL[4] + x * t::ACOSH_ONE_CL[5])))));
    let (y1, y2) = polydd3(x, 0.0, &t::ACOSH_ONE_CH, y2);
    let (y1, y2) = mulddd(y1, y2, x);
    let (y0, y1) = fasttwosum(1.0, y1);
    let y1 = y1 + y2;
    let (y0, y1) = muldd_acc2(y0, y1, sh, sl);
    y0 + y1
}

/// `as_acosh_refine`: the triple-double refinement.
fn acosh_refine(x: f64, a: f64) -> f64 {
    let ix = x.to_bits();
    let (zh, zl) = if ix < 0x4190_0000_0000_0000 {
        let x2h = x * x;
        let x2l = x.mul_add(x, -x2h);
        let (wh, wl) = fasttwosum(x2h - 1.0, x2l);
        let s = wh.sqrt();
        let ish = 0.5 / wh;
        let sl = (ish * s) * (wl - s.mul_add(s, -wh));
        let (zh, zl) = fasttwosum(x, s);
        fasttwosum(zh, zl + sl)
    } else if ix < 0x4330_0000_0000_0000 {
        (2.0 * x, -0.5 / x)
    } else {
        (x, 0.0)
    };

    let mut t_bits = zh.to_bits();
    let e = (t_bits >> 52) as i32 - 0x3ff + i32::from(zl == 0.0);
    t_bits &= !0u64 >> 12;
    t_bits |= 0x3ffu64 << 52;

    let (xh, mut xl, sh0, sl0, th, l, el1, el2) = refine_reduce(t_bits, e, a);
    if zl != 0.0 {
        let b = zl.to_bits().wrapping_sub((e as i64 as u64) << 52);
        xl += th * f64::from_bits(b);
    }
    let (xh, xl) = adddd(xh, xl, sh0, sl0);

    let sl =
        xh * (t::ACOSH_REFINE_CL[0] + xh * (t::ACOSH_REFINE_CL[1] + xh * t::ACOSH_REFINE_CL[2]));
    let (sh, sl) = polydd3(xh, xl, &t::ACOSH_REFINE_CH, sl);
    let (sh, sl) = muldd_acc2(xh, xl, sh, sl);
    let (sh, sl) = adddd(sh, sl, el1, el2);
    let (sh, sl) = adddd(sh, sl, l[1], l[2]);

    let (v0, v2) = fasttwosum(l[0], sh);
    let (v1, v2) = fasttwosum(v2, sl);
    let (v0, mut v1, v2) = (v0 * 2.0, v1 * 2.0, v2 * 2.0);

    let mut bits = v1.to_bits();
    if bits & (!0u64 >> 12) == 0 {
        if (v2.to_bits() ^ bits) >> 63 != 0 {
            bits -= 1;
        } else {
            bits += 1;
        }
        v1 = f64::from_bits(bits);
    }
    let t0 = v0.to_bits();
    let er = bits.wrapping_add(7) & (!0u64 >> 12);
    let de = ((t0 >> 52) & 0x7ff).wrapping_sub((bits >> 52) & 0x7ff);
    let res = v0 + v1;
    if de > 102 || er < 15 {
        return database(&t::ACOSH_DB, x, 1.0, res);
    }
    res
}

/// `acosh(x)`, correctly rounded.
pub fn acosh(x: f64) -> f64 {
    let ix = x.to_bits();

    if ix >= 0x7ff0_0000_0000_0000 {
        // Negative, or infinite, or a NaN — `ix` being unsigned puts all
        // three past this one comparison.
        let aix = ix << 1;
        if ix == 0x7ff0_0000_0000_0000 || aix > (0x7ffu64 << 53) {
            return x + x; // +inf, or a NaN to quieten
        }
        return invalid(x);
    }
    if ix <= 0x3ff0_0000_0000_0000 {
        if ix == 0x3ff0_0000_0000_0000 {
            return 0.0; // acosh(1)
        }
        return invalid(x);
    }

    let mut off = 0x3fe;
    let mut t = ix;
    let g;

    if ix < 0x3ff1_e83e_425a_ee63 {
        // Just above 1, where `sqrt(x^2 - 1)` cancels: factor out the
        // `sqrt(2(x-1))` that carries the singularity and fit the rest.
        let z = x - 1.0;
        let iz = (-0.25) / z;
        let zt = 2.0 * z;
        let sh = zt.sqrt();
        let sl = sh.mul_add(sh, -zt) * (sh * iz);
        let z2 = z * z;
        let z4 = z2 * z2;
        let ds = (sh * z).mul_add(
            ACOSH_S1[0]
                + z * (((ACOSH_S1[1] + z * ACOSH_S1[2]) + z2 * (ACOSH_S1[3] + z * ACOSH_S1[4]))
                    + z4 * ((ACOSH_S1[5] + z * ACOSH_S1[6])
                        + z2 * (ACOSH_S1[7] + z * ACOSH_S1[8]))),
            sl,
        );
        let eps = ds * f64::from_bits(0x3ccfc00000000000) - f64::from_bits(0x3970000000000000) * sh;
        let lb = sh + (ds - eps);
        let ub = sh + (ds + eps);
        if lb == ub {
            return lb;
        }
        return acosh_one(z, sh, sl);
    } else if ix < 0x405b_f000_0000_0000 {
        // Up to `0x1.bfp+6` the square root still needs its low word, and
        // the logarithm's argument is `x + sqrt(x^2 - 1)` in full.
        off = 0x3ff;
        let x2h = x * x;
        let wh = x2h - 1.0;
        let wl = x.mul_add(x, -x2h);
        let sh = wh.sqrt();
        let ish = 0.5 / wh;
        let sl = (wl - sh.mul_add(sh, -wh)) * (sh * ish);
        let (th, tl) = fasttwosum(x, sh);
        t = th.to_bits();
        g = (tl + sl) / th;
    } else if ix < 0x4087_1000_0000_0000 {
        // Past it, `x + sqrt(x^2 - 1)` is `2x` to a `double` and the
        // difference becomes an asymptotic series in `1/x^2` added to the
        // logarithm rather than to its argument.
        let z = 1.0 / (x * x);
        g = ACOSH_S2[0] + z * (ACOSH_S2[1] + z * (ACOSH_S2[2] + z * ACOSH_S2[3]));
    } else if ix < 0x40e0_1000_0000_0000 {
        let z = 1.0 / (x * x);
        g = ACOSH_S3[0] + z * (ACOSH_S3[1] + z * ACOSH_S3[2]);
    } else if ix < 0x41ea_0000_0000_0000 {
        let z = 1.0 / (x * x);
        g = ACOSH_S4[0] + z * ACOSH_S4[1];
    } else {
        g = 0.0;
    }

    let (e, i1, i2, dx, f) = fast_log_parts(f64::from_bits(t), off);
    let ed = f64::from(e);
    let lh = (ft::L1[i1][1] + ft::L2[i2][1]) + L2H * ed;
    // Four separate accumulations, in `e_acosh.c`'s own order — not the same
    // grouping `asinh` uses just above, and the difference shows in the last
    // bit.
    let mut ll = dx + L2L * ed;
    ll += g;
    ll += ft::L1[i1][0] + ft::L2[i2][0];
    ll += f;

    let eps = f64::from_bits(0x3c14a90ceafff9df);
    let lb = lh + (ll - eps);
    let ub = lh + (ll + eps);
    if lb == ub {
        return lb;
    }
    acosh_refine(x, INV_LN2 * lb)
}
