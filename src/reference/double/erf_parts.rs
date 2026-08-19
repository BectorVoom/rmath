//! `erf`, and the double-double arithmetic it and `erfc` are built from.
//!
//! # Why this one is not like the others
//!
//! Everywhere else in this crate, bit-exactness is a claim about a platform:
//! reproduce glibc's operation schedule and you match glibc, and a different
//! `libm` would need a different port. `erf` is the exception. glibc's is
//! CORE-MATH's, and CORE-MATH's is **correctly rounded** — it returns the
//! representable value nearest the true `erf(x)` for every input. Correct
//! rounding is a property of the *answer*, not of the route to it, so any
//! correctly-rounded implementation agrees with any other on every input. The
//! port below is therefore bit-exact to glibc *and* to every other correctly
//! rounded `erf`, on every platform, forever.
//!
//! # The two-step scheme
//!
//! Correct rounding is reached the way CORE-MATH reaches it, and the shape is
//! what makes it vectorisable:
//!
//! 1. A **fast path** evaluates `erf` as a double-double `h + l` with a proven
//!    relative error bound `err` — about `2^-69`. That path is straight-line
//!    arithmetic over one table lookup, so it runs eight lanes wide.
//! 2. A **rounding test** asks whether `h + l` is far enough from a rounding
//!    boundary that the bound settles which way it goes: round `h + l - err`
//!    and `h + l + err` and see if they agree. They almost always do — the
//!    bound is some `2^15` times narrower than an ulp, so the test fails on
//!    roughly one input in thirty thousand.
//! 3. Those few lanes take an **accurate path**: a degree-18 double-double
//!    polynomial, and a short table of arguments that even it cannot resolve.
//!
//! Step 3 is scalar, and that is the right trade — it runs on 0.003% of lanes,
//! and writing it in vector form would slow the other 99.997% down to pay for
//! it.

use crate::tables::double::erf as t;

/// `CH + CL` is `2/sqrt(pi)` to double-double precision.
pub(crate) const CH: f64 = f64::from_bits(0x3ff20dd750429b6d);
/// The low half of `2/sqrt(pi)`. See [`CH`].
pub(crate) const CL: f64 = f64::from_bits(0x3c71ae3a914fed80);

/// The largest `x` with `erf(x) != 1` after rounding: `0x1.7afb48dc96626p+2`.
pub(crate) const ERF_MAX: u64 = 0x4017afb48dc96626;

/// Below this magnitude `erf(x)` is `2/sqrt(pi) * x` and nothing else:
/// `0x1p-61`.
pub(crate) const ERF_TINY: f64 = f64::from_bits(0x3c20000000000000);

/// The relative error the fast path holds to for `z < 1/16`: `0x1.78p-69`.
pub(crate) const ERR_SMALL: f64 = f64::from_bits(0x3ba7800000000000);
/// The relative error the fast path holds to for `z >= 1/16`: `0x1.11p-69`.
pub(crate) const ERR_TABLE: f64 = f64::from_bits(0x3ba1100000000000);

// ---------------------------------------------------------------------------
// Double-double primitives
// ---------------------------------------------------------------------------
//
// Four operations, each exact or nearly so, and each with a precondition that
// is part of its contract rather than a nicety. They are `#[inline(always)]`
// and written as expressions over `f64` so that the vector kernels can spell
// the same sequence over `V: Simd` without a second copy drifting from this
// one -- the *sequence* is what has to match, not the types.

/// `a + b` as a double-double, **assuming `|a| >= |b|`**.
///
/// Exact in round-to-nearest. Three operations; [`two_sum`] costs six and
/// drops the precondition.
#[inline(always)]
pub(crate) fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let e = hi - a; // exact
    (hi, b - e) // exact
}

/// `a + b` as a double-double, for any `a` and `b`.
#[inline(always)]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let aa = hi - b;
    let bb = hi - aa;
    (hi, (a - aa) + (b - bb))
}

/// `a * b` as a double-double. Exact, and it is the FMA that makes it so.
#[inline(always)]
pub(crate) fn a_mul(a: f64, b: f64) -> (f64, f64) {
    let hi = a * b;
    (hi, a.mul_add(b, -hi))
}

/// `a * (bh + bl)`, dropping the `a * bl` rounding error.
#[inline(always)]
pub(crate) fn s_mul(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = a_mul(a, bh);
    (hi, a.mul_add(bl, lo))
}

/// `(ah + al) * (bh + bl)`, dropping the `al * bl` term.
#[inline(always)]
pub(crate) fn d_mul(ah: f64, al: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = a_mul(ah, bh);
    let lo = ah.mul_add(bl, lo);
    (hi, al.mul_add(bh, lo))
}

/// `a + (bh + bl)`, **assuming `|a| >= |bh|`**.
#[inline(always)]
pub(crate) fn fast_sum(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = fast_two_sum(a, bh);
    (hi, lo + bl)
}

// ---------------------------------------------------------------------------
// The fast path
// ---------------------------------------------------------------------------

/// `erf(z)` as `(h, l, err)` for `0 <= z <= 0x1.7afb48dc96626p+2`.
///
/// `err` is the *relative* error bound: `|(h + l)/erf(z) - 1| < err`.
/// CORE-MATH's `cr_erf_fast`.
pub(crate) fn erf_fast(z: f64) -> (f64, f64, f64) {
    if z < 0.0625 {
        // Evaluated at zero rather than at the midpoint: `z - 1/32` would not
        // be exact for a tiny `z`, and a midpoint fit loses all its relative
        // accuracy there anyway.
        let (z2h, z2l) = a_mul(z, z);
        let z4 = z2h * z2h;
        let c9 = t::C0[7].mul_add(z2h, t::C0[6]);
        let c5 = t::C0[5].mul_add(z2h, t::C0[4]);
        let c5 = c9.mul_add(z4, c5);

        let (th, tl) = a_mul(z2h, c5);
        let (mut h, mut l) = fast_two_sum(t::C0[2], th);
        l += tl + t::C0[3];

        let h_copy = h;
        let (th, tl) = a_mul(z2h, h);
        let tl = tl + z2h.mul_add(l, t::C0[1]);
        let (h2, l2) = fast_two_sum(t::C0[0], th);
        h = h2;
        l = l2 + z2l.mul_add(h_copy, tl);

        let (h3, tl) = a_mul(h, z);
        return (h3, l.mul_add(z, tl), ERR_SMALL);
    }

    let v = (16.0 * z).floor();
    let i = (16.0 * z) as usize;
    // `z - 0.03125` is exact — 0.03125 is an integer multiple of `ulp(z)` in
    // `z`'s binade, or both are multiples of the smaller binade's ulp — and
    // subtracting `0.0625 * v` is exact for the same reason.
    let z = (z - 0.03125) - 0.0625 * v;
    let c = &t::C[i - 1];

    let z2 = z * z;
    let z4 = z2 * z2;
    let c9 = c[12].mul_add(z, c[11]);
    let c7 = c[10].mul_add(z, c[9]);
    let c5 = c[8].mul_add(z, c[7]);
    let (mut c3h, mut c3l) = fast_two_sum(c[5], z * c[6]);
    let c7 = c9.mul_add(z2, c7);

    let (h, tl) = fast_two_sum(c3h, c5 * z2);
    c3h = h;
    c3l += tl;
    let (h, tl) = fast_two_sum(c3h, c7 * z4);
    c3h = h;
    c3l += tl;

    let (th, tl) = a_mul(z, c3h);
    let (c2h, c2l) = fast_two_sum(c[4], th);
    let c2l = c2l + z.mul_add(c3l, tl);

    let (th, tl) = a_mul(z, c2h);
    let (h, l) = fast_two_sum(c[2], th);
    let l = l + (tl + z.mul_add(c2l, c[3]));

    let (th, tl) = a_mul(z, h);
    let tl = z.mul_add(l, tl);
    let (h, l) = fast_two_sum(c[0], th);
    (h, l + tl + c[1], ERR_TABLE)
}

// ---------------------------------------------------------------------------
// The accurate path
// ---------------------------------------------------------------------------

/// `erf(z)` as a double-double, for the lanes the rounding test could not
/// settle. CORE-MATH's `cr_erf_accurate`.
pub(crate) fn erf_accurate(z: f64) -> (f64, f64) {
    for e in t::EXCEPTIONS.iter() {
        if z == e[0] {
            return (e[1], e[2]);
        }
    }
    if z < 0.125 {
        return erf_accurate_tiny(z, true);
    }
    let v = (8.0 * z).floor();
    let i = (8.0 * z) as usize;
    let z = (z - 0.0625) - 0.125 * v;
    let p = &t::C2[i - 1];

    let mut h = p[26]; // degree 18
    for j in (11..=17).rev() {
        h = h.mul_add(z, p[8 + j]);
    }
    let mut l = 0.0_f64;
    for j in (8..=10).rev() {
        let (th, tl) = a_mul(h, z);
        let tl = l.mul_add(z, tl);
        let (nh, nl) = two_sum(p[8 + j], th);
        h = nh;
        l = nl + tl;
    }
    for j in (0..=7).rev() {
        let (th, tl) = a_mul(h, z);
        let tl = l.mul_add(z, tl);
        // `two_sum`, not `fast_two_sum`: for `i = 3` the degree-7 coefficient
        // is `0x1.060b78c935b8ep-13` against a degree-8 of `0x1.678b51a9c4b0ap-7`,
        // so the precondition `|a| >= |b|` does not hold.
        let (nh, nl) = two_sum(p[2 * j], th);
        h = nh;
        l = nl + (p[2 * j + 1] + tl);
    }
    (h, l)
}

/// The accurate path for `|z| < 1/8`. CORE-MATH's `cr_erf_accurate_tiny`.
///
/// `exceptions` selects whether the hard-case table is consulted: `erf` needs
/// it, `erfc` does not, because the two round at different places.
pub(crate) fn erf_accurate_tiny(z: f64, exceptions: bool) -> (f64, f64) {
    if exceptions {
        // Bisection rather than a scan: the table is sorted, and this runs on
        // the slow path of the slow path, where an extra branch costs nothing.
        let (mut i, mut j) = (0usize, t::EXCEPTIONS_TINY.len());
        while i + 1 < j {
            let k = (i + j) / 2;
            if t::EXCEPTIONS_TINY[k][0] <= z {
                i = k;
            } else {
                j = k;
            }
        }
        if z == t::EXCEPTIONS_TINY[i][0] {
            return (t::EXCEPTIONS_TINY[i][1], t::EXCEPTIONS_TINY[i][2]);
        }
    }

    let z2 = z * z;
    let mut h = t::P[14]; // degree 21
    for a in (13..=19).rev().step_by(2) {
        h = h.mul_add(z2, t::P[a / 2 + 4]);
    }
    let mut l = 0.0_f64;
    for a in [11usize, 9] {
        let (th, tl) = a_mul(h, z);
        let tl = l.mul_add(z, tl);
        let (nh, nl) = a_mul(th, z);
        h = nh;
        l = tl.mul_add(z, nl);
        let (nh, tl) = fast_two_sum(t::P[a / 2 + 4], h);
        h = nh;
        l += tl;
    }
    for a in [7usize, 5, 3, 1] {
        let (th, tl) = a_mul(h, z);
        let tl = l.mul_add(z, tl);
        let (nh, nl) = a_mul(th, z);
        h = nh;
        l = tl.mul_add(z, nl);
        let (nh, tl) = fast_two_sum(t::P[a - 1], h);
        h = nh;
        l += t::P[a] + tl;
    }
    let (h, tl) = a_mul(h, z);
    (h, l.mul_add(z, tl))
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// `erf(x)`, correctly rounded — and so bit-identical to glibc's.
pub fn erf(x: f64) -> f64 {
    let z = x.abs();
    let ux = z.to_bits();

    if ux > ERF_MAX {
        let os = 1.0f64.copysign(x);
        const INF: u64 = 0x7ff0_0000_0000_0000;
        if ux > INF {
            return x + x; // NaN, propagating the payload
        }
        if ux == INF {
            return os;
        }
        // Just short of 1: the correctly-rounded value is `nextbelow(1)`.
        return os - f64::from_bits(0x3c90000000000000) * os;
    }

    if z < ERF_TINY {
        // The code below would return `+0` for `x = -0`.
        if x == 0.0 {
            return x;
        }
        // `erf(x) = 2/sqrt(pi) * x + O(x^3)`, and the cubic term is 2^-123 of
        // the linear one here. Scaling by 2^106 keeps the double-double
        // correction out of the subnormal range, where it would be flushed.
        let y = CH * x;
        let sx = x * f64::from_bits(0x4690000000000000); // 0x1p106
        let (h, l) = a_mul(CH, sx);
        let l = CL.mul_add(sx, l);
        // `h - y*2^106` is exact: the two are within an ulp of each other.
        let l = l + (h - y * f64::from_bits(0x4690000000000000));
        return l.mul_add(f64::from_bits(0x3950000000000000), y); // 0x1p-106
    }

    let (h, l, err) = erf_fast(z);
    // Re-apply the sign by bit surgery: `erf` is odd, so the whole
    // double-double negates.
    let sign = x.to_bits() & 0x8000_0000_0000_0000;
    let uf = f64::from_bits(h.to_bits() ^ sign);
    let vf = f64::from_bits(l.to_bits() ^ sign);

    let left = uf + err.mul_add(-uf, vf);
    let right = uf + err.mul_add(uf, vf);
    if left == right {
        return left;
    }

    let (h, l) = erf_accurate(z);
    if x >= 0.0 { h + l } else { (-h) + (-l) }
}
