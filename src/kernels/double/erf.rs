//! The error function.
//!
//! # What `BitExact` means here, and why it is stronger than usual
//!
//! Everywhere else in this crate, bit-exactness is a claim about *this*
//! platform's `libm`. For `erf` it is not: glibc's `erf` is correctly rounded,
//! so the value it returns is a property of the mathematics rather than of the
//! algorithm, and the kernel below matches it — and any other correctly
//! rounded `erf` — on every input, on every platform. See
//! [`crate::reference::double::erf`].
//!
//! # The shape, and why it vectorises
//!
//! * The **fast path** produces `erf(|x|)` as a double-double `h + l` with a
//!   proven relative error near `2^-69`. It is straight-line arithmetic over
//!   one 13-wide table gather, so it runs at full lane width.
//! * A **rounding test** compares `h + l - err` with `h + l + err`. When they
//!   round the same way — which is about 99.997% of inputs — that value is the
//!   correctly rounded answer and the kernel is done.
//! * The lanes where they disagree go to the scalar accurate path through
//!   [`crate::simd::patch_lanes`], the same mechanism every other kernel uses
//!   for its rare inputs. Here the "rare input" is not an infinity or a
//!   subnormal but a *hard rounding case*, which is a nicer thing to have to
//!   fall back on: it is decided by a test rather than by a guess about where
//!   the algorithm stops being valid.
//!
//! # `Fast`
//!
//! Drops the rounding test and returns `h + l` rounded once. Measured error
//! below 0.51 ulp, which is to say it is correctly rounded on all but about
//! one input in thirty thousand — for a third to a half of the cost, since no
//! lane ever leaves the vector.
//!
//! `Finite` means `0x1p-61 <= |x| <= 0x1.7afb48dc96626p+2`. Below that range
//! `erf(x)` is `2/sqrt(pi) x` and above it `erf(x)` is `+-1`, so the guarantee
//! is cheap to keep and the kernel does not test for either.

use crate::kernels::double::dd::{a_mul, fast_two_sum};
use crate::policy::{Accuracy, Domain};
use crate::reference::double::erf as reference;
use crate::reference::double::erf_parts as r;
use crate::simd::{Lanes, Mask, Simd, patch_lanes};
use crate::tables::double::erf as t;

/// `0x1.7afb48dc96626p+2`, above which `erf` rounds to `+-1`.
const ERF_MAX: f64 = f64::from_bits(0x4017afb48dc96626);

/// `erf(x)` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    let z = x.abs();
    let (h, l, err) = fast(z);

    // `erf` is odd, so the sign of `x` applies to the whole double-double by
    // bit surgery — no branch, no negation of a NaN payload.
    let sgn = x.and_bits(V::splat(f64::from_bits(0x8000_0000_0000_0000)));
    let uf = h.xor_bits(sgn);
    let vf = l.xor_bits(sgn);

    if !A::BIT_EXACT {
        let y = uf + vf;
        return if D::CHECKED {
            patch_lanes(x, y, outside(z), reference)
        } else {
            y
        };
    }

    let left = uf + err.mul_add(-uf, vf);
    let right = uf + err.mul_add(uf, vf);
    // Lanes the error bound does not settle, plus the lanes the fast path was
    // never valid for. Both go to the same scalar reference, which handles
    // each properly, so they are one mask rather than two passes.
    let mut repair = left.eq_mask(right).not();
    if D::CHECKED {
        repair = repair.or(outside(z));
    }
    patch_lanes(x, left, repair, reference)
}

/// Lanes outside the fast path's range: below `0x1p-61`, above
/// `0x1.7afb48dc96626p+2`, or NaN.
///
/// Spelled as the negation of "inside" so NaN — which compares false against
/// everything — is caught rather than waved through.
#[inline(always)]
fn outside<V: Simd<Elem = f64>>(z: V) -> V::Mask {
    z.ge_mask(V::splat(r::ERF_TINY))
        .and(z.le_mask(V::splat(ERF_MAX)))
        .not()
}

/// `erf(z)` as the double-double `(h, l)` and its relative error bound.
///
/// Exposed so [`super::erfc`] can share it: `erfc(x) = 1 -+ erf(|x|)` for
/// arguments small enough that the subtraction does not cancel, and running
/// `erf`'s fast path there rather than a second polynomial is what keeps the
/// two functions agreeing on `erf(x) + erfc(x) = 1`.
#[inline(always)]
pub(crate) fn fast_parts<V: Simd<Elem = f64>>(z: V) -> (V, V, V) {
    fast(z)
}

/// `erf(z)` as `(h, l, err)`, lane-for-lane identical to
/// [`crate::reference::double::erf_parts::erf_fast`].
///
/// Both of that function's branches are evaluated and blended, rather than one
/// of them being deferred to the scalar path. The `z < 1/16` branch is not a
/// rare case to be repaired — a buffer of small arguments would be *entirely*
/// that branch — and it is the cheaper of the two, so computing it
/// unconditionally costs less than losing the vector would.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(z: V) -> (V, V, V) {
    let (sh, sl) = small(z);
    let (th, tl) = table(z);
    let is_small = z.lt_mask(V::splat(0.0625));
    (
        V::select(is_small, sh, th),
        V::select(is_small, sl, tl),
        V::select(is_small, V::splat(r::ERR_SMALL), V::splat(r::ERR_TABLE)),
    )
}

/// The `z < 1/16` branch: a series in `z^2` evaluated *at zero*.
///
/// Not at the interval midpoint, which is what every other interval uses:
/// `z - 1/32` is not exact for a tiny `z`, and a midpoint fit has no relative
/// accuracy near the origin where `erf` has a zero.
#[inline(always)]
fn small<V: Simd<Elem = f64>>(z: V) -> (V, V) {
    let c0 = &t::C0;
    let (z2h, z2l) = a_mul(z, z);
    let z4 = z2h * z2h;
    let c9 = V::splat(c0[7]).mul_add(z2h, V::splat(c0[6]));
    let c5 = V::splat(c0[5]).mul_add(z2h, V::splat(c0[4]));
    let c5 = c9.mul_add(z4, c5);

    let (th, tl) = a_mul(z2h, c5);
    let (h, l) = fast_two_sum(V::splat(c0[2]), th);
    let l = l + (tl + V::splat(c0[3]));

    let h_copy = h;
    let (th, tl) = a_mul(z2h, h);
    let tl = tl + z2h.mul_add(l, V::splat(c0[1]));
    let (h, l) = fast_two_sum(V::splat(c0[0]), th);
    let l = l + z2l.mul_add(h_copy, tl);

    let (h, tl) = a_mul(h, z);
    (h, l.mul_add(z, tl))
}

/// The `1/16 <= z` branch: a degree-10 minimax polynomial per sixteenth.
#[inline(always)]
fn table<V: Simd<Elem = f64>>(z: V) -> (V, V) {
    let scaled = z * V::splat(16.0);
    let v = scaled.floor();
    // The gather. `i` is clamped rather than trusted: the `z < 1/16` lanes
    // index 0 and the out-of-range lanes index anything at all, and both are
    // discarded downstream — but they must not index out of the table first.
    let c = gather::<V>(&scaled);

    // `z - 0.03125` is exact, and so is `- 0.0625 * v`: both constants are
    // integer multiples of `ulp(z)` in `z`'s binade, or `z` is a multiple of
    // theirs. So the reduced argument carries no rounding error into a
    // polynomial that assumes it has none.
    let z = (z - V::splat(0.03125)) - V::splat(0.0625) * v;

    let z2 = z * z;
    let z4 = z2 * z2;
    let c9 = c[12].mul_add(z, c[11]);
    let c7 = c[10].mul_add(z, c[9]);
    let c5 = c[8].mul_add(z, c[7]);
    let (mut c3h, mut c3l) = fast_two_sum(c[5], z * c[6]);
    let c7 = c9.mul_add(z2, c7);

    let (nh, tl) = fast_two_sum(c3h, c5 * z2);
    c3h = nh;
    c3l = c3l + tl;
    let (nh, tl) = fast_two_sum(c3h, c7 * z4);
    c3h = nh;
    c3l = c3l + tl;

    let (th, tl) = a_mul(z, c3h);
    let (c2h, c2l) = fast_two_sum(c[4], th);
    let c2l = c2l + z.mul_add(c3l, tl);

    let (th, tl) = a_mul(z, c2h);
    let (h, l) = fast_two_sum(c[2], th);
    let l = l + (tl + z.mul_add(c2l, c[3]));

    let (th, tl) = a_mul(z, h);
    let tl = z.mul_add(l, tl);
    let (h, l) = fast_two_sum(c[0], th);
    (h, l + (tl + c[1]))
}

/// The thirteen coefficients of `C[floor(16 z) - 1]`, one vector each.
///
/// Thirteen scalar loads per lane, and the one part of this kernel that does
/// not vectorise — the same trade every table-driven kernel here makes. The
/// loop is fixed-length and branch-free, which is the shape LLVM handles well.
#[inline(always)]
fn gather<V: Simd<Elem = f64>>(scaled: &V) -> [V; 13] {
    let s = scaled.to_array();
    let mut out = [V::Floats::filled_default(); 13];
    for lane in 0..V::LANES {
        // `f64 as usize` saturates in Rust, so an infinite or NaN lane lands
        // at one end of the range rather than reading off the table.
        let i = (s.as_slice()[lane] as usize).clamp(1, t::C.len());
        let row = &t::C[i - 1];
        for (k, o) in out.iter_mut().enumerate() {
            o.as_mut_slice()[lane] = row[k];
        }
    }
    out.map(V::from_array)
}
