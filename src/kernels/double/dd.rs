//! Double-double arithmetic, in vector form.
//!
//! A double-double is an unevaluated sum `h + l` with `|l| <= ulp(h)/2`, which
//! carries about 106 significand bits. `erf` and `erfc` need it because they
//! are *correctly rounded*: to decide the last bit of a `double` you have to
//! know the true value to rather more than a `double`'s worth of digits, and
//! then prove you do.
//!
//! Each operation below is exact, or exact but for a stated dropped term, and
//! each has a precondition that is part of its contract rather than a nicety —
//! `fast_two_sum` is wrong without `|a| >= |b|`, and it is used anyway,
//! wherever that ordering is known, because it costs half what `two_sum`
//! costs.
//!
//! These are the same sequences [`crate::reference::double::erf_parts`] spells
//! over `f64`, and deliberately so. Every IEEE-754 operation rounds
//! identically at any lane count, so replaying the schedule lane-parallel
//! returns the scalar bits exactly — which is the whole reason a bit-exact
//! vector kernel is possible at all.

use crate::simd::Simd;

/// `a + b` as a double-double, **assuming `|a| >= |b|`**. Exact.
#[inline(always)]
pub(crate) fn fast_two_sum<V: Simd>(a: V, b: V) -> (V, V) {
    let hi = a + b;
    let e = hi - a;
    (hi, b - e)
}

/// `a + b` as a double-double, for any `a` and `b`. Exact, at twice the cost.
///
/// Unused by the vector kernels as they stand — every double-double sum they
/// perform has a known ordering — but kept, because the scalar reference's
/// accurate paths use it and a set of primitives with one member missing is
/// the kind of gap that gets filled wrongly later.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn two_sum<V: Simd>(a: V, b: V) -> (V, V) {
    let hi = a + b;
    let aa = hi - b;
    let bb = hi - aa;
    (hi, (a - aa) + (b - bb))
}

/// `a * b` as a double-double. Exact, and it is the FMA that makes it so.
#[inline(always)]
pub(crate) fn a_mul<V: Simd>(a: V, b: V) -> (V, V) {
    let hi = a * b;
    (hi, a.mul_add(b, -hi))
}

/// `a * (bh + bl)`, dropping the rounding error of `a * bl`.
#[inline(always)]
pub(crate) fn s_mul<V: Simd>(a: V, bh: V, bl: V) -> (V, V) {
    let (hi, lo) = a_mul(a, bh);
    (hi, a.mul_add(bl, lo))
}

/// `(ah + al) * (bh + bl)`, dropping the `al * bl` term.
#[inline(always)]
pub(crate) fn d_mul<V: Simd>(ah: V, al: V, bh: V, bl: V) -> (V, V) {
    let (hi, lo) = a_mul(ah, bh);
    let lo = ah.mul_add(bl, lo);
    (hi, al.mul_add(bh, lo))
}

/// `a + (bh + bl)`, **assuming `|a| >= |bh|`**.
#[inline(always)]
pub(crate) fn fast_sum<V: Simd>(a: V, bh: V, bl: V) -> (V, V) {
    let (hi, lo) = fast_two_sum(a, bh);
    (hi, lo + bl)
}
