//! `f64` as a one-lane vector.
//!
//! This is not a courtesy implementation. It is what makes the crate work
//! without any SIMD backend at all, and it is what
//! [`crate::Function::eval_slice`] runs on the ragged tail of a slice — so the
//! tail is handled by the *same* kernel source as the body, not by a second
//! hand-written path that could drift from it.

use super::{Mask, Simd};

impl Mask for bool {
    const LANES: usize = 1;
    type Bools = [bool; 1];

    #[inline(always)]
    fn any(self) -> bool {
        self
    }
    #[inline(always)]
    fn all(self) -> bool {
        self
    }
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        self & other
    }
    #[inline(always)]
    fn or(self, other: Self) -> Self {
        self | other
    }
    #[inline(always)]
    fn not(self) -> Self {
        !self
    }
    #[inline(always)]
    fn to_bools(self) -> Self::Bools {
        [self]
    }
}

impl Simd for f64 {
    const LANES: usize = 1;
    type Mask = bool;
    type Floats = [f64; 1];
    type Bits = [u64; 1];
    type Bools = [bool; 1];

    #[inline(always)]
    fn splat(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn to_array(self) -> Self::Floats {
        [self]
    }
    #[inline(always)]
    fn from_array(a: Self::Floats) -> Self {
        a[0]
    }
    #[inline(always)]
    fn to_bits(self) -> Self::Bits {
        [f64::to_bits(self)]
    }
    #[inline(always)]
    fn from_bits(b: Self::Bits) -> Self {
        f64::from_bits(b[0])
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        // `f64::mul_add` is a genuine fused multiply-add: one rounding, from
        // the hardware instruction where there is one and a correctly-rounded
        // software routine where there is not. Both satisfy the bit-exact
        // kernels' requirement; only `self * mul + add` would not.
        f64::mul_add(self, mul, add)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        // Bit clear rather than `f64::abs`, which is `std`-only on older
        // toolchains. Identical for every input, NaN payloads included.
        f64::from_bits(f64::to_bits(self) & 0x7fff_ffff_ffff_ffff)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline(always)]
    fn lt_mask(self, other: Self) -> bool {
        self < other
    }
    #[inline(always)]
    fn le_mask(self, other: Self) -> bool {
        self <= other
    }
    #[inline(always)]
    fn gt_mask(self, other: Self) -> bool {
        self > other
    }
    #[inline(always)]
    fn ge_mask(self, other: Self) -> bool {
        self >= other
    }
    #[inline(always)]
    fn eq_mask(self, other: Self) -> bool {
        self == other
    }

    #[inline(always)]
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
}
