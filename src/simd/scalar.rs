//! `f64` and `f32` as one-lane vectors.
//!
//! This is not a courtesy implementation. It is what makes the crate work
//! without any SIMD backend at all, and it is what
//! [`crate::Function::eval_slice`] runs on the ragged tail of a slice — so the
//! tail is handled by the *same* kernel source as the body, not by a second
//! hand-written path that could drift from it. It is also the entire
//! implementation of the scalar-fallback kernels: a function whose bit-exact
//! schedule has not been vectorised runs here, one lane at a time, and is
//! still exactly the function it claims to be.

use super::{Mask, Real, Simd};

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
    #[inline(always)]
    fn to_bitmask(self) -> u32 {
        self as u32
    }
}

macro_rules! impl_scalar {
    (
        $elem:ty, $uint:ty, $name:literal, $widest:ty,
        $signmask:literal, $mant:literal, $expb:literal, $bias:literal
    ) => {
        impl Simd for $elem {
            type Elem = $elem;
            const LANES: usize = 1;
            type Mask = bool;
            type Floats = [$elem; 1];
            type Bits = [$uint; 1];
            type Bools = [bool; 1];
            type Wide = f64;

            #[inline(always)]
            fn widen(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn narrow(wide: f64) -> Self {
                wide as $elem
            }

            #[inline(always)]
            fn splat(v: $elem) -> Self {
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
                [<$elem>::to_bits(self)]
            }
            #[inline(always)]
            fn from_bits(b: Self::Bits) -> Self {
                <$elem>::from_bits(b[0])
            }

            #[inline(always)]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                // A genuine fused multiply-add: one rounding, from the
                // hardware instruction where there is one and a
                // correctly-rounded software routine where there is not. Both
                // satisfy the bit-exact kernels' requirement; only
                // `self * mul + add` would not.
                <$elem>::mul_add(self, mul, add)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                // Bit clear rather than the `std` method, and identical for
                // every input, NaN payloads included.
                <$elem>::from_bits(<$elem>::to_bits(self) & !$signmask)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$elem>::sqrt(self)
            }

            #[inline(always)]
            fn floor(self) -> Self {
                <$elem>::floor(self)
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                <$elem>::ceil(self)
            }
            #[inline(always)]
            fn trunc(self) -> Self {
                <$elem>::trunc(self)
            }
            #[inline(always)]
            fn round_ties_even(self) -> Self {
                <$elem>::round_ties_even(self)
            }

            #[inline(always)]
            fn and_bits(self, other: Self) -> Self {
                <$elem>::from_bits(<$elem>::to_bits(self) & <$elem>::to_bits(other))
            }
            #[inline(always)]
            fn or_bits(self, other: Self) -> Self {
                <$elem>::from_bits(<$elem>::to_bits(self) | <$elem>::to_bits(other))
            }
            #[inline(always)]
            fn xor_bits(self, other: Self) -> Self {
                <$elem>::from_bits(<$elem>::to_bits(self) ^ <$elem>::to_bits(other))
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

        impl Real for $elem {
            type Uint = $uint;
            type Widest = $widest;

            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const HALF: Self = 0.5;
            const MIN_POSITIVE: Self = <$elem>::MIN_POSITIVE;
            const INFINITY: Self = <$elem>::INFINITY;
            const NEG_INFINITY: Self = <$elem>::NEG_INFINITY;
            const MAX: Self = <$elem>::MAX;
            const NAN: Self = <$elem>::NAN;
            const NAME: &'static str = $name;

            const MANT_BITS: u32 = $mant;
            const EXP_BITS: u32 = $expb;
            const EXP_BIAS: i32 = $bias;
            const TWO_POW_MANT: Self = <$elem>::from_bits((($bias + $mant) as $uint) << $mant);

            #[inline(always)]
            fn bits(self) -> $uint {
                <$elem>::to_bits(self)
            }
            #[inline(always)]
            fn of_bits(b: $uint) -> Self {
                <$elem>::from_bits(b)
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn of_f64(v: f64) -> Self {
                v as $elem
            }
        }
    };
}

#[cfg(feature = "wide")]
impl_scalar!(
    f64,
    u64,
    "f64",
    wide::f64x8,
    0x8000_0000_0000_0000u64,
    52,
    11,
    1023
);
#[cfg(feature = "wide")]
impl_scalar!(f32, u32, "f32", wide::f32x8, 0x8000_0000u32, 23, 8, 127);

#[cfg(not(feature = "wide"))]
impl_scalar!(f64, u64, "f64", f64, 0x8000_0000_0000_0000u64, 52, 11, 1023);
#[cfg(not(feature = "wide"))]
impl_scalar!(f32, u32, "f32", f32, 0x8000_0000u32, 23, 8, 127);
