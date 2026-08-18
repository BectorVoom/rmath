//! The vector abstraction every kernel is written against.
//!
//! A kernel is generic over [`Simd`], so one source implementation serves
//! every width of its element type — `f64`, `f64x2`, `f64x4`, `f64x8` for a
//! double-precision kernel, `f32`, `f32x4`, `f32x8`, `f32x16` for a
//! single-precision one — and a future backend only has to implement this
//! trait to inherit every function in the crate.
//!
//! # Two levels, on purpose
//!
//! [`Real`] is the *element*: `f64` or `f32`. [`Simd`] is a vector of them,
//! and `Real: Simd<Elem = Self>` — every element type is also a one-lane
//! vector of itself. That is what lets the generic kernels serve as their own
//! scalar fallback, including for the ragged tail of
//! [`crate::Function::eval_slice`], without a second implementation that could
//! drift from the first.
//!
//! Kernels are generic over *width* but not over *precision*: a
//! double-precision kernel is written `V: Simd<Elem = f64>`. That is not an
//! oversight. The algorithms genuinely differ between precisions — different
//! tables, different polynomial degrees, different reduction constants — so
//! sharing source between them would mean parameterising over every constant,
//! which buys nothing and hides which variant is running.
//!
//! # Both styles of access
//!
//! The trait deliberately exposes both:
//!
//! * **Whole-vector arithmetic** (`mul_add`, `select`, the operators, the
//!   bitwise and rounding primitives) for the parts that vectorise.
//! * **Per-lane arrays** ([`Simd::to_bits`] / [`Simd::from_bits`]) for the
//!   parts that cannot: table gathers and exponent surgery. These compile to
//!   a fixed-length, branch-free loop, which LLVM handles well — but they are
//!   the reason [`crate::policy::Fast`] exists, since avoiding the gather is
//!   the single biggest lever on throughput.

use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Shl, Shr, Sub};

#[cfg(feature = "wide")]
mod wide_backend;

mod scalar;

/// A fixed-length lane array, `[T; LANES]`.
///
/// Exists because `Simd::LANES` cannot size an array in a trait signature on
/// stable Rust; an associated type sidesteps that without `generic_const_exprs`.
pub trait Lanes<T: Copy>: Copy {
    /// The lanes, in order.
    fn as_slice(&self) -> &[T];
    /// The lanes, mutably.
    fn as_mut_slice(&mut self) -> &mut [T];
    /// An array of `T::default()`, to be filled in.
    fn filled_default() -> Self;
}

impl<T: Copy + Default, const N: usize> Lanes<T> for [T; N] {
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self
    }
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
    #[inline(always)]
    fn filled_default() -> Self {
        [T::default(); N]
    }
}

/// The per-lane boolean result of a comparison.
pub trait Mask: Copy {
    /// Lane count, equal to that of the vector that produced it.
    const LANES: usize;
    /// `[bool; LANES]`.
    type Bools: Lanes<bool>;

    /// True if any lane is set.
    fn any(self) -> bool;
    /// True if every lane is set.
    fn all(self) -> bool;
    /// Lane-wise conjunction.
    fn and(self, other: Self) -> Self;
    /// Lane-wise disjunction.
    fn or(self, other: Self) -> Self;
    /// Lane-wise negation.
    fn not(self) -> Self;
    /// Unpack to booleans, for driving a per-lane fallback.
    fn to_bools(self) -> Self::Bools;

    /// True if no lane is set.
    #[inline(always)]
    fn none(self) -> bool {
        !self.any()
    }
}

/// The unsigned integer that holds a [`Real`]'s bit pattern.
///
/// Sealed, and deliberately minimal: exactly the operations the
/// exponent-surgery functions need — `frexp`, `ldexp`, `nextafter`, `fmod`,
/// `remainder` — so that each of them is written once rather than once per
/// precision. Anything a *specific* precision's kernel needs it takes from
/// concrete `u64`/`u32` code instead.
pub trait Uint:
    Copy
    + Default
    + Eq
    + Ord
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + private::Sealed
{
    /// `0`.
    const ZERO: Self;
    /// `1`.
    const ONE: Self;

    /// `self + other`, wrapping on overflow.
    fn wrapping_add(self, other: Self) -> Self;
    /// `self - other`, wrapping on underflow.
    fn wrapping_sub(self, other: Self) -> Self;
    /// Widen a small constant.
    fn from_u32(v: u32) -> Self;
    /// Narrow to `u32`, discarding the high bits.
    fn as_u32(self) -> u32;
}

impl Uint for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        u64::wrapping_add(self, other)
    }
    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        u64::wrapping_sub(self, other)
    }
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        v as u64
    }
    #[inline(always)]
    fn as_u32(self) -> u32 {
        self as u32
    }
}

impl Uint for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        u32::wrapping_add(self, other)
    }
    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        u32::wrapping_sub(self, other)
    }
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        v
    }
    #[inline(always)]
    fn as_u32(self) -> u32 {
        self
    }
}

/// A scalar element type: `f64` or `f32`.
///
/// Sealed. Everything a kernel needs about a *specific* precision it gets from
/// concrete `f64`/`f32` code; what lives here is only what genuinely generic
/// code needs — the buffer helpers on [`crate::Function`], the special-lane
/// repair in [`patch_lanes`], and the domain predicates in
/// [`crate::kernels`].
pub trait Real: Simd<Elem = Self> + Default + PartialEq + PartialOrd + private::Sealed {
    /// The unsigned integer of the same width: `u64` for `f64`, `u32` for `f32`.
    type Uint: Uint;

    /// The widest vector of this element the enabled backend provides.
    ///
    /// What [`crate::Function::eval_slice`] steps a buffer with. Without the
    /// `wide` feature it is the element itself, and the buffer helpers
    /// degenerate to a scalar loop that still runs the same kernel source.
    type Widest: Simd<Elem = Self>;

    /// `0.0`.
    const ZERO: Self;
    /// `1.0`.
    const ONE: Self;
    /// `0.5`.
    const HALF: Self;
    /// The smallest positive *normal* value.
    const MIN_POSITIVE: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;
    /// The largest finite value.
    const MAX: Self;
    /// A quiet NaN.
    const NAN: Self;
    /// Human-readable name, for `Debug` output and test messages.
    const NAME: &'static str;

    /// Stored significand bits: 52 for `f64`, 23 for `f32`.
    const MANT_BITS: u32;
    /// Exponent field width: 11 for `f64`, 8 for `f32`.
    const EXP_BITS: u32;
    /// Exponent bias: 1023 for `f64`, 127 for `f32`.
    const EXP_BIAS: i32;
    /// `2` raised to `MANT_BITS`, the scale that normalises a subnormal.
    const TWO_POW_MANT: Self;

    /// Reinterpret as an unsigned integer.
    fn bits(self) -> Self::Uint;
    /// Reinterpret an unsigned integer as a float.
    fn of_bits(b: Self::Uint) -> Self;
    /// Widen to `f64`, exactly.
    fn to_f64(self) -> f64;
    /// Round an `f64` to this precision, once, to nearest-even.
    fn of_f64(v: f64) -> Self;
}

/// A vector of [`Real`] lanes.
///
/// Implemented by the element types themselves with `LANES == 1`, and by each
/// backend's vector types.
pub trait Simd:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The lane type.
    type Elem: Real;
    /// Number of lanes.
    const LANES: usize;
    /// The comparison result type.
    type Mask: Mask<Bools = Self::Bools>;
    /// `[Elem; LANES]`.
    type Floats: Lanes<Self::Elem>;
    /// `[Elem::Uint; LANES]`.
    type Bits: Lanes<<Self::Elem as Real>::Uint>;
    /// `[bool; LANES]`.
    type Bools: Lanes<bool>;

    /// The same lanes, at double precision.
    ///
    /// `Self` for a double-precision vector; `f64xN` for an `f32xN`. This is
    /// not a convenience — it is how the single-precision kernels are
    /// bit-exact *and* vectorised. The platform's `expf`, `logf`, `sinf` and
    /// `powf` all do their arithmetic in `double` and round once at the end,
    /// so reproducing their schedule means widening first, replaying it
    /// lane-parallel, and narrowing once. Without this the only bit-exact
    /// option would be a scalar loop.
    type Wide: Simd<Elem = f64>;

    /// Widen every lane to `f64`. Exact.
    fn widen(self) -> Self::Wide;
    /// Round every lane of a double-precision vector to this precision.
    ///
    /// One rounding, to nearest-even. For a double-precision vector this is
    /// the identity.
    fn narrow(wide: Self::Wide) -> Self;

    /// Every lane set to `v`.
    fn splat(v: Self::Elem) -> Self;
    /// Unpack the lanes.
    fn to_array(self) -> Self::Floats;
    /// Pack lanes into a vector.
    fn from_array(a: Self::Floats) -> Self;
    /// Reinterpret each lane as an unsigned integer.
    fn to_bits(self) -> Self::Bits;
    /// Reinterpret each unsigned integer as a lane.
    fn from_bits(b: Self::Bits) -> Self;

    /// `self * mul + add`, as one correctly-rounded fused operation.
    ///
    /// Must be a true FMA (one rounding). The bit-exact kernels depend on it:
    /// evaluating it as `a * b + c` rounds twice and changes the result.
    fn mul_add(self, mul: Self, add: Self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Correctly-rounded square root.
    fn sqrt(self) -> Self;

    /// Largest integer not greater than each lane.
    fn floor(self) -> Self;
    /// Smallest integer not less than each lane.
    fn ceil(self) -> Self;
    /// Each lane truncated towards zero.
    fn trunc(self) -> Self;
    /// Each lane rounded to the nearest integer, ties to even.
    ///
    /// Ties-to-even, not ties-away — this is the IEEE `roundToIntegralTiesToEven`
    /// operation and the one hardware provides. `f64::round`'s ties-away
    /// behaviour is built from it in [`crate::kernels`].
    fn round_ties_even(self) -> Self;

    /// Lane-wise bitwise AND of the representations.
    fn and_bits(self, other: Self) -> Self;
    /// Lane-wise bitwise OR of the representations.
    fn or_bits(self, other: Self) -> Self;
    /// Lane-wise bitwise XOR of the representations.
    fn xor_bits(self, other: Self) -> Self;

    /// Lane-wise `self < other`.
    fn lt_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self <= other`.
    fn le_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self > other`.
    fn gt_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self >= other`.
    fn ge_mask(self, other: Self) -> Self::Mask;
    /// Lane-wise `self == other`.
    fn eq_mask(self, other: Self) -> Self::Mask;

    /// Lane-wise `if mask { if_true } else { if_false }`.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    /// Lane-wise "is NaN", derived from the IEEE rule that NaN != NaN.
    #[inline(always)]
    fn is_nan(self) -> Self::Mask {
        self.eq_mask(self).not()
    }

    /// The sign bit of `sign`, applied to the magnitude of `self`.
    ///
    /// Pure bit manipulation, so it is exact for every input including zeros,
    /// infinities and NaN — which is what the C `copysign` contract requires.
    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        let mask = Self::splat(Self::Elem::of_bits(sign_mask::<Self::Elem>()));
        self.abs().or_bits(sign.and_bits(mask))
    }

    /// Every lane set to the same constant, widened from `f64`.
    ///
    /// A convenience for constants that are exactly representable in both
    /// precisions (small integers, powers of two). Anything else must be
    /// spelled at the precision it is used in — rounding a decimal literal
    /// through `f64` is exactly the sort of silent change this crate exists
    /// to avoid.
    #[inline(always)]
    fn splat_f64(v: f64) -> Self {
        Self::splat(Self::Elem::of_f64(v))
    }
}

/// The sign-bit pattern for an element type.
#[inline(always)]
fn sign_mask<E: Real>() -> E::Uint {
    // Built from `-0.0` rather than a literal, so it is correct for any
    // element width without a per-precision constant.
    (-E::ZERO).bits()
}

/// Recompute the lanes selected by `mask` with a scalar reference function.
///
/// The pattern every `FullRange` kernel ends with: compute the main path
/// across all lanes, then repair the rare ones. Lanes outside the mask keep
/// the vector result untouched, so this cannot perturb the common case, and
/// the loop is skipped entirely when no lane is set.
#[inline(always)]
pub fn patch_lanes<V: Simd>(
    x: V,
    y: V,
    mask: V::Mask,
    reference: impl Fn(V::Elem) -> V::Elem,
) -> V {
    if mask.none() {
        return y;
    }
    let xs = x.to_array();
    let mut ys = y.to_array();
    let flags = mask.to_bools();
    for i in 0..V::LANES {
        if flags.as_slice()[i] {
            ys.as_mut_slice()[i] = reference(xs.as_slice()[i]);
        }
    }
    V::from_array(ys)
}

/// [`patch_lanes`] for a two-argument function.
#[inline(always)]
pub fn patch_lanes2<V: Simd>(
    x: V,
    y: V,
    z: V,
    mask: V::Mask,
    reference: impl Fn(V::Elem, V::Elem) -> V::Elem,
) -> V {
    if mask.none() {
        return z;
    }
    let xs = x.to_array();
    let ys = y.to_array();
    let mut zs = z.to_array();
    let flags = mask.to_bools();
    for i in 0..V::LANES {
        if flags.as_slice()[i] {
            zs.as_mut_slice()[i] = reference(xs.as_slice()[i], ys.as_slice()[i]);
        }
    }
    V::from_array(zs)
}

/// Evaluate a scalar function on every lane.
///
/// What a kernel with no vectorised schedule for its policy falls back to.
/// See the accuracy table in the crate documentation for which functions this
/// applies to, and why.
#[inline(always)]
pub fn map_lanes<V: Simd>(x: V, f: impl Fn(V::Elem) -> V::Elem) -> V {
    let xs = x.to_array();
    let mut ys = V::Floats::filled_default();
    for i in 0..V::LANES {
        ys.as_mut_slice()[i] = f(xs.as_slice()[i]);
    }
    V::from_array(ys)
}

/// [`map_lanes`] for a two-argument function.
#[inline(always)]
pub fn map_lanes2<V: Simd>(x: V, y: V, f: impl Fn(V::Elem, V::Elem) -> V::Elem) -> V {
    let xs = x.to_array();
    let ys = y.to_array();
    let mut zs = V::Floats::filled_default();
    for i in 0..V::LANES {
        zs.as_mut_slice()[i] = f(xs.as_slice()[i], ys.as_slice()[i]);
    }
    V::from_array(zs)
}

/// [`map_lanes`] for a function returning two values.
#[inline(always)]
pub fn map_lanes_pair<V: Simd>(x: V, f: impl Fn(V::Elem) -> (V::Elem, V::Elem)) -> (V, V) {
    let xs = x.to_array();
    let mut a = V::Floats::filled_default();
    let mut b = V::Floats::filled_default();
    for i in 0..V::LANES {
        let (p, q) = f(xs.as_slice()[i]);
        a.as_mut_slice()[i] = p;
        b.as_mut_slice()[i] = q;
    }
    (V::from_array(a), V::from_array(b))
}

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for u64 {}
    impl Sealed for u32 {}
}
