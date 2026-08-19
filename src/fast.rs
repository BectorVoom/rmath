//! Fast-path free functions configured with [`Fast`] accuracy and [`FullRange`] domain.
//!
//! These functions provide direct access to the vectorized approximation kernels (e.g.
//! Cody-Waite polynomial reductions, table-free exponentials/logarithms, etc.)
//! without needing to construct a builder at each call site.
//!
//! ```
//! use rmath::fast;
//!
//! let y = fast::exp(1.0_f64);
//! assert!((y - 1.0_f64.exp()).abs() < 1e-14);
//! ```

use crate::function::*;
use crate::policy::{Fast, FullRange};
use crate::simd::Simd;

// ─────────────────────────────────────────────────────────────────────────────
// 1-Argument Functions
// ─────────────────────────────────────────────────────────────────────────────

/// `acos(x)`, fast approximation.
#[inline(always)]
pub fn acos<V: Simd>(x: V) -> V
where
    Acos<Fast, FullRange>: Function<V::Elem>,
{
    <Acos<Fast, FullRange> as Function<V::Elem>>::eval(&Acos::default(), x)
}

/// `acosh(x)`, fast approximation.
#[inline(always)]
pub fn acosh<V: Simd>(x: V) -> V
where
    Acosh<Fast, FullRange>: Function<V::Elem>,
{
    <Acosh<Fast, FullRange> as Function<V::Elem>>::eval(&Acosh::default(), x)
}

/// `asin(x)`, fast approximation.
#[inline(always)]
pub fn asin<V: Simd>(x: V) -> V
where
    Asin<Fast, FullRange>: Function<V::Elem>,
{
    <Asin<Fast, FullRange> as Function<V::Elem>>::eval(&Asin::default(), x)
}

/// `asinh(x)`, fast approximation.
#[inline(always)]
pub fn asinh<V: Simd>(x: V) -> V
where
    Asinh<Fast, FullRange>: Function<V::Elem>,
{
    <Asinh<Fast, FullRange> as Function<V::Elem>>::eval(&Asinh::default(), x)
}

/// `atan(x)`, fast approximation.
#[inline(always)]
pub fn atan<V: Simd>(x: V) -> V
where
    Atan<Fast, FullRange>: Function<V::Elem>,
{
    <Atan<Fast, FullRange> as Function<V::Elem>>::eval(&Atan::default(), x)
}

/// `atanh(x)`, fast approximation.
#[inline(always)]
pub fn atanh<V: Simd>(x: V) -> V
where
    Atanh<Fast, FullRange>: Function<V::Elem>,
{
    <Atanh<Fast, FullRange> as Function<V::Elem>>::eval(&Atanh::default(), x)
}

/// `cbrt(x)`, fast approximation.
#[inline(always)]
pub fn cbrt<V: Simd>(x: V) -> V
where
    Cbrt<Fast, FullRange>: Function<V::Elem>,
{
    <Cbrt<Fast, FullRange> as Function<V::Elem>>::eval(&Cbrt::default(), x)
}

/// `ceil(x)`.
#[inline(always)]
pub fn ceil<V: Simd>(x: V) -> V
where
    Ceil<Fast, FullRange>: Function<V::Elem>,
{
    <Ceil<Fast, FullRange> as Function<V::Elem>>::eval(&Ceil::default(), x)
}

/// `cos(x)`, fast vectorized trigonometric approximation.
#[inline(always)]
pub fn cos<V: Simd>(x: V) -> V
where
    Cos<Fast, FullRange>: Function<V::Elem>,
{
    <Cos<Fast, FullRange> as Function<V::Elem>>::eval(&Cos::default(), x)
}

/// `cosh(x)`, fast approximation.
#[inline(always)]
pub fn cosh<V: Simd>(x: V) -> V
where
    Cosh<Fast, FullRange>: Function<V::Elem>,
{
    <Cosh<Fast, FullRange> as Function<V::Elem>>::eval(&Cosh::default(), x)
}

/// `erf(x)`, fast error function approximation.
#[inline(always)]
pub fn erf<V: Simd>(x: V) -> V
where
    Erf<Fast, FullRange>: Function<V::Elem>,
{
    <Erf<Fast, FullRange> as Function<V::Elem>>::eval(&Erf::default(), x)
}

/// `erfc(x)`, fast complementary error function approximation.
#[inline(always)]
pub fn erfc<V: Simd>(x: V) -> V
where
    Erfc<Fast, FullRange>: Function<V::Elem>,
{
    <Erfc<Fast, FullRange> as Function<V::Elem>>::eval(&Erfc::default(), x)
}

/// `exp(x)`, fast table-free exponential approximation.
#[inline(always)]
pub fn exp<V: Simd>(x: V) -> V
where
    Exp<Fast, FullRange>: Function<V::Elem>,
{
    <Exp<Fast, FullRange> as Function<V::Elem>>::eval(&Exp::default(), x)
}

/// `exp2(x)`, fast table-free base-2 exponential approximation.
#[inline(always)]
pub fn exp2<V: Simd>(x: V) -> V
where
    Exp2<Fast, FullRange>: Function<V::Elem>,
{
    <Exp2<Fast, FullRange> as Function<V::Elem>>::eval(&Exp2::default(), x)
}

/// `exp10(x)`, fast table-free base-10 exponential approximation.
#[inline(always)]
pub fn exp10<V: Simd>(x: V) -> V
where
    Exp10<Fast, FullRange>: Function<V::Elem>,
{
    <Exp10<Fast, FullRange> as Function<V::Elem>>::eval(&Exp10::default(), x)
}

/// `expm1(x)`, fast approximation for small `x`.
#[inline(always)]
pub fn expm1<V: Simd>(x: V) -> V
where
    Expm1<Fast, FullRange>: Function<V::Elem>,
{
    <Expm1<Fast, FullRange> as Function<V::Elem>>::eval(&Expm1::default(), x)
}

/// `floor(x)`.
#[inline(always)]
pub fn floor<V: Simd>(x: V) -> V
where
    Floor<Fast, FullRange>: Function<V::Elem>,
{
    <Floor<Fast, FullRange> as Function<V::Elem>>::eval(&Floor::default(), x)
}

/// `ilogb(x)`.
#[inline(always)]
pub fn ilogb<V: Simd>(x: V) -> V
where
    Ilogb<Fast, FullRange>: Function<V::Elem>,
{
    <Ilogb<Fast, FullRange> as Function<V::Elem>>::eval(&Ilogb::default(), x)
}

/// `j0(x)`, fast vectorized Bessel J0 function.
#[inline(always)]
pub fn j0<V: Simd>(x: V) -> V
where
    J0<Fast, FullRange>: Function<V::Elem>,
{
    <J0<Fast, FullRange> as Function<V::Elem>>::eval(&J0::default(), x)
}

/// `j1(x)`, fast vectorized Bessel J1 function.
#[inline(always)]
pub fn j1<V: Simd>(x: V) -> V
where
    J1<Fast, FullRange>: Function<V::Elem>,
{
    <J1<Fast, FullRange> as Function<V::Elem>>::eval(&J1::default(), x)
}

/// `lgamma(x)`, fast log-gamma function.
#[inline(always)]
pub fn lgamma<V: Simd>(x: V) -> V
where
    LGamma<Fast, FullRange>: Function<V::Elem>,
{
    <LGamma<Fast, FullRange> as Function<V::Elem>>::eval(&LGamma::default(), x)
}

/// `ln(x)`, fast natural logarithm approximation.
#[inline(always)]
pub fn ln<V: Simd>(x: V) -> V
where
    Ln<Fast, FullRange>: Function<V::Elem>,
{
    <Ln<Fast, FullRange> as Function<V::Elem>>::eval(&Ln::default(), x)
}

/// `log1p(x)`, fast approximation for `ln(1 + x)`.
#[inline(always)]
pub fn log1p<V: Simd>(x: V) -> V
where
    Log1p<Fast, FullRange>: Function<V::Elem>,
{
    <Log1p<Fast, FullRange> as Function<V::Elem>>::eval(&Log1p::default(), x)
}

/// `log2(x)`, fast base-2 logarithm approximation.
#[inline(always)]
pub fn log2<V: Simd>(x: V) -> V
where
    Log2<Fast, FullRange>: Function<V::Elem>,
{
    <Log2<Fast, FullRange> as Function<V::Elem>>::eval(&Log2::default(), x)
}

/// `log10(x)`, fast base-10 logarithm approximation.
#[inline(always)]
pub fn log10<V: Simd>(x: V) -> V
where
    Log10<Fast, FullRange>: Function<V::Elem>,
{
    <Log10<Fast, FullRange> as Function<V::Elem>>::eval(&Log10::default(), x)
}

/// `rint(x)`.
#[inline(always)]
pub fn rint<V: Simd>(x: V) -> V
where
    Rint<Fast, FullRange>: Function<V::Elem>,
{
    <Rint<Fast, FullRange> as Function<V::Elem>>::eval(&Rint::default(), x)
}

/// `round(x)`.
#[inline(always)]
pub fn round<V: Simd>(x: V) -> V
where
    Round<Fast, FullRange>: Function<V::Elem>,
{
    <Round<Fast, FullRange> as Function<V::Elem>>::eval(&Round::default(), x)
}

/// `sin(x)`, fast vectorized trigonometric sine approximation.
#[inline(always)]
pub fn sin<V: Simd>(x: V) -> V
where
    Sin<Fast, FullRange>: Function<V::Elem>,
{
    <Sin<Fast, FullRange> as Function<V::Elem>>::eval(&Sin::default(), x)
}

/// `sinh(x)`, fast hyperbolic sine approximation.
#[inline(always)]
pub fn sinh<V: Simd>(x: V) -> V
where
    Sinh<Fast, FullRange>: Function<V::Elem>,
{
    <Sinh<Fast, FullRange> as Function<V::Elem>>::eval(&Sinh::default(), x)
}

/// `sqrt(x)`, correctly-rounded square root.
#[inline(always)]
pub fn sqrt<V: Simd>(x: V) -> V
where
    Sqrt<Fast, FullRange>: Function<V::Elem>,
{
    <Sqrt<Fast, FullRange> as Function<V::Elem>>::eval(&Sqrt::default(), x)
}

/// `tan(x)`, fast vectorized tangent approximation.
#[inline(always)]
pub fn tan<V: Simd>(x: V) -> V
where
    Tan<Fast, FullRange>: Function<V::Elem>,
{
    <Tan<Fast, FullRange> as Function<V::Elem>>::eval(&Tan::default(), x)
}

/// `tanh(x)`, fast hyperbolic tangent approximation.
#[inline(always)]
pub fn tanh<V: Simd>(x: V) -> V
where
    Tanh<Fast, FullRange>: Function<V::Elem>,
{
    <Tanh<Fast, FullRange> as Function<V::Elem>>::eval(&Tanh::default(), x)
}

/// `tgamma(x)`, fast gamma function.
#[inline(always)]
pub fn tgamma<V: Simd>(x: V) -> V
where
    TGamma<Fast, FullRange>: Function<V::Elem>,
{
    <TGamma<Fast, FullRange> as Function<V::Elem>>::eval(&TGamma::default(), x)
}

/// `trunc(x)`.
#[inline(always)]
pub fn trunc<V: Simd>(x: V) -> V
where
    Trunc<Fast, FullRange>: Function<V::Elem>,
{
    <Trunc<Fast, FullRange> as Function<V::Elem>>::eval(&Trunc::default(), x)
}

/// `y0(x)`, fast vectorized Bessel Y0 function.
#[inline(always)]
pub fn y0<V: Simd>(x: V) -> V
where
    Y0<Fast, FullRange>: Function<V::Elem>,
{
    <Y0<Fast, FullRange> as Function<V::Elem>>::eval(&Y0::default(), x)
}

/// `y1(x)`, fast vectorized Bessel Y1 function.
#[inline(always)]
pub fn y1<V: Simd>(x: V) -> V
where
    Y1<Fast, FullRange>: Function<V::Elem>,
{
    <Y1<Fast, FullRange> as Function<V::Elem>>::eval(&Y1::default(), x)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-Argument Functions
// ─────────────────────────────────────────────────────────────────────────────

/// `atan2(y, x)`, fast approximation.
#[inline(always)]
pub fn atan2<V: Simd>(y: V, x: V) -> V
where
    Atan2<Fast, FullRange>: Function2<V::Elem>,
{
    <Atan2<Fast, FullRange> as Function2<V::Elem>>::eval(&Atan2::default(), y, x)
}

/// `copysign(mag, sign)`.
#[inline(always)]
pub fn copysign<V: Simd>(mag: V, sign: V) -> V
where
    CopySign<Fast, FullRange>: Function2<V::Elem>,
{
    <CopySign<Fast, FullRange> as Function2<V::Elem>>::eval(&CopySign::default(), mag, sign)
}

/// `fdim(x, y)`.
#[inline(always)]
pub fn fdim<V: Simd>(x: V, y: V) -> V
where
    Fdim<Fast, FullRange>: Function2<V::Elem>,
{
    <Fdim<Fast, FullRange> as Function2<V::Elem>>::eval(&Fdim::default(), x, y)
}

/// `fmax(x, y)`.
#[inline(always)]
pub fn fmax<V: Simd>(x: V, y: V) -> V
where
    Fmax<Fast, FullRange>: Function2<V::Elem>,
{
    <Fmax<Fast, FullRange> as Function2<V::Elem>>::eval(&Fmax::default(), x, y)
}

/// `fmin(x, y)`.
#[inline(always)]
pub fn fmin<V: Simd>(x: V, y: V) -> V
where
    Fmin<Fast, FullRange>: Function2<V::Elem>,
{
    <Fmin<Fast, FullRange> as Function2<V::Elem>>::eval(&Fmin::default(), x, y)
}

/// `fmod(x, y)`.
#[inline(always)]
pub fn fmod<V: Simd>(x: V, y: V) -> V
where
    Fmod<Fast, FullRange>: Function2<V::Elem>,
{
    <Fmod<Fast, FullRange> as Function2<V::Elem>>::eval(&Fmod::default(), x, y)
}

/// `hypot(x, y)`, fast approximation.
#[inline(always)]
pub fn hypot<V: Simd>(x: V, y: V) -> V
where
    Hypot<Fast, FullRange>: Function2<V::Elem>,
{
    <Hypot<Fast, FullRange> as Function2<V::Elem>>::eval(&Hypot::default(), x, y)
}

/// `jn(n, x)`, fast Bessel Jn function.
#[inline(always)]
pub fn jn<V: Simd>(n: V, x: V) -> V
where
    Jn<Fast, FullRange>: Function2<V::Elem>,
{
    <Jn<Fast, FullRange> as Function2<V::Elem>>::eval(&Jn::default(), n, x)
}

/// `ldexp(x, exp)`.
#[inline(always)]
pub fn ldexp<V: Simd>(x: V, exp: V) -> V
where
    Ldexp<Fast, FullRange>: Function2<V::Elem>,
{
    <Ldexp<Fast, FullRange> as Function2<V::Elem>>::eval(&Ldexp::default(), x, exp)
}

/// `nextafter(from, to)`.
#[inline(always)]
pub fn nextafter<V: Simd>(from: V, to: V) -> V
where
    NextAfter<Fast, FullRange>: Function2<V::Elem>,
{
    <NextAfter<Fast, FullRange> as Function2<V::Elem>>::eval(&NextAfter::default(), from, to)
}

/// `pow(x, y)`, fast table-free power approximation.
#[inline(always)]
pub fn pow<V: Simd>(x: V, y: V) -> V
where
    Pow<Fast, FullRange>: Function2<V::Elem>,
{
    <Pow<Fast, FullRange> as Function2<V::Elem>>::eval(&Pow::default(), x, y)
}

/// `remainder(x, y)`.
#[inline(always)]
pub fn remainder<V: Simd>(x: V, y: V) -> V
where
    Remainder<Fast, FullRange>: Function2<V::Elem>,
{
    <Remainder<Fast, FullRange> as Function2<V::Elem>>::eval(&Remainder::default(), x, y)
}

/// `scalbn(x, n)`.
#[inline(always)]
pub fn scalbn<V: Simd>(x: V, n: V) -> V
where
    Scalbn<Fast, FullRange>: Function2<V::Elem>,
{
    <Scalbn<Fast, FullRange> as Function2<V::Elem>>::eval(&Scalbn::default(), x, n)
}

/// `yn(n, x)`, fast Bessel Yn function.
#[inline(always)]
pub fn yn<V: Simd>(n: V, x: V) -> V
where
    Yn<Fast, FullRange>: Function2<V::Elem>,
{
    <Yn<Fast, FullRange> as Function2<V::Elem>>::eval(&Yn::default(), n, x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pair Functions
// ─────────────────────────────────────────────────────────────────────────────

/// `sincos(x)`, fast vectorized sine and cosine pair.
#[inline(always)]
pub fn sincos<V: Simd>(x: V) -> (V, V)
where
    SinCos<Fast, FullRange>: FunctionPair<V::Elem>,
{
    <SinCos<Fast, FullRange> as FunctionPair<V::Elem>>::eval(&SinCos::default(), x)
}

/// `frexp(x)`.
#[inline(always)]
pub fn frexp<V: Simd>(x: V) -> (V, V)
where
    Frexp<Fast, FullRange>: FunctionPair<V::Elem>,
{
    <Frexp<Fast, FullRange> as FunctionPair<V::Elem>>::eval(&Frexp::default(), x)
}

/// `modf(x)`.
#[inline(always)]
pub fn modf<V: Simd>(x: V) -> (V, V)
where
    Modf<Fast, FullRange>: FunctionPair<V::Elem>,
{
    <Modf<Fast, FullRange> as FunctionPair<V::Elem>>::eval(&Modf::default(), x)
}

/// `lgamma_r(x)`.
#[inline(always)]
pub fn lgamma_r<V: Simd>(x: V) -> (V, V)
where
    LGammaR<Fast, FullRange>: FunctionPair<V::Elem>,
{
    <LGammaR<Fast, FullRange> as FunctionPair<V::Elem>>::eval(&LGammaR::default(), x)
}

/// `remquo(x, y)`.
#[inline(always)]
pub fn remquo<V: Simd>(x: V, y: V) -> (V, V)
where
    Remquo<Fast, FullRange>: Function2Pair<V::Elem>,
{
    <Remquo<Fast, FullRange> as Function2Pair<V::Elem>>::eval(&Remquo::default(), x, y)
}
