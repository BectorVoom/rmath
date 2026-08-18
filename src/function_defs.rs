// The function catalogue: one macro invocation per function.
//
// `include!`d into `function.rs` rather than kept as a submodule, so the
// generated types land at `rmath::function::*` where callers expect them,
// while the file that lists them stays separate from the machinery that
// generates them.

math_fn! {
    /// `e^x`.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let f = Exp::new();
    /// assert_eq!(f.eval(0.0_f64), 1.0);
    /// assert_eq!(f.eval(0.0_f32), 1.0);
    /// ```
    name: Exp,
    builder: ExpBuilder,
    free: exp,
    k64: crate::kernels::double::exp,
    k32: crate::kernels::single::exp,
    /// `e^x`, bit-exact and safe on any input.
    ///
    /// The shorthand for `Exp::new().eval(x)`.
}

math_fn! {
    /// `2^x`.
    name: Exp2,
    builder: Exp2Builder,
    free: exp2,
    k64: crate::kernels::double::exp2,
    k32: crate::kernels::single::exp2,
    /// `2^x`, bit-exact and safe on any input.
}

math_fn! {
    /// Natural logarithm.
    name: Ln,
    builder: LnBuilder,
    free: ln,
    k64: crate::kernels::double::ln,
    k32: crate::kernels::single::ln,
    /// `ln(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Cube root.
    ///
    /// The accuracy axis is accepted but has no effect; see
    /// [`crate::kernels::double::cbrt`].
    name: Cbrt,
    builder: CbrtBuilder,
    free: cbrt,
    k64: crate::kernels::double::cbrt,
    k32: crate::kernels::single::cbrt,
    /// `x^(1/3)`, bit-exact and safe on any input.
}

math_fn! {
    /// Square root. Always correctly rounded; both policy axes are no-ops.
    name: Sqrt,
    builder: SqrtBuilder,
    free: sqrt,
    k64: crate::kernels::exact::sqrt,
    k32: crate::kernels::exact::sqrt,
    /// `sqrt(x)`, correctly rounded.
}

// ---------------------------------------------------------------------------
// Exact: rounding, sign and exponent manipulation.
//
// One kernel serves both precisions, because there is no approximation in any
// of them to specialise. See `crate::kernels::exact`.
// ---------------------------------------------------------------------------

math_fn! {
    /// Largest integer not greater than `x`.
    name: Floor,
    builder: FloorBuilder,
    free: floor,
    k64: crate::kernels::exact::floor,
    k32: crate::kernels::exact::floor,
    /// `floor(x)`, exact.
}

math_fn! {
    /// Smallest integer not less than `x`.
    name: Ceil,
    builder: CeilBuilder,
    free: ceil,
    k64: crate::kernels::exact::ceil,
    k32: crate::kernels::exact::ceil,
    /// `ceil(x)`, exact.
}

math_fn! {
    /// `x` rounded to the nearest integer, ties away from zero.
    name: Round,
    builder: RoundBuilder,
    free: round,
    k64: crate::kernels::exact::round,
    k32: crate::kernels::exact::round,
    /// `round(x)`, ties away from zero, exact.
}

math_fn! {
    /// `x` truncated towards zero.
    name: Trunc,
    builder: TruncBuilder,
    free: trunc,
    k64: crate::kernels::exact::trunc,
    k32: crate::kernels::exact::trunc,
    /// `trunc(x)`, exact.
}

math_fn2! {
    /// The magnitude of `x` with the sign of `y`.
    name: CopySign,
    builder: CopySignBuilder,
    free: copysign,
    k64: crate::kernels::exact::copysign,
    k32: crate::kernels::exact::copysign,
    /// `copysign(x, y)`, exact for every input including zeros and NaN.
}

math_fn2! {
    /// `x` reduced modulo `y`, with the sign of `x`.
    name: Fmod,
    builder: FmodBuilder,
    free: fmod,
    k64: crate::kernels::exact::fmod,
    k32: crate::kernels::exact::fmod,
    /// `fmod(x, y)`, exact.
}

math_fn2! {
    /// The IEEE-754 remainder of `x` and `y`.
    name: Remainder,
    builder: RemainderBuilder,
    free: remainder,
    k64: crate::kernels::exact::remainder,
    k32: crate::kernels::exact::remainder,
    /// `remainder(x, y)`, exact.
}

math_fn2! {
    /// The next representable value after `x` towards `y`.
    name: NextAfter,
    builder: NextAfterBuilder,
    free: nextafter,
    k64: crate::kernels::exact::nextafter,
    k32: crate::kernels::exact::nextafter,
    /// `nextafter(x, y)`, exact.
}

math_fn2! {
    /// `x * 2^n`, with `n` carried in float lanes.
    ///
    /// See [`crate::kernels::exact::ldexp`] for why the exponent is a float.
    name: Ldexp,
    builder: LdexpBuilder,
    free: ldexp,
    k64: crate::kernels::exact::ldexp,
    k32: crate::kernels::exact::ldexp,
    /// `ldexp(x, n)`, exact.
}

math_fn_pair! {
    /// Split `x` into a significand in `[0.5, 1)` and a power of two.
    name: Frexp,
    builder: FrexpBuilder,
    free: frexp,
    k64: crate::kernels::exact::frexp,
    k32: crate::kernels::exact::frexp,
    /// `frexp(x)`, returning `(fraction, exponent)`. Exact.
}

// ---------------------------------------------------------------------------
// Power and logarithm extensions.
// ---------------------------------------------------------------------------

math_fn! {
    /// Base-2 logarithm.
    name: Log2,
    builder: Log2Builder,
    free: log2,
    k64: crate::kernels::double::logx::log2,
    k32: crate::kernels::single::log2,
    /// `log2(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Base-10 logarithm.
    name: Log10,
    builder: Log10Builder,
    free: log10,
    k64: crate::kernels::double::logx::log10,
    k32: crate::kernels::single::log10,
    /// `log10(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// `ln(1 + x)`, accurate for small `x`.
    name: Log1p,
    builder: Log1pBuilder,
    free: log1p,
    k64: crate::kernels::double::logx::log1p,
    k32: crate::kernels::single::log1p,
    /// `ln(1 + x)`, bit-exact and safe on any input.
}

math_fn! {
    /// `e^x - 1`, accurate for small `x`.
    name: Expm1,
    builder: Expm1Builder,
    free: expm1,
    k64: crate::kernels::double::expm1,
    k32: crate::kernels::single::expm1,
    /// `e^x - 1`, bit-exact and safe on any input.
}

math_fn2! {
    /// `x` raised to the power `y`.
    name: Pow,
    builder: PowBuilder,
    free: pow,
    k64: crate::kernels::double::pow,
    k32: crate::kernels::single::pow,
    /// `x^y`, bit-exact and safe on any input.
}

math_fn2! {
    /// `sqrt(x^2 + y^2)`, without intermediate overflow.
    name: Hypot,
    builder: HypotBuilder,
    free: hypot,
    k64: crate::kernels::double::hypot,
    k32: crate::kernels::single::hypot,
    /// `hypot(x, y)`, bit-exact and safe on any input.
}

// ---------------------------------------------------------------------------
// Trigonometric family.
// ---------------------------------------------------------------------------

math_fn! {
    /// Sine. Argument in radians.
    name: Sin,
    builder: SinBuilder,
    free: sin,
    k64: crate::kernels::double::trig::sin,
    k32: crate::kernels::single::sin,
    /// `sin(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Cosine. Argument in radians.
    name: Cos,
    builder: CosBuilder,
    free: cos,
    k64: crate::kernels::double::trig::cos,
    k32: crate::kernels::single::cos,
    /// `cos(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Tangent. Argument in radians.
    name: Tan,
    builder: TanBuilder,
    free: tan,
    k64: crate::kernels::double::trig::tan,
    k32: crate::kernels::single::tan,
    /// `tan(x)`, bit-exact and safe on any input.
}

math_fn_pair! {
    /// Sine and cosine of the same argument.
    ///
    /// Cheaper than calling both: the argument reduction and both polynomials
    /// are shared, so the pair costs one blend more than either alone.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let (s, c) = SinCos::new().eval(0.0_f64);
    /// assert_eq!((s, c), (0.0, 1.0));
    /// ```
    name: SinCos,
    builder: SinCosBuilder,
    free: sincos,
    k64: crate::kernels::double::trig::sincos,
    k32: crate::kernels::single::sincos,
    /// `(sin(x), cos(x))`, bit-exact and safe on any input.
}

math_fn! {
    /// Arc sine, in radians.
    name: Asin,
    builder: AsinBuilder,
    free: asin,
    k64: crate::kernels::double::invtrig::asin,
    k32: crate::kernels::single::asin,
    /// `asin(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Arc cosine, in radians.
    name: Acos,
    builder: AcosBuilder,
    free: acos,
    k64: crate::kernels::double::invtrig::acos,
    k32: crate::kernels::single::acos,
    /// `acos(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Arc tangent, in radians.
    name: Atan,
    builder: AtanBuilder,
    free: atan,
    k64: crate::kernels::double::invtrig::atan,
    k32: crate::kernels::single::atan,
    /// `atan(x)`, bit-exact and safe on any input.
}

math_fn2! {
    /// Two-argument arc tangent: the angle of the point `(x, y)`.
    ///
    /// Argument order follows C and Rust: `atan2(y, x)`.
    name: Atan2,
    builder: Atan2Builder,
    free: atan2,
    k64: crate::kernels::double::invtrig::atan2,
    k32: crate::kernels::single::atan2,
    /// `atan2(y, x)`, bit-exact and safe on any input.
}

// ---------------------------------------------------------------------------
// Hyperbolic family.
// ---------------------------------------------------------------------------

math_fn! {
    /// Hyperbolic sine.
    name: Sinh,
    builder: SinhBuilder,
    free: sinh,
    k64: crate::kernels::double::hyper::sinh,
    k32: crate::kernels::single::sinh,
    /// `sinh(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Hyperbolic cosine.
    name: Cosh,
    builder: CoshBuilder,
    free: cosh,
    k64: crate::kernels::double::hyper::cosh,
    k32: crate::kernels::single::cosh,
    /// `cosh(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Hyperbolic tangent.
    name: Tanh,
    builder: TanhBuilder,
    free: tanh,
    k64: crate::kernels::double::hyper::tanh,
    k32: crate::kernels::single::tanh,
    /// `tanh(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Inverse hyperbolic sine.
    name: Asinh,
    builder: AsinhBuilder,
    free: asinh,
    k64: crate::kernels::double::hyper::asinh,
    k32: crate::kernels::single::asinh,
    /// `asinh(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Inverse hyperbolic cosine.
    name: Acosh,
    builder: AcoshBuilder,
    free: acosh,
    k64: crate::kernels::double::hyper::acosh,
    k32: crate::kernels::single::acosh,
    /// `acosh(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Inverse hyperbolic tangent.
    ///
    /// Matches Rust's `atanh`, which is not the C library's; see
    /// [`crate::reference::double::atanh`].
    name: Atanh,
    builder: AtanhBuilder,
    free: atanh,
    k64: crate::kernels::double::hyper::atanh,
    k32: crate::kernels::single::atanh,
    /// `atanh(x)`, bit-exact and safe on any input.
}

// ---------------------------------------------------------------------------
// Gamma.
// ---------------------------------------------------------------------------

math_fn! {
    /// The natural logarithm of `|Gamma(x)|`.
    ///
    /// Both policy axes are no-ops; see [`crate::kernels::double::gamma`] for
    /// why this function makes an accuracy claim rather than a bit-exactness
    /// one.
    name: LGamma,
    builder: LGammaBuilder,
    free: lgamma,
    k64: crate::kernels::double::gamma::lgamma,
    k32: crate::kernels::single::lgamma,
    /// `ln|Gamma(x)|`, safe on any input.
}

math_fn! {
    /// The Gamma function.
    ///
    /// Both policy axes are no-ops; see [`crate::kernels::double::gamma`].
    name: TGamma,
    builder: TGammaBuilder,
    free: tgamma,
    k64: crate::kernels::double::gamma::tgamma,
    k32: crate::kernels::single::tgamma,
    /// `Gamma(x)`, safe on any input.
}
