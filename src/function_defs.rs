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
    /// `10^x`.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let f = Exp10::new();
    /// assert_eq!(f.eval(3.0_f64), 1000.0);
    /// assert_eq!(f.eval(3.0_f32), 1000.0);
    /// ```
    name: Exp10,
    builder: Exp10Builder,
    free: exp10,
    k64: crate::kernels::double::exp10,
    k32: crate::kernels::single::exp10,
    /// `10^x`, bit-exact and safe on any input.
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

// ---------------------------------------------------------------------------
// The error function.
// ---------------------------------------------------------------------------

math_fn! {
    /// The error function, `2/sqrt(pi) * integral of e^(-t^2) from 0 to x`.
    ///
    /// Unusually for this crate, [`crate::policy::BitExact`] here is not a
    /// claim about *this* platform: the result is correctly rounded, so it
    /// agrees with any correctly-rounded `erf` anywhere. See
    /// [`crate::kernels::double::erf`].
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let f = Erf::new();
    /// assert_eq!(f.eval(0.0_f64), 0.0);
    /// assert_eq!(f.eval(f64::INFINITY), 1.0);
    /// ```
    name: Erf,
    builder: ErfBuilder,
    free: erf,
    k64: crate::kernels::double::erf,
    k32: crate::kernels::single::erf,
    /// `erf(x)`, correctly rounded and safe on any input.
}

math_fn! {
    /// The complementary error function, `1 - erf(x)`.
    ///
    /// Computed as its own function, not as `1 - erf(x)`: that subtraction
    /// cancels away every significant digit for `x` above about 1, and returns
    /// exactly zero well before `erfc` does. Correctly rounded; see
    /// [`crate::kernels::double::erfc`].
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let f = Erfc::new();
    /// assert_eq!(f.eval(0.0_f64), 1.0);
    /// // 1.0 - Erf::new().eval(20.0) is 0; the true value is not.
    /// assert!(f.eval(20.0_f64) > 0.0);
    /// ```
    name: Erfc,
    builder: ErfcBuilder,
    free: erfc,
    k64: crate::kernels::double::erfc,
    k32: crate::kernels::single::erfc,
    /// `erfc(x)`, correctly rounded and safe on any input.
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

// ---------------------------------------------------------------------------
// The IEEE-754 numeric helpers.
//
// Exact like the block above, and precision-generic for the same reason: there
// is nothing to approximate in any of them.
// ---------------------------------------------------------------------------

math_fn! {
    /// `x` rounded to the nearest integer, ties to even.
    ///
    /// The ties rule is the difference from [`Round`], and it is the one
    /// hardware implements.
    name: Rint,
    builder: RintBuilder,
    free: rint,
    k64: crate::kernels::exact::rint,
    k32: crate::kernels::exact::rint,
    /// `rint(x)`, exact.
}

math_fn2! {
    /// `x * 2^n`. [`Ldexp`] under its other name.
    name: Scalbn,
    builder: ScalbnBuilder,
    free: scalbn,
    k64: crate::kernels::exact::scalbn,
    k32: crate::kernels::exact::scalbn,
    /// `scalbn(x, n)`, exact.
}

math_fn2! {
    /// The positive difference: `x - y` if `x > y`, else `+0`.
    name: Fdim,
    builder: FdimBuilder,
    free: fdim,
    k64: crate::kernels::exact::fdim,
    k32: crate::kernels::exact::fdim,
    /// `fdim(x, y)`, exact.
}

math_fn2! {
    /// The larger of `x` and `y`, ignoring NaN.
    ///
    /// C's `fmax`: a NaN argument is discarded rather than propagated. See
    /// [`crate::kernels::exact::fmax`] for the signed-zero convention.
    name: Fmax,
    builder: FmaxBuilder,
    free: fmax,
    k64: crate::kernels::exact::fmax,
    k32: crate::kernels::exact::fmax,
    /// `fmax(x, y)`, exact.
}

math_fn2! {
    /// The smaller of `x` and `y`, ignoring NaN.
    name: Fmin,
    builder: FminBuilder,
    free: fmin,
    k64: crate::kernels::exact::fmin,
    k32: crate::kernels::exact::fmin,
    /// `fmin(x, y)`, exact.
}

math_fn! {
    /// The binary exponent of `x`, as an integer in float lanes.
    ///
    /// See [`crate::kernels::exact::ilogb`] for the three sentinel values and
    /// what they cost in `f32`.
    name: Ilogb,
    builder: IlogbBuilder,
    free: ilogb,
    k64: crate::kernels::exact::ilogb,
    k32: crate::kernels::exact::ilogb,
    /// `ilogb(x)`, exact.
}

math_fn_pair! {
    /// Split `x` into its fractional and integral parts.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let (frac, int) = Modf::new().eval(-3.75_f64);
    /// assert_eq!((frac, int), (-0.75, -3.0));
    /// ```
    name: Modf,
    builder: ModfBuilder,
    free: modf,
    k64: crate::kernels::exact::modf,
    k32: crate::kernels::exact::modf,
    /// `modf(x)`, returning `(fraction, integral)`. Exact.
}

math_fn2_pair! {
    /// `x` reduced modulo `y`, with the low bits of the quotient.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let (rem, quo) = Remquo::new().eval(9.0_f64, 4.0_f64);
    /// assert_eq!((rem, quo), (1.0, 2.0));
    /// ```
    name: Remquo,
    builder: RemquoBuilder,
    free: remquo,
    k64: crate::kernels::exact::remquo,
    k32: crate::kernels::exact::remquo,
    /// `remquo(x, y)`, returning `(remainder, quotient)`. Exact.
}

math_fn_pair! {
    /// `ln|Gamma(x)|` together with the sign of `Gamma(x)`.
    ///
    /// C's `lgamma_r`, the reentrant form: the sign comes back as a second
    /// result rather than in the global `signgam`. Both policy axes are no-ops
    /// — see [`crate::kernels::double::gamma`] for why the value makes an
    /// accuracy claim rather than a bit-exactness one. The *sign* is exact.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// let (v, s) = LGammaR::new().eval(-0.5_f64);
    /// assert_eq!(s, -1.0);                 // Gamma(-0.5) < 0
    /// assert!((v - 1.2655121234846454).abs() < 1e-14);
    /// ```
    name: LGammaR,
    builder: LGammaRBuilder,
    free: lgamma_r,
    k64: crate::kernels::double::gamma::lgamma_r,
    k32: crate::kernels::single::lgamma_r,
    /// `(ln|Gamma(x)|, sign(Gamma(x)))`, safe on any input.
}

// ---------------------------------------------------------------------------
// The Bessel family.
//
// Ports of fdlibm, which is what glibc still runs for these. See
// `crate::kernels::double::bessel` for which parts vectorise and which do not.
// ---------------------------------------------------------------------------

math_fn! {
    /// Bessel function of the first kind, order 0.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// assert_eq!(J0::new().eval(0.0_f64), 1.0);
    /// ```
    name: J0,
    builder: J0Builder,
    free: j0,
    k64: crate::kernels::double::bessel::j0,
    k32: crate::kernels::single::bessel::j0,
    /// `j0(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Bessel function of the first kind, order 1.
    name: J1,
    builder: J1Builder,
    free: j1,
    k64: crate::kernels::double::bessel::j1,
    k32: crate::kernels::single::bessel::j1,
    /// `j1(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Bessel function of the second kind, order 0.
    ///
    /// Defined on the positive reals: `y0(0)` is `-inf` and `y0(x)` is NaN for
    /// `x < 0`.
    name: Y0,
    builder: Y0Builder,
    free: y0,
    k64: crate::kernels::double::bessel::y0,
    k32: crate::kernels::single::bessel::y0,
    /// `y0(x)`, bit-exact and safe on any input.
}

math_fn! {
    /// Bessel function of the second kind, order 1.
    name: Y1,
    builder: Y1Builder,
    free: y1,
    k64: crate::kernels::double::bessel::y1,
    k32: crate::kernels::single::bessel::y1,
    /// `y1(x)`, bit-exact and safe on any input.
}

math_fn2! {
    /// Bessel function of the first kind, order `n`.
    ///
    /// Argument order follows C: `jn(n, x)`, with the order in the first
    /// vector. It travels in a float lane for the same reason
    /// [`Ldexp`]'s exponent does, and is read by truncation towards zero.
    ///
    /// ```
    /// use rmath::prelude::*;
    ///
    /// assert_eq!(Jn::new().eval(0.0_f64, 0.0_f64), 1.0);   // jn(0, 0) = j0(0)
    /// assert_eq!(Jn::new().eval(2.0_f64, 0.0_f64), 0.0);
    /// ```
    name: Jn,
    builder: JnBuilder,
    free: jn,
    k64: crate::kernels::double::bessel::jn,
    k32: crate::kernels::single::bessel::jn,
    /// `jn(n, x)`, bit-exact and safe on any input.
}

math_fn2! {
    /// Bessel function of the second kind, order `n`. See [`Jn`] for the
    /// argument order.
    name: Yn,
    builder: YnBuilder,
    free: yn,
    k64: crate::kernels::double::bessel::yn,
    k32: crate::kernels::single::bessel::yn,
    /// `yn(n, x)`, bit-exact and safe on any input.
}
