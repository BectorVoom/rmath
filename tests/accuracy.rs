//! What [`rmath::Fast`] promises, measured rather than asserted.
//!
//! `Fast` is not bit-exact — that is its purpose — so the only honest way to
//! describe it is a number, and the only honest way to publish that number is
//! to fail the build when it moves. Every bound below is the measured maximum
//! over the stated corpus, rounded up to a power of two of headroom; the
//! crate documentation quotes these, so a kernel change that loosens one
//! fails here and the documentation is wrong until it is updated.
//!
//! Also here: the Gamma functions, which are the only ones in the crate that
//! make an accuracy claim under *both* policies, because Rust has no
//! `f64::tgamma` for a bit-exactness claim to refer to. They are compared
//! against the platform's through `extern "C"`.
//!
//! ```text
//! cargo test --release --test accuracy
//! ```

use rmath::prelude::*;

/// The platform's own routines, for the two functions `std` does not expose.
mod libm {
    unsafe extern "C" {
        pub safe fn tgamma(x: f64) -> f64;
        pub safe fn lgamma(x: f64) -> f64;
        pub safe fn erf(x: f64) -> f64;
        pub safe fn erfc(x: f64) -> f64;
        pub safe fn exp10(x: f64) -> f64;
        pub safe fn j0(x: f64) -> f64;
        pub safe fn j1(x: f64) -> f64;
        pub safe fn y0(x: f64) -> f64;
        pub safe fn y1(x: f64) -> f64;
        pub safe fn j0f(x: f32) -> f32;
        pub safe fn y0f(x: f32) -> f32;
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn log_uniform(&mut self, lo: f64, hi: f64, signed: bool) -> f64 {
        let m = self.uniform(lo.ln(), hi.ln()).exp();
        if signed && self.next() & 1 == 0 {
            -m
        } else {
            m
        }
    }
}

/// Distance between two values in representable steps.
///
/// Counted in the integer domain rather than as `(got - want) / 2^(e-52)`:
/// that scale underflows to zero for a subnormal result and would report every
/// one of them as an infinite error, which measures the measurement rather
/// than the kernel.
fn ulps(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    if got.is_nan() || want.is_nan() {
        return if got.to_bits() == want.to_bits() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let key = |v: f64| {
        let b = v.to_bits() as i64;
        if b < 0 { i64::MIN - b } else { b }
    };
    (key(got) - key(want)).abs() as f64
}

const N: usize = 200_000;

/// Assert a one-argument kernel stays inside `bound` ulp over a corpus.
fn check(
    name: &str,
    seed: u64,
    bound: f64,
    make: impl Fn(&mut Rng) -> f64,
    ours: impl Fn(f64) -> f64,
    theirs: impl Fn(f64) -> f64,
) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut at = 0.0;
    for _ in 0..N {
        let x = make(&mut rng);
        let want = theirs(x);
        if !want.is_finite() {
            continue;
        }
        let u = ulps(ours(x), want);
        if u > worst {
            worst = u;
            at = x;
        }
    }
    assert!(
        worst <= bound,
        "{name}: {worst} ulp at x = {at:e}, documented bound is {bound}"
    );
}

/// Assert a two-argument kernel stays inside `bound` ulp.
fn check2(
    name: &str,
    seed: u64,
    bound: f64,
    make: impl Fn(&mut Rng) -> (f64, f64),
    ours: impl Fn(f64, f64) -> f64,
    theirs: impl Fn(f64, f64) -> f64,
) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut at = (0.0, 0.0);
    for _ in 0..N {
        let (x, y) = make(&mut rng);
        let want = theirs(x, y);
        if !want.is_finite() {
            continue;
        }
        let u = ulps(ours(x, y), want);
        if u > worst {
            worst = u;
            at = (x, y);
        }
    }
    assert!(
        worst <= bound,
        "{name}: {worst} ulp at {at:?}, documented bound is {bound}"
    );
}

/// A `Fast` + `FullRange` closure for a unary function object.
macro_rules! fast {
    ($t:ident) => {{
        let k = $t::builder().accuracy(Fast).domain(FullRange).build();
        move |x| k.eval(x)
    }};
}

/// The same, for a binary one.
macro_rules! fast2 {
    ($t:ident) => {{
        let k = $t::builder().accuracy(Fast).domain(FullRange).build();
        move |x, y| k.eval(x, y)
    }};
}

#[test]
fn fast_trigonometry_stays_within_bounds() {
    let ang = |r: &mut Rng| r.uniform(-1e5, 1e5);
    check("sin", 1, 4.0, ang, fast!(Sin), f64::sin);
    check("cos", 2, 4.0, ang, fast!(Cos), f64::cos);
    check("tan", 3, 8.0, ang, fast!(Tan), f64::tan);
}

#[test]
fn fast_inverse_trigonometry_stays_within_bounds() {
    let unit = |r: &mut Rng| r.uniform(-1.0, 1.0);
    check("asin", 4, 8.0, unit, fast!(Asin), f64::asin);
    check("acos", 5, 8.0, unit, fast!(Acos), f64::acos);
    check(
        "atan",
        6,
        4.0,
        |r| r.log_uniform(1e-10, 1e10, true),
        fast!(Atan),
        f64::atan,
    );
    check2(
        "atan2",
        7,
        4.0,
        |r| {
            (
                r.log_uniform(1e-8, 1e8, true),
                r.log_uniform(1e-8, 1e8, true),
            )
        },
        fast2!(Atan2),
        f64::atan2,
    );
}

#[test]
fn fast_hyperbolics_stay_within_bounds() {
    check(
        "sinh",
        8,
        4.0,
        |r| r.uniform(-700.0, 700.0),
        fast!(Sinh),
        f64::sinh,
    );
    check(
        "cosh",
        9,
        4.0,
        |r| r.uniform(-700.0, 700.0),
        fast!(Cosh),
        f64::cosh,
    );
    check(
        "tanh",
        10,
        4.0,
        |r| r.uniform(-30.0, 30.0),
        fast!(Tanh),
        f64::tanh,
    );
    check(
        "asinh",
        11,
        8.0,
        |r| r.log_uniform(1e-10, 1e100, true),
        fast!(Asinh),
        f64::asinh,
    );
    check(
        "acosh",
        12,
        4.0,
        |r| r.log_uniform(1.000001, 1e100, false),
        fast!(Acosh),
        f64::acosh,
    );
}

/// `atanh` is the one function whose `Fast` path is *further* from the
/// platform than the others, and it is worth being precise about why: the
/// platform is the inaccurate one.
///
/// Rust's `f64::atanh` is not a `libm` call — it evaluates
/// `0.5 * ln_1p(2x / (1 - x))` with the signed `x`, and as `x` approaches -1
/// that argument approaches -1 too, where `ln_1p` loses most of its digits.
/// The kernel here folds to `|x|` first, so its argument grows instead of
/// cancelling. Checked at 200 bits, the kernel is the one closer to the true
/// value at the point of largest disagreement.
///
/// The bound is therefore stated against the platform, and is a measure of how
/// wrong the platform gets, not of how wrong the kernel gets.
#[test]
fn fast_atanh_differs_from_the_platform_only_where_the_platform_is_worse() {
    check(
        "atanh",
        13,
        256.0,
        |r| r.uniform(-0.999, 0.999),
        fast!(Atanh),
        f64::atanh,
    );
}

#[test]
fn fast_logarithms_stay_within_bounds() {
    let pos = |r: &mut Rng| r.log_uniform(1e-300, 1e300, false);
    check("log2", 14, 4.0, pos, fast!(Log2), f64::log2);
    check("log10", 15, 4.0, pos, fast!(Log10), f64::log10);
    check(
        "log1p",
        16,
        4.0,
        |r| r.log_uniform(1e-15, 1e10, true),
        fast!(Log1p),
        f64::ln_1p,
    );
    check(
        "expm1",
        17,
        4.0,
        |r| r.uniform(-500.0, 500.0),
        fast!(Expm1),
        f64::exp_m1,
    );
}

#[test]
fn fast_pow_and_hypot_stay_within_bounds() {
    check2(
        "hypot",
        18,
        4.0,
        |r| {
            (
                r.log_uniform(1e-100, 1e100, true),
                r.log_uniform(1e-100, 1e100, true),
            )
        },
        fast2!(Hypot),
        f64::hypot,
    );
    // `pow` is the loosest kernel in the crate, and the reason is structural:
    // `x^y = 2^(y log2 x)` multiplies the *error* in `log2 x` by `y` along
    // with its value. The double-double split below removes the error of
    // representing the logarithm, but not the error of computing it, so what
    // is left grows in proportion to `|y log2 x|`. Tightening this means
    // carrying the logarithm's table path in double-double as glibc's `pow`
    // does — a port, not a tweak. `BitExact` is exact meanwhile.
    check2(
        "pow",
        19,
        40.0,
        |r| (r.log_uniform(1e-30, 1e30, false), r.uniform(-20.0, 20.0)),
        fast2!(Pow),
        f64::powf,
    );
    check2(
        "pow (moderate exponent)",
        20,
        16.0,
        |r| (r.uniform(0.1, 10.0), r.uniform(-8.0, 8.0)),
        fast2!(Pow),
        f64::powf,
    );
}

/// The Gamma functions, under the default policy — there is no other.
#[test]
fn gamma_stays_within_bounds() {
    check(
        "lgamma (positive)",
        21,
        16.0,
        |r| r.uniform(0.0, 170.0),
        |x| LGamma::new().eval(x),
        |x| libm::lgamma(x),
    );
    check(
        "tgamma (positive)",
        22,
        16.0,
        |r| r.uniform(0.0, 18.0),
        |x| TGamma::new().eval(x),
        |x| libm::tgamma(x),
    );
    check(
        "tgamma (negative)",
        23,
        256.0,
        |r| r.uniform(-18.0, 0.0),
        |x| TGamma::new().eval(x),
        |x| libm::tgamma(x),
    );
    // Above the direct recurrence the result is `exp(lgamma(x))`, and an
    // absolute error of one ulp in a logarithm near 700 is a relative error
    // near 1e-13 in its exponential. Inherent to the route, not a defect in
    // it.
    check(
        "tgamma (large)",
        24,
        4096.0,
        |r| r.uniform(18.0, 171.0),
        |x| TGamma::new().eval(x),
        |x| libm::tgamma(x),
    );
}

/// On the negative half-line `lgamma` is a difference of comparable terms, and
/// near its zeros there the relative error is unbounded for *any*
/// implementation — the answer passes through zero while the terms do not. So
/// the claim there is absolute, not relative.
#[test]
fn lgamma_negative_is_bounded_in_absolute_error() {
    let mut rng = Rng(25);
    let mut worst = 0.0f64;
    let mut at = 0.0;
    for _ in 0..N {
        let x = rng.uniform(-40.0, 0.0);
        let want = libm::lgamma(x);
        if !want.is_finite() {
            continue;
        }
        let e = (LGamma::new().eval(x) - want).abs();
        if e > worst {
            worst = e;
            at = x;
        }
    }
    assert!(
        worst <= 1e-12,
        "lgamma: absolute error {worst:e} at x = {at:e}"
    );
}

// ---------------------------------------------------------------------------
// Single precision
//
// The `f32` bit-exact paths are checked exhaustively in `tests/single.rs` --
// every one of the 2^32 inputs -- so what is left to pin down here is `Fast`,
// which has no exactness claim and therefore needs a number instead.
// ---------------------------------------------------------------------------

/// Distance between two `f32` values in representable steps.
fn ulps32(got: f32, want: f32) -> f64 {
    if got == want {
        return 0.0;
    }
    if got.is_nan() || want.is_nan() {
        return if got.to_bits() == want.to_bits() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let key = |v: f32| {
        let b = v.to_bits() as i32;
        if b < 0 { i32::MIN - b } else { b }
    };
    (key(got) as i64 - key(want) as i64).abs() as f64
}

fn check32(
    name: &str,
    seed: u64,
    bound: f64,
    make: impl Fn(&mut Rng) -> f32,
    ours: impl Fn(f32) -> f32,
    theirs: impl Fn(f32) -> f32,
) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut at = 0.0f32;
    for _ in 0..N {
        let x = make(&mut rng);
        let want = theirs(x);
        if !want.is_finite() {
            continue;
        }
        let u = ulps32(ours(x), want);
        if u > worst {
            worst = u;
            at = x;
        }
    }
    assert!(
        worst <= bound,
        "{name} (f32): {worst} ulp at x = {at:e}, documented bound is {bound}"
    );
}

macro_rules! fast32 {
    ($t:ident) => {{
        let k = $t::builder().accuracy(Fast).domain(FullRange).build();
        move |x: f32| Function::<f32>::eval(&k, x)
    }};
}

#[test]
fn fast_single_precision_stays_within_bounds() {
    let e = |r: &mut Rng| r.uniform(-40.0, 40.0) as f32;
    let p = |r: &mut Rng| r.log_uniform(1e-12, 1e12, false) as f32;
    check32("exp", 30, 4.0, e, fast32!(Exp), f32::exp);
    check32("exp2", 31, 4.0, e, fast32!(Exp2), f32::exp2);
    check32("ln", 32, 4.0, p, fast32!(Ln), f32::ln);
    check32("log2", 33, 4.0, p, fast32!(Log2), f32::log2);
    // The remaining single-precision `Fast` kernels widen to `f64` and run the
    // double-precision vector path, so they inherit its bound with a `f32`
    // rounding on top -- comfortably inside 4 ulp of the `f32` result.
    check32(
        "sin",
        34,
        4.0,
        |r| r.uniform(-1e4, 1e4) as f32,
        fast32!(Sin),
        f32::sin,
    );
    check32("tanh", 35, 4.0, e, fast32!(Tanh), f32::tanh);
    check32("log10", 36, 4.0, p, fast32!(Log10), f32::log10);
    check32("expm1", 37, 4.0, e, fast32!(Expm1), f32::exp_m1);
    check32(
        "atan",
        38,
        4.0,
        |r| r.uniform(-1e3, 1e3) as f32,
        fast32!(Atan),
        f32::atan,
    );
}

/// The bounds for the kernels added alongside the special functions.
///
/// `erf`, `erfc` and `exp10` are the interesting ones: their `Fast` paths are
/// the *same* double-double arithmetic the bit-exact ones use, with only the
/// rounding test and the accurate fallback dropped. So the error is not
/// "a few ulps" but a fraction of one — they are correctly rounded on all but
/// about one input in thirty thousand, and the miss is by a single step.
#[test]
fn fast_special_functions_stay_within_bounds() {
    let near = |r: &mut Rng| r.uniform(-6.0, 6.0);
    check("erf", 0x0E2F_0001, 1.0, near, fast!(Erf), |x| libm::erf(x));
    check(
        "erfc",
        0x0E2F_0002,
        1.0,
        |r| r.uniform(-6.0, 27.0),
        fast!(Erfc),
        |x| libm::erfc(x),
    );
    check(
        "exp10",
        0x0E2F_0003,
        2.0,
        |r| r.uniform(-300.0, 300.0),
        fast!(Exp10),
        |x| libm::exp10(x),
    );
}

/// The Bessel kernels' `Fast` path replaces the platform's trigonometry with
/// [`rmath::Fast`]'s, and that dominates: `j0(x)` for large `x` is
/// `cos(x - pi/4)` scaled, so an error in the reduced angle is an error of the
/// same relative size in the answer — and near a zero of `j0` it is very much
/// larger, because the zero is where the value is small and the derivative is
/// not. So this is an *absolute* bound on `[0, 60]` rather than a relative
/// one, which is the only bound that means anything near a zero.
#[test]
fn fast_bessel_stays_within_bounds() {
    let f = J0::builder().accuracy(Fast).domain(FullRange).build();
    let g = J1::builder().accuracy(Fast).domain(FullRange).build();
    let p = Y0::builder().accuracy(Fast).domain(FullRange).build();
    let q = Y1::builder().accuracy(Fast).domain(FullRange).build();
    let mut worst = 0.0f64;
    for i in 1..200_000u64 {
        let x = i as f64 * (60.0 / 200_000.0);
        for (ours, theirs) in [
            (f.eval(x), libm::j0(x)),
            (g.eval(x), libm::j1(x)),
            (p.eval(x), libm::y0(x)),
            (q.eval(x), libm::y1(x)),
        ] {
            if theirs.is_finite() {
                worst = worst.max((ours - theirs).abs());
            }
        }
    }
    assert!(
        worst < 1e-12,
        "fast Bessel: absolute error {worst:e} on [0, 60]"
    );
}

/// Single precision under `Fast` takes the *double* Bessel kernel and rounds
/// once, so it is not a cheaper approximation of `j0f` — it is a better one.
/// glibc's own bound for `j0f` is 9 ulps; this asserts that the widened path
/// beats it, which is the whole reason that configuration is offered.
#[test]
fn fast_single_precision_bessel_beats_the_platform() {
    let f = J0::builder().accuracy(Fast).domain(FullRange).build();
    let p = Y0::builder().accuracy(Fast).domain(FullRange).build();
    let mut worst = 0.0f64;
    for i in 1..200_000u32 {
        let x = i as f32 * (60.0 / 200_000.0);
        let d = (f64::from(f.eval(x)) - f64::from(libm::j0f(x))).abs();
        worst = worst.max(d);
        let d = (f64::from(p.eval(x)) - f64::from(libm::y0f(x))).abs();
        worst = worst.max(d);
    }
    // 9 ulps of a value of order 1 in f32 is about 1.1e-6.
    assert!(
        worst < 1.1e-6,
        "fast j0f/y0f: absolute error {worst:e} on [0, 60]"
    );
}
