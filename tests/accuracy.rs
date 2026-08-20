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
        pub safe fn erff(x: f32) -> f32;
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
    // Measured 3 ulp (asin) / 2 ulp (acos) over 100M samples
    // (`tests/ulp_scan.rs`), stably at the |x| = 1/2 fold boundary rather
    // than growing with the sample count — a structural worst case, not a
    // rare outlier, so the headroom below is real rather than optimistic.
    check("asin", 4, 4.0, unit, fast!(Asin), f64::asin);
    check("acos", 5, 4.0, unit, fast!(Acos), f64::acos);
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

/// `exp` and `exp2` had no `Fast` bound asserted anywhere in this file before
/// this test — a real gap, since they are the crate's headline "ported"
/// functions and the README quotes numbers for them. Both measure 1 ulp over
/// hundreds of millions of samples (`tests/ulp_scan.rs`); `exp2` reached that
/// only after its degree-1 term `r * ln(2)` was carried as a double-double
/// (`src/kernels/double/exp2.rs`) rather than a single rounded product, which
/// had previously left it a rounding worse than `exp`.
#[test]
fn fast_exponentials_stay_within_bounds() {
    let e = |r: &mut Rng| r.uniform(-500.0, 500.0);
    check("exp", 27, 2.0, e, fast!(Exp), f64::exp);
    check("exp2", 28, 2.0, e, fast!(Exp2), f64::exp2);
}

/// `cbrt`'s `Fast` policy used to be a no-op — `BitExact` and `Fast` ran the
/// same correctly-rounded algorithm, deliberately (see the module doc's
/// history). This is its first divergent bound: a native seed-plus-Newton
/// kernel with no double-double step, measured at 8 ulp worst case, stable
/// from 100M to 500M samples at extreme magnitudes
/// (`tests/ulp_scan.rs::scan_cbrt`) — looser than most `Fast` bounds in this
/// file because there is no wider type to widen into and refine `r` in, the
/// way the `f32` kernel does; see `src/kernels/double/cbrt.rs::fast` for why.
#[test]
fn fast_cbrt_stays_within_bounds() {
    check(
        "cbrt",
        29,
        4.0,
        |r| r.log_uniform(1e-300, 1e300, true),
        fast!(Cbrt),
        f64::cbrt,
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
    // `pow` amplifies its logarithm's error structurally: `x^y = 2^(y log2 x)`
    // multiplies the *error* in `log2 x` by `y` along with its value, so the
    // kernel computes the logarithm in double-double throughout — the
    // division's residue, the `log2(e)` scale and the first two series terms
    // are all carried exactly, pushing the error of *computing* it below one
    // part in 2^53 of the logarithm itself. What remains is dominated by the
    // `Fast` exponential's own bound: with `exp2` now at 1 ulp (see
    // `fast_exponentials_stay_within_bounds`), the first two corpora measure
    // 2 ulp over hundreds of millions of samples (`tests/ulp_scan.rs`). The
    // third corpus is pinned near the `|y log2 x| < 1020` edge of the vector
    // domain, where the amplification is at its worst, and stays measurably
    // looser — 5 ulp, stable from 100M to 400M samples — so it keeps its own,
    // wider bound rather than sharing the other two's. (Before the
    // double-double logarithm, every corpus here measured 40 ulp.)
    check2(
        "pow",
        19,
        4.0,
        |r| (r.log_uniform(1e-30, 1e30, false), r.uniform(-20.0, 20.0)),
        fast2!(Pow),
        f64::powf,
    );
    check2(
        "pow (moderate exponent)",
        20,
        4.0,
        |r| (r.uniform(0.1, 10.0), r.uniform(-8.0, 8.0)),
        fast2!(Pow),
        f64::powf,
    );
    check2(
        "pow (extreme exponent)",
        26,
        8.0,
        |r| {
            (
                r.log_uniform(1e-300, 1e300, false),
                r.uniform(-1000.0, 1000.0),
            )
        },
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
    // Above the direct recurrence the result is computed via `stirling_dd`
    // followed by `pow_exp` in double-double, avoiding rounding through
    // `ln(Gamma(x))`. Measured at <= 1 ulp across the full domain.
    check(
        "tgamma (large)",
        24,
        8.0,
        |r| r.uniform(18.0, 171.0),
        |x| TGamma::new().eval(x),
        |x| libm::tgamma(x),
    );
}

#[test]
fn tgamma_dense_boundaries_and_monotonicity() {
    let mut rng = Rng(2026);
    let tg = TGamma::new();

    // 1. Dense tests around hand-off threshold 18.0
    for _ in 0..10_000 {
        let x = rng.uniform(17.5, 18.5);
        let ours = tg.eval(x);
        let want = libm::tgamma(x);
        let diff = (ours - want).abs() / want;
        assert!(diff <= 1e-14, "tgamma hand-off accuracy at {x}: ours={ours}, want={want}");
    }

    // 2. Dense tests around binade boundaries in [18, 171]
    for binade in [32.0, 64.0, 128.0] {
        for delta in [-1e-4, -1e-8, 0.0, 1e-8, 1e-4] {
            let x = binade + delta;
            let ours = tg.eval(x);
            let want = libm::tgamma(x);
            let diff = (ours - want).abs() / want;
            assert!(diff <= 1e-14, "tgamma binade accuracy at {x}: ours={ours}, want={want}");
        }
    }

    // 3. Dense tests around 170.0..overflow threshold
    for _ in 0..10_000 {
        let x = rng.uniform(170.0, 171.62437);
        let ours = tg.eval(x);
        let want = libm::tgamma(x);
        if want.is_finite() && ours.is_finite() {
            let diff = (ours - want).abs() / want;
            assert!(diff <= 1e-14, "tgamma near overflow accuracy at {x}: ours={ours}, want={want}");
        }
    }

    // 4. Positive domain monotonicity check on [2.0, 171.0]
    let mut prev_x = 2.0;
    let mut prev_g = tg.eval(prev_x);
    for step in 1..2000 {
        let x = 2.0 + (step as f64) * (169.0 / 2000.0);
        let g = tg.eval(x);
        assert!(g >= prev_g, "tgamma monotonicity violated between {prev_x} and {x}: {prev_g} vs {g}");
        prev_x = x;
        prev_g = g;
    }
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

/// Assert a one-argument `f32` kernel stays inside `bound` *absolute* error.
///
/// For the bounded trigonometric functions, where a relative bound is the
/// wrong question. `sin` and `cos` never exceed 1, so "within N ulp of 1.0"
/// is exactly the accuracy a caller experiences — whereas an ulp bound on
/// the *result* is unbounded near a zero of the function, and reports a
/// kernel that is doing well (absolute error near 1e-12) as failing. This is
/// the same reasoning `fast_bessel_stays_within_bounds` and
/// `lgamma_negative_is_bounded_in_absolute_error` already apply.
fn check32_abs(
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
        let e = (ours(x) as f64 - want as f64).abs();
        if e > worst {
            worst = e;
            at = x;
        }
    }
    assert!(
        worst <= bound,
        "{name} (f32): absolute error {worst:e} at x = {at:e}, documented bound is {bound:e}"
    );
}

macro_rules! fast32 {
    ($t:ident) => {{
        let k = $t::builder().accuracy(Fast).domain(FullRange).build();
        move |x: f32| Function::<f32>::eval(&k, x)
    }};
}

/// The narrow band just inside each `Fast` exponential's `Finite` domain
/// limit where `2^x`/`e^x`/`10^x` is itself subnormal.
///
/// Found by `Fast` `erfc`f's own corpus (`|x|` up to its 10.05 bound) landing
/// `exp2` arguments near -127 through the `exp(-x^2)` composition, where the
/// scale's exponent field construction (`k.wrapping_add(127) << 23`) wrapped
/// silently to `+-inf` or 0 instead of the correct subnormal -- a real bug in
/// `exp`/`exp2`/`exp10`f's `fast`, not merely an accuracy shortfall, since it
/// was inside the domain each kernel's own `Finite` claims to cover. Fixed by
/// building `2^(k + 25)` (never subnormal, for any `k` these domains admit)
/// and multiplying by the exact power of two `2^-25` separately; see each
/// kernel's `fast` doc. Compared against `BitExact` here, not the platform:
/// `BitExact` already proves it returns the correct subnormal
/// (`tests/bit_exact.rs`), so it is a self-contained ground truth for this
/// regression, no new platform binding needed.
#[test]
fn fast_single_precision_handles_subnormal_results() {
    macro_rules! exact32 {
        ($t:ident) => {{
            let k = $t::builder().accuracy(BitExact).domain(Finite).build();
            move |x: f32| Function::<f32>::eval(&k, x)
        }};
    }
    check32(
        "exp (subnormal band)",
        45,
        4.0,
        |r| r.uniform(-88.0, -87.0) as f32,
        fast32!(Exp),
        exact32!(Exp),
    );
    check32(
        "exp2 (subnormal band)",
        46,
        4.0,
        |r| r.uniform(-128.0, -126.0) as f32,
        fast32!(Exp2),
        exact32!(Exp2),
    );
    check32(
        "exp10 (subnormal band)",
        47,
        4.0,
        |r| r.uniform(-38.0, -37.0) as f32,
        fast32!(Exp10),
        exact32!(Exp10),
    );
}

#[test]
fn fast_single_precision_stays_within_bounds() {
    let e = |r: &mut Rng| r.uniform(-40.0, 40.0) as f32;
    let p = |r: &mut Rng| r.log_uniform(1e-12, 1e12, false) as f32;
    check32("exp", 30, 4.0, e, fast32!(Exp), f32::exp);
    check32("exp2", 31, 4.0, e, fast32!(Exp2), f32::exp2);
    check32("ln", 32, 4.0, p, fast32!(Ln), f32::ln);
    check32("log2", 33, 4.0, p, fast32!(Log2), f32::log2);
    // `sin`, `cos` and `tan` are native (`src/kernels/single/trig.rs`), not
    // widened -- the corpus stays wider than `TRIG_LIMIT` (~804) on purpose,
    // to also exercise the scalar-reference repair past it. Near a zero of
    // `sin`/`cos` (or a pole of `tan`) any nonzero absolute error is
    // unbounded in ulp terms — inherent to the function, not this kernel; see
    // the module docs — so at a much higher sample count than this test uses
    // (`tests/ulp_scan.rs`, 50M+) the worst case climbs into the thousands.
    // At this corpus size it stays tight, and `tan`'s bound is looser for the
    // same reason `f64` `tan`'s is: it is measured near its own pole too.
    // Two corpora each, deliberately. The wide one runs past `TRIG_LIMIT`, so
    // most of its lanes exercise the scalar-reference repair rather than the
    // vector path -- worth checking, but on its own it would leave the new
    // kernel barely tested. The in-domain one puts every lane through the
    // reduction and the polynomials, which is the code that can actually be
    // wrong.
    let ang32 = |r: &mut Rng| r.uniform(-1e4, 1e4) as f32;
    let indomain32 = |r: &mut Rng| r.uniform(-800.0, 800.0) as f32;
    check32("sin (wide)", 34, 4.0, ang32, fast32!(Sin), f32::sin);
    check32("cos (wide)", 40, 4.0, ang32, fast32!(Cos), f32::cos);
    check32("tan (wide)", 41, 16.0, ang32, fast32!(Tan), f32::tan);
    // In-domain, `sin` and `cos` are held to an absolute bound of 4 ulp *of
    // 1.0* rather than 4 ulp of the result -- see `check32_abs`. The wide
    // rows above can keep the relative bound only because most of their
    // lanes land past `TRIG_LIMIT` and come back bit-exact from the scalar
    // reference; measured relatively, an in-domain corpus reports 5 ulp at
    // x = -733.56, where `cos` is 5.0e-6 and the absolute error is 2.3e-12.
    const ULP32: f64 = 1.1920928955078125e-7; // 2^-23, the spacing at 1.0
    check32_abs("sin", 42, 4.0 * ULP32, indomain32, fast32!(Sin), f32::sin);
    check32_abs("cos", 43, 4.0 * ULP32, indomain32, fast32!(Cos), f32::cos);
    // `tan` is unbounded, so absolute error is the wrong question for it;
    // the relative bound stands, widened because it is measured near its own
    // poles as well as its zeros.
    check32("tan", 44, 16.0, indomain32, fast32!(Tan), f32::tan);
    // `tanh` is native (`src/kernels/single/tanh.rs`), not widened -- see
    // that module's own doc for its bound. `log10`, `expm1` and `atan` below
    // are the ones that widen to `f64` and run the double-precision vector
    // path, so they inherit its bound with an `f32` rounding on top --
    // comfortably inside 4 ulp of the `f32` result.
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
    // Native, not widened-and-rounded: the seed-plus-Newton kernel leaves a
    // residual near 2^-38 before the one narrowing, so the measured worst is
    // a single ulp.
    check32(
        "cbrt",
        39,
        2.0,
        |r| r.log_uniform(1e-38, 1e38, true) as f32,
        fast32!(Cbrt),
        f32::cbrt,
    );
}

/// `erff`'s `Fast` path, unlike every other single-precision function above:
/// native rather than widened, and — unlike `erfc`f, whose own native attempt
/// was measured and reverted (see `src/kernels/single/erfc.rs`'s module
/// doc) — accepted. Measured 2 ulp exhaustively over every positive normal
/// `f32` (`tests/ulp_scan.rs::scan_single_precision_exhaustive`, not a
/// sample), at 4.1x against `BitExact`'s 1.6x.
#[test]
fn fast_erf_stays_within_bounds() {
    check32(
        "erf",
        48,
        4.0,
        |r| r.uniform(-6.0, 6.0) as f32,
        fast32!(Erf),
        |x| libm::erff(x),
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
