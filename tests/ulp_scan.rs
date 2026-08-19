//! A permanent, ignored-by-default measurement harness.
//!
//! `tests/accuracy.rs` *asserts* a bound; this *measures* one, over a much
//! larger corpus, without failing the build. It exists so "measure before
//! asserting" — the step every kernel change in this crate needs — is one
//! command instead of a throwaway script rebuilt each time:
//!
//! ```text
//! cargo test --release --test ulp_scan -- --ignored --nocapture
//! RMATH_SCAN_N=100000000 cargo test --release --test ulp_scan -- --ignored --nocapture scan_pow
//! ```
//!
//! `RMATH_SCAN_N` overrides the sample count per corpus (default 30,000,000).
//! Each `scan_*` test prints the worst ulp found and where, for every corpus
//! it covers, then passes unconditionally — the number is the deliverable.

use rmath::prelude::*;
use std::sync::OnceLock;

/// The platform's own routines, for the functions `std` does not expose.
mod libm {
    unsafe extern "C" {
        pub safe fn exp10(x: f64) -> f64;
        pub safe fn lgamma(x: f64) -> f64;
        pub safe fn tgamma(x: f64) -> f64;
        pub safe fn erff(x: f32) -> f32;
        pub safe fn erfcf(x: f32) -> f32;
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

fn n() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RMATH_SCAN_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30_000_000)
    })
}

fn ulps(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    if got.is_nan() || want.is_nan() {
        return f64::INFINITY;
    }
    let key = |v: f64| {
        let b = v.to_bits() as i64;
        if b < 0 { i64::MIN - b } else { b }
    };
    (key(got).wrapping_sub(key(want))).abs() as f64
}

fn ulps32(got: f32, want: f32) -> f64 {
    if got == want {
        return 0.0;
    }
    if got.is_nan() || want.is_nan() {
        return f64::INFINITY;
    }
    let key = |v: f32| {
        let b = v.to_bits() as i32;
        if b < 0 { i32::MIN - b } else { b }
    };
    (key(got) as i64 - key(want) as i64).abs() as f64
}

/// Scan a one-argument `f64` corpus and print the worst ulp found.
fn scan(
    label: &str,
    seed: u64,
    make: impl Fn(&mut Rng) -> f64,
    ours: impl Fn(f64) -> f64,
    theirs: impl Fn(f64) -> f64,
) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut at = 0.0;
    let mut skipped = 0u64;
    for _ in 0..n() {
        let x = make(&mut rng);
        let want = theirs(x);
        if !want.is_finite() {
            skipped += 1;
            continue;
        }
        let u = ulps(ours(x), want);
        if u > worst {
            worst = u;
            at = x;
        }
    }
    println!("{label:<24} worst {worst:>8} ulp at x = {at:e}  (skipped {skipped})");
}

/// Scan a one-argument `f64` corpus by *absolute* error.
///
/// For the one function in the crate whose relative error is structurally
/// unbounded at its zeros — `lgamma`'s negative half-line, see
/// `tests/accuracy.rs` — so `scan`'s ulp metric would report a blow-up that
/// is not a regression.
fn scan_abs(
    label: &str,
    seed: u64,
    make: impl Fn(&mut Rng) -> f64,
    ours: impl Fn(f64) -> f64,
    theirs: impl Fn(f64) -> f64,
) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut at = 0.0;
    for _ in 0..n() {
        let x = make(&mut rng);
        let want = theirs(x);
        if !want.is_finite() {
            continue;
        }
        let e = (ours(x) - want).abs();
        if e > worst {
            worst = e;
            at = x;
        }
    }
    println!("{label:<24} worst {worst:>12e} abs at x = {at:e}");
}

/// Scan a two-argument `f64` corpus and print the worst ulp found.
fn scan2(
    label: &str,
    seed: u64,
    make: impl Fn(&mut Rng) -> (f64, f64),
    ours: impl Fn(f64, f64) -> f64,
    theirs: impl Fn(f64, f64) -> f64,
) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut at = (0.0, 0.0);
    let mut skipped = 0u64;
    for _ in 0..n() {
        let (x, y) = make(&mut rng);
        let want = theirs(x, y);
        if !want.is_finite() {
            skipped += 1;
            continue;
        }
        let u = ulps(ours(x, y), want);
        if u > worst {
            worst = u;
            at = (x, y);
        }
    }
    println!("{label:<24} worst {worst:>8} ulp at {at:?}  (skipped {skipped})");
}

/// Scan a one-argument `f32` corpus and print the worst ulp found.
fn scan32(
    label: &str,
    seed: u64,
    make: impl Fn(&mut Rng) -> f32,
    ours: impl Fn(f32) -> f32,
    theirs: impl Fn(f32) -> f32,
) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut at = 0.0f32;
    let mut skipped = 0u64;
    for _ in 0..n() {
        let x = make(&mut rng);
        let want = theirs(x);
        if !want.is_finite() {
            skipped += 1;
            continue;
        }
        let u = ulps32(ours(x), want);
        if u > worst {
            worst = u;
            at = x;
        }
    }
    println!("{label:<24} worst {worst:>8} ulp at x = {at:e}  (skipped {skipped})");
}

/// Scan *every* `f32` whose bits fall in `lo..=hi`, exhaustively rather than
/// by sampling — cheap at `f32` width (at most 2^32 inputs), and the only way
/// to be sure a worst case measured by sampling was actually the worst one.
fn scan32_exhaustive(
    label: &str,
    lo: u32,
    hi: u32,
    ours: impl Fn(f32) -> f32,
    theirs: impl Fn(f32) -> f32,
) {
    let mut worst = 0.0f64;
    let mut at = 0.0f32;
    let mut skipped = 0u64;
    let mut n = 0u64;
    for bits in lo..=hi {
        let x = f32::from_bits(bits);
        n += 1;
        let want = theirs(x);
        if !want.is_finite() {
            skipped += 1;
            continue;
        }
        let u = ulps32(ours(x), want);
        if u > worst {
            worst = u;
            at = x;
        }
    }
    println!("{label:<24} worst {worst:>8} ulp at x = {at:e}  ({n} checked, {skipped} skipped)");
}

macro_rules! fast {
    ($t:ident) => {{
        let k = $t::builder().accuracy(Fast).domain(FullRange).build();
        move |x| k.eval(x)
    }};
}
macro_rules! fast2 {
    ($t:ident) => {{
        let k = $t::builder().accuracy(Fast).domain(FullRange).build();
        move |x, y| k.eval(x, y)
    }};
}
macro_rules! fast32 {
    ($t:ident) => {{
        let k = $t::builder().accuracy(Fast).domain(FullRange).build();
        move |x: f32| Function::<f32>::eval(&k, x)
    }};
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_pow() {
    scan2(
        "pow wide",
        19,
        |r| (r.log_uniform(1e-30, 1e30, false), r.uniform(-20.0, 20.0)),
        fast2!(Pow),
        f64::powf,
    );
    scan2(
        "pow moderate",
        20,
        |r| (r.uniform(0.1, 10.0), r.uniform(-8.0, 8.0)),
        fast2!(Pow),
        f64::powf,
    );
    scan2(
        "pow extreme",
        26,
        |r| {
            (
                r.log_uniform(1e-300, 1e300, false),
                r.uniform(-1000.0, 1000.0),
            )
        },
        fast2!(Pow),
        f64::powf,
    );
    scan2(
        "pow near-1 big-y",
        78,
        |r| (r.uniform(0.9, 1.1), r.log_uniform(1.0, 1e15, true)),
        fast2!(Pow),
        f64::powf,
    );
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_exponentials() {
    let e = |r: &mut Rng| r.uniform(-700.0, 700.0);
    scan("exp2", 100, e, fast!(Exp2), f64::exp2);
    scan("exp", 101, e, fast!(Exp), f64::exp);
    scan(
        "exp10",
        102,
        |r| r.uniform(-300.0, 300.0),
        fast!(Exp10),
        |x| libm::exp10(x),
    );
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_cbrt() {
    // Rust's `f64::cbrt`, not the platform's -- see `src/kernels/double/cbrt.rs`.
    scan(
        "cbrt",
        103,
        |r| r.log_uniform(1e-300, 1e300, true),
        fast!(Cbrt),
        f64::cbrt,
    );
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_gamma() {
    scan(
        "tgamma large",
        200,
        |r| r.uniform(18.0, 171.0),
        |x| TGamma::new().eval(x),
        |x| libm::tgamma(x),
    );
    scan(
        "tgamma near overflow",
        202,
        |r| r.uniform(170.0, 171.63),
        |x| TGamma::new().eval(x),
        |x| libm::tgamma(x),
    );
    scan_abs(
        "lgamma negative",
        201,
        |r| r.uniform(-40.0, 0.0),
        |x| LGamma::new().eval(x),
        |x| libm::lgamma(x),
    );
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_trig_f64_sanity() {
    let ang = |r: &mut Rng| r.uniform(-6000.0, 6000.0);
    scan("sin(f64)", 500, ang, fast!(Sin), f64::sin);
    scan("cos(f64)", 501, ang, fast!(Cos), f64::cos);
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_inverse_trig() {
    let unit = |r: &mut Rng| r.uniform(-1.0, 1.0);
    scan("asin", 300, unit, fast!(Asin), f64::asin);
    scan("acos", 301, unit, fast!(Acos), f64::acos);
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_inverse_hyperbolic() {
    scan(
        "asinh",
        310,
        |r| r.log_uniform(1e-10, 1e100, true),
        fast!(Asinh),
        f64::asinh,
    );
}

#[test]
#[ignore = "sweeps all positive-normal f32 inputs; minutes, not seconds"]
fn scan_single_precision_exhaustive() {
    // Every positive normal `f32`, bit for bit -- not a sample. `f32` is
    // small enough that "exhaustive" is affordable, and it is the only way
    // to be sure a `Fast`-path worst case measured elsewhere is the actual
    // worst case rather than the worst one sampling happened to find.
    scan32_exhaustive(
        "log2f",
        0x0080_0000, // smallest positive normal
        0x7f7f_ffff, // largest finite
        fast32!(Log2),
        f32::log2,
    );
    // `erff`'s native `Fast` path (`src/kernels/single/erf.rs`): every
    // positive normal, including the saturation band past `ERF_MAX_BITS`
    // and the `FullRange` patch logic, not just the `fast` function in
    // isolation.
    scan32_exhaustive("erff", 0x0080_0000, 0x7f7f_ffff, fast32!(Erf), |x| {
        libm::erff(x)
    });
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_erf_single_precision() {
    scan32(
        "erff",
        410,
        |r| r.uniform(-6.0, 6.0) as f32,
        fast32!(Erf),
        |x| libm::erff(x),
    );
    scan32(
        "erfcf",
        411,
        |r| r.uniform(-6.0, 10.0) as f32,
        fast32!(Erfc),
        |x| libm::erfcf(x),
    );
}

#[test]
#[ignore = "measurement, not a check; run explicitly"]
fn scan_single_precision() {
    let p = |r: &mut Rng| r.log_uniform(1e-38, 1e38, false) as f32;
    let ps = |r: &mut Rng| r.log_uniform(1e-38, 1e38, true) as f32;
    scan32("log2f", 400, p, fast32!(Log2), f32::log2);
    scan32("cbrtf", 401, ps, fast32!(Cbrt), f32::cbrt);
    scan32(
        "tanhf",
        402,
        |r| r.uniform(-40.0, 40.0) as f32,
        fast32!(Tanh),
        f32::tanh,
    );
    // At this sample count, expect worst-case ulp in the thousands here --
    // inherent near a zero of sin/cos or a pole of tan, not a regression; see
    // `src/kernels/single/trig.rs`. Absolute error is what to compare across
    // runs for these three.
    let ang = |r: &mut Rng| r.uniform(-800.0, 800.0) as f32;
    scan32("sinf", 403, ang, fast32!(Sin), f32::sin);
    scan32("cosf", 404, ang, fast32!(Cos), f32::cos);
    scan32("tanf", 405, ang, fast32!(Tan), f32::tan);
}
