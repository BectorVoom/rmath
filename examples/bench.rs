//! Throughput against the scalar `libm` this crate replaces.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench
//! ```
//!
//! Pass `--csv=PATH` (or bare `--csv` for stdout) to also emit every row as
//! `function,metric,speedup`, machine-readable input for
//! `tools/bench_diff.py`. The pretty tables below are unaffected either way.
//!
//! `target-cpu=native` matters and is not a detail: without the `fma` target
//! feature, `wide` has no fused multiply-add, and rmath substitutes a
//! per-lane scalar FMA to keep its bit-exactness promise. That is correct but
//! several times slower, and it is not the configuration anyone should
//! measure. The harness says so at startup rather than silently reporting bad
//! numbers.
//!
//! Method: best-of-N over a buffer larger than L2, timing `eval_slice`
//! against a plain scalar loop over the same data. Best-of rather than mean,
//! because the quantity of interest is the achievable rate and the noise here
//! is all one-directional.
//!
//! # Reading the table
//!
//! The baseline column is the platform's scalar routine, in nanoseconds per
//! element. The rest are speedups over it — above 1.00 is faster than the
//! `libm` call it replaces.
//!
//! `BitExact` is only expected to win for the functions with a vectorised
//! schedule. For the delegating ones it evaluates the same scalar routine one
//! lane at a time, so parity is the *ceiling* there, and `Fast` is where the
//! vector path lives. The crate documentation has the per-function table; this
//! is the measurement behind it.

use rmath::prelude::*;
use std::time::Instant;

/// The platform's own routines, for the functions `std` does not expose.
mod libm {
    unsafe extern "C" {
        pub safe fn erf(x: f64) -> f64;
        pub safe fn erfc(x: f64) -> f64;
        pub safe fn exp10(x: f64) -> f64;
        pub safe fn j0(x: f64) -> f64;
        pub safe fn j1(x: f64) -> f64;
        pub safe fn y0(x: f64) -> f64;
        pub safe fn y1(x: f64) -> f64;
        pub safe fn erff(x: f32) -> f32;
        pub safe fn erfcf(x: f32) -> f32;
        pub safe fn exp10f(x: f32) -> f32;
    }
}

const N: usize = 1 << 20;
const REPS: usize = 8;

/// Every `(function, metric, speedup)` triple printed this run, for `--csv`.
///
/// A global rather than an accumulator threaded through every row-printing
/// site: the recorder is purely additive and this is a benchmark binary, so
/// the shared mutable state buys real simplicity here in a way it would not
/// in the library. `Mutex` rather than `RefCell` only because a `static`
/// demands `Sync`; there is no contention to speak of.
static RECORD: std::sync::Mutex<Vec<(&'static str, &'static str, f64)>> =
    std::sync::Mutex::new(Vec::new());

/// Record one measurement alongside printing it, for `--csv` / `tools/bench_diff.py`.
fn record(function: &'static str, metric: &'static str, speedup: f64) {
    RECORD.lock().unwrap().push((function, metric, speedup));
}

/// Write every recorded row as CSV — to a file if `--csv=PATH` was given, to
/// stdout if bare `--csv` was given, and not at all otherwise.
///
/// Kept a separate, explicit pass over the whole run's data rather than
/// streamed row by row: the pretty tables above are what a human reads while
/// the benchmark runs, and the CSV is what `tools/bench_diff.py` reads
/// afterwards, so interleaving them would make either harder to follow.
fn write_csv() {
    let flag = std::env::args().find(|a| a == "--csv" || a.starts_with("--csv="));
    let Some(flag) = flag else { return };
    let mut out = String::from("function,metric,speedup\n");
    for (function, metric, speedup) in RECORD.lock().unwrap().iter() {
        out.push_str(&format!("{function},{metric},{speedup}\n"));
    }
    if let Some(path) = flag.strip_prefix("--csv=") {
        std::fs::write(path, out).expect("writing --csv output");
        eprintln!("wrote {path}");
    } else {
        print!("{out}");
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
}

fn time(src: &[f64], dst: &mut [f64], mut run: impl FnMut(&[f64], &mut [f64])) -> f64 {
    run(src, dst);
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        run(src, dst);
        best = best.min(t.elapsed().as_secs_f64());
    }
    // Keep the result observable so nothing is optimised away.
    std::hint::black_box(dst[0]);
    best * 1e9 / src.len() as f64
}

fn header() {
    println!(
        "\n{:<10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "function", "libm ns", "exact", "exact/F", "fast", "fast/F"
    );
    println!("{}", "-".repeat(60));
}

/// One unary function, every configuration, against the scalar baseline.
macro_rules! row {
    ($name:literal, $src:expr, $dst:expr, $scalar:expr, $obj:ident) => {{
        let base = time($src, $dst, |s, d| {
            for (x, o) in s.iter().zip(d.iter_mut()) {
                *o = $scalar(*x);
            }
        });
        let a = $obj::builder().accuracy(BitExact).domain(FullRange).build();
        let b = $obj::builder().accuracy(BitExact).domain(Finite).build();
        let c = $obj::builder().accuracy(Fast).domain(FullRange).build();
        let e = $obj::builder().accuracy(Fast).domain(Finite).build();
        let ta = time($src, $dst, |s, d| a.eval_slice(s, d));
        let tb = time($src, $dst, |s, d| b.eval_slice(s, d));
        let tc = time($src, $dst, |s, d| c.eval_slice(s, d));
        let te = time($src, $dst, |s, d| e.eval_slice(s, d));
        println!(
            "{:<10} {base:>10.2} {:>8.2}x {:>8.2}x {:>8.2}x {:>8.2}x",
            $name,
            base / ta,
            base / tb,
            base / tc,
            base / te
        );
        record($name, "exact", base / ta);
        record($name, "exact/F", base / tb);
        record($name, "fast", base / tc);
        record($name, "fast/F", base / te);
    }};
}

fn main() {
    if !cfg!(target_feature = "fma") {
        eprintln!(
            "WARNING: built without the `fma` target feature. rmath is falling back to a\n\
             per-lane scalar FMA to stay bit-exact, so these numbers are not representative.\n\
             Re-run with RUSTFLAGS=\"-C target-cpu=native\".\n"
        );
    }
    println!(
        "widest f64 vector: {} lanes    widest f32 vector: {} lanes    fma: {}",
        <rmath::Widest as Simd>::LANES,
        <rmath::WidestF32 as Simd>::LANES,
        cfg!(target_feature = "fma")
    );

    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    // Exponential arguments: a range that neither overflows nor is trivially tiny.
    let expargs: Vec<f64> = (0..N).map(|_| rng.uniform(-40.0, 40.0)).collect();
    // Positive magnitudes spanning many decades, as physical data tends to be.
    let posargs: Vec<f64> = (0..N)
        .map(|_| rng.uniform((1e-12f64).ln(), (1e12f64).ln()).exp())
        .collect();
    // Angles, in the range a reduction can handle cheaply.
    let angargs: Vec<f64> = (0..N).map(|_| rng.uniform(-1e4, 1e4)).collect();
    // The unit interval, for the inverse trigonometric functions.
    let unitargs: Vec<f64> = (0..N).map(|_| rng.uniform(-1.0, 1.0)).collect();
    // Moderate arguments for the hyperbolic family.
    let hypargs: Vec<f64> = (0..N).map(|_| rng.uniform(-20.0, 20.0)).collect();
    let mut dst = vec![0.0; N];

    header();
    row!("exp", &expargs, &mut dst, f64::exp, Exp);
    row!("exp2", &expargs, &mut dst, f64::exp2, Exp2);
    row!("expm1", &expargs, &mut dst, f64::exp_m1, Expm1);
    row!("ln", &posargs, &mut dst, f64::ln, Ln);
    row!("log2", &posargs, &mut dst, f64::log2, Log2);
    row!("log10", &posargs, &mut dst, f64::log10, Log10);
    row!("log1p", &posargs, &mut dst, f64::ln_1p, Log1p);
    row!("cbrt", &posargs, &mut dst, f64::cbrt, Cbrt);
    row!("sqrt", &posargs, &mut dst, f64::sqrt, Sqrt);

    header();
    row!("sin", &angargs, &mut dst, f64::sin, Sin);
    row!("cos", &angargs, &mut dst, f64::cos, Cos);
    row!("tan", &angargs, &mut dst, f64::tan, Tan);
    row!("asin", &unitargs, &mut dst, f64::asin, Asin);
    row!("acos", &unitargs, &mut dst, f64::acos, Acos);
    row!("atan", &angargs, &mut dst, f64::atan, Atan);

    header();
    row!("sinh", &hypargs, &mut dst, f64::sinh, Sinh);
    row!("cosh", &hypargs, &mut dst, f64::cosh, Cosh);
    row!("tanh", &hypargs, &mut dst, f64::tanh, Tanh);
    row!("asinh", &hypargs, &mut dst, f64::asinh, Asinh);
    row!("acosh", &posargs, &mut dst, f64::acosh, Acosh);
    row!("atanh", &unitargs, &mut dst, f64::atanh, Atanh);

    header();
    // `erf`'s argument is interesting only on about [-6, 6]; outside it the
    // answer is +-1 and every implementation short-circuits.
    let erfargs: Vec<f64> = (0..N).map(|_| rng.uniform(-6.0, 6.0)).collect();
    // `erfc`'s tail is the half that matters, so the corpus leans into it.
    let erfcargs: Vec<f64> = (0..N).map(|_| rng.uniform(-6.0, 25.0)).collect();
    let besselargs: Vec<f64> = (0..N).map(|_| rng.uniform(0.0, 60.0)).collect();
    row!("exp10", &expargs, &mut dst, |x| libm::exp10(x), Exp10);
    row!("erf", &erfargs, &mut dst, |x| libm::erf(x), Erf);
    row!("erfc", &erfcargs, &mut dst, |x| libm::erfc(x), Erfc);

    header();
    row!("j0", &besselargs, &mut dst, |x| libm::j0(x), J0);
    row!("j1", &besselargs, &mut dst, |x| libm::j1(x), J1);
    row!("y0", &besselargs, &mut dst, |x| libm::y0(x), Y0);
    row!("y1", &besselargs, &mut dst, |x| libm::y1(x), Y1);

    header();
    row!("floor", &expargs, &mut dst, f64::floor, Floor);
    row!("round", &expargs, &mut dst, f64::round, Round);
    row!("trunc", &expargs, &mut dst, f64::trunc, Trunc);

    // Binary functions and the Gamma pair do not fit the unary macro.
    println!(
        "\n{:<10} {:>10} {:>9} {:>9}",
        "function", "libm ns", "exact", "fast"
    );
    println!("{}", "-".repeat(42));
    binary(
        "pow",
        &posargs,
        &expargs,
        &mut dst,
        f64::powf,
        Pow::new(),
        Pow::builder().accuracy(Fast).domain(FullRange).build(),
    );
    binary(
        "atan2",
        &angargs,
        &posargs,
        &mut dst,
        f64::atan2,
        Atan2::new(),
        Atan2::builder().accuracy(Fast).domain(FullRange).build(),
    );
    binary(
        "hypot",
        &posargs,
        &angargs,
        &mut dst,
        f64::hypot,
        Hypot::new(),
        Hypot::builder().accuracy(Fast).domain(FullRange).build(),
    );

    println!("\n{:<10} {:>10} {:>9}", "function", "libm ns", "rmath");
    println!("{}", "-".repeat(32));
    gamma("lgamma", &hypargs, &mut dst);
    let mut grng = Rng(0x9E37_79B9_7F4A_7C15);
    let tgargs_direct: Vec<f64> = (0..N).map(|_| grng.uniform(0.0, 18.0)).collect();
    let tgargs_stirling: Vec<f64> = (0..N).map(|_| grng.uniform(18.0, 171.0)).collect();
    tgamma_bench("tgamma (direct)", &tgargs_direct, &mut dst);
    tgamma_bench("tgamma (stirling)", &tgargs_stirling, &mut dst);

    single_precision();

    write_csv();
}

fn binary<A: Function2<f64>, B: Function2<f64>>(
    name: &'static str,
    xs: &[f64],
    ys: &[f64],
    dst: &mut [f64],
    scalar: impl Fn(f64, f64) -> f64,
    exact: A,
    fast: B,
) {
    let base = {
        let t = Instant::now();
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t0 = Instant::now();
            for i in 0..xs.len() {
                dst[i] = scalar(xs[i], ys[i]);
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let _ = t;
        best * 1e9 / xs.len() as f64
    };
    type Binary<'a> = &'a dyn Fn(&[f64], &[f64], &mut [f64]);
    let mut one = |f: Binary<'_>| {
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            f(xs, ys, dst);
            best = best.min(t.elapsed().as_secs_f64());
        }
        best * 1e9 / xs.len() as f64
    };
    let a = one(&|x, y, d| exact.eval_slice(x, y, d));
    let b = one(&|x, y, d| fast.eval_slice(x, y, d));
    println!(
        "{name:<10} {base:>10.2} {:>8.2}x {:>8.2}x",
        base / a,
        base / b
    );
    record(name, "exact", base / a);
    record(name, "fast", base / b);
}

fn gamma(name: &'static str, src: &[f64], dst: &mut [f64]) {
    unsafe extern "C" {
        safe fn lgamma(x: f64) -> f64;
    }
    let base = time(src, dst, |s, d| {
        for (x, o) in s.iter().zip(d.iter_mut()) {
            *o = lgamma(*x);
        }
    });
    let k = LGamma::new();
    let t = time(src, dst, |s, d| k.eval_slice(s, d));
    println!("{name:<10} {base:>10.2} {:>8.2}x", base / t);
    record(name, "rmath", base / t);
}

/// Separate from [`gamma`]: shows the `TG_DIRECT_LIMIT` guard's win directly
/// by running each side of it as its own row, rather than one corpus blending
/// both costs together.
fn tgamma_bench(name: &'static str, src: &[f64], dst: &mut [f64]) {
    unsafe extern "C" {
        safe fn tgamma(x: f64) -> f64;
    }
    let base = time(src, dst, |s, d| {
        for (x, o) in s.iter().zip(d.iter_mut()) {
            *o = tgamma(*x);
        }
    });
    let k = TGamma::new();
    let t = time(src, dst, |s, d| k.eval_slice(s, d));
    println!("{name:<10} {base:>10.2} {:>8.2}x", base / t);
    record(name, "rmath", base / t);
}

fn single_precision() {
    println!(
        "\nsingle precision\n{:<10} {:>10} {:>9} {:>9}",
        "function", "libm ns", "exact", "fast"
    );
    println!("{}", "-".repeat(42));
    let mut rng = Rng(0xB5AD_4ECE_DA10_80CC);
    let e: Vec<f32> = (0..N).map(|_| rng.uniform(-40.0, 40.0) as f32).collect();
    let p: Vec<f32> = (0..N)
        .map(|_| rng.uniform((1e-12f64).ln(), (1e12f64).ln()).exp() as f32)
        .collect();
    let mut d = vec![0.0f32; N];

    macro_rules! row32 {
        ($name:literal, $src:expr, $scalar:expr, $obj:ident) => {{
            let base = {
                let mut best = f64::INFINITY;
                for _ in 0..REPS {
                    let t = Instant::now();
                    for (x, o) in $src.iter().zip(d.iter_mut()) {
                        *o = $scalar(*x);
                    }
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best * 1e9 / N as f64
            };
            let mut run = |f: &dyn Fn(&[f32], &mut [f32])| {
                let mut best = f64::INFINITY;
                for _ in 0..REPS {
                    let t = Instant::now();
                    f($src, &mut d);
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best * 1e9 / N as f64
            };
            let a = $obj::new();
            let c = $obj::builder().accuracy(Fast).domain(FullRange).build();
            let ta = run(&|s, o| Function::<f32>::eval_slice(&a, s, o));
            let tc = run(&|s, o| Function::<f32>::eval_slice(&c, s, o));
            println!(
                "{:<10} {base:>10.2} {:>8.2}x {:>8.2}x",
                $name,
                base / ta,
                base / tc
            );
            record($name, "exact", base / ta);
            record($name, "fast", base / tc);
        }};
    }
    row32!("expf", &e, f32::exp, Exp);
    row32!("exp2f", &e, f32::exp2, Exp2);
    row32!("logf", &p, f32::ln, Ln);
    row32!("log2f", &p, f32::log2, Log2);
    row32!("sqrtf", &p, f32::sqrt, Sqrt);
    row32!("cbrtf", &p, f32::cbrt, Cbrt);
    // Not `&e`: that corpus is +-40, and `tanh` saturates to +-1 (indistinguishable
    // at `f32` precision) by about +-9 -- `src/kernels/single/tanh.rs` routes
    // anything past that to the scalar reference, same as every other kernel's
    // rare tail, so a +-40 corpus would spend ~78% of its lanes there and
    // measure the reference call's cost, not the native kernel's.
    let tanh32args: Vec<f32> = (0..N).map(|_| rng.uniform(-9.0, 9.0) as f32).collect();
    row32!("tanhf", &tanh32args, f32::tanh, Tanh);
    // `TRIG_LIMIT` is ~804 (`src/kernels/single/trig.rs`); stay inside it for
    // the same corpus-artifact reason as `tanhf` above.
    let ang32: Vec<f32> = (0..N).map(|_| rng.uniform(-700.0, 700.0) as f32).collect();
    row32!("sinf", &ang32, f32::sin, Sin);
    row32!("cosf", &ang32, f32::cos, Cos);
    row32!("tanf", &ang32, f32::tan, Tan);
    let erf32: Vec<f32> = (0..N).map(|_| rng.uniform(-6.0, 6.0) as f32).collect();
    let erfc32: Vec<f32> = (0..N).map(|_| rng.uniform(-6.0, 10.0) as f32).collect();
    // Not `&e`: that corpus is +-40, which straddles `exp10f`'s +-38 main-path
    // limit (`src/kernels/single/exp10.rs`), so roughly 5% of its lanes would
    // take the scalar repair under both policies and understate both numbers
    // identically -- a corpus artifact, not a cost of either kernel.
    let exp10args: Vec<f32> = (0..N).map(|_| rng.uniform(-35.0, 35.0) as f32).collect();
    row32!("exp10f", &exp10args, |x| libm::exp10f(x), Exp10);
    row32!("erff", &erf32, |x| libm::erff(x), Erf);
    row32!("erfcf", &erfc32, |x| libm::erfcf(x), Erfc);
}
