//! Throughput against the scalar `libm` this crate replaces.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench
//! ```
//!
//! Pass `--csv=PATH` (or bare `--csv` for stdout) to also emit every row as
//! `function,metric,speedup,ns_per_elem`, machine-readable input for
//! `tools/bench_diff.py`. The pretty tables below are unaffected either way.
//!
//! `target-cpu=native` matters and is not a detail: without the `fma` target
//! feature, `wide` has no fused multiply-add, and rmath substitutes a
//! per-lane scalar FMA to keep its bit-exactness promise. That is correct but
//! several times slower, and it is not the configuration anyone should
//! measure. The harness says so at startup rather than silently reporting bad
//! numbers.
//!
//! Method: warm-up iterations followed by timed samples reporting median
//! over a buffer, timing `eval_slice` against a plain scalar loop over the same data.
//!
//! # Options
//! - `--size=N`: custom buffer size (default 1048576)
//! - `--corpus=NAME`: `default`, `in-domain`, `boundary`, `random-bit`, `coherent`, `special`, `mixed-special`
//! - `--suite=NAME`: `default`, `traversal`, `repair`, `all`
//! - `--csv=PATH` or `--csv`: emit CSV output with metadata header

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

const DEFAULT_N: usize = 1 << 20;
const WARMUP: usize = 2;
const SAMPLES: usize = 10;

/// Every `(function, metric, speedup, ns_per_elem)` printed this run, for `--csv`.
static RECORD: std::sync::Mutex<Vec<(String, String, f64, f64)>> =
    std::sync::Mutex::new(Vec::new());

/// Record one measurement alongside printing it, for `--csv` / `tools/bench_diff.py`.
fn record(function: impl Into<String>, metric: impl Into<String>, speedup: f64, ns_per_elem: f64) {
    RECORD
        .lock()
        .unwrap()
        .push((function.into(), metric.into(), speedup, ns_per_elem));
}

fn get_cpu_model() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if line.starts_with("model name") {
                if let Some((_, model)) = line.split_once(':') {
                    return model.trim().to_string();
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

/// Compute median and min ns/element from sampled times.
fn compute_stats(mut times: Vec<f64>, len: usize) -> (f64, f64) {
    if len == 0 || times.is_empty() {
        return (0.0, 0.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let median = if times.len() % 2 == 0 {
        0.5 * (times[times.len() / 2 - 1] + times[times.len() / 2])
    } else {
        times[times.len() / 2]
    };
    let min = times[0];
    (median * 1e9 / len as f64, min * 1e9 / len as f64)
}

/// Write every recorded row as CSV — to a file if `--csv=PATH` was given, to
/// stdout if bare `--csv` was given, and not at all otherwise.
fn write_csv(corpus: &str, size: usize, suite: &str) {
    let flag = std::env::args().find(|a| a == "--csv" || a.starts_with("--csv="));
    let Some(flag) = flag else { return };
    let mut out = String::new();
    out.push_str(&format!("# target_arch: {}\n", std::env::consts::ARCH));
    out.push_str(&format!("# cpu_model: {}\n", get_cpu_model()));
    out.push_str(&format!(
        "# widest_f64_lanes: {}\n",
        <rmath::Widest as Simd>::LANES
    ));
    out.push_str(&format!(
        "# widest_f32_lanes: {}\n",
        <rmath::WidestF32 as Simd>::LANES
    ));
    out.push_str(&format!("# fma: {}\n", cfg!(target_feature = "fma")));
    out.push_str(&format!("# suite: {}\n", suite));
    out.push_str(&format!("# size: {}\n", size));
    out.push_str(&format!("# corpus: {}\n", corpus));
    out.push_str(&format!(
        "# repetition_policy: warmup={},samples={},statistic=median\n",
        WARMUP, SAMPLES
    ));
    if let Some(rustc) = option_env!("RMATH_RUSTC_VERBOSE") {
        out.push_str(&format!("# rustc: {}\n", rustc));
    }
    if let Some(flags) = option_env!("RUSTFLAGS") {
        out.push_str(&format!("# rustflags: {}\n", flags));
    }
    out.push_str("function,metric,speedup,ns_per_elem\n");
    for (function, metric, speedup, ns) in RECORD.lock().unwrap().iter() {
        out.push_str(&format!("{function},{metric},{speedup:.4},{ns:.4}\n"));
    }
    if let Some(path) = flag.strip_prefix("--csv=") {
        std::fs::write(path, out).expect("writing --csv output");
        eprintln!("wrote {path}");
    } else {
        print!("{out}");
    }
}

#[derive(Clone)]
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
    if src.is_empty() {
        return 0.0;
    }
    for _ in 0..WARMUP {
        run(src, dst);
    }
    let mut times = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t = Instant::now();
        run(src, dst);
        times.push(t.elapsed().as_secs_f64());
    }
    // Keep the result observable so nothing is optimised away.
    std::hint::black_box(dst[0]);
    compute_stats(times, src.len()).0
}

fn header() {
    println!(
        "\n{:<10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "function", "libm ns", "exact", "exact/F", "fast", "fast/F"
    );
    println!("{}", "-".repeat(60));
}

/// Same shape as [`time`], for a two-output [`FunctionPair`] run.
fn time_pair(
    src: &[f64],
    a: &mut [f64],
    b: &mut [f64],
    mut run: impl FnMut(&[f64], &mut [f64], &mut [f64]),
) -> f64 {
    if src.is_empty() {
        return 0.0;
    }
    for _ in 0..WARMUP {
        run(src, a, b);
    }
    let mut times = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t = Instant::now();
        run(src, a, b);
        times.push(t.elapsed().as_secs_f64());
    }
    std::hint::black_box((a[0], b[0]));
    compute_stats(times, src.len()).0
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
        record($name, "exact", base / ta, ta);
        record($name, "exact/F", base / tb, tb);
        record($name, "fast", base / tc, tc);
        record($name, "fast/F", base / te, te);
    }};
}

/// [`row!`] for a two-output [`FunctionPair`] such as `SinCos`.
macro_rules! row_pair {
    ($name:literal, $src:expr, $da:expr, $db:expr, $scalar:expr, $obj:ident) => {{
        let base = time_pair($src, $da, $db, |s, oa, ob| {
            for ((x, ra), rb) in s.iter().zip(oa.iter_mut()).zip(ob.iter_mut()) {
                let (sa, sb) = $scalar(*x);
                *ra = sa;
                *rb = sb;
            }
        });
        let a = $obj::builder().accuracy(BitExact).domain(FullRange).build();
        let b = $obj::builder().accuracy(BitExact).domain(Finite).build();
        let c = $obj::builder().accuracy(Fast).domain(FullRange).build();
        let e = $obj::builder().accuracy(Fast).domain(Finite).build();
        let ta = time_pair($src, $da, $db, |s, oa, ob| a.eval_slice(s, oa, ob));
        let tb = time_pair($src, $da, $db, |s, oa, ob| b.eval_slice(s, oa, ob));
        let tc = time_pair($src, $da, $db, |s, oa, ob| c.eval_slice(s, oa, ob));
        let te = time_pair($src, $da, $db, |s, oa, ob| e.eval_slice(s, oa, ob));
        println!(
            "{:<10} {base:>10.2} {:>8.2}x {:>8.2}x {:>8.2}x {:>8.2}x",
            $name,
            base / ta,
            base / tb,
            base / tc,
            base / te
        );
        record($name, "exact", base / ta, ta);
        record($name, "exact/F", base / tb, tb);
        record($name, "fast", base / tc, tc);
        record($name, "fast/F", base / te, te);
    }};
}

struct Corpora {
    expargs: Vec<f64>,
    posargs: Vec<f64>,
    angargs: Vec<f64>,
    unitargs: Vec<f64>,
    hypargs: Vec<f64>,
    erfargs: Vec<f64>,
    erfcargs: Vec<f64>,
    besselargs: Vec<f64>,
}

fn generate_corpora(profile: &str, n: usize) -> Corpora {
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    match profile {
        "in-domain" => Corpora {
            expargs: (0..n).map(|_| rng.uniform(-20.0, 20.0)).collect(),
            posargs: (0..n).map(|_| rng.uniform(0.1, 100.0)).collect(),
            angargs: (0..n).map(|_| rng.uniform(-std::f64::consts::PI, std::f64::consts::PI)).collect(),
            unitargs: (0..n).map(|_| rng.uniform(-0.9, 0.9)).collect(),
            hypargs: (0..n).map(|_| rng.uniform(-5.0, 5.0)).collect(),
            erfargs: (0..n).map(|_| rng.uniform(-3.0, 3.0)).collect(),
            erfcargs: (0..n).map(|_| rng.uniform(-3.0, 10.0)).collect(),
            besselargs: (0..n).map(|_| rng.uniform(0.5, 30.0)).collect(),
        },
        "boundary" => {
            let special_vals = [
                0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1e-15, -1e-15,
                f64::MIN_POSITIVE, 709.78, -708.39, f64::MAX,
            ];
            let build_vec = |rng: &mut Rng, lo: f64, hi: f64| -> Vec<f64> {
                (0..n)
                    .map(|i| {
                        if i % 8 < special_vals.len() && i % 8 != 0 {
                            special_vals[i % special_vals.len()]
                        } else {
                            rng.uniform(lo, hi)
                        }
                    })
                    .collect()
            };
            Corpora {
                expargs: build_vec(&mut rng, -700.0, 700.0),
                posargs: build_vec(&mut rng, 1e-300, 1e300),
                angargs: build_vec(&mut rng, -1e15, 1e15),
                unitargs: build_vec(&mut rng, -1.0, 1.0),
                hypargs: build_vec(&mut rng, -700.0, 700.0),
                erfargs: build_vec(&mut rng, -6.0, 6.0),
                erfcargs: build_vec(&mut rng, -6.0, 25.0),
                besselargs: build_vec(&mut rng, 0.0, 100.0),
            }
        }
        "random-bit" => {
            let build_vec = |rng: &mut Rng| -> Vec<f64> {
                (0..n).map(|_| f64::from_bits(rng.next())).collect()
            };
            Corpora {
                expargs: build_vec(&mut rng),
                posargs: build_vec(&mut rng),
                angargs: build_vec(&mut rng),
                unitargs: build_vec(&mut rng),
                hypargs: build_vec(&mut rng),
                erfargs: build_vec(&mut rng),
                erfcargs: build_vec(&mut rng),
                besselargs: build_vec(&mut rng),
            }
        }
        "coherent" | "sorted" => {
            let mut c = Corpora {
                expargs: (0..n).map(|i| -40.0 + 80.0 * (i as f64 / n.max(1) as f64)).collect(),
                posargs: (0..n).map(|i| ((1e-12f64).ln() + ((1e12f64).ln() - (1e-12f64).ln()) * (i as f64 / n.max(1) as f64)).exp()).collect(),
                angargs: (0..n).map(|i| -1e4 + 2e4 * (i as f64 / n.max(1) as f64)).collect(),
                unitargs: (0..n).map(|i| -1.0 + 2.0 * (i as f64 / n.max(1) as f64)).collect(),
                hypargs: (0..n).map(|i| -20.0 + 40.0 * (i as f64 / n.max(1) as f64)).collect(),
                erfargs: (0..n).map(|i| -6.0 + 12.0 * (i as f64 / n.max(1) as f64)).collect(),
                erfcargs: (0..n).map(|i| -6.0 + 31.0 * (i as f64 / n.max(1) as f64)).collect(),
                besselargs: (0..n).map(|i| 60.0 * (i as f64 / n.max(1) as f64)).collect(),
            };
            c.expargs.sort_by(|a, b| a.total_cmp(b));
            c.posargs.sort_by(|a, b| a.total_cmp(b));
            c.angargs.sort_by(|a, b| a.total_cmp(b));
            c.unitargs.sort_by(|a, b| a.total_cmp(b));
            c.hypargs.sort_by(|a, b| a.total_cmp(b));
            c.erfargs.sort_by(|a, b| a.total_cmp(b));
            c.erfcargs.sort_by(|a, b| a.total_cmp(b));
            c.besselargs.sort_by(|a, b| a.total_cmp(b));
            c
        }
        "special" | "mixed-special" => {
            let specials = [
                0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
                f64::MIN_POSITIVE * 0.25, f64::MAX, -f64::MAX,
            ];
            let build_vec = |rng: &mut Rng, lo: f64, hi: f64| -> Vec<f64> {
                (0..n)
                    .map(|i| {
                        if i % 4 == 0 {
                            specials[(i / 4) % specials.len()]
                        } else {
                            rng.uniform(lo, hi)
                        }
                    })
                    .collect()
            };
            Corpora {
                expargs: build_vec(&mut rng, -40.0, 40.0),
                posargs: build_vec(&mut rng, 0.1, 1000.0),
                angargs: build_vec(&mut rng, -1e4, 1e4),
                unitargs: build_vec(&mut rng, -1.0, 1.0),
                hypargs: build_vec(&mut rng, -20.0, 20.0),
                erfargs: build_vec(&mut rng, -6.0, 6.0),
                erfcargs: build_vec(&mut rng, -6.0, 25.0),
                besselargs: build_vec(&mut rng, 0.0, 60.0),
            }
        }
        _ => Corpora {
            // Default
            expargs: (0..n).map(|_| rng.uniform(-40.0, 40.0)).collect(),
            posargs: (0..n)
                .map(|_| rng.uniform((1e-12f64).ln(), (1e12f64).ln()).exp())
                .collect(),
            angargs: (0..n).map(|_| rng.uniform(-1e4, 1e4)).collect(),
            unitargs: (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect(),
            hypargs: (0..n).map(|_| rng.uniform(-20.0, 20.0)).collect(),
            erfargs: (0..n).map(|_| rng.uniform(-6.0, 6.0)).collect(),
            erfcargs: (0..n).map(|_| rng.uniform(-6.0, 25.0)).collect(),
            besselargs: (0..n).map(|_| rng.uniform(0.0, 60.0)).collect(),
        },
    }
}

fn parse_arg(prefix: &str) -> Option<String> {
    std::env::args().find_map(|a| {
        if a.starts_with(prefix) {
            Some(a.strip_prefix(prefix).unwrap().to_string())
        } else {
            None
        }
    })
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

    let n: usize = parse_arg("--size=")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N);
    let corpus_name = parse_arg("--corpus=")
        .or_else(|| parse_arg("--profile="))
        .unwrap_or_else(|| "default".to_string());
    let suite = parse_arg("--suite=").unwrap_or_else(|| "default".to_string());

    let corpora = generate_corpora(&corpus_name, n);
    let mut dst = vec![0.0; n];
    let mut dst2 = vec![0.0; n];

    if suite == "default" || suite == "all" || suite == "standard" {
        header();
        row!("exp", &corpora.expargs, &mut dst, f64::exp, Exp);
        row!("exp2", &corpora.expargs, &mut dst, f64::exp2, Exp2);
        row!("expm1", &corpora.expargs, &mut dst, f64::exp_m1, Expm1);
        row!("ln", &corpora.posargs, &mut dst, f64::ln, Ln);
        row!("log2", &corpora.posargs, &mut dst, f64::log2, Log2);
        row!("log10", &corpora.posargs, &mut dst, f64::log10, Log10);
        row!("log1p", &corpora.posargs, &mut dst, f64::ln_1p, Log1p);
        row!("cbrt", &corpora.posargs, &mut dst, f64::cbrt, Cbrt);
        row!("sqrt", &corpora.posargs, &mut dst, f64::sqrt, Sqrt);

        header();
        row!("sin", &corpora.angargs, &mut dst, f64::sin, Sin);
        row!("cos", &corpora.angargs, &mut dst, f64::cos, Cos);
        row_pair!(
            "sincos",
            &corpora.angargs,
            &mut dst,
            &mut dst2,
            f64::sin_cos,
            SinCos
        );
        row!("tan", &corpora.angargs, &mut dst, f64::tan, Tan);
        row!("asin", &corpora.unitargs, &mut dst, f64::asin, Asin);
        row!("acos", &corpora.unitargs, &mut dst, f64::acos, Acos);
        row!("atan", &corpora.angargs, &mut dst, f64::atan, Atan);

        header();
        row!("sinh", &corpora.hypargs, &mut dst, f64::sinh, Sinh);
        row!("cosh", &corpora.hypargs, &mut dst, f64::cosh, Cosh);
        row!("tanh", &corpora.hypargs, &mut dst, f64::tanh, Tanh);
        row!("asinh", &corpora.hypargs, &mut dst, f64::asinh, Asinh);
        row!("acosh", &corpora.posargs, &mut dst, f64::acosh, Acosh);
        row!("atanh", &corpora.unitargs, &mut dst, f64::atanh, Atanh);

        header();
        row!("exp10", &corpora.expargs, &mut dst, |x| libm::exp10(x), Exp10);
        row!("erf", &corpora.erfargs, &mut dst, |x| libm::erf(x), Erf);
        row!("erfc", &corpora.erfcargs, &mut dst, |x| libm::erfc(x), Erfc);

        header();
        row!("j0", &corpora.besselargs, &mut dst, |x| libm::j0(x), J0);
        row!("j1", &corpora.besselargs, &mut dst, |x| libm::j1(x), J1);
        row!("y0", &corpora.besselargs, &mut dst, |x| libm::y0(x), Y0);
        row!("y1", &corpora.besselargs, &mut dst, |x| libm::y1(x), Y1);

        header();
        row!("floor", &corpora.expargs, &mut dst, f64::floor, Floor);
        row!("round", &corpora.expargs, &mut dst, f64::round, Round);
        row!("trunc", &corpora.expargs, &mut dst, f64::trunc, Trunc);

        // Binary functions and the Gamma pair do not fit the unary macro.
        println!(
            "\n{:<10} {:>10} {:>9} {:>9}",
            "function", "libm ns", "exact", "fast"
        );
        println!("{}", "-".repeat(42));
        binary(
            "pow",
            &corpora.posargs,
            &corpora.expargs,
            &mut dst,
            f64::powf,
            Pow::new(),
            Pow::builder().accuracy(Fast).domain(FullRange).build(),
        );
        binary(
            "atan2",
            &corpora.angargs,
            &corpora.posargs,
            &mut dst,
            f64::atan2,
            Atan2::new(),
            Atan2::builder().accuracy(Fast).domain(FullRange).build(),
        );
        binary(
            "hypot",
            &corpora.posargs,
            &corpora.angargs,
            &mut dst,
            f64::hypot,
            Hypot::new(),
            Hypot::builder().accuracy(Fast).domain(FullRange).build(),
        );

        println!("\n{:<10} {:>10} {:>9}", "function", "libm ns", "rmath");
        println!("{}", "-".repeat(32));
        gamma("lgamma", &corpora.hypargs, &mut dst);
        let mut grng = Rng(0x9E37_79B9_7F4A_7C15);
        let tgargs_direct: Vec<f64> = (0..n).map(|_| grng.uniform(0.0, 18.0)).collect();
        let tgargs_stirling: Vec<f64> = (0..n).map(|_| grng.uniform(18.0, 171.0)).collect();
        tgamma_bench("tgamma (direct)", &tgargs_direct, &mut dst);
        tgamma_bench("tgamma (stirling)", &tgargs_stirling, &mut dst);

        single_precision(n);
    }

    if suite == "traversal" || suite == "all" {
        traversal_benchmarks();
    }

    if suite == "repair" || suite == "all" {
        repair_benchmarks(n);
    }

    write_csv(&corpus_name, n, &suite);
}

#[allow(clippy::type_complexity)]
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
        for _ in 0..WARMUP {
            for i in 0..xs.len() {
                dst[i] = scalar(xs[i], ys[i]);
            }
        }
        let mut times = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let t0 = Instant::now();
            for i in 0..xs.len() {
                dst[i] = scalar(xs[i], ys[i]);
            }
            times.push(t0.elapsed().as_secs_f64());
        }
        compute_stats(times, xs.len()).0
    };
    let mut one = |f: &dyn Fn(&[f64], &[f64], &mut [f64])| {
        for _ in 0..WARMUP {
            f(xs, ys, dst);
        }
        let mut times = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let t = Instant::now();
            f(xs, ys, dst);
            times.push(t.elapsed().as_secs_f64());
        }
        compute_stats(times, xs.len()).0
    };
    let a = one(&|x, y, d| exact.eval_slice(x, y, d));
    let b = one(&|x, y, d| fast.eval_slice(x, y, d));
    println!(
        "{name:<10} {base:>10.2} {:>8.2}x {:>8.2}x",
        base / a,
        base / b
    );
    record(name, "exact", base / a, a);
    record(name, "fast", base / b, b);
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
    record(name, "rmath", base / t, t);
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
    record(name, "rmath", base / t, t);
}

fn single_precision(n: usize) {
    println!(
        "\nsingle precision\n{:<10} {:>10} {:>9} {:>9}",
        "function", "libm ns", "exact", "fast"
    );
    println!("{}", "-".repeat(42));
    let mut rng = Rng(0xB5AD_4ECE_DA10_80CC);
    let e: Vec<f32> = (0..n).map(|_| rng.uniform(-40.0, 40.0) as f32).collect();
    let p: Vec<f32> = (0..n)
        .map(|_| rng.uniform((1e-12f64).ln(), (1e12f64).ln()).exp() as f32)
        .collect();
    let mut d = vec![0.0f32; n];

    macro_rules! row32 {
        ($name:literal, $src:expr, $scalar:expr, $obj:ident) => {{
            let base = {
                for _ in 0..WARMUP {
                    for (x, o) in $src.iter().zip(d.iter_mut()) {
                        *o = $scalar(*x);
                    }
                }
                let mut times = Vec::with_capacity(SAMPLES);
                for _ in 0..SAMPLES {
                    let t = Instant::now();
                    for (x, o) in $src.iter().zip(d.iter_mut()) {
                        *o = $scalar(*x);
                    }
                    times.push(t.elapsed().as_secs_f64());
                }
                compute_stats(times, n).0
            };
            let mut run = |f: &dyn Fn(&[f32], &mut [f32])| {
                for _ in 0..WARMUP {
                    f($src, &mut d);
                }
                let mut times = Vec::with_capacity(SAMPLES);
                for _ in 0..SAMPLES {
                    let t = Instant::now();
                    f($src, &mut d);
                    times.push(t.elapsed().as_secs_f64());
                }
                compute_stats(times, n).0
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
            record($name, "exact", base / ta, ta);
            record($name, "fast", base / tc, tc);
        }};
    }
    row32!("expf", &e, f32::exp, Exp);
    row32!("exp2f", &e, f32::exp2, Exp2);
    row32!("logf", &p, f32::ln, Ln);
    row32!("log2f", &p, f32::log2, Log2);
    row32!("sqrtf", &p, f32::sqrt, Sqrt);
    row32!("cbrtf", &p, f32::cbrt, Cbrt);
    let tanh32args: Vec<f32> = (0..n).map(|_| rng.uniform(-9.0, 9.0) as f32).collect();
    row32!("tanhf", &tanh32args, f32::tanh, Tanh);
    let ang32: Vec<f32> = (0..n).map(|_| rng.uniform(-700.0, 700.0) as f32).collect();
    row32!("sinf", &ang32, f32::sin, Sin);
    row32!("cosf", &ang32, f32::cos, Cos);
    row32!("tanf", &ang32, f32::tan, Tan);
    let erf32: Vec<f32> = (0..n).map(|_| rng.uniform(-6.0, 6.0) as f32).collect();
    let erfc32: Vec<f32> = (0..n).map(|_| rng.uniform(-6.0, 10.0) as f32).collect();
    let exp10args: Vec<f32> = (0..n).map(|_| rng.uniform(-35.0, 35.0) as f32).collect();
    row32!("exp10f", &exp10args, |x| libm::exp10f(x), Exp10);
    row32!("erff", &erf32, |x| libm::erff(x), Erf);
    row32!("erfcf", &erfc32, |x| libm::erfcf(x), Erfc);
}

#[derive(Copy, Clone)]
struct Identity;
impl<E: Real> Function<E> for Identity {
    #[inline(always)]
    fn eval<V: Simd<Elem = E>>(&self, x: V) -> V {
        x
    }
}

#[derive(Copy, Clone)]
struct Identity2;
impl<E: Real> Function2<E> for Identity2 {
    #[inline(always)]
    fn eval<V: Simd<Elem = E>>(&self, x: V, _y: V) -> V {
        x
    }
}

#[derive(Copy, Clone)]
struct IdentityPair;
impl<E: Real> FunctionPair<E> for IdentityPair {
    #[inline(always)]
    fn eval<V: Simd<Elem = E>>(&self, x: V) -> (V, V) {
        (x, x)
    }
}

#[derive(Copy, Clone)]
struct Identity2Pair;
impl<E: Real> Function2Pair<E> for Identity2Pair {
    #[inline(always)]
    fn eval<V: Simd<Elem = E>>(&self, x: V, y: V) -> (V, V) {
        (x, y)
    }
}

fn traversal_benchmarks() {
    println!("\ntraversal and buffer shapes (ns/elem)");
    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "workload", "len=0", "len=1", "len=lanes-1", "len=lanes", "len=lanes+1", "len=64", "len=4096", "len=65536"
    );
    println!("{}", "-".repeat(102));

    let lanes = <rmath::Widest as Simd>::LANES;
    let l_minus_1 = lanes.saturating_sub(1).max(1);
    let l_plus_1 = lanes + 1;
    let lens = [0, 1, l_minus_1, lanes, l_plus_1, 64, 4096, 65536];

    let identity = Identity;
    let identity2 = Identity2;
    let identity_pair = IdentityPair;
    let identity_2pair = Identity2Pair;

    let exp = Exp::new();
    let sqrt = Sqrt::new();
    let floor = Floor::new();
    let pow = Pow::new();
    let sincos = SinCos::new();
    let remquo = Remquo::new();

    let time_iters = |len: usize, mut run: Box<dyn FnMut()>| -> f64 {
        if len == 0 {
            return 0.0;
        }
        let iters = (65536 / len.max(1)).clamp(10, 10000);
        for _ in 0..WARMUP {
            for _ in 0..iters {
                run();
            }
        }
        let mut times = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let t = Instant::now();
            for _ in 0..iters {
                run();
            }
            times.push(t.elapsed().as_secs_f64() / iters as f64);
        }
        compute_stats(times, len).0
    };

    let bench_lens = |name: &str, mut make_runner: Box<dyn FnMut(usize) -> Box<dyn FnMut()>>| {
        let times: Vec<f64> = lens.iter().map(|&len| time_iters(len, make_runner(len))).collect();
        println!(
            "{:<30} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            name, times[0], times[1], times[2], times[3], times[4], times[5], times[6], times[7]
        );
        for (&len, &t) in lens.iter().zip(times.iter()) {
            record(format!("{name}_len{len}"), "ns_per_elem", 1.0, t);
        }
    };

    // Identity floor benchmarks
    bench_lens("identity_unary", Box::new(|len| {
        let src = vec![1.5f64; len];
        let mut dst = vec![0.0f64; len];
        Box::new(move || {
            identity.eval_slice(&src, &mut dst);
            std::hint::black_box(dst.first().copied());
        })
    }));

    bench_lens("identity_in_place", Box::new(|len| {
        let mut buf = vec![1.5f64; len];
        Box::new(move || {
            identity.eval_in_place(&mut buf);
            std::hint::black_box(buf.first().copied());
        })
    }));

    bench_lens("identity_binary", Box::new(|len| {
        let a = vec![2.0f64; len];
        let b = vec![3.0f64; len];
        let mut dst = vec![0.0f64; len];
        Box::new(move || {
            identity2.eval_slice(&a, &b, &mut dst);
            std::hint::black_box(dst.first().copied());
        })
    }));

    bench_lens("identity_scalar_second", Box::new(|len| {
        let a = vec![2.0f64; len];
        let mut dst = vec![0.0f64; len];
        Box::new(move || {
            identity2.eval_slice_scalar(&a, 3.0, &mut dst);
            std::hint::black_box(dst.first().copied());
        })
    }));

    bench_lens("identity_pair", Box::new(|len| {
        let src = vec![1.0f64; len];
        let mut d1 = vec![0.0f64; len];
        let mut d2 = vec![0.0f64; len];
        Box::new(move || {
            identity_pair.eval_slice(&src, &mut d1, &mut d2);
            std::hint::black_box(d1.first().copied());
        })
    }));

    bench_lens("identity_2pair", Box::new(|len| {
        let a = vec![1.0f64; len];
        let b = vec![2.0f64; len];
        let mut d1 = vec![0.0f64; len];
        let mut d2 = vec![0.0f64; len];
        Box::new(move || {
            identity_2pair.eval_slice(&a, &b, &mut d1, &mut d2);
            std::hint::black_box(d1.first().copied());
        })
    }));

    // Real kernels
    bench_lens("sqrt_unary", Box::new(|len| {
        let src = vec![4.0f64; len];
        let mut dst = vec![0.0f64; len];
        Box::new(move || {
            sqrt.eval_slice(&src, &mut dst);
            std::hint::black_box(dst.first().copied());
        })
    }));

    bench_lens("floor_unary", Box::new(|len| {
        let src = vec![4.5f64; len];
        let mut dst = vec![0.0f64; len];
        Box::new(move || {
            floor.eval_slice(&src, &mut dst);
            std::hint::black_box(dst.first().copied());
        })
    }));

    bench_lens("exp_unary", Box::new(|len| {
        let src = vec![1.5f64; len];
        let mut dst = vec![0.0f64; len];
        Box::new(move || {
            exp.eval_slice(&src, &mut dst);
            std::hint::black_box(dst.first().copied());
        })
    }));

    bench_lens("pow_binary", Box::new(|len| {
        let a = vec![2.0f64; len];
        let b = vec![3.0f64; len];
        let mut dst = vec![0.0f64; len];
        Box::new(move || {
            pow.eval_slice(&a, &b, &mut dst);
            std::hint::black_box(dst.first().copied());
        })
    }));

    bench_lens("sincos_pair", Box::new(|len| {
        let src = vec![1.0f64; len];
        let mut d1 = vec![0.0f64; len];
        let mut d2 = vec![0.0f64; len];
        Box::new(move || {
            sincos.eval_slice(&src, &mut d1, &mut d2);
            std::hint::black_box(d1.first().copied());
        })
    }));

    bench_lens("remquo_2pair", Box::new(|len| {
        let a = vec![5.0f64; len];
        let b = vec![2.0f64; len];
        let mut d1 = vec![0.0f64; len];
        let mut d2 = vec![0.0f64; len];
        Box::new(move || {
            remquo.eval_slice(&a, &b, &mut d1, &mut d2);
            std::hint::black_box(d1.first().copied());
        })
    }));
}

fn repair_benchmarks(n: usize) {
    println!("\nscalar repair density overhead (ns/elem)");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "kernel", "0% repair", "1-lane", "25% repair", "50% repair", "100% repair"
    );
    println!("{}", "-".repeat(76));

    let lanes = <rmath::Widest as Simd>::LANES;
    let exp = Exp::new();
    let erf = Erf::new();
    let sin = Sin::new();
    let pow = Pow::new();
    let atan2 = Atan2::new();
    let sincos = SinCos::new();
    let modf = Modf::new();

    let run_density = |name: &str, make_vec: &dyn Fn(f64) -> Vec<f64>| {
        let densities = [0.0, 1.0 / (lanes as f64), 0.25, 0.50, 1.0];
        let mut times = [0.0; 5];
        for (i, &d) in densities.iter().enumerate() {
            let src = make_vec(d);
            let mut dst = vec![0.0; n];
            let mut dst2 = vec![0.0; n];
            let mut times_samples = Vec::with_capacity(SAMPLES);
            for _ in 0..WARMUP {
                match name {
                    "exp" => exp.eval_slice(&src, &mut dst),
                    "erf" => erf.eval_slice(&src, &mut dst),
                    "sin" => sin.eval_slice(&src, &mut dst),
                    "pow" => pow.eval_slice(&src, &src, &mut dst),
                    "atan2" => atan2.eval_slice(&src, &src, &mut dst),
                    "sincos" => sincos.eval_slice(&src, &mut dst, &mut dst2),
                    "modf" => modf.eval_slice(&src, &mut dst, &mut dst2),
                    _ => {}
                }
            }
            for _ in 0..SAMPLES {
                let t = Instant::now();
                match name {
                    "exp" => exp.eval_slice(&src, &mut dst),
                    "erf" => erf.eval_slice(&src, &mut dst),
                    "sin" => sin.eval_slice(&src, &mut dst),
                    "pow" => pow.eval_slice(&src, &src, &mut dst),
                    "atan2" => atan2.eval_slice(&src, &src, &mut dst),
                    "sincos" => sincos.eval_slice(&src, &mut dst, &mut dst2),
                    "modf" => modf.eval_slice(&src, &mut dst, &mut dst2),
                    _ => {}
                }
                times_samples.push(t.elapsed().as_secs_f64());
            }
            times[i] = compute_stats(times_samples, n).0;
            record(
                format!("repair_{name}_{:.0}pct", d * 100.0),
                "ns_per_elem",
                1.0,
                times[i],
            );
        }
        println!(
            "{:<20} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            name, times[0], times[1], times[2], times[3], times[4]
        );
    };

    // exp repairs on x > 709.7 or x < -745.0 or NaN
    run_density("exp", &|pct| {
        let step = if pct <= 0.0 { usize::MAX } else { (1.0 / pct).round() as usize };
        (0..n).map(|i| if step > 0 && i % step == 0 { 750.0 } else { 1.5 }).collect()
    });

    // erf repairs on |x| >= 6.0 or special
    run_density("erf", &|pct| {
        let step = if pct <= 0.0 { usize::MAX } else { (1.0 / pct).round() as usize };
        (0..n).map(|i| if step > 0 && i % step == 0 { 10.0 } else { 1.5 }).collect()
    });

    // sin repairs on |x| >= 804.0 (for bit exact reduction)
    run_density("sin", &|pct| {
        let step = if pct <= 0.0 { usize::MAX } else { (1.0 / pct).round() as usize };
        (0..n).map(|i| if step > 0 && i % step == 0 { 10000.0 } else { 1.5 }).collect()
    });

    // pow repairs on non-normal/extreme
    run_density("pow", &|pct| {
        let step = if pct <= 0.0 { usize::MAX } else { (1.0 / pct).round() as usize };
        (0..n).map(|i| if step > 0 && i % step == 0 { f64::NAN } else { 2.0 }).collect()
    });

    // atan2 repairs on zero/infinite/NaN
    run_density("atan2", &|pct| {
        let step = if pct <= 0.0 { usize::MAX } else { (1.0 / pct).round() as usize };
        (0..n).map(|i| if step > 0 && i % step == 0 { 0.0 } else { 2.0 }).collect()
    });

    // sincos repairs on large angles
    run_density("sincos", &|pct| {
        let step = if pct <= 0.0 { usize::MAX } else { (1.0 / pct).round() as usize };
        (0..n).map(|i| if step > 0 && i % step == 0 { 1e8 } else { 1.5 }).collect()
    });

    // modf repairs on inf/nan
    run_density("modf", &|pct| {
        let step = if pct <= 0.0 { usize::MAX } else { (1.0 / pct).round() as usize };
        (0..n).map(|i| if step > 0 && i % step == 0 { f64::INFINITY } else { 1.5 }).collect()
    });
}
