//! Throughput of each configuration against the scalar `libm` it replaces.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench
//! ```
//!
//! `target-cpu=native` matters and is not a detail: without the `fma` target
//! feature, `wide` has no fused multiply-add, and rmath substitutes a
//! per-lane scalar FMA to keep its bit-exactness promise. That is correct but
//! several times slower, and it is not the configuration anyone should measure.
//! The harness says so at startup rather than silently reporting bad numbers.
//!
//! Method: best-of-N over a buffer larger than L2, timing `eval_slice` against
//! a plain scalar loop over the same data. Best-of rather than mean, because
//! the quantity of interest is the achievable rate, and the noise here is all
//! one-directional.

use rmath::prelude::*;
use std::time::Instant;

const N: usize = 1 << 20;
const REPS: usize = 12;

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

fn bench(
    label: &str,
    src: &[f64],
    dst: &mut [f64],
    mut run: impl FnMut(&[f64], &mut [f64]),
) -> f64 {
    run(src, dst); // warm
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        run(src, dst);
        best = best.min(t.elapsed().as_secs_f64());
    }
    let ns = best * 1e9 / src.len() as f64;
    let checksum: f64 = dst.iter().filter(|v| v.is_finite()).sum();
    println!("    {label:<34} {ns:7.3} ns/elem    (sum {checksum:+.9e})");
    ns
}

/// One function, every configuration, against the scalar baseline.
macro_rules! suite {
    ($name:literal, $src:expr, $dst:expr, $scalar:expr, $ctor:expr) => {{
        println!("\n{} ({} elements, best of {REPS}):", $name, N);
        let base = bench("scalar std loop (the baseline)", $src, $dst, |s, d| {
            for (x, o) in s.iter().zip(d.iter_mut()) {
                *o = $scalar(*x);
            }
        });
        let exact_checked = $ctor(BitExact, FullRange);
        let exact_finite = $ctor(BitExact, Finite);
        let fast_checked = $ctor(Fast, FullRange);
        let fast_finite = $ctor(Fast, Finite);

        let a = bench("BitExact + FullRange", $src, $dst, |s, d| exact_checked.eval_slice(s, d));
        let b = bench("BitExact + Finite", $src, $dst, |s, d| exact_finite.eval_slice(s, d));
        let c = bench("Fast     + FullRange", $src, $dst, |s, d| fast_checked.eval_slice(s, d));
        let e = bench("Fast     + Finite", $src, $dst, |s, d| fast_finite.eval_slice(s, d));
        println!(
            "    -> speedup vs scalar: BitExact {:.2}x / {:.2}x finite, Fast {:.2}x / {:.2}x finite",
            base / a,
            base / b,
            base / c,
            base / e
        );
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
        "widest vector: {} lanes    fma: {}",
        <rmath::Widest as Simd>::LANES,
        cfg!(target_feature = "fma")
    );

    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    // exp/exp2 arguments: a range that neither overflows nor is trivially tiny.
    let expargs: Vec<f64> = (0..N).map(|_| rng.uniform(-40.0, 40.0)).collect();
    // ln/cbrt arguments: log-uniform positive magnitudes, as physical data tends to be.
    let posargs: Vec<f64> = (0..N)
        .map(|_| rng.uniform((1e-12f64).ln(), (1e12f64).ln()).exp())
        .collect();
    let mut dst = vec![0.0; N];

    suite!("exp", &expargs, &mut dst, f64::exp, |a, d| {
        Exp::builder().accuracy(a).domain(d).build()
    });
    suite!("exp2", &expargs, &mut dst, f64::exp2, |a, d| {
        Exp2::builder().accuracy(a).domain(d).build()
    });
    suite!("ln", &posargs, &mut dst, f64::ln, |a, d| {
        Ln::builder().accuracy(a).domain(d).build()
    });
    suite!("cbrt", &posargs, &mut dst, f64::cbrt, |a, d| {
        Cbrt::builder().accuracy(a).domain(d).build()
    });
    suite!("sqrt", &posargs, &mut dst, f64::sqrt, |a, d| {
        Sqrt::builder().accuracy(a).domain(d).build()
    });
}
