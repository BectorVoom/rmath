//! Direct `f64x8` `cbrt` timing, bypassing `Function::eval_slice`'s
//! buffer-walk indirection, to isolate `src/kernels/double/cbrt.rs`'s
//! vectorised decompose/recompose from any load_slice/store_slice/chunking
//! overhead the general harness (`examples/bench.rs`) adds. Prints one
//! number, ns/elem, best of `REPS`:
//!
//!     RUSTFLAGS="-C target-cpu=native" cargo run --release --example cbrt_direct
//!
//! Useful as a before/after: `git stash -- src/kernels/double/cbrt.rs`
//! between two runs isolates a change to that one file from everything
//! else. On a busy machine, run it several times — background load on
//! other cores can still perturb a single-threaded wall-clock loop.
use rmath::prelude::*;
#[cfg(feature = "wide")]
use wide::f64x8;
use std::time::Instant;

const N: usize = 1 << 20;
const REPS: usize = 20;

fn main() {
    let mut seed = 0x243F6A8885A308D3u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let xs: Vec<f64> = (0..N)
        .map(|_| {
            let u = (next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            0.1 + u * 99.9
        })
        .collect();

    let f = Cbrt::<BitExact, FullRange>::new();
    let mut best = f64::INFINITY;
    let mut sum = 0.0f64;
    for _ in 0..REPS {
        let t = Instant::now();
        let mut s = 0.0f64;
        for chunk in xs.chunks_exact(8) {
            let mut b = [0.0; 8];
            b.copy_from_slice(chunk);
            let v = f64x8::new(b);
            let r = f.eval(v);
            let a: [f64; 8] = r.into();
            s += a.iter().sum::<f64>();
        }
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
        sum = s;
    }
    assert!(sum.is_finite());
    println!("{:.3}", best * 1e9 / N as f64);
}
