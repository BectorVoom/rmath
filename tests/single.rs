//! Single-precision bit-exactness, checked against the platform `libm`.
//!
//! `f32` admits a much stronger test than `f64` does: there are only 2^32
//! inputs, so a unary function can be checked on *every* one of them rather
//! than on a sample. The full sweep is behind `--ignored` because it takes
//! minutes; what runs by default is the same sweep at a stride, plus every
//! special value and branch boundary, which is already far more coverage than
//! the double-precision sweeps get.
//!
//! ```text
//! cargo test --release --test single
//! cargo test --release --test single -- --ignored --nocapture   # all 2^32
//! ```
//!
//! The comparison is against `libm` through `extern "C"`, not through Rust's
//! `f32` methods. Those agree on this platform, but the claim rmath makes is
//! about the C library, so that is what it is held to.

use rmath::prelude::*;

/// The platform's own `float` routines.
///
/// Declared rather than depended on: `std` already links `libm`, so this needs
/// no dev-dependency, and it names exactly the functions the bit-exactness
/// claim is about.
mod libm {
    unsafe extern "C" {
        pub safe fn expf(x: f32) -> f32;
        pub safe fn exp2f(x: f32) -> f32;
        pub safe fn logf(x: f32) -> f32;
        pub safe fn log2f(x: f32) -> f32;
        pub safe fn log10f(x: f32) -> f32;
        pub safe fn log1pf(x: f32) -> f32;
        pub safe fn expm1f(x: f32) -> f32;
        pub safe fn cbrtf(x: f32) -> f32;
        pub safe fn sqrtf(x: f32) -> f32;
        pub safe fn sinf(x: f32) -> f32;
        pub safe fn cosf(x: f32) -> f32;
        pub safe fn tanf(x: f32) -> f32;
        pub safe fn asinf(x: f32) -> f32;
        pub safe fn acosf(x: f32) -> f32;
        pub safe fn atanf(x: f32) -> f32;
        pub safe fn sinhf(x: f32) -> f32;
        pub safe fn coshf(x: f32) -> f32;
        pub safe fn tanhf(x: f32) -> f32;
        pub safe fn asinhf(x: f32) -> f32;
        pub safe fn acoshf(x: f32) -> f32;
        pub safe fn roundf(x: f32) -> f32;
        pub safe fn sincosf(x: f32, s: *mut f32, c: *mut f32);
        pub safe fn atan2f(y: f32, x: f32) -> f32;
        pub safe fn powf(x: f32, y: f32) -> f32;
        pub safe fn hypotf(x: f32, y: f32) -> f32;
    }
}

/// Every `f32` whose bit pattern is `start + k * stride`, wrapping once.
fn sweep(stride: u32, mut f: impl FnMut(f32)) {
    let mut b: u32 = 0;
    loop {
        f(f32::from_bits(b));
        let (next, carry) = b.overflowing_add(stride);
        if carry {
            break;
        }
        b = next;
    }
}

/// Compare one configured function against the platform routine.
///
/// Both arguments go through `black_box`. Without it LLVM recognises `expf`
/// and friends as libcalls it may evaluate itself, and it evaluates them
/// correctly rounded — which is *not* what the platform does, so the
/// comparison quietly stops testing anything. That is not hypothetical: it hid
/// a real one-ulp difference in `expf` here until the barrier was added.
fn check(name: &str, stride: u32, ours: impl Fn(f32) -> f32, theirs: impl Fn(f32) -> f32) {
    use std::hint::black_box;
    let mut bad = 0u64;
    let mut shown = 0;
    let mut n = 0u64;
    sweep(stride, |x| {
        n += 1;
        let (a, b) = (ours(black_box(x)), theirs(black_box(x)));
        if a.to_bits() != b.to_bits() {
            bad += 1;
            if shown < 8 {
                shown += 1;
                eprintln!(
                    "{name}: x = {x:e} ({:#010x}) => ours {a:e} ({:#010x}), libm {b:e} ({:#010x})",
                    x.to_bits(),
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    });
    assert_eq!(
        bad, 0,
        "{name}: {bad} of {n} inputs differ from the platform libm"
    );
}

/// The stride the default (non-`--ignored`) run uses.
///
/// Coprime with 2^32 so the sweep visits every exponent and a well-spread set
/// of significands rather than a lattice of round numbers.
const STRIDE: u32 = 9_973;

macro_rules! suite {
    ($fast:ident, $slow:ident, $name:literal, $f:expr, $libm:path) => {
        #[test]
        fn $fast() {
            check($name, STRIDE, $f, |x| $libm(x));
        }

        #[test]
        #[ignore = "sweeps all 2^32 f32 inputs; minutes, not seconds"]
        fn $slow() {
            check($name, 1, $f, |x| $libm(x));
        }
    };
}

suite!(
    exp_matches_libm_sampled,
    exp_matches_libm_exhaustive,
    "expf",
    |x| Exp::new().eval(x),
    libm::expf
);
suite!(
    exp2_matches_libm_sampled,
    exp2_matches_libm_exhaustive,
    "exp2f",
    |x| Exp2::new().eval(x),
    libm::exp2f
);
suite!(
    ln_matches_libm_sampled,
    ln_matches_libm_exhaustive,
    "logf",
    |x| Ln::new().eval(x),
    libm::logf
);
suite!(
    cbrt_matches_libm_sampled,
    cbrt_matches_libm_exhaustive,
    "cbrtf",
    |x| Cbrt::new().eval(x),
    libm::cbrtf
);
suite!(
    sqrt_matches_libm_sampled,
    sqrt_matches_libm_exhaustive,
    "sqrtf",
    |x| Sqrt::new().eval(x),
    libm::sqrtf
);

suite!(
    log2_matches_libm_sampled,
    log2_matches_libm_exhaustive,
    "log2f",
    |x| Log2::new().eval(x),
    libm::log2f
);
suite!(
    log10_matches_libm_sampled,
    log10_matches_libm_exhaustive,
    "log10f",
    |x| Log10::new().eval(x),
    libm::log10f
);
suite!(
    log1p_matches_libm_sampled,
    log1p_matches_libm_exhaustive,
    "log1pf",
    |x| Log1p::new().eval(x),
    libm::log1pf
);
suite!(
    expm1_matches_libm_sampled,
    expm1_matches_libm_exhaustive,
    "expm1f",
    |x| Expm1::new().eval(x),
    libm::expm1f
);
suite!(
    sin_matches_libm_sampled,
    sin_matches_libm_exhaustive,
    "sinf",
    |x| Sin::new().eval(x),
    libm::sinf
);
suite!(
    cos_matches_libm_sampled,
    cos_matches_libm_exhaustive,
    "cosf",
    |x| Cos::new().eval(x),
    libm::cosf
);
suite!(
    tan_matches_libm_sampled,
    tan_matches_libm_exhaustive,
    "tanf",
    |x| Tan::new().eval(x),
    libm::tanf
);
suite!(
    asin_matches_libm_sampled,
    asin_matches_libm_exhaustive,
    "asinf",
    |x| Asin::new().eval(x),
    libm::asinf
);
suite!(
    acos_matches_libm_sampled,
    acos_matches_libm_exhaustive,
    "acosf",
    |x| Acos::new().eval(x),
    libm::acosf
);
suite!(
    atan_matches_libm_sampled,
    atan_matches_libm_exhaustive,
    "atanf",
    |x| Atan::new().eval(x),
    libm::atanf
);
suite!(
    sinh_matches_libm_sampled,
    sinh_matches_libm_exhaustive,
    "sinhf",
    |x| Sinh::new().eval(x),
    libm::sinhf
);
suite!(
    cosh_matches_libm_sampled,
    cosh_matches_libm_exhaustive,
    "coshf",
    |x| Cosh::new().eval(x),
    libm::coshf
);
suite!(
    tanh_matches_libm_sampled,
    tanh_matches_libm_exhaustive,
    "tanhf",
    |x| Tanh::new().eval(x),
    libm::tanhf
);
suite!(
    asinh_matches_libm_sampled,
    asinh_matches_libm_exhaustive,
    "asinhf",
    |x| Asinh::new().eval(x),
    libm::asinhf
);
suite!(
    acosh_matches_libm_sampled,
    acosh_matches_libm_exhaustive,
    "acoshf",
    |x| Acosh::new().eval(x),
    libm::acoshf
);
// `atanh` is compared against Rust's `f32::atanh`, not against the C
// `atanhf`, and that is the whole point rather than a dodge: Rust does not
// forward `atanh` to `libm`, it evaluates `0.5 * ln_1p(2x / (1 - x))` itself,
// and the two disagree on roughly one input in ten. The call a Rust caller was
// already making is `f32::atanh`, so that is what rmath must not change.
suite!(
    atanh_matches_rust_sampled,
    atanh_matches_rust_exhaustive,
    "atanhf",
    |x| Atanh::new().eval(x),
    f32::atanh
);
suite!(
    round_matches_libm_sampled,
    round_matches_libm_exhaustive,
    "roundf",
    |x| Round::new().eval(x),
    libm::roundf
);

// ---------------------------------------------------------------------------
// The functions `suite!` cannot express
// ---------------------------------------------------------------------------
//
// `sincosf` returns a pair, and `atan2f`/`powf`/`hypotf` take two arguments —
// a 2^64 domain, which no sweep can exhaust. They were delegating calls until
// the platform routines were ported, so they need their own coverage here or
// nothing in the repository would catch a regression in them.

/// `sincosf` through `extern "C"`, one half at a time.
fn platform_sincosf(x: f32) -> (f32, f32) {
    let (mut s, mut c) = (0.0f32, 0.0f32);
    libm::sincosf(x, &raw mut s, &raw mut c);
    (s, c)
}

/// Both halves at once, so a swapped pair cannot pass by testing one of them.
///
/// `suite!` takes a `path` for its reference and `sincosf` needs a wrapper, so
/// this spells the two tests out rather than going through the macro.
fn check_sincos(stride: u32) {
    check(
        "sincosf (sin)",
        stride,
        |x| SinCos::new().eval(x).0,
        |x| platform_sincosf(x).0,
    );
    check(
        "sincosf (cos)",
        stride,
        |x| SinCos::new().eval(x).1,
        |x| platform_sincosf(x).1,
    );
}

#[test]
fn sincos_matches_libm_sampled() {
    check_sincos(STRIDE);
}

#[test]
#[ignore = "sweeps all 2^32 f32 inputs; minutes, not seconds"]
fn sincos_matches_libm_exhaustive() {
    check_sincos(1);
}

/// xorshift64*, for the two-argument corpora.
struct Rng2(u64);

impl Rng2 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Compare a two-argument function over a structured random corpus.
///
/// Four shapes, cycled: raw bit patterns (which land on NaN and infinity far
/// more often than any realistic distribution), a raw base against a small
/// integer exponent, two non-negative bit patterns, and two values built from
/// a uniformly random exponent — which is the only one of the four that
/// reliably lands in the bands where `powf` over- and underflows.
fn check2(name: &str, n: u64, ours: impl Fn(f32, f32) -> f32, theirs: impl Fn(f32, f32) -> f32) {
    use std::hint::black_box;
    let mut g = Rng2(0x1234_5678_9abc_def1);
    let mut bad = 0u64;
    let mut shown = 0;
    for i in 0..n {
        let (x, y) = match i % 4 {
            0 => (
                f32::from_bits(g.next() as u32),
                f32::from_bits(g.next() as u32),
            ),
            1 => (
                f32::from_bits(g.next() as u32),
                ((g.next() % 128) as i32 - 64) as f32,
            ),
            2 => (
                f32::from_bits(g.next() as u32 & 0x7fff_ffff),
                f32::from_bits(g.next() as u32 & 0x7fff_ffff),
            ),
            _ => {
                let e = |v: u64| {
                    f32::from_bits(
                        ((v % 254 + 1) << 23) as u32 | (v as u32 >> 9) | ((v as u32 & 1) << 31),
                    )
                };
                (e(g.next()), e(g.next()))
            }
        };
        let (a, b) = (
            ours(black_box(x), black_box(y)),
            theirs(black_box(x), black_box(y)),
        );
        if a.to_bits() != b.to_bits() {
            bad += 1;
            if shown < 8 {
                shown += 1;
                eprintln!(
                    "{name}: x = {:#010x}, y = {:#010x} => ours {:#010x}, libm {:#010x}",
                    x.to_bits(),
                    y.to_bits(),
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
    assert_eq!(
        bad, 0,
        "{name}: {bad} of {n} input pairs differ from the platform libm"
    );
}

/// The pair count the default run uses. `--ignored` runs a hundred times more.
const PAIRS: u64 = 2_000_000;

macro_rules! suite2 {
    ($fast:ident, $slow:ident, $name:literal, $f:expr, $libm:expr) => {
        #[test]
        fn $fast() {
            check2($name, PAIRS, $f, $libm);
        }

        #[test]
        #[ignore = "200M input pairs; minutes, not seconds"]
        fn $slow() {
            check2($name, 100 * PAIRS, $f, $libm);
        }
    };
}

suite2!(
    atan2_matches_libm_sampled,
    atan2_matches_libm_large,
    "atan2f",
    |y, x| Atan2::new().eval(y, x),
    |y, x| libm::atan2f(y, x)
);
suite2!(
    pow_matches_libm_sampled,
    pow_matches_libm_large,
    "powf",
    |x, y| Pow::new().eval(x, y),
    |x, y| libm::powf(x, y)
);
suite2!(
    hypot_matches_libm_sampled,
    hypot_matches_libm_large,
    "hypotf",
    |x, y| Hypot::new().eval(x, y),
    |x, y| libm::hypotf(x, y)
);
