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
