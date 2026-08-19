//! `BitExact` must be bit-exact for *every* function, not just the ported ones.
//!
//! The functions in this file reach that guarantee by delegation rather than
//! by replaying a schedule (see [`rmath::reference`]), which makes it cheap to
//! state and easy to get wrong in a way no accuracy test would catch: a kernel
//! that quietly took its `Fast` path under `BitExact` would still look fine to
//! within a few ulp. So the check here is `to_bits()` equality against the
//! platform, at every vector width, exactly as for the ported functions.
//!
//! Lane independence matters as much as the values: a delegating kernel walks
//! the lane array by hand, and an off-by-one there would show up only at some
//! widths.

use rmath::prelude::*;
use rmath::simd::Lanes;

/// xorshift64*, so the corpus is reproducible without a dev-dependency.
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

/// Inputs every function is fed, whatever its domain.
fn universal() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 2.0,
        f64::from_bits(1),
        f64::MAX,
        f64::MIN,
    ]
}

fn check_width<V, F, G>(name: &str, vals: &[f64], scalar: F, vector: G)
where
    V: Simd<Elem = f64>,
    F: Fn(f64) -> f64,
    G: Fn(V) -> V,
{
    let mut bad = 0usize;
    let mut shown = 0;
    for chunk in vals.chunks(V::LANES) {
        let mut lanes = V::Floats::filled_default();
        for slot in lanes.as_mut_slice() {
            *slot = chunk[0];
        }
        lanes.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
        let got = vector(V::from_array(lanes)).to_array();
        for i in 0..chunk.len() {
            let x = lanes.as_slice()[i];
            let (want, have) = (scalar(x), got.as_slice()[i]);
            if want.to_bits() != have.to_bits() {
                bad += 1;
                if shown < 5 {
                    shown += 1;
                    eprintln!(
                        "{name} @{} lanes: x = {x:e} => want {want:e} ({:#018x}), got {have:e} ({:#018x})",
                        V::LANES,
                        want.to_bits(),
                        have.to_bits()
                    );
                }
            }
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} of {} lanes differ", vals.len());
}

macro_rules! all_widths {
    ($name:expr, $vals:expr, $scalar:expr, $f:expr) => {{
        let f = $f;
        check_width::<f64, _, _>($name, $vals, $scalar, |x| f.eval(x));
        #[cfg(feature = "wide")]
        {
            check_width::<wide::f64x2, _, _>($name, $vals, $scalar, |x| f.eval(x));
            check_width::<wide::f64x4, _, _>($name, $vals, $scalar, |x| f.eval(x));
            check_width::<wide::f64x8, _, _>($name, $vals, $scalar, |x| f.eval(x));
        }
    }};
}

/// A corpus spanning the whole real line plus the specials.
fn wide_corpus(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for _ in 0..120_000 {
        v.push(rng.uniform(-700.0, 700.0));
    }
    for _ in 0..120_000 {
        v.push(rng.log_uniform(1e-300, 1e300, true));
    }
    for _ in 0..60_000 {
        v.push(f64::from_bits(rng.next()));
    }
    v
}

/// A corpus concentrated on `[-1, 1]`, for the functions defined there.
fn unit_corpus(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for _ in 0..150_000 {
        v.push(rng.uniform(-1.0, 1.0));
    }
    for _ in 0..60_000 {
        v.push(rng.uniform(-1.5, 1.5));
    }
    for _ in 0..60_000 {
        v.push(f64::from_bits(rng.next()));
    }
    v
}

macro_rules! suite {
    ($test:ident, $name:literal, $obj:ident, $corpus:expr, $scalar:expr) => {
        #[test]
        fn $test() {
            all_widths!($name, &$corpus, $scalar, $obj::new());
        }
    };
}

suite!(tan_is_bit_exact, "tan", Tan, wide_corpus(3), f64::tan);
suite!(asin_is_bit_exact, "asin", Asin, unit_corpus(4), f64::asin);
suite!(acos_is_bit_exact, "acos", Acos, unit_corpus(5), f64::acos);
suite!(atan_is_bit_exact, "atan", Atan, wide_corpus(6), f64::atan);
suite!(sinh_is_bit_exact, "sinh", Sinh, wide_corpus(7), f64::sinh);
suite!(cosh_is_bit_exact, "cosh", Cosh, wide_corpus(8), f64::cosh);
suite!(tanh_is_bit_exact, "tanh", Tanh, wide_corpus(9), f64::tanh);
suite!(
    asinh_is_bit_exact,
    "asinh",
    Asinh,
    wide_corpus(10),
    f64::asinh
);
suite!(
    acosh_is_bit_exact,
    "acosh",
    Acosh,
    wide_corpus(11),
    f64::acosh
);
suite!(
    atanh_is_bit_exact,
    "atanh",
    Atanh,
    unit_corpus(12),
    f64::atanh
);
suite!(log2_is_bit_exact, "log2", Log2, wide_corpus(13), f64::log2);
suite!(
    log10_is_bit_exact,
    "log10",
    Log10,
    wide_corpus(14),
    f64::log10
);
suite!(
    log1p_is_bit_exact,
    "log1p",
    Log1p,
    wide_corpus(15),
    f64::ln_1p
);
suite!(
    expm1_is_bit_exact,
    "expm1",
    Expm1,
    wide_corpus(16),
    f64::exp_m1
);

/// Compare a two-argument function against the platform, at one width.
fn check_width2<V, F, G>(name: &str, xs: &[f64], ys: &[f64], scalar: F, vector: G)
where
    V: Simd<Elem = f64>,
    F: Fn(f64, f64) -> f64,
    G: Fn(V, V) -> V,
{
    let mut bad = 0usize;
    let mut shown = 0;
    for (cx, cy) in xs.chunks(V::LANES).zip(ys.chunks(V::LANES)) {
        let mut lx = V::Floats::filled_default();
        let mut ly = V::Floats::filled_default();
        for k in 0..V::LANES {
            lx.as_mut_slice()[k] = cx[0];
            ly.as_mut_slice()[k] = cy[0];
        }
        lx.as_mut_slice()[..cx.len()].copy_from_slice(cx);
        ly.as_mut_slice()[..cy.len()].copy_from_slice(cy);
        let got = vector(V::from_array(lx), V::from_array(ly)).to_array();
        for i in 0..cx.len().min(cy.len()) {
            let (x, y) = (lx.as_slice()[i], ly.as_slice()[i]);
            let (want, have) = (scalar(x, y), got.as_slice()[i]);
            if want.to_bits() != have.to_bits() {
                bad += 1;
                if shown < 5 {
                    shown += 1;
                    eprintln!(
                        "{name} @{} lanes: ({x:e}, {y:e}) => want {want:e} ({:#018x}), got {have:e} ({:#018x})",
                        V::LANES,
                        want.to_bits(),
                        have.to_bits()
                    );
                }
            }
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} lanes differ");
}

macro_rules! all_widths2 {
    ($name:expr, $xs:expr, $ys:expr, $scalar:expr, $f:expr) => {{
        let f = $f;
        check_width2::<f64, _, _>($name, $xs, $ys, $scalar, |a, b| f.eval(a, b));
        #[cfg(feature = "wide")]
        {
            check_width2::<wide::f64x2, _, _>($name, $xs, $ys, $scalar, |a, b| f.eval(a, b));
            check_width2::<wide::f64x4, _, _>($name, $xs, $ys, $scalar, |a, b| f.eval(a, b));
            check_width2::<wide::f64x8, _, _>($name, $xs, $ys, $scalar, |a, b| f.eval(a, b));
        }
    }};
}

/// The two-argument functions, at every width.
///
/// `pow` is a port, so this is the same class of check `tests/bit_exact.rs`
/// makes of `exp` — and the width sweep matters more here than for the
/// delegating pair, because the vector code walks two lane arrays by hand.
#[test]
fn binary_functions_are_bit_exact() {
    let vals = wide_corpus(18);
    let n = vals.len() / 2;
    let (xs, ys) = (&vals[..n], &vals[n..2 * n]);
    all_widths2!("pow", xs, ys, f64::powf, Pow::new());
    all_widths2!("atan2", xs, ys, f64::atan2, Atan2::new());
    all_widths2!("hypot", xs, ys, f64::hypot, Hypot::new());
}

/// `pow` over the ranges its own main path actually covers.
///
/// The shared corpus is dominated by inputs that leave the vector path
/// immediately — random bit patterns are NaN or out of range far more often
/// than not — so this adds a sweep that stays inside it, where the ported
/// schedule is what is being tested rather than the repair.
#[test]
fn pow_is_bit_exact_on_its_main_path() {
    let mut rng = Rng(0xF00D_BABE_1234_5678);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..250_000 {
        xs.push(rng.log_uniform(1e-300, 1e300, false));
        ys.push(rng.uniform(-40.0, 40.0));
    }
    for _ in 0..100_000 {
        xs.push(rng.uniform(0.0, 4.0));
        ys.push(rng.uniform(-200.0, 200.0));
    }
    // Exact powers, where a seeded reconstruction is most likely to be off.
    for i in 1..500u64 {
        xs.push(i as f64);
        ys.push(((i % 7) + 1) as f64);
    }
    all_widths2!("pow/main", &xs, &ys, f64::powf, Pow::new());
}
