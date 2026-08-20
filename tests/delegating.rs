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

