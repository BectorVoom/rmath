//! What the two policy axes actually promise.
//!
//! * [`rmath::Fast`] is not bit-exact — that is its purpose — so it is held to
//!   a measured ulp bound instead. The bounds asserted here are the ones the
//!   README quotes; if a change to a kernel moves them, this fails and the
//!   README is wrong.
//! * [`rmath::Finite`] must agree with [`rmath::FullRange`] on every input
//!   inside the documented domain, and is allowed to differ only outside it.
//! * The builder must be genuinely zero-cost.

use rmath::prelude::*;
use rmath::reference;

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

/// Error in units in the last place of the correctly-rounded result.
fn ulp_error(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    if !got.is_finite() || !want.is_finite() {
        return if got.to_bits() == want.to_bits() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    // ulp at `want`, taken from its exponent.
    let e = ((want.abs().to_bits() >> 52) as i64) - 1023;
    let ulp = f64::from_bits((((e - 52) + 1023) as u64) << 52);
    (got - want).abs() / ulp
}

fn worst_ulp(vals: &[f64], f: impl Fn(f64) -> f64, reference: impl Fn(f64) -> f64) -> (f64, f64) {
    let mut worst = 0.0;
    let mut at = 0.0;
    for &x in vals {
        let e = ulp_error(f(x), reference(x));
        if e > worst {
            worst = e;
            at = x;
        }
    }
    (worst, at)
}

#[test]
fn fast_exp_stays_within_its_documented_bound() {
    let mut rng = Rng(0x1234_5678);
    let vals: Vec<f64> = (0..300_000).map(|_| rng.uniform(-500.0, 500.0)).collect();
    let f = Exp::builder().accuracy(Fast).build();
    let (worst, at) = worst_ulp(&vals, |x| f.eval(x), reference::exp);
    assert!(
        worst <= 2.0,
        "Fast exp worst error {worst} ulp at x = {at:e} (bound: 2 ulp)"
    );
    eprintln!("Fast exp: worst {worst:.3} ulp");
}

#[test]
fn fast_exp2_stays_within_its_documented_bound() {
    let mut rng = Rng(0x8765_4321);
    let vals: Vec<f64> = (0..300_000).map(|_| rng.uniform(-500.0, 500.0)).collect();
    let f = Exp2::builder().accuracy(Fast).build();
    let (worst, at) = worst_ulp(&vals, |x| f.eval(x), reference::exp2);
    assert!(
        worst <= 2.0,
        "Fast exp2 worst error {worst} ulp at x = {at:e} (bound: 2 ulp)"
    );
    eprintln!("Fast exp2: worst {worst:.3} ulp");
}

#[test]
fn fast_ln_stays_within_its_documented_bound() {
    let mut rng = Rng(0xABCD_EF01);
    let vals: Vec<f64> = (0..300_000)
        .map(|_| {
            let m = rng.uniform((1e-300f64).ln(), (1e300f64).ln());
            m.exp()
        })
        .collect();
    let f = Ln::builder().accuracy(Fast).build();
    let (worst, at) = worst_ulp(&vals, |x| f.eval(x), reference::ln);
    assert!(
        worst <= 4.0,
        "Fast ln worst error {worst} ulp at x = {at:e} (bound: 4 ulp)"
    );
    eprintln!("Fast ln: worst {worst:.3} ulp");
}

/// Inside the documented domain, dropping the range check must change nothing.
#[test]
fn finite_agrees_with_full_range_inside_the_domain() {
    let mut rng = Rng(0x0F1E_2D3C);

    let checked = Exp::new();
    let unchecked = Exp::builder().domain(Finite).build();
    for _ in 0..200_000 {
        let x = rng.uniform(-500.0, 500.0); // inside |x| < 512
        assert_eq!(
            checked.eval(x).to_bits(),
            unchecked.eval(x).to_bits(),
            "exp disagrees at x = {x:e}"
        );
    }

    let checked = Ln::new();
    let unchecked = Ln::builder().domain(Finite).build();
    for _ in 0..200_000 {
        let x = rng.uniform(1e-280, 1e280).abs().max(f64::MIN_POSITIVE);
        assert_eq!(
            checked.eval(x).to_bits(),
            unchecked.eval(x).to_bits(),
            "ln disagrees at x = {x:e}"
        );
    }
}

/// The builder is typestate over zero-sized types; none of it may cost
/// anything at runtime.
#[test]
fn configuration_is_free() {
    let default = Exp::new();
    let configured = Exp::builder().accuracy(Fast).domain(Finite).build();
    assert_eq!(size_of_val(&default), 0);
    assert_eq!(size_of_val(&configured), 0);
    assert_eq!(size_of::<rmath::ExpBuilder<BitExact, FullRange>>(), 0);

    // The configuration is visible in the type, which is what makes the
    // choice reviewable at a call site rather than buried in a runtime flag.
    assert_eq!(format!("{default:?}"), "Exp<BitExact, FullRange>");
    assert_eq!(format!("{configured:?}"), "Exp<Fast, Finite>");
}

/// The order of builder calls must not matter.
#[test]
fn builder_is_order_independent() {
    let a = Exp::builder().accuracy(Fast).domain(Finite).build();
    let b = Exp::builder().domain(Finite).accuracy(Fast).build();
    let mut rng = Rng(0x5A5A_5A5A);
    for _ in 0..10_000 {
        let x = rng.uniform(-50.0, 50.0);
        assert_eq!(a.eval(x).to_bits(), b.eval(x).to_bits());
    }
}

/// A function object works at any width, and the width must not change the
/// answer — the same guarantee `bit_exact.rs` makes, restated for `Fast`,
/// where there is no external reference to compare against.
#[test]
#[cfg(feature = "wide")]
fn fast_results_are_width_independent() {
    use wide::{f64x2, f64x4, f64x8};

    let mut rng = Rng(0xC0FF_EE00);
    let f = Ln::builder().accuracy(Fast).build();
    for _ in 0..50_000 {
        let x = rng.uniform(0.01, 1000.0);
        let s = f.eval(x);
        assert_eq!(f.eval(f64x2::splat(x)).to_array()[0].to_bits(), s.to_bits());
        assert_eq!(f.eval(f64x4::splat(x)).to_array()[0].to_bits(), s.to_bits());
        assert_eq!(f.eval(f64x8::splat(x)).to_array()[0].to_bits(), s.to_bits());
    }
}
