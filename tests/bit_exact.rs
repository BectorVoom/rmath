//! The claim that gives this crate its name: under
//! [`rmath::BitExact`], every lane is bit-identical to the platform `libm`.
//!
//! These are `to_bits()` comparisons, not tolerances. A tolerance test would
//! pass on an implementation that is merely close, which is exactly the
//! property the default policy exists to rule out.
//!
//! Three things are checked for every function:
//!
//! 1. **Coverage.** Millions of inputs — physical magnitudes, the full
//!    exponent range, subnormals, specials, and uniformly random bit
//!    patterns, which land on NaN and infinity far more often than any
//!    "realistic" distribution would.
//! 2. **Branch boundaries.** Every constant a kernel branches on, together
//!    with the two representable values either side of it. Off-by-one-ulp
//!    boundary errors are invisible to random sampling and are exactly the
//!    kind of mistake this port could make.
//! 3. **Lane independence.** The same inputs at one, two, four and eight
//!    lanes must agree, so a result cannot depend on which lane it landed in
//!    or on how a batch was chunked.
//!
//! Run with `cargo test --release`; the sweeps are large.

// Threshold constants are quoted at the precision glibc's source quotes them.
#![allow(clippy::excessive_precision)]

use rmath::prelude::*;
use rmath::reference;

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

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Log-uniform magnitude in `[lo, hi]`, optionally with a random sign.
    fn log_uniform(&mut self, lo: f64, hi: f64, signed: bool) -> f64 {
        let m = self.uniform(lo.ln(), hi.ln()).exp();
        if signed && self.next() & 1 == 0 {
            -m
        } else {
            m
        }
    }
}

/// `x`, and the two representable values on either side of it.
fn around(x: f64) -> impl Iterator<Item = f64> {
    (-2i64..=2).map(move |d| f64::from_bits(x.to_bits().wrapping_add(d as u64)))
}

/// Inputs every function is fed, whatever its domain.
fn universal() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 2.0,               // subnormal
        f64::from_bits(1),                     // smallest subnormal
        f64::from_bits(0x000f_ffff_ffff_ffff), // largest subnormal
        f64::MAX,
        f64::MIN,
    ];
    for c in [1.0, 2.0, 0.5, 1e-300, 1e300] {
        v.extend(around(c));
        v.extend(around(-c));
    }
    v
}

/// Compare every lane against `scalar`, at the given vector width.
fn check_width<V, F, G>(name: &str, vals: &[f64], scalar: F, vector: G)
where
    V: Simd<Elem = f64>,
    F: Fn(f64) -> f64,
    G: Fn(V) -> V,
{
    use rmath::simd::Lanes;
    let mut shown = 0usize;
    let mut bad = 0usize;

    for chunk in vals.chunks(V::LANES) {
        let mut lanes = V::Floats::filled_default();
        // Pad the ragged tail by repeating the first element rather than with
        // zero: zero is a special case for several of these functions, and
        // padding with it would quietly test that case instead of nothing.
        for slot in lanes.as_mut_slice() {
            *slot = chunk[0];
        }
        lanes.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);

        let got = vector(V::from_array(lanes)).to_array();
        for i in 0..chunk.len() {
            let x = lanes.as_slice()[i];
            let want = scalar(x);
            let have = got.as_slice()[i];
            if want.to_bits() != have.to_bits() {
                bad += 1;
                if shown < 8 {
                    shown += 1;
                    eprintln!(
                        "{name} @{} lanes: x = {x:e} ({:#018x}) => want {want:e} ({:#018x}), got {have:e} ({:#018x})",
                        V::LANES,
                        x.to_bits(),
                        want.to_bits(),
                        have.to_bits()
                    );
                }
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "{name}: {bad} of {} lanes differ at width {}",
        vals.len(),
        V::LANES
    );
}

/// Run the comparison at every supported width.
macro_rules! check_all_widths {
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

/// [`check_width`] for a two-argument function.
fn check_width2<V, F, G>(name: &str, xs: &[f64], ys: &[f64], scalar: F, vector: G)
where
    V: Simd<Elem = f64>,
    F: Fn(f64, f64) -> f64,
    G: Fn(V, V) -> V,
{
    use rmath::simd::Lanes;
    let mut shown = 0usize;
    let mut bad = 0usize;

    for (cx, cy) in xs.chunks(V::LANES).zip(ys.chunks(V::LANES)) {
        let mut lx = V::Floats::filled_default();
        let mut ly = V::Floats::filled_default();
        for slot in lx.as_mut_slice() {
            *slot = cx[0];
        }
        for slot in ly.as_mut_slice() {
            *slot = cy[0];
        }
        lx.as_mut_slice()[..cx.len()].copy_from_slice(cx);
        ly.as_mut_slice()[..cy.len()].copy_from_slice(cy);

        let got = vector(V::from_array(lx), V::from_array(ly)).to_array();
        for i in 0..cx.len() {
            let (x, y) = (lx.as_slice()[i], ly.as_slice()[i]);
            let want = scalar(x, y);
            let have = got.as_slice()[i];
            if want.to_bits() != have.to_bits() {
                bad += 1;
                if shown < 8 {
                    shown += 1;
                    eprintln!(
                        "{name} @{} lanes: (x, y) = ({x:e}, {y:e}) => want {want:e} ({:#018x}), got {have:e} ({:#018x})",
                        V::LANES,
                        want.to_bits(),
                        have.to_bits()
                    );
                }
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "{name}: {bad} of {} lanes differ at width {}",
        xs.len(),
        V::LANES
    );
}

/// Run the two-argument comparison at every supported width.
macro_rules! check_all_widths2 {
    ($name:expr, $xs:expr, $ys:expr, $scalar:expr, $f:expr) => {{
        let f = $f;
        check_width2::<f64, _, _>($name, $xs, $ys, $scalar, |x, y| f.eval(x, y));
        #[cfg(feature = "wide")]
        {
            check_width2::<wide::f64x2, _, _>($name, $xs, $ys, $scalar, |x, y| f.eval(x, y));
            check_width2::<wide::f64x4, _, _>($name, $xs, $ys, $scalar, |x, y| f.eval(x, y));
            check_width2::<wide::f64x8, _, _>($name, $xs, $ys, $scalar, |x, y| f.eval(x, y));
        }
    }};
}

// ---------------------------------------------------------------------------
// The reference ports themselves must match the platform libm. If these fail,
// nothing else in the file is meaningful — the vector kernels would be
// faithfully reproducing the wrong thing.
// ---------------------------------------------------------------------------

fn corpus_exp(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for c in [
        f64::from_bits(0x3c90000000000000), // 0x1p-54, the tiny cutoff
        512.0,                              // main-path limit
        1024.0,
        709.782712893383973096, // overflow threshold
        -708.39641853226410622, // underflow threshold
        -745.13321910194110842, // flush-to-zero threshold
        928.0,
        -1075.0,
    ] {
        v.extend(around(c));
        v.extend(around(-c));
    }
    for _ in 0..800_000 {
        v.push(rng.uniform(-40.0, 40.0)); // ordinary use
    }
    for _ in 0..400_000 {
        v.push(rng.uniform(-760.0, 720.0)); // includes over/underflow
    }
    for _ in 0..300_000 {
        v.push(rng.log_uniform(1e-30, 1e3, true)); // tiny-|x| branch
    }
    for _ in 0..300_000 {
        v.push(f64::from_bits(rng.next())); // adversarial
    }
    v
}

fn corpus_log(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for c in [
        0.9375,                             // near-1.0 window, low
        f64::from_bits(0x3ff1090000000000), // near-1.0 window, high
        f64::from_bits(0x3fe6000000000000), // table-centring offset
    ] {
        v.extend(around(c));
    }
    for _ in 0..800_000 {
        v.push(rng.log_uniform(1e-30, 1e10, false));
    }
    for _ in 0..400_000 {
        v.push(rng.uniform(0.85, 1.15)); // dense across the near-1.0 window
    }
    for _ in 0..400_000 {
        v.push(rng.log_uniform(1e-320, 1e308, true)); // full range, signed
    }
    for _ in 0..300_000 {
        v.push(f64::from_bits(rng.next()));
    }
    v
}

/// `sinh`/`cosh`'s own vectorised bit-exact schedule (`src/kernels/double/hyper.rs`)
/// had no dedicated corpus here before this — only the generic `universal()`
/// values and whatever `tests/accuracy.rs`'s `Fast`-only sampling happened to
/// cross. Hardens the overflow boundary specifically: `OVERFLOW = 710.0` is
/// where the kernel hands off to the scalar reference, and the true
/// mathematical overflow (`sinh`/`cosh(x) ~ e^x/2`) sits at `x ~= 709.78`, so
/// the gap between the two is exactly the band a boundary-placement mistake
/// would hide in.
fn corpus_hyper(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for c in [
        710.0,                  // OVERFLOW: the kernel's own domain boundary
        709.782712893383973096, // true overflow threshold (same as exp's)
        700.0,
        720.0,
    ] {
        v.extend(around(c));
        v.extend(around(-c));
    }
    for _ in 0..800_000 {
        v.push(rng.uniform(-700.0, 700.0)); // ordinary use
    }
    for _ in 0..400_000 {
        v.push(rng.uniform(-720.0, 720.0)); // straddles the overflow band
    }
    for _ in 0..300_000 {
        v.push(f64::from_bits(rng.next())); // adversarial
    }
    v
}

fn corpus_cbrt(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for _ in 0..800_000 {
        v.push(rng.log_uniform(1e-30, 1e10, true));
    }
    for _ in 0..400_000 {
        v.push(rng.log_uniform(1e-320, 1e300, true));
    }
    for _ in 0..300_000 {
        v.push(f64::from_bits(rng.next()));
    }
    // Exact cubes, where a seeded iteration is most likely to land off by one.
    for i in 1..2000u64 {
        let n = i as f64;
        v.push(n * n * n);
        v.push(-(n * n * n));
    }
    v
}

/// `sin`/`cos`/`sincos`'s bands: `TINY`/`POLY`/`MID`/`TABLE_LIMIT`, at the
/// exact bit patterns `src/reference/double/trig.rs` and
/// `src/kernels/double/trig.rs` branch on (`cos`/`sincos` share one `TINY`,
/// `sin` has its own). Quadrant boundaries (multiples of `pi/2`, where the
/// table-band reduction's `n` wraps) and the `__branred` handoff edge get the
/// same `around()` treatment as every other threshold here — an off-by-one
/// there is exactly the class of bug this file exists to catch.
fn corpus_trig(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for c in [
        f64::from_bits(0x3e500000_00000000), // sin's TINY
        f64::from_bits(0x3e400000_00000000), // cos/sincos's TINY
        f64::from_bits(0x3feb6000_00000000), // POLY: |x| < 0.855469
        f64::from_bits(0x400368fd_00000000), // MID: |x| < 2.426265
        f64::from_bits(0x419921fb_00000000), // TABLE_LIMIT / __branred handoff
    ] {
        v.extend(around(c));
        v.extend(around(-c));
    }
    // Quadrant boundaries: `reduce_sincos`'s `n` steps at every multiple of
    // pi/2, both in the ordinary table band and pushed out near its edge.
    for k in 1..2000i64 {
        let q = k as f64 * std::f64::consts::FRAC_PI_2;
        v.extend(around(q));
        v.extend(around(-q));
    }
    for k in [7000i32, 33_550_336, 67_100_672, 134_201_344] {
        let q = k as f64 * std::f64::consts::FRAC_PI_2;
        v.extend(around(q));
        v.extend(around(-q));
    }
    for _ in 0..800_000 {
        v.push(rng.uniform(-1000.0, 1000.0)); // ordinary use, table band
    }
    for _ in 0..400_000 {
        v.push(rng.uniform(-105_500_000.0, 105_500_000.0)); // straddles __branred
    }
    for _ in 0..300_000 {
        v.push(rng.log_uniform(1e-320, 1e300, true)); // subnormals through huge
    }
    for _ in 0..300_000 {
        v.push(f64::from_bits(rng.next())); // adversarial
    }
    v
}

/// `atan`'s five bands (`A <= u < B <= u < C <= u < D <= u < E <= u`), at the
/// exact bit patterns `src/reference/double/invtrig.rs` and
/// `src/kernels/double/invtrig.rs::bit_exact` branch on.
fn corpus_atan(seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut v = universal();
    for c in [
        f64::from_bits(0x3e4b_b67a_0000_0000), // A
        1.0 / 16.0,                            // B
        1.0,                                   // C
        16.0,                                  // D
        5.805e15,                              // E
    ] {
        v.extend(around(c));
        v.extend(around(-c));
    }
    for _ in 0..800_000 {
        v.push(rng.uniform(-20.0, 20.0)); // straddles every band below D
    }
    for _ in 0..400_000 {
        v.push(rng.log_uniform(1e-320, 1e300, true)); // subnormals through huge
    }
    for _ in 0..400_000 {
        v.push(f64::from_bits(rng.next())); // adversarial
    }
    v
}

/// `atan2`'s quadrants and its own extra edges: the `1/16` band shared with
/// `atan`, the extreme-exponent-difference short circuit
/// (`de = +-57 * 16^5`, i.e. `|y| ~= |x| * 2^57`), and the extreme-magnitude
/// rescale bands (`2^-500`, `2^500`) — all four of which
/// `bit_exact::atan2_needs_repair` routes to the reference rather than the
/// vector path, so this corpus exists to prove that routing is right, not to
/// exercise the vector arithmetic at those specific points.
fn corpus_atan2(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng(seed);
    let mut ys = universal();
    let mut xs = universal();
    while xs.len() < ys.len() {
        xs.push(1.0);
    }
    let mut push = |y: f64, x: f64| {
        ys.push(y);
        xs.push(x);
    };
    for &s in &[1.0, -1.0] {
        for &t in &[1.0, -1.0] {
            for c in [
                0.0,
                1.0 / 16.0,
                1.0,
                1e-10,
                1e10,
                f64::from_bits(0x20b0_0000_0000_0000), // 2^-500
                f64::from_bits(0x5f30_0000_0000_0000), // 2^500
            ] {
                for d in [1e-6, 1.0, 1e6] {
                    push(s * c, t * d);
                    push(s * d, t * c);
                }
            }
            // The exponent-difference threshold itself, and just either
            // side of it.
            for k in -2i32..=2 {
                let big = f64::from_bits((0x7fdu64 << 52).wrapping_add(k as u64));
                push(s * big, t * 1.0);
                push(s * 1.0, t * big);
            }
        }
    }
    for _ in 0..600_000 {
        let y = rng.log_uniform(1e-300, 1e300, true);
        let x = rng.log_uniform(1e-300, 1e300, true);
        push(y, x);
    }
    for _ in 0..300_000 {
        push(f64::from_bits(rng.next()), f64::from_bits(rng.next()));
    }
    (ys, xs)
}

fn assert_reference(
    name: &str,
    vals: &[f64],
    ours: impl Fn(f64) -> f64,
    libm: impl Fn(f64) -> f64,
) {
    let mut bad = 0usize;
    let mut shown = 0;
    for &x in vals {
        let (a, b) = (ours(x), libm(x));
        if a.to_bits() != b.to_bits() {
            bad += 1;
            if shown < 8 {
                shown += 1;
                eprintln!(
                    "reference::{name}: x = {x:e} ({:#018x}) => ours {a:e} ({:#018x}), libm {b:e} ({:#018x})",
                    x.to_bits(),
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "reference::{name} differs from the platform libm on {bad} of {} inputs. \
         rmath's BitExact policy is a claim about this platform's libm; if that \
         library is built differently (a non-FMA variant, a different version), \
         this is where it shows up.",
        vals.len()
    );
}

/// [`assert_reference`] for a two-argument function.
fn assert_reference2(
    name: &str,
    ys: &[f64],
    xs: &[f64],
    ours: impl Fn(f64, f64) -> f64,
    libm: impl Fn(f64, f64) -> f64,
) {
    let mut bad = 0usize;
    let mut shown = 0;
    for (&y, &x) in ys.iter().zip(xs) {
        let (a, b) = (ours(y, x), libm(y, x));
        if a.to_bits() != b.to_bits() {
            bad += 1;
            if shown < 8 {
                shown += 1;
                eprintln!(
                    "reference::{name}: (y, x) = ({y:e}, {x:e}) => ours {a:e} ({:#018x}), libm {b:e} ({:#018x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "reference::{name} differs from the platform libm on {bad} of {} inputs.",
        ys.len()
    );
}

#[test]
fn reference_exp_matches_platform_libm() {
    assert_reference(
        "exp",
        &corpus_exp(0x9E37_79B9_7F4A_7C15),
        reference::exp,
        f64::exp,
    );
}

#[test]
fn reference_exp2_matches_platform_libm() {
    assert_reference(
        "exp2",
        &corpus_exp(0x1234_5678_9ABC_DEF0),
        reference::exp2,
        f64::exp2,
    );
}

#[test]
fn reference_ln_matches_platform_libm() {
    assert_reference(
        "ln",
        &corpus_log(0xD1B5_4A32_D192_ED03),
        reference::ln,
        f64::ln,
    );
}

#[test]
fn reference_cbrt_matches_platform_libm() {
    assert_reference(
        "cbrt",
        &corpus_cbrt(0xA076_1D64_78BD_642F),
        reference::cbrt,
        f64::cbrt,
    );
}

#[test]
fn reference_sinh_matches_platform_libm() {
    assert_reference(
        "sinh",
        &corpus_hyper(0x9E37_79B9_7F4A_7C16),
        reference::sinh,
        f64::sinh,
    );
}

#[test]
fn reference_cosh_matches_platform_libm() {
    assert_reference(
        "cosh",
        &corpus_hyper(0x1234_5678_9ABC_DEF1),
        reference::cosh,
        f64::cosh,
    );
}

#[test]
fn reference_sin_matches_platform_libm() {
    assert_reference(
        "sin",
        &corpus_trig(0x9E37_79B9_7F4A_7C17),
        reference::sin,
        f64::sin,
    );
}

#[test]
fn reference_cos_matches_platform_libm() {
    assert_reference(
        "cos",
        &corpus_trig(0x1234_5678_9ABC_DEF2),
        reference::cos,
        f64::cos,
    );
}

#[test]
fn reference_atan_matches_platform_libm() {
    assert_reference(
        "atan",
        &corpus_atan(0x2545_F491_4F6C_DD21),
        reference::atan,
        f64::atan,
    );
}

#[test]
fn reference_atan2_matches_platform_libm() {
    let (ys, xs) = corpus_atan2(0x0BAD_C0DE_1234_5678);
    assert_reference2("atan2", &ys, &xs, reference::atan2, f64::atan2);
}

/// `reference::sincos` must agree with `reference::sin`/`cos` taken
/// separately — the module doc explains why `sin`'s and `sincos`'s mid-band
/// are genuinely different computations, so this is not redundant with the
/// two tests above.
#[test]
fn reference_sincos_matches_platform_libm() {
    let vals = corpus_trig(0xC2B2_AE3D_27D4_EB50);
    let mut bad = 0usize;
    let mut shown = 0;
    for &x in &vals {
        let (s, c) = reference::sincos(x);
        let (ws, wc) = (f64::sin(x), f64::cos(x));
        if s.to_bits() != ws.to_bits() || c.to_bits() != wc.to_bits() {
            bad += 1;
            if shown < 8 {
                shown += 1;
                eprintln!(
                    "reference::sincos: x = {x:e} ({:#018x}) => ours ({s:e}, {c:e}), libm ({ws:e}, {wc:e})",
                    x.to_bits()
                );
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "reference::sincos differs from the platform libm on {bad} of {} inputs",
        vals.len()
    );
}

// ---------------------------------------------------------------------------
// The vector kernels must match the reference at every width.
// ---------------------------------------------------------------------------

#[test]
fn exp_bit_exact_at_every_width() {
    let vals = corpus_exp(0x0BAD_C0DE_DEAD_BEEF);
    check_all_widths!("exp", &vals, f64::exp, Exp::new());
}

#[test]
fn exp2_bit_exact_at_every_width() {
    let vals = corpus_exp(0xFEED_FACE_CAFE_B0BA);
    check_all_widths!("exp2", &vals, f64::exp2, Exp2::new());
}

#[test]
fn ln_bit_exact_at_every_width() {
    let vals = corpus_log(0x5DEE_CE66_D1B5_4A32);
    check_all_widths!("ln", &vals, f64::ln, Ln::new());
}

#[test]
fn sinh_bit_exact_at_every_width() {
    let vals = corpus_hyper(0x5133_5E36_2B36_1D62);
    check_all_widths!("sinh", &vals, f64::sinh, Sinh::new());
}

#[test]
fn cosh_bit_exact_at_every_width() {
    let vals = corpus_hyper(0xC2B2_AE3D_27D4_EB4F);
    check_all_widths!("cosh", &vals, f64::cosh, Cosh::new());
}

#[test]
fn cbrt_bit_exact_at_every_width() {
    let vals = corpus_cbrt(0x2545_F491_4F6C_DD1D);
    check_all_widths!("cbrt", &vals, f64::cbrt, Cbrt::new());
}

#[test]
fn atan_bit_exact_at_every_width() {
    let vals = corpus_atan(0xC2B2_AE3D_27D4_EB53);
    check_all_widths!("atan", &vals, f64::atan, Atan::new());
}

#[test]
fn atan2_bit_exact_at_every_width() {
    let (ys, xs) = corpus_atan2(0x5DEE_CE66_D1B5_4A37);
    check_all_widths2!("atan2", &ys, &xs, f64::atan2, Atan2::new());
}

#[test]
fn sin_bit_exact_at_every_width() {
    let vals = corpus_trig(0x0BAD_C0DE_DEAD_C0DE);
    check_all_widths!("sin", &vals, f64::sin, Sin::new());
}

#[test]
fn cos_bit_exact_at_every_width() {
    let vals = corpus_trig(0xFEED_FACE_CAFE_F00D);
    check_all_widths!("cos", &vals, f64::cos, Cos::new());
}

/// `sincos` must agree with `sin`/`cos` taken separately at every width — the
/// pair object cannot go through `check_all_widths!` directly (it returns two
/// values), so each half is compared through a shim that keeps only that
/// half, same pattern as `tests/delegating.rs` used before this test moved.
#[test]
fn sincos_bit_exact_at_every_width() {
    let vals = corpus_trig(0x5DEE_CE66_D1B5_4A33);

    #[derive(Clone, Copy)]
    struct Sin1;
    impl Function<f64> for Sin1 {
        fn eval<V: Simd<Elem = f64>>(&self, x: V) -> V {
            SinCos::new().eval(x).0
        }
    }
    #[derive(Clone, Copy)]
    struct Cos1;
    impl Function<f64> for Cos1 {
        fn eval<V: Simd<Elem = f64>>(&self, x: V) -> V {
            SinCos::new().eval(x).1
        }
    }
    check_all_widths!("sincos.sin", &vals, f64::sin, Sin1);
    check_all_widths!("sincos.cos", &vals, f64::cos, Cos1);
}

#[test]
fn sqrt_is_correctly_rounded_at_every_width() {
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    let mut vals = universal();
    for _ in 0..500_000 {
        vals.push(rng.log_uniform(1e-300, 1e300, true));
    }
    for _ in 0..200_000 {
        vals.push(f64::from_bits(rng.next()));
    }
    check_all_widths!("sqrt", &vals, f64::sqrt, Sqrt::new());
}

/// A special lane must not perturb its neighbours: the repair path writes only
/// the lanes its mask selects.
#[test]
fn special_lanes_do_not_leak_into_neighbours() {
    let cases: [[f64; 8]; 3] = [
        [1.3, f64::NAN, -2.0, 0.0, 700.0, -745.0, 1e-300, 2.5],
        [
            f64::INFINITY,
            0.9375,
            1.0,
            -0.0,
            5e-324,
            513.0,
            -1.0,
            1.000001,
        ],
        [
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
            1e-310,
            8.0,
            27.0,
            -64.0,
            0.5,
        ],
    ];
    for lanes in cases {
        check_all_widths!("exp/mixed", &lanes, f64::exp, Exp::new());
        check_all_widths!("exp2/mixed", &lanes, f64::exp2, Exp2::new());
        check_all_widths!("ln/mixed", &lanes, f64::ln, Ln::new());
        check_all_widths!("cbrt/mixed", &lanes, f64::cbrt, Cbrt::new());
    }
}

/// `eval_slice` must agree with lane-by-lane evaluation, including on lengths
/// that do not divide the vector width.
#[test]
fn eval_slice_matches_scalar_including_ragged_tails() {
    let mut rng = Rng(0xB5AD_4ECE_DA10_80CC);
    for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1000, 1001] {
        let src: Vec<f64> = (0..len).map(|_| rng.uniform(-30.0, 30.0)).collect();
        let mut dst = vec![0.0; len];
        Exp::new().eval_slice(&src, &mut dst);
        for i in 0..len {
            assert_eq!(
                dst[i].to_bits(),
                src[i].exp().to_bits(),
                "len {len}, index {i}"
            );
        }

        let mut in_place = src.clone();
        Exp::new().eval_in_place(&mut in_place);
        assert_eq!(
            in_place, dst,
            "eval_in_place disagrees with eval_slice at len {len}"
        );
    }
}
