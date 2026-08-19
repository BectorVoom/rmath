//! Bit-exactness against the C library itself, for the functions Rust's `std`
//! does not expose.
//!
//! `tests/bit_exact.rs` compares against `f64::exp` and friends, which is the
//! right reference for anything `std` has a method for. The functions checked
//! here — `erf`, `erfc`, the Bessel family, `exp10`, `remquo`, `modf`,
//! `ilogb`, `fdim`, `fmax`, `fmin`, `rint`, `scalbn` — have no `std` method,
//! so the reference is the platform `libm`, reached through `extern "C"`.
//!
//! That link lives *here* and not in the crate: the crate's own references are
//! self-contained ports, precisely so that a libm replacement does not call
//! the libm it replaces. A test may.
//!
//! Run with `cargo test --release`; the sweeps are large.
#![cfg(unix)]
// Branch boundaries are quoted at the precision the C sources quote them.
#![allow(clippy::excessive_precision)]

use rmath::prelude::*;
use rmath::simd::{Lanes, Simd};

unsafe extern "C" {
    fn lgammaf_r(x: f32, s: *mut i32) -> f32;
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
    fn erff(x: f32) -> f32;
    fn erfcf(x: f32) -> f32;
    fn j0(x: f64) -> f64;
    fn j1(x: f64) -> f64;
    fn jn(n: i32, x: f64) -> f64;
    fn y0(x: f64) -> f64;
    fn y1(x: f64) -> f64;
    fn yn(n: i32, x: f64) -> f64;
    fn j0f(x: f32) -> f32;
    fn j1f(x: f32) -> f32;
    fn jnf(n: i32, x: f32) -> f32;
    fn y0f(x: f32) -> f32;
    fn y1f(x: f32) -> f32;
    fn ynf(n: i32, x: f32) -> f32;
    fn exp10(x: f64) -> f64;
    fn exp10f(x: f32) -> f32;
    fn remquo(x: f64, y: f64, q: *mut i32) -> f64;
    fn remquof(x: f32, y: f32, q: *mut i32) -> f32;
    fn modf(x: f64, i: *mut f64) -> f64;
    fn modff(x: f32, i: *mut f32) -> f32;
    fn ilogb(x: f64) -> i32;
    fn ilogbf(x: f32) -> i32;
    fn fdim(x: f64, y: f64) -> f64;
    fn fdimf(x: f32, y: f32) -> f32;
    fn fmax(x: f64, y: f64) -> f64;
    fn fmaxf(x: f32, y: f32) -> f32;
    fn fmin(x: f64, y: f64) -> f64;
    fn fminf(x: f32, y: f32) -> f32;
    fn rint(x: f64) -> f64;
    fn rintf(x: f32) -> f32;
    fn scalbn(x: f64, n: i32) -> f64;
    fn scalbnf(x: f32, n: i32) -> f32;
    fn lgamma_r(x: f64, s: *mut i32) -> f64;
}

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

/// The `f64` corpus: physical magnitudes, the full exponent range, specials,
/// and uniformly random bit patterns.
fn corpus64() -> Vec<f64> {
    let mut v: Vec<f64> = Vec::with_capacity(300_000);
    v.extend([
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::from_bits(1),
        -f64::from_bits(1),
    ]);
    let mut r = Rng(0x1234_5678_9abc_def1);
    for _ in 0..60_000 {
        v.push(r.uniform(-40.0, 40.0));
    }
    for _ in 0..40_000 {
        v.push(r.log_uniform(1e-300, 1e300, true));
    }
    for _ in 0..40_000 {
        v.push(f64::from_bits(r.next()));
    }
    for i in 0..20_000u64 {
        v.push(i as f64 * 0.001);
        v.push(-(i as f64) * 0.001);
    }
    v
}

fn corpus32() -> Vec<f32> {
    let mut v: Vec<f32> = corpus64().into_iter().map(|x| x as f32).collect();
    let mut r = Rng(0xfeed_face_cafe_0001);
    for _ in 0..80_000 {
        v.push(f32::from_bits(r.next() as u32));
    }
    v
}

/// Compare `to_bits()`, treating any two NaNs as equal.
fn same64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}
fn same32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

macro_rules! check1 {
    ($name:literal, $obj:expr, $c:expr, $corpus:expr, $same:expr) => {{
        let f = $obj;
        let mut bad = 0;
        for &x in $corpus.iter() {
            let got = f.eval(x);
            let want = unsafe { $c(x) };
            if !$same(got, want) {
                if bad < 5 {
                    eprintln!(
                        "{}({:?} = {:#x}) = {:?}, libm {:?}",
                        $name,
                        x,
                        x.to_bits(),
                        got,
                        want
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{}: {} mismatches", $name, bad);
    }};
}

macro_rules! check2 {
    ($name:literal, $obj:expr, $c:expr, $corpus:expr, $same:expr) => {{
        let f = $obj;
        let mut bad = 0;
        let n = $corpus.len();
        for i in 0..n {
            let x = $corpus[i];
            let y = $corpus[(i * 7 + 13) % n];
            let got = f.eval(x, y);
            let want = unsafe { $c(x, y) };
            if !$same(got, want) {
                if bad < 5 {
                    eprintln!("{}({:?}, {:?}) = {:?}, libm {:?}", $name, x, y, got, want);
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{}: {} mismatches", $name, bad);
    }};
}

#[test]
fn ieee_helpers_f64() {
    let c = corpus64();
    check1!("rint", Rint::new(), rint, c, same64);
    check1!("ilogb", Ilogb::new(), |x| ilogb(x) as f64, c, same64);
    check2!("fdim", Fdim::new(), fdim, c, same64);
    check2!("fmax", Fmax::new(), fmax, c, same64);
    check2!("fmin", Fmin::new(), fmin, c, same64);
}

#[test]
fn ieee_helpers_f32() {
    let c = corpus32();
    check1!("rintf", Rint::new(), rintf, c, same32);
    check1!("ilogbf", Ilogb::new(), |x| ilogbf(x) as f32, c, same32);
    check2!("fdimf", Fdim::new(), fdimf, c, same32);
    check2!("fmaxf", Fmax::new(), fmaxf, c, same32);
    check2!("fminf", Fmin::new(), fminf, c, same32);
}

#[test]
fn scalbn_matches() {
    let c = corpus64();
    let f = Scalbn::new();
    for (i, &x) in c.iter().enumerate() {
        for n in [-2000i32, -1075, -1022, -53, -1, 0, 1, 53, 1023, 1024, 2000] {
            let got: f64 = f.eval(x, n as f64);
            let want = unsafe { scalbn(x, n) };
            assert!(
                same64(got, want),
                "scalbn({x:?}, {n}) = {got:?}, libm {want:?} (i={i})"
            );
        }
    }
    let c = corpus32();
    for &x in c.iter() {
        for n in [-400i32, -150, -126, -24, -1, 0, 1, 24, 127, 128, 400] {
            let got: f32 = f.eval(x, n as f32);
            let want = unsafe { scalbnf(x, n) };
            assert!(
                same32(got, want),
                "scalbnf({x:?}, {n}) = {got:?}, libm {want:?}"
            );
        }
    }
}

#[test]
fn modf_matches() {
    let f = Modf::new();
    for &x in corpus64().iter() {
        let (fr, int) = f.eval(x);
        let mut ci = 0.0f64;
        let cf = unsafe { modf(x, &mut ci) };
        assert!(
            same64(fr, cf) && same64(int, ci),
            "modf({x:?}) = ({fr:?}, {int:?}), libm ({cf:?}, {ci:?})"
        );
    }
    for &x in corpus32().iter() {
        let (fr, int) = f.eval(x);
        let mut ci = 0.0f32;
        let cf = unsafe { modff(x, &mut ci) };
        assert!(
            same32(fr, cf) && same32(int, ci),
            "modff({x:?}) = ({fr:?}, {int:?}), libm ({cf:?}, {ci:?})"
        );
    }
}

#[test]
fn remquo_matches() {
    let f = Remquo::new();
    let c = corpus64();
    let n = c.len();
    for i in 0..n {
        let (x, y) = (c[i], c[(i * 7 + 13) % n]);
        let (r, q) = f.eval(x, y);
        let mut cq = 0i32;
        let cr = unsafe { remquo(x, y, &mut cq) };
        assert!(
            same64(r, cr),
            "remquo({x:?}, {y:?}) rem = {r:?}, libm {cr:?}"
        );
        if r.is_nan() {
            continue;
        }
        assert_eq!(q, cq as f64, "remquo({x:?}, {y:?}) quo = {q:?}, libm {cq}");
    }
    let c = corpus32();
    let n = c.len();
    for i in 0..n {
        let (x, y) = (c[i], c[(i * 7 + 13) % n]);
        let (r, q) = f.eval(x, y);
        let mut cq = 0i32;
        let cr = unsafe { remquof(x, y, &mut cq) };
        assert!(
            same32(r, cr),
            "remquof({x:?}, {y:?}) rem = {r:?}, libm {cr:?}"
        );
        if r.is_nan() {
            continue;
        }
        assert_eq!(q, cq as f32, "remquof({x:?}, {y:?}) quo = {q:?}, libm {cq}");
    }
}

#[test]
fn exp10_matches() {
    let mut c = corpus64();
    // Every branch boundary of the scalar routine, and its neighbours.
    for b in [
        0.0f64,
        256.0,
        -256.0,
        308.25471555991675,
        -350.0,
        2f64.powi(-57),
        -(2f64.powi(-57)),
        1.0,
        -1.0,
        22.0,
        23.0,
        308.0,
        309.0,
        -324.0,
        -400.0,
    ] {
        for d in -2i32..=2 {
            let mut v = b;
            for _ in 0..d.abs() {
                v = if d < 0 {
                    f64::from_bits(v.to_bits() - 1)
                } else {
                    f64::from_bits(v.to_bits() + 1)
                };
            }
            c.push(v);
        }
    }
    check1!("exp10", Exp10::new(), exp10, c, same64);

    let mut c = corpus32();
    for b in [
        0.0f32, 38.0, -38.0, 38.531837, -45.154499, 38.6, -45.2, 1.0, -1.0,
    ] {
        c.push(b);
        c.push(f32::from_bits(b.to_bits().wrapping_add(1)));
        c.push(f32::from_bits(b.to_bits().wrapping_sub(1)));
    }
    check1!("exp10f", Exp10::new(), exp10f, c, same32);
}

/// Every one of the 2^32 `f32` inputs, against the platform's `exp10f`.
#[test]
#[ignore = "exhaustive: 2^32 inputs, minutes not seconds"]
fn exp10f_exhaustive() {
    let f = Exp10::new();
    let mut bad = 0u64;
    for u in 0..=u32::MAX {
        let x = f32::from_bits(u);
        let got: f32 = f.eval(x);
        let want = unsafe { exp10f(x) };
        if !same32(got, want) {
            if bad < 10 {
                eprintln!("exp10f({x:?} = {u:#010x}) = {got:?}, libm {want:?}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "exp10f: {bad} mismatches over 2^32 inputs");
}

/// `lgamma_r`'s sign is exact and is checked against the platform bit for
/// bit. Its *value* is [`rmath::LGamma`]'s, which carries an accuracy claim
/// rather than a bit-exactness one — `tests/accuracy.rs` owns that bound — so
/// what is checked here is that the two agree exactly, plus that the poles,
/// the infinities and NaN come back as the platform spells them.
#[test]
fn lgamma_r_sign_and_value() {
    let f = LGammaR::new();
    let g = LGamma::new();
    let mut c = corpus64();
    for n in -40i32..=40 {
        c.push(n as f64);
        c.push(n as f64 + 0.5);
        c.push(f64::from_bits((n as f64).to_bits().wrapping_add(1)));
        c.push(f64::from_bits((n as f64).to_bits().wrapping_sub(1)));
    }
    for &x in c.iter() {
        let (v, s) = f.eval(x);
        let mut cs = 0i32;
        let cv = unsafe { lgamma_r(x, &mut cs) };
        assert_eq!(s, cs as f64, "lgamma_r({x:?}) sign = {s}, libm {cs}");
        assert!(
            same64(v, g.eval(x)),
            "lgamma_r({x:?}) value differs from LGamma"
        );
        if !cv.is_finite() {
            assert!(same64(v, cv), "lgamma_r({x:?}) = {v:?}, libm {cv:?}");
        }
    }

    // The sign is exact in single precision too, and glibc draws its
    // huge-magnitude convention differently there.
    let mut c = corpus32();
    for n in -40i32..=40 {
        c.push(n as f32);
        c.push(n as f32 + 0.5);
    }
    for &x in c.iter() {
        let (_, s): (f32, f32) = f.eval(x);
        let mut cs = 0i32;
        unsafe { lgammaf_r(x, &mut cs) };
        assert_eq!(s, cs as f32, "lgammaf_r({x:?}) sign = {s}, libm {cs}");
    }
}

/// The scalar reference for `erf`, against the platform, before any vector
/// code is involved. A correctly-rounded reference is what the whole `erf`
/// design rests on, so it is checked on its own.
#[test]
fn reference_erf_matches_platform() {
    let mut c = corpus64();
    for b in [
        0.0f64,
        0.0625,
        0.125,
        1.0,
        5.9215871960644,
        6.0,
        2f64.powi(-61),
        0.5,
        2.0,
        3.0,
        4.0,
    ] {
        for k in -3i64..=3 {
            c.push(f64::from_bits((b.to_bits() as i64 + k) as u64));
            c.push(-f64::from_bits((b.to_bits() as i64 + k) as u64));
        }
    }
    // Dense sweeps through every table interval and every 1/8 sub-interval.
    for i in 0..400_000u64 {
        c.push(i as f64 * (6.0 / 400_000.0));
    }
    let mut bad = 0;
    for &x in c.iter() {
        let got = rmath::reference::erf(x);
        let want = unsafe { erf(x) };
        if !same64(got, want) {
            if bad < 8 {
                eprintln!(
                    "reference::erf({x:?} = {:#x}) = {got:?}, libm {want:?}",
                    x.to_bits()
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "reference::erf: {bad} mismatches");
}

/// `erf` at every width, against the platform. `BitExact` here is a claim
/// about correct rounding rather than about this `libm`, but the check is the
/// same one: `to_bits()` equality, no tolerance.
#[test]
fn erf_is_bit_exact() {
    let mut c = corpus64();
    for b in [
        0.0f64,
        0.0625,
        0.125,
        1.0,
        5.9215871960644,
        6.0,
        0.5,
        2.0,
        3.0,
        4.0,
    ] {
        for k in -3i64..=3 {
            let v = f64::from_bits((b.to_bits() as i64 + k) as u64);
            c.push(v);
            c.push(-v);
        }
    }
    for i in 0..200_000u64 {
        c.push(i as f64 * (6.0 / 200_000.0));
        c.push(-(i as f64) * (6.0 / 200_000.0));
    }
    check1!("erf", Erf::new(), erf, c, same64);

    let mut c = corpus32();
    for i in 0..200_000u32 {
        c.push(i as f32 * (4.0 / 200_000.0));
        c.push(-(i as f32) * (4.0 / 200_000.0));
    }
    check1!("erff", Erf::new(), erff, c, same32);
}

/// Every one of the 2^32 `f32` inputs, against the platform's `erff`.
#[test]
#[ignore = "exhaustive: 2^32 inputs, minutes not seconds"]
fn erff_exhaustive() {
    let f = Erf::new();
    let mut bad = 0u64;
    for u in 0..=u32::MAX {
        let x = f32::from_bits(u);
        let got: f32 = f.eval(x);
        let want = unsafe { erff(x) };
        if !same32(got, want) {
            if bad < 10 {
                eprintln!("erff({x:?} = {u:#010x}) = {got:?}, libm {want:?}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "erff: {bad} mismatches over 2^32 inputs");
}

/// The scalar reference for `erfc`, against the platform. `erfc` is the one
/// function here with a genuinely separate algorithm above and below its
/// threshold, so the corpus leans on the crossing and on the far tail, where
/// the result is subnormal.
#[test]
fn reference_erfc_matches_platform() {
    let mut c = corpus64();
    for b in [
        0.0f64,
        2.8846,
        2.8846404621299175,
        0.4771,
        1.7109,
        27.2,
        27.226017,
        -5.8636,
        26.6,
        6.0,
        1.0,
        -1.0,
        0.5,
    ] {
        for k in -3i64..=3 {
            let v = f64::from_bits((b.to_bits() as i64 + k) as u64);
            c.push(v);
            c.push(-v);
        }
    }
    for i in 0..300_000u64 {
        c.push(i as f64 * (28.0 / 300_000.0));
        c.push(-(i as f64) * (6.0 / 300_000.0));
    }
    let mut bad = 0;
    for &x in c.iter() {
        let got = rmath::reference::erfc(x);
        let want = unsafe { erfc(x) };
        if !same64(got, want) {
            if bad < 8 {
                eprintln!(
                    "reference::erfc({x:?} = {:#x}) = {got:?}, libm {want:?}",
                    x.to_bits()
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "reference::erfc: {bad} mismatches");
}

/// `erfc` at every width, against the platform.
#[test]
fn erfc_is_bit_exact() {
    let mut c = corpus64();
    for b in [
        0.0f64,
        2.8846404621299175,
        0.4771,
        27.226017,
        -5.8636,
        6.0,
        1.0,
        0.5,
        26.6,
    ] {
        for k in -3i64..=3 {
            let v = f64::from_bits((b.to_bits() as i64 + k) as u64);
            c.push(v);
            c.push(-v);
        }
    }
    for i in 0..300_000u64 {
        c.push(i as f64 * (28.0 / 300_000.0));
        c.push(-(i as f64) * (6.0 / 300_000.0));
    }
    check1!("erfc", Erfc::new(), erfc, c, same64);

    let mut c = corpus32();
    for i in 0..300_000u32 {
        c.push(i as f32 * (11.0 / 300_000.0));
        c.push(-(i as f32) * (4.0 / 300_000.0));
    }
    check1!("erfcf", Erfc::new(), erfcf, c, same32);
}

/// Every one of the 2^32 `f32` inputs, against the platform's `erfcf`.
#[test]
#[ignore = "exhaustive: 2^32 inputs, minutes not seconds"]
fn erfcf_exhaustive() {
    let f = Erfc::new();
    let mut bad = 0u64;
    for u in 0..=u32::MAX {
        let x = f32::from_bits(u);
        let got: f32 = f.eval(x);
        let want = unsafe { erfcf(x) };
        if !same32(got, want) {
            if bad < 10 {
                eprintln!("erfcf({x:?} = {u:#010x}) = {got:?}, libm {want:?}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "erfcf: {bad} mismatches over 2^32 inputs");
}

/// The scalar Bessel references, against the platform. These are fdlibm
/// ports, so unlike `erf` this *is* a claim about the host's `libm` — and it
/// leans on `sin`, `cos` and `log` agreeing too, which they do only because
/// the crate delegates them.
#[test]
fn reference_bessel_matches_platform() {
    let mut c: Vec<f64> = corpus64();
    for i in 0..400_000u64 {
        c.push(i as f64 * (60.0 / 400_000.0));
    }
    for b in [
        0.0f64,
        2.0,
        8.0,
        4.5454,
        2.8571,
        1e300,
        2f64.powi(-27),
        2f64.powi(-54),
        2f64.powi(129),
    ] {
        for k in -3i64..=3 {
            let v = f64::from_bits((b.to_bits() as i64 + k) as u64);
            c.push(v);
            c.push(-v);
        }
    }
    for (name, ours, theirs) in [
        (
            "j0",
            rmath::reference::j0 as fn(f64) -> f64,
            j0 as unsafe extern "C" fn(f64) -> f64,
        ),
        ("j1", rmath::reference::j1, j1),
        ("y0", rmath::reference::y0, y0),
        ("y1", rmath::reference::y1, y1),
    ] {
        let mut bad = 0;
        for &x in c.iter() {
            let got = ours(x);
            let want = unsafe { theirs(x) };
            if !same64(got, want) {
                if bad < 5 {
                    eprintln!(
                        "reference::{name}({x:?} = {:#x}) = {got:?}, libm {want:?}",
                        x.to_bits()
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "reference::{name}: {bad} mismatches of {}", c.len());
    }
}

/// `jn` and `yn` against the platform, over orders that exercise each of the
/// three regimes: forward recurrence, backward recurrence with its run-time
/// continued fraction, and the tiny-argument series.
#[test]
fn reference_jn_yn_matches_platform() {
    let mut xs: Vec<f64> = Vec::new();
    for i in 0..4_000u64 {
        xs.push(i as f64 * (60.0 / 4_000.0));
    }
    xs.extend([
        0.0,
        1e-10,
        1e-30,
        2f64.powi(-29),
        2f64.powi(-30),
        1.0,
        2.0,
        8.0,
        100.0,
        1e10,
        2f64.powi(303),
        f64::INFINITY,
        f64::NAN,
        -1.0,
        -100.0,
        f64::MIN_POSITIVE,
    ]);
    let ns = [
        -300i32, -33, -5, -2, -1, 0, 1, 2, 3, 5, 10, 33, 34, 50, 200, 1000,
    ];
    let mut bad = 0;
    for &n in ns.iter() {
        for &x in xs.iter() {
            let g = rmath::reference::jn(n, x);
            let w = unsafe { jn(n, x) };
            if !same64(g, w) {
                if bad < 6 {
                    eprintln!("reference::jn({n}, {x:?}) = {g:?}, libm {w:?}");
                }
                bad += 1;
            }
            let g = rmath::reference::yn(n, x);
            let w = unsafe { yn(n, x) };
            if !same64(g, w) {
                if bad < 6 {
                    eprintln!("reference::yn({n}, {x:?}) = {g:?}, libm {w:?}");
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "jn/yn: {bad} mismatches");
}

/// The single-precision Bessel references, against the platform.
#[test]
fn reference_bessel_f32_matches_platform() {
    let mut c: Vec<f32> = corpus32();
    for i in 0..400_000u32 {
        c.push(i as f32 * (300.0 / 400_000.0));
    }
    for b in [
        0.0f32,
        2.0,
        0.8935760259628296,
        2.404825448989868,
        3.8317060470581055,
        2.197141408920288,
        1.7568,
        203.0,
        1e30,
        1e38,
    ] {
        for k in -3i32..=3 {
            let v = f32::from_bits((b.to_bits() as i32 + k) as u32);
            c.push(v);
            c.push(-v);
        }
    }
    for (name, ours, theirs) in [
        (
            "j0f",
            rmath::reference::single::j0 as fn(f32) -> f32,
            j0f as unsafe extern "C" fn(f32) -> f32,
        ),
        ("j1f", rmath::reference::single::j1, j1f),
        ("y0f", rmath::reference::single::y0, y0f),
        ("y1f", rmath::reference::single::y1, y1f),
    ] {
        let mut bad = 0;
        for &x in c.iter() {
            let got = ours(x);
            let want = unsafe { theirs(x) };
            if !same32(got, want) {
                if bad < 6 {
                    eprintln!(
                        "reference::single::{name}({x:?} = {:#x}) = {got:?}, libm {want:?}",
                        x.to_bits()
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad,
            0,
            "reference::single::{name}: {bad} mismatches of {}",
            c.len()
        );
    }
}

/// `jnf` and `ynf` against the platform.
#[test]
fn reference_jnf_ynf_matches_platform() {
    let mut xs: Vec<f32> = Vec::new();
    for i in 0..4_000u32 {
        xs.push(i as f32 * (60.0 / 4_000.0));
    }
    xs.extend([
        0.0,
        1e-10,
        1e-30,
        1.0,
        2.0,
        8.0,
        100.0,
        1e10,
        1e30,
        f32::INFINITY,
        f32::NAN,
        -1.0,
        -100.0,
        f32::MIN_POSITIVE,
    ]);
    let ns = [
        -300i32, -33, -5, -2, -1, 0, 1, 2, 3, 5, 10, 33, 34, 50, 200, 1000,
    ];
    let mut bad = 0;
    for &n in ns.iter() {
        for &x in xs.iter() {
            let g = rmath::reference::single::jn(n, x);
            let w = unsafe { jnf(n, x) };
            if !same32(g, w) {
                if bad < 6 {
                    eprintln!("reference::single::jn({n}, {x:?}) = {g:?}, libm {w:?}");
                }
                bad += 1;
            }
            let g = rmath::reference::single::yn(n, x);
            let w = unsafe { ynf(n, x) };
            if !same32(g, w) {
                if bad < 6 {
                    eprintln!("reference::single::yn({n}, {x:?}) = {g:?}, libm {w:?}");
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "jnf/ynf: {bad} mismatches");
}

/// The Bessel kernels at every width, against the platform. The order-0 and
/// order-1 double-precision ones are vectorised, so this is where lane
/// independence gets checked for them.
#[test]
fn bessel_is_bit_exact() {
    let mut c: Vec<f64> = corpus64();
    for i in 0..400_000u64 {
        c.push(i as f64 * (60.0 / 400_000.0));
    }
    for b in [
        0.0f64,
        1.0,
        2.0,
        8.0,
        4.5454,
        2.8571,
        2f64.powi(-13),
        2f64.powi(-27),
        2f64.powi(-54),
        2f64.powi(28),
        2f64.powi(129),
    ] {
        for k in -3i64..=3 {
            let v = f64::from_bits((b.to_bits() as i64 + k) as u64);
            c.push(v);
            c.push(-v);
        }
    }
    check1!("j0", J0::new(), j0, c, same64);
    check1!("j1", J1::new(), j1, c, same64);
    check1!("y0", Y0::new(), y0, c, same64);
    check1!("y1", Y1::new(), y1, c, same64);

    let mut c: Vec<f32> = corpus32();
    for i in 0..200_000u32 {
        c.push(i as f32 * (300.0 / 200_000.0));
    }
    check1!("j0f", J0::new(), j0f, c, same32);
    check1!("j1f", J1::new(), j1f, c, same32);
    check1!("y0f", Y0::new(), y0f, c, same32);
    check1!("y1f", Y1::new(), y1f, c, same32);
}

/// `jn` / `yn` at every width, and the order travelling in a float lane.
#[test]
fn jn_yn_is_bit_exact() {
    let f = Jn::new();
    let g = Yn::new();
    let mut xs: Vec<f64> = Vec::new();
    for i in 0..3_000u64 {
        xs.push(i as f64 * (60.0 / 3_000.0));
    }
    xs.extend([
        0.0,
        1e-30,
        1.0,
        100.0,
        1e10,
        f64::INFINITY,
        f64::NAN,
        -1.0,
        -100.0,
    ]);
    for n in [-33i32, -2, 0, 1, 2, 5, 34, 200] {
        for &x in xs.iter() {
            let got: f64 = f.eval(n as f64, x);
            assert!(same64(got, unsafe { jn(n, x) }), "jn({n}, {x:?}) = {got:?}");
            let got: f64 = g.eval(n as f64, x);
            assert!(same64(got, unsafe { yn(n, x) }), "yn({n}, {x:?}) = {got:?}");
            let got: f32 = f.eval(n as f32, x as f32);
            assert!(
                same32(got, unsafe { jnf(n, x as f32) }),
                "jnf({n}, {x:?}) = {got:?}"
            );
            let got: f32 = g.eval(n as f32, x as f32);
            assert!(
                same32(got, unsafe { ynf(n, x as f32) }),
                "ynf({n}, {x:?}) = {got:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Lane independence
// ---------------------------------------------------------------------------

/// Run a vectorised kernel at one width, comparing every lane against the
/// platform.
///
/// The ragged tail is padded by repeating the first element rather than with
/// zero, because zero is a special case for most of these functions and
/// padding with it would quietly test that case instead of nothing.
fn check_width<V, F, G>(name: &str, vals: &[f64], scalar: F, vector: G)
where
    V: Simd<Elem = f64>,
    F: Fn(f64) -> f64,
    G: Fn(V) -> V,
{
    let mut bad = 0usize;
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
            if !same64(want, have) {
                if bad < 6 {
                    eprintln!(
                        "{name} @{} lanes: x = {x:e} ({:#018x}) => want {want:e}, got {have:e}",
                        V::LANES,
                        x.to_bits()
                    );
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} lanes differ at width {}", V::LANES);
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

/// The same inputs at one, two, four and eight lanes must agree with the
/// platform — so a result cannot depend on which lane it landed in, or on how
/// a batch happened to be chunked.
///
/// This matters most for the kernels here that *branch per vector*: `erfc`
/// takes a different path when every lane is in its tail than when the lanes
/// are mixed, and the Bessel kernels blend two branches. Mixed vectors are
/// therefore built deliberately, not just sampled.
#[test]
fn new_kernels_are_lane_independent() {
    let mut vals = corpus64();
    // Deliberately interleave the branches each kernel chooses between.
    for i in 0..20_000u64 {
        let t = i as f64;
        vals.push(t * 0.001);
        vals.push(-(t * 0.001));
        vals.push(2.0 + t * 0.003);
        vals.push(28.0 - t * 0.0014);
        vals.push(t * 0.02);
    }
    all_widths!("exp10", &vals, rmath::reference::exp10, Exp10::new());
    all_widths!("erf", &vals, rmath::reference::erf, Erf::new());
    all_widths!("erfc", &vals, rmath::reference::erfc, Erfc::new());
    all_widths!("j0", &vals, rmath::reference::j0, J0::new());
    all_widths!("j1", &vals, rmath::reference::j1, J1::new());
    all_widths!("y0", &vals, rmath::reference::y0, Y0::new());
    all_widths!("y1", &vals, rmath::reference::y1, Y1::new());
    all_widths!("rint", &vals, |x: f64| unsafe { rint(x) }, Rint::new());
    all_widths!(
        "ilogb",
        &vals,
        |x: f64| unsafe { ilogb(x) as f64 },
        Ilogb::new()
    );
}

/// Every one of the 2^32 `f32` inputs, for the whole single-precision Bessel
/// family.
///
/// Worth the minutes: `j0f` and friends switch between three algorithms on
/// tests that no sampled corpus reliably straddles — a bracket coming out
/// small, an argument landing inside one of 64 tabulated intervals — and a
/// boundary that is off by one representable value is invisible to anything
/// less than this.
#[test]
#[ignore = "exhaustive: 2^32 inputs per function, minutes not seconds"]
fn bessel_f32_exhaustive() {
    for (name, ours, theirs) in [
        (
            "j0f",
            rmath::reference::single::j0 as fn(f32) -> f32,
            j0f as unsafe extern "C" fn(f32) -> f32,
        ),
        ("j1f", rmath::reference::single::j1, j1f),
        ("y0f", rmath::reference::single::y0, y0f),
        ("y1f", rmath::reference::single::y1, y1f),
    ] {
        let mut bad = 0u64;
        for u in 0..=u32::MAX {
            let x = f32::from_bits(u);
            let got = ours(x);
            let want = unsafe { theirs(x) };
            if !same32(got, want) {
                if bad < 10 {
                    eprintln!("{name}({x:?} = {u:#010x}) = {got:?}, libm {want:?}");
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{name}: {bad} mismatches over 2^32 inputs");
    }
}

/// The buffer helpers for the two-result shapes, including the ragged tail.
///
/// `Function2Pair` is the arity `remquo` introduced, and it is the one with no
/// prior coverage: a bug in its chunking would show up only here, since every
/// other test calls `eval` directly.
#[test]
fn pair_slice_helpers_match_scalar() {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    for len in [0usize, 1, 3, 7, 8, 9, 15, 17, 1001] {
        let a: Vec<f64> = (0..len).map(|_| r.uniform(-30.0, 30.0)).collect();
        let b: Vec<f64> = (0..len).map(|_| r.uniform(-8.0, 8.0)).collect();

        let (mut rem, mut quo) = (vec![0.0; len], vec![0.0; len]);
        Remquo::new().eval_slice(&a, &b, &mut rem, &mut quo);
        for i in 0..len {
            let mut cq = 0i32;
            let cr = unsafe { remquo(a[i], b[i], &mut cq) };
            assert!(same64(rem[i], cr), "remquo len {len} index {i}");
            assert_eq!(quo[i], cq as f64, "remquo quo len {len} index {i}");
        }

        let (mut frac, mut int) = (vec![0.0; len], vec![0.0; len]);
        Modf::new().eval_slice(&a, &mut frac, &mut int);
        for i in 0..len {
            let mut ci = 0.0f64;
            let cf = unsafe { modf(a[i], &mut ci) };
            assert!(
                same64(frac[i], cf) && same64(int[i], ci),
                "modf len {len} index {i}"
            );
        }

        let (mut v, mut sg) = (vec![0.0; len], vec![0.0; len]);
        LGammaR::new().eval_slice(&a, &mut v, &mut sg);
        for i in 0..len {
            let mut cs = 0i32;
            unsafe { lgamma_r(a[i], &mut cs) };
            assert_eq!(sg[i], cs as f64, "lgamma_r sign len {len} index {i}");
        }
    }
}
