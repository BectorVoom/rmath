//! Single-precision scalar references.
//!
//! Ports of what the platform's `libm` runs for `float`, which on this target
//! is ARM's optimized-routines — the same source glibc compiles into `__expf`,
//! `__logf` and friends. Every one of them evaluates in `double` over a small
//! table and rounds once at the end, so the ports below take and return `f32`
//! but do their arithmetic in `f64`.
//!
//! That detail is what makes the single-precision kernels worth having: the
//! schedule is plain double-precision arithmetic, so [`crate::kernels::single`]
//! replays it eight lanes at a time by widening `f32x8` to `f64x8`, rather than
//! falling back to a scalar loop.
//!
//! # Which functions are here
//!
//! Only the ones whose platform implementation is *not* correctly rounded, and
//! which therefore need their exact schedule reproduced: `expf`, `exp2f`,
//! `logf`, `log2f`, `powf`, `sinf`, `cosf`. Everything else in single
//! precision is correctly rounded on this platform, which is a far easier
//! target: computing in `f64` and rounding once lands on the correctly-rounded
//! `f32` result, and `tests/exhaustive.rs` checks that over *every* one of the
//! 2^32 inputs rather than a sample.

pub mod bessel;
pub mod erf_parts;

pub use bessel::{j0, j1, jn, y0, y1, yn};
pub use erf_parts::{erf, erfc};

use crate::tables::single::exp as et;
use crate::tables::single::exp10 as x10t;
use crate::tables::single::log as lt;
use crate::tables::single::log2 as l2t;

/// The sign and exponent bits, as ARM's `top12`.
#[inline(always)]
fn top12(x: f32) -> u32 {
    x.to_bits() >> 20
}

// ---------------------------------------------------------------------------
// expf / exp2f
// ---------------------------------------------------------------------------

/// `e^x`, bit-identical to the platform's `expf`.
pub fn exp(x: f32) -> f32 {
    let abstop = top12(x) & 0x7ff;
    if abstop >= top12(88.0) {
        // |x| >= 88, or NaN.
        if x.to_bits() == f32::NEG_INFINITY.to_bits() {
            return 0.0;
        }
        if abstop >= top12(f32::INFINITY) {
            return x + x; // +inf, or NaN (propagating the payload)
        }
        if x > f32::from_bits(0x42b17217) {
            // x > log(0x1p128) ~= 88.72: overflow.
            return f32::INFINITY;
        }
        if x < -f32::from_bits(0x42cff1b4) {
            // x < log(0x1p-150) ~= -103.97: underflow to zero.
            return 0.0;
        }
        // Large but representable: fall through to the main path.
    }
    exp_core(x as f64) as f32
}

/// The shared main path of `expf`, in double precision.
#[inline(always)]
fn exp_core(xd: f64) -> f64 {
    // x*N/ln2 = k + r with |r| <= 1/2 and integer k.
    //
    // Both uses of `INVLN2_SCALED * xd` are fused, so the product is never
    // rounded to a double of its own — `vfmadd132sd` then `vfmsub132sd` in
    // the disassembly of `__expf`'s FMA variant. Computing the product once
    // into a variable and reusing it is the obvious transcription of the C
    // source and is wrong on two inputs out of 2^32, which is precisely the
    // kind of difference only an exhaustive check finds.
    let kd_s = et::INVLN2_SCALED.mul_add(xd, et::SHIFT);
    let ki = kd_s.to_bits();
    let kd = kd_s - et::SHIFT;
    let r = et::INVLN2_SCALED.mul_add(xd, -kd);

    // 2^(k/N), with the exponent folded in by one integer add.
    let t = et::TAB[(ki % 32) as usize].wrapping_add(ki << 47);
    let s = f64::from_bits(t);

    let z = r.mul_add(et::CS0, et::CS1);
    let r2 = r * r;
    let y = r.mul_add(et::CS2, 1.0);
    let y = r2.mul_add(z, y);
    y * s
}

/// `2^x`, bit-identical to the platform's `exp2f`.
pub fn exp2(x: f32) -> f32 {
    let abstop = top12(x) & 0x7ff;
    if abstop >= top12(128.0) {
        if x.to_bits() == f32::NEG_INFINITY.to_bits() {
            return 0.0;
        }
        if abstop >= top12(f32::INFINITY) {
            return x + x;
        }
        if x > 0.0 {
            return f32::INFINITY;
        }
        if x <= -150.0 {
            return 0.0;
        }
    }
    exp2_core(x as f64) as f32
}

/// The shared main path of `exp2f`, in double precision.
#[inline(always)]
fn exp2_core(xd: f64) -> f64 {
    let kd_s = xd + et::SHIFT_SCALED;
    let ki = kd_s.to_bits();
    let kd = kd_s - et::SHIFT_SCALED; // k/N, for integer k
    let r = xd - kd;

    let t = et::TAB[(ki % 32) as usize].wrapping_add(ki << 47);
    let s = f64::from_bits(t);

    let z = r.mul_add(et::C0, et::C1);
    let r2 = r * r;
    let y = r.mul_add(et::C2, 1.0);
    let y = r2.mul_add(z, y);
    y * s
}

// ---------------------------------------------------------------------------
// exp10f
// ---------------------------------------------------------------------------

/// `10^x`, bit-identical to the platform's `exp10f`.
///
/// `exp2f` with one constant changed: the same 32-entry table, the same three
/// scaled coefficients, and a reduction scale of `N*log2(10)` rather than `N`.
/// Unlike [`exp`] it uses **no** fused multiply-adds — glibc ships no FMA
/// variant of `exp10f`, and the compiled routine is `mulsd`/`addsd`
/// throughout.
pub fn exp10(x: f32) -> f32 {
    let abstop = (x.to_bits() >> 19) & 0xfff;
    if abstop >= x10t::BIG_TOP13 {
        // |x| >= 38, or NaN.
        if x.to_bits() == f32::NEG_INFINITY.to_bits() {
            return 0.0;
        }
        if abstop >= x10t::INF_TOP13 {
            return x + x; // +inf, or NaN (propagating the payload)
        }
        if x > x10t::OFLOW_BOUND {
            return f32::INFINITY;
        }
        if x < x10t::UFLOW_BOUND {
            return 0.0;
        }
        // Large but representable: fall through to the main path.
    }
    exp10_core(x as f64) as f32
}

/// The shared main path of `exp10f`, in double precision.
#[inline(always)]
fn exp10_core(xd: f64) -> f64 {
    let z = x10t::INVLN10N * xd;
    let kd_s = z + et::SHIFT;
    let ki = kd_s.to_bits();
    let kd = kd_s - et::SHIFT;
    let r = z - kd;

    let t = et::TAB[(ki % 32) as usize].wrapping_add(ki << 47);
    let s = f64::from_bits(t);

    let z = et::CS0 * r + et::CS1;
    let r2 = r * r;
    let y = et::CS2 * r + 1.0;
    let y = z * r2 + y;
    y * s
}

// ---------------------------------------------------------------------------
// logf / log2f
// ---------------------------------------------------------------------------

/// `logf`'s table-centring offset, `bits(0x1.66p-1)`.
const LOG_OFF: u32 = 0x3f33_0000;

/// `ln(x)`, bit-identical to the platform's `logf`.
pub fn ln(x: f32) -> f32 {
    let mut ix = x.to_bits();
    // Fix the sign of zero at x == 1 under directed rounding.
    if ix == 0x3f80_0000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 {
        // x < 0x1p-126, or inf, or NaN.
        if ix.wrapping_mul(2) == 0 {
            return f32::NEG_INFINITY;
        }
        if ix == 0x7f80_0000 {
            return x;
        }
        if (ix & 0x8000_0000) != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
            #[allow(clippy::eq_op)]
            // Not `f32::NAN`: that is the *positive* quiet NaN, while
            // `0/0` on x86 yields the negative one, and this is a
            // bit-exactness contract — the sign of a NaN counts.
            return (x - x) / (x - x);
        }
        // Subnormal: normalise and correct the exponent.
        ix = (x * f32::from_bits(0x4b00_0000)).to_bits(); // 0x1p23
        ix = ix.wrapping_sub(23 << 23);
    }
    ln_core(ix) as f32
}

/// The main path of `logf`, taking already-normalised bits.
#[inline(always)]
fn ln_core(ix: u32) -> f64 {
    // x = 2^k z, with z in [OFF, 2*OFF) and exact.
    let tmp = ix.wrapping_sub(LOG_OFF);
    let i = ((tmp >> 19) % 16) as usize;
    let k = (tmp as i32) >> 23;
    let iz = ix.wrapping_sub(tmp & 0xff80_0000);

    let invc = lt::TAB[2 * i];
    let logc = lt::TAB[2 * i + 1];
    let z = f32::from_bits(iz) as f64;

    // log(x) = log1p(z/c - 1) + log(c) + k*ln2
    let r = z.mul_add(invc, -1.0);
    let y0 = (k as f64).mul_add(lt::LN2, logc);

    let r2 = r * r;
    let y = r.mul_add(lt::A1, lt::A2);
    let y = r2.mul_add(lt::A0, y);
    y.mul_add(r2, y0 + r)
}

/// `log2(x)`, bit-identical to the platform's `log2f`.
pub fn log2(x: f32) -> f32 {
    let mut ix = x.to_bits();
    if ix == 0x3f80_0000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 {
        if ix.wrapping_mul(2) == 0 {
            return f32::NEG_INFINITY;
        }
        if ix == 0x7f80_0000 {
            return x;
        }
        if (ix & 0x8000_0000) != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
            #[allow(clippy::eq_op)]
            return (x - x) / (x - x);
        }
        ix = (x * f32::from_bits(0x4b00_0000)).to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }
    log2_core(ix) as f32
}

/// The main path of `log2f`, taking already-normalised bits.
#[inline(always)]
fn log2_core(ix: u32) -> f64 {
    let tmp = ix.wrapping_sub(LOG_OFF);
    let i = ((tmp >> 19) % 16) as usize;
    let top = tmp & 0xff80_0000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23;

    let invc = l2t::TAB[2 * i];
    let logc = l2t::TAB[2 * i + 1];
    let z = f32::from_bits(iz) as f64;

    // log2(x) = log2(z/c) + log2(c) + k
    let r = z.mul_add(invc, -1.0);
    let y0 = logc + k as f64;

    let r2 = r * r;
    let y = r.mul_add(l2t::A1, l2t::A2);
    let p = r.mul_add(l2t::A3, y0);
    let y = r2.mul_add(l2t::A0, y);
    y.mul_add(r2, p)
}

// ---------------------------------------------------------------------------
// Delegating references
//
// Same reasoning as `reference::double`: where the platform's routine has not
// been reproduced here, `BitExact` stays a true statement by forwarding to it.
// The `f32` methods are used rather than `extern "C"` so this is portable to
// any target the crate builds on.
// ---------------------------------------------------------------------------

/// Generate a reference that forwards to the platform routine.
macro_rules! delegate {
    ($(#[$doc:meta])* $name:ident => $method:ident) => {
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(x: f32) -> f32 {
            f32::$method(x)
        }
    };
}

delegate! { /// `sin(x)`, the platform's. `x` in radians.
sin => sin }
delegate! { /// `cos(x)`, the platform's. `x` in radians.
cos => cos }
delegate! { /// `tan(x)`, the platform's. `x` in radians.
tan => tan }
delegate! { /// `asin(x)`, the platform's.
asin => asin }
delegate! { /// `acos(x)`, the platform's.
acos => acos }
delegate! { /// `atan(x)`, the platform's.
atan => atan }
delegate! { /// `sinh(x)`, the platform's.
sinh => sinh }
delegate! { /// `cosh(x)`, the platform's.
cosh => cosh }
delegate! { /// `tanh(x)`, the platform's.
tanh => tanh }
delegate! { /// `asinh(x)`, the platform's.
asinh => asinh }
delegate! { /// `acosh(x)`, the platform's.
acosh => acosh }
delegate! { /// `atanh(x)`, matching Rust's `f32::atanh`.
atanh => atanh }
delegate! { /// `log10(x)`, the platform's.
log10 => log10 }
delegate! { /// `log1p(x)`, the platform's.
log1p => ln_1p }
delegate! { /// `expm1(x)`, the platform's.
expm1 => exp_m1 }
delegate! { /// `cbrt(x)`, the platform's.
cbrt => cbrt }

/// `x^y`, the platform's `powf`.
#[inline(always)]
pub fn pow(x: f32, y: f32) -> f32 {
    f32::powf(x, y)
}

/// `atan2(y, x)`, the platform's.
#[inline(always)]
pub fn atan2(y: f32, x: f32) -> f32 {
    f32::atan2(y, x)
}

/// `hypot(x, y)`, the platform's.
#[inline(always)]
pub fn hypot(x: f32, y: f32) -> f32 {
    f32::hypot(x, y)
}

/// `(sin(x), cos(x))`, the platform's.
#[inline(always)]
pub fn sincos(x: f32) -> (f32, f32) {
    (f32::sin(x), f32::cos(x))
}
