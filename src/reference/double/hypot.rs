//! `hypot(x, y) = sqrt(x^2 + y^2)`, without the intermediate overflow.
//!
//! A port of glibc's `__ieee754_hypot`/`__hypot` (`sysdeps/ieee754/dbl-64/
//! e_hypot.c`) — this host has no separate FMA ifunc for `hypot` (confirmed:
//! `__ieee754_hypot` and `__hypot` share one address), and the algorithm
//! itself is the modern (2021+) Borges "MyHypot3" correction, not the older
//! Dekker/`dla.h` style EFT this crate's own roadmap once assumed. It
//! deliberately runs the *non*-FMA branch even on an FMA-capable host: the
//! error-free-transformation identities it uses depend on separately-rounded
//! `mul`/`add`/`sub`, so fusing any of them would not just change the last
//! bit, it would break the error extraction outright — confirmed on this
//! host's disassembly, which has zero `vfmadd`/`vfnmadd`/`vfmsub`
//! instructions anywhere in the function.

/// `2^-600`: the down-scale for the huge-`ax` branch.
const SCALE: f64 = f64::from_bits(0x1a70000000000000);
/// `2^511`: above this, squaring could overflow even after scaling down.
const LARGE_VAL: f64 = f64::from_bits(0x5fe0000000000000);
/// `2^-459`: below this, squaring could underflow to zero.
const TINY_VAL: f64 = f64::from_bits(0x2340000000000000);
/// `2^-54`: below this ratio, the smaller argument cannot affect the result.
const EPS: f64 = f64::from_bits(0x3c90000000000000);

/// The compensated correction, given `ax >= ay >= 0` scaled so that squaring
/// neither overflows nor underflows.
///
/// Every operation here is a separate rounding by design (see the module
/// doc); replaying it with `mul_add` anywhere would be a different,
/// unrelated algorithm that happens to also approximate `hypot`.
fn kernel(ax: f64, ay: f64) -> f64 {
    let mut h = (ax * ax + ay * ay).sqrt();
    let (t1, t2);
    if h <= 2.0 * ay {
        let delta = h - ay;
        t1 = ax * (2.0 * delta - ax);
        t2 = (delta - 2.0 * (ax - ay)) * delta;
    } else {
        let delta = h - ax;
        t1 = 2.0 * delta * (ax - 2.0 * ay);
        t2 = (4.0 * delta - ay) * ay + delta * delta;
    }
    h -= (t1 + t2) / (2.0 * h);
    h
}

/// `hypot(x, y)`, the platform's.
pub fn hypot(x: f64, y: f64) -> f64 {
    if !x.is_finite() || !y.is_finite() {
        if (x.is_infinite() || y.is_infinite()) && !is_signaling_nan(x) && !is_signaling_nan(y) {
            return f64::INFINITY;
        }
        return x + y;
    }

    let x = x.abs();
    let y = y.abs();
    let ax = x.max(y);
    let ay = x.min(y);

    if ax > LARGE_VAL {
        if ay <= ax * EPS {
            return ax + ay;
        }
        return kernel(ax * SCALE, ay * SCALE) / SCALE;
    }
    if ay < TINY_VAL {
        if ax >= ay / EPS {
            return ax + ay;
        }
        return kernel(ax / SCALE, ay / SCALE) * SCALE;
    }
    if ay <= ax * EPS {
        return ax + ay;
    }
    kernel(ax, ay)
}

/// A signaling NaN has its mantissa's top bit clear (and at least one other
/// mantissa bit set, to remain a NaN rather than an infinity).
fn is_signaling_nan(x: f64) -> bool {
    let bits = x.to_bits();
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let exp_all_ones = (bits >> 52) & 0x7ff == 0x7ff;
    exp_all_ones && mantissa != 0 && (mantissa & (1 << 51)) == 0
}
