//! `sqrt(x^2 + y^2)`, without the intermediate overflow.
//!
//! Delegating under `BitExact`. Under `Fast`, the point of the function is
//! that `x*x + y*y` overflows for perfectly representable answers — and
//! underflows to zero for perfectly representable small ones — so the vector
//! path never forms it. It divides instead, which is exact in the exponent and
//! cannot leave the range.

use crate::kernels::{dispatch2, not_normal};
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Mask, Simd};

/// Above this, `a * sqrt(1 + t^2)` could still overflow.
const SCALE_LIMIT: f64 = 1.0e308 / 2.0;

/// `hypot(x, y)` for vectors of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V, y: V) -> V {
    dispatch2::<V, A, D>(x, y, reference::hypot, fast, |x, y| {
        // Zeros, subnormals and non-finites all go to the reference: the
        // quotient below is `0/0` for a zero pair, and the special-value rules
        // (`hypot(inf, NaN) == inf`) are not worth restating in vector form.
        let a = x.abs();
        let b = y.abs();
        let big = V::select(a.gt_mask(b), a, b);
        not_normal(x)
            .or(not_normal(y))
            .or(big.gt_mask(V::splat(SCALE_LIMIT)))
    })
}

/// Measured error: below 2 ulp for normal, non-zero arguments.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V, y: V) -> V {
    let a = x.abs();
    let b = y.abs();
    let big = V::select(a.gt_mask(b), a, b);
    let small = V::select(a.gt_mask(b), b, a);
    // `t` is in [0, 1], so `1 + t*t` is between 1 and 2 and cannot leave the
    // exponent range whatever the inputs were.
    let t = small / big;
    big * t.mul_add(t, V::splat(1.0)).sqrt()
}
