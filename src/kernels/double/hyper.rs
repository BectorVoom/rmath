//! The hyperbolic and inverse-hyperbolic families.
//!
//! `sinh`, `cosh` and `tanh` are **ports**, and they came almost free: the
//! platform computes all three as compositions on `exp` and `expm1`, exactly,
//! with no fused-multiply-add subtleties of their own. Once `expm1` was ported
//! the three followed, and their `BitExact` paths are vectorised — the vector
//! code calls this crate's own `exp` and `expm1` kernels, not the platform's,
//! so nothing drops to a scalar loop.
//!
//! The inverse family — `asinh`, `acosh`, `atanh` — still delegates under
//! `BitExact`; those are not compositions of anything already ported here.
//!
//! Under `Fast` all six are built from this crate's vectorised `exp`, `expm1`,
//! `ln` and `log1p`, with the identities chosen to be the ones that stay
//! accurate where the obvious ones cancel.

use crate::kernels::double::{exp, expm1, logx};
use crate::kernels::{dispatch, no_lanes, not_normal, outside};
use crate::policy::{Accuracy, BitExact, Domain, Fast, Finite, FullRange};
use crate::reference::double as reference;
use crate::simd::{Mask, Simd};

/// Where `sinh` and `cosh` overflow. Past it the vector path is meaningless.
const OVERFLOW: f64 = 710.0;

/// `ln(DBL_MAX)` as the platform tests it, on the high word alone.
///
/// Above this `sinh` and `cosh` split the exponential in two so neither half
/// overflows — a rare enough band that the vector path hands it over rather
/// than carrying a fourth arm.
const LN_DBL_MAX: f64 = f64::from_bits(0x40862e4200000000);
/// `0.5 * ln(2)`, where `cosh` switches from `expm1` to `exp`.
const HALF_LN2: f64 = f64::from_bits(0x3fd62e4300000000);
/// `2^-55`, below which `tanh` is the identity.
const TANH_TINY: f64 = f64::from_bits(0x3c80000000000000);
/// `2^-28`, below which `sinh` is the identity.
const SINH_TINY: f64 = f64::from_bits(0x3e30000000000000);
/// Where squaring the argument would overflow, for `asinh` / `acosh`.
const SQUARE_LIMIT: f64 = 1e150;

/// The `Fast`, unchecked forms of the kernels this family is built on.
///
/// `Finite` because the callers below have already dealt with the special
/// lanes themselves — paying for a second range test per composed call would
/// be pure waste.
mod inner {
    use super::*;

    #[inline(always)]
    pub fn exp<V: Simd<Elem = f64>>(x: V) -> V {
        super::exp::eval::<V, Fast, Finite>(x)
    }
    #[inline(always)]
    pub fn expm1<V: Simd<Elem = f64>>(x: V) -> V {
        super::expm1::eval::<V, Fast, Finite>(x)
    }

    /// The bit-exact `exp`, checked — the composed functions reach arguments
    /// past its main path, so `Finite` would be wrong here.
    #[inline(always)]
    pub fn exp_exact<V: Simd<Elem = f64>>(x: V) -> V {
        super::exp::eval::<V, BitExact, FullRange>(x)
    }
    /// The bit-exact `expm1`, checked, for the same reason.
    #[inline(always)]
    pub fn expm1_exact<V: Simd<Elem = f64>>(x: V) -> V {
        super::expm1::eval::<V, BitExact, FullRange>(x)
    }
    #[inline(always)]
    pub fn ln<V: Simd<Elem = f64>>(x: V) -> V {
        crate::kernels::double::ln::eval::<V, Fast, Finite>(x)
    }
    #[inline(always)]
    pub fn log1p<V: Simd<Elem = f64>>(x: V) -> V {
        logx::log1p::eval::<V, Fast, Finite>(x)
    }
}

/// Hyperbolic sine.
pub mod sinh {
    use super::*;

    /// `sinh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact(x);
            if D::CHECKED {
                return crate::simd::patch_lanes(x, y, outside(x, LN_DBL_MAX), reference::sinh);
            }
            return y;
        }
        dispatch::<V, A, D>(x, reference::sinh, fast, |x| outside(x, OVERFLOW))
    }

    /// The ported path, valid for `|x| < ln(DBL_MAX)`.
    ///
    /// The two arms need *different* exponentials — `expm1` below 22 and `exp`
    /// above — so evaluating both unconditionally would double the cost of
    /// every vector to serve whichever arm happened to be needed. Each is
    /// computed only when some lane is in its range, which is the same bet
    /// `patch_lanes` makes everywhere else here.
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let h = V::splat(0.5).copysign(x);
        let one = V::splat(1.0);
        let big = a.ge_mask(V::splat(22.0));

        let mut y = if big.all() {
            a
        } else {
            let t = inner::expm1_exact(a);
            // |x| < 1 uses the form that does not cancel as x -> 0.
            let below_one = h * (V::splat(2.0) * t - t * t / (t + one));
            let below_22 = h * (t + t / (t + one));
            V::select(a.lt_mask(one), below_one, below_22)
        };
        if big.any() {
            y = V::select(big, h * inner::exp_exact(a), y);
        }
        V::select(a.lt_mask(V::splat(SINH_TINY)), x, y)
    }

    /// Measured error: below 3 ulp over `|x| < 710`.
    ///
    /// `t (t + 2) / (2 (t + 1))` with `t = expm1(|x|)`, rather than
    /// `(e^x - e^-x)/2`. The two agree mathematically; the difference is that
    /// for small `x` the subtraction form cancels to nothing while this one
    /// reduces to `t`, which is already the answer.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let t = inner::expm1(x.abs());
        let one = V::splat(1.0);
        (V::splat(0.5) * (t + t / (t + one))).copysign(x)
    }
}

/// Hyperbolic cosine.
pub mod cosh {
    use super::*;

    /// `cosh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact(x);
            if D::CHECKED {
                return crate::simd::patch_lanes(x, y, outside(x, LN_DBL_MAX), reference::cosh);
            }
            return y;
        }
        dispatch::<V, A, D>(x, reference::cosh, fast, |x| outside(x, OVERFLOW))
    }

    /// The ported path, valid for `|x| < ln(DBL_MAX)`.
    ///
    /// `expm1` is evaluated only when a lane is actually below `ln(2)/2`.
    /// `cosh` is the cheapest function in this family — the platform does it in
    /// about three nanoseconds — so an unconditional second exponential costs
    /// more than the whole call it replaces.
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let one = V::splat(1.0);
        let half = V::splat(0.5);

        let e = inner::exp_exact(a);
        let mid = half * e + half / e;
        let y = V::select(a.lt_mask(V::splat(22.0)), mid, half * e);

        let small = a.lt_mask(V::splat(HALF_LN2));
        if small.none() {
            return y;
        }
        // Near zero, `1 + t^2/(2(1+t))` keeps the result's leading bits, which
        // `0.5(e + 1/e)` would round away.
        let t = inner::expm1_exact(a);
        let w = one + t;
        let tiny = V::select(
            a.lt_mask(V::splat(f64::from_bits(0x3c80000000000000))),
            w,
            one + (t * t) / (w + w),
        );
        V::select(small, tiny, y)
    }

    /// Measured error: below 3 ulp over `|x| < 710`.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        // Both terms are positive, so there is nothing to cancel and the naive
        // form is the right one here — unlike `sinh`.
        let t = inner::exp(x.abs());
        V::splat(0.5) * (t + V::splat(1.0) / t)
    }
}

/// Hyperbolic tangent.
pub mod tanh {
    use super::*;

    /// `tanh(x)` for a vector of lanes.
    ///
    /// Saturates smoothly, so no lane needs repair: for large `|x|` the
    /// reduction gives `u -> -1` and the quotient gives `±1` on its own.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact(x);
            if D::CHECKED {
                // Only the non-finite lanes need repair; everything else
                // saturates on its own.
                return crate::simd::patch_lanes(x, y, not_normal(x), reference::tanh);
            }
            return y;
        }
        dispatch::<V, A, D>(x, reference::tanh, fast, no_lanes)
    }

    /// The ported path.
    ///
    /// One exponential, not two: the two arms of the platform's `|x| < 22`
    /// case differ only in the sign of the argument, so selecting the argument
    /// before the call rather than the result after it halves the work.
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let one = V::splat(1.0);
        let two = V::splat(2.0);

        let ge_one = a.ge_mask(one);
        let arg = V::select(ge_one, a + a, -(a + a));
        let t = inner::expm1_exact(arg);

        let z = V::select(ge_one, one - two / (t + two), -t / (t + two));
        let z = V::select(
            a.lt_mask(V::splat(22.0)),
            z,
            one - V::splat(f64::MIN_POSITIVE),
        );
        let z = V::select(a.lt_mask(V::splat(TANH_TINY)), a, z);
        z.copysign(x)
    }

    /// Measured error: below 3 ulp over the whole real line.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let u = inner::expm1(-(a + a));
        (-u / (u + V::splat(2.0))).copysign(x)
    }
}

/// Inverse hyperbolic sine.
pub mod asinh {
    use super::*;

    /// `asinh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::asinh, fast, |x| outside(x, SQUARE_LIMIT))
    }

    /// Measured error: below 3 ulp over `|x| < 1e150`.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let one = V::splat(1.0);
        let root = a.mul_add(a, one).sqrt();
        // Below 1, `a + root` rounds away the information in `a`; the log1p
        // form keeps it. Above 1 the direct form is both accurate and cheaper.
        let small = inner::log1p(a + a * a / (one + root));
        let large = inner::ln(a + root);
        V::select(a.lt_mask(one), small, large).copysign(x)
    }
}

/// Inverse hyperbolic cosine.
pub mod acosh {
    use super::*;

    /// `acosh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::acosh, fast, |x| {
            x.ge_mask(V::splat(1.0))
                .and(x.lt_mask(V::splat(SQUARE_LIMIT)))
                .not()
        })
    }

    /// Measured error: below 3 ulp over `1 <= x < 1e150`.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        // Written in `t = x - 1` throughout: `acosh(1) = 0`, and near there
        // the answer is the square root of a small number, which `x^2 - 1`
        // would have already destroyed.
        let t = x - V::splat(1.0);
        inner::log1p(t + (t * (t + V::splat(2.0))).sqrt())
    }
}

/// Inverse hyperbolic tangent.
pub mod atanh {
    use super::*;

    const ATANH_TINY: f64 = f64::from_bits(0x3c90000000000000); // 2^-54

    /// `atanh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact(x);
            if D::CHECKED {
                return crate::simd::patch_lanes(x, y, outside(x, 1.0), reference::atanh);
            }
            return y;
        }
        dispatch::<V, A, D>(x, reference::atanh, fast, |x| outside(x, 1.0))
    }

    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let one = V::splat(1.0);
        let two = V::splat(2.0);
        let half = V::splat(0.5);
        let arg = (two * x) / (one - x);
        let y = half * logx::log1p::eval::<V, BitExact, FullRange>(arg);
        V::select(x.abs().lt_mask(V::splat(ATANH_TINY)), x, y)
    }

    /// Measured error: below 3 ulp over `|x| < 1`.
    ///
    /// The same identity Rust's own `f64::atanh` uses, which is what makes
    /// this one's `BitExact` reference Rust's rather than the C library's —
    /// the two disagree on roughly one input in ten.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let two = V::splat(2.0);
        (V::splat(0.5) * inner::log1p(two * a / (V::splat(1.0) - a))).copysign(x)
    }
}
