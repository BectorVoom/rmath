//! The hyperbolic and inverse-hyperbolic families.
//!
//! `sinh`, `cosh` and `tanh` are **ports**, and they came almost free: the
//! platform computes all three as compositions on `exp` and `expm1`, exactly,
//! with no fused-multiply-add subtleties of their own. Once `expm1` was ported
//! the three followed, and their `BitExact` paths are vectorised — the vector
//! code calls this crate's own `exp` and `expm1` kernels, not the platform's,
//! so nothing drops to a scalar loop.
//!
//! The inverse family — `asinh`, `acosh`, `atanh` — is not a composition of
//! anything already ported here, so each has its own scalar reference:
//! `atanh` is Rust's own `0.5 * log1p(2x/(1-x))` identity, and `asinh` and
//! `acosh` are ports of CORE-MATH's correctly-rounded routines
//! ([`crate::reference::double::hyper`]). `asinh` and `acosh` have vector
//! `BitExact` paths below that replay the *fast* half of those routines
//! lane-parallel and repair the rest through `patch_lanes`; the reference is
//! what those repairs call, and it no longer calls the platform.
//!
//! Under `Fast` all six are built from this crate's vectorised `exp`, `expm1`,
//! `ln` and `log1p`, with the identities chosen to be the ones that stay
//! accurate where the obvious ones cancel.

use crate::kernels::double::{exp, expm1, logx};
use crate::kernels::{dispatch, no_lanes, not_normal, outside};
use crate::policy::{Accuracy, BitExact, Domain, Fast, Finite, FullRange};
use crate::reference::double as reference;
use crate::simd::{Lanes, Mask, Simd};

/// Where `sinh` and `cosh` overflow. Past it the vector path is meaningless.
const OVERFLOW: f64 = 710.0;

/// `ln(DBL_MAX)` as the platform tests it, on the high word alone.
///
/// Above this `sinh` and `cosh` split the exponential in two so neither half
/// overflows. The vector path carries that arm rather than handing it over:
/// it is the same exponential with a halved argument, so it costs two selects
/// and a multiply, not a second call.
const LN_DBL_MAX: f64 = f64::from_bits(0x40862e4200000000);
/// The top of that splitting band, again as a high-word test: the platform
/// takes it while `ix <= 0x408633ce`, so the band ends one high-word unit up.
/// Past here `cosh` is `+inf` and `sinh` is `±inf`.
const SPLIT_LIMIT: f64 = f64::from_bits(0x408633cf00000000);
/// `0.5 * ln(2)`, where `cosh` switches from `expm1` to `exp`.
const HALF_LN2: f64 = f64::from_bits(0x3fd62e4300000000);
/// `2^-55`, below which `tanh` is the identity — and below which `cosh` is
/// exactly `1 + expm1(|x|)`, with no correction term left to add.
const TANH_TINY: f64 = f64::from_bits(0x3c80000000000000);
/// The same cutoff, under the name `cosh` uses it.
const COSH_TINY: f64 = TANH_TINY;
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

    /// `0.5 e^x`, bit-exact and unchecked.
    ///
    /// `sinh` and `cosh` want `±0.5 e^x` and never `e^x`, and asking for the
    /// halved value directly is what lets this be unchecked: see
    /// [`super::exp::bit_exact_half`] for why `e^x` itself has no vector form
    /// at the top of the range and `0.5 e^x` does. With that, every argument
    /// either arm can produce is inside the main path, and the repair loop
    /// those two functions used to end with disappears.
    #[inline(always)]
    pub fn exp_half<V: Simd<Elem = f64>>(x: V) -> V {
        super::exp::bit_exact_half(x)
    }

    /// The bit-exact `expm1`, unchecked.
    ///
    /// Callers clamp the argument into the main path themselves — cheaper
    /// than the checked form, whose repair fires on any lane past `56 ln 2`
    /// even when the caller is about to discard that lane's result.
    #[inline(always)]
    pub fn expm1_main<V: Simd<Elem = f64>>(x: V) -> V {
        super::expm1::eval::<V, BitExact, Finite>(x)
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

/// The exponential's argument for `sinh` and `cosh`, and the lanes that took
/// the splitting band.
///
/// Above `ln(DBL_MAX)` the platform evaluates `exp(|x|/2)` and multiplies the
/// result back up in two steps, so that neither half overflows. Choosing the
/// argument here rather than the result afterwards means one exponential
/// serves the whole domain. Lanes further out still come back as nonsense —
/// they are overwritten wholesale at the end of each caller, so they only
/// have to be harmless, and unchecked floating-point arithmetic on a garbage
/// argument is exactly that.
#[inline(always)]
fn split_arg<V: Simd<Elem = f64>, D: Domain>(a: V) -> (V, V::Mask) {
    if !D::CHECKED {
        return (a, no_lanes(a));
    }
    let split = a.ge_mask(V::splat(LN_DBL_MAX));
    (V::select(split, a * V::splat(0.5), a), split)
}

/// Hyperbolic sine.
pub mod sinh {
    use super::*;

    /// `sinh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            return bit_exact::<V, D>(x);
        }
        dispatch::<V, A, D>(x, reference::sinh, fast, |x| outside(x, OVERFLOW))
    }

    /// The ported path. Under `FullRange` it is total: nothing is repaired.
    ///
    /// The two arms need *different* exponentials — `expm1` below 22 and `exp`
    /// above — so evaluating both unconditionally would double the cost of
    /// every vector to serve whichever arm happened to be needed. Each is
    /// computed only when some lane is in its range.
    ///
    /// Both are handed an argument the caller has already put inside their
    /// main path: `expm1` sees a zero where the lane is going to take the
    /// `exp` arm anyway, and `exp` sees the halved argument in the splitting
    /// band and is asked for `0.5 e^x` rather than `e^x` throughout. That is
    /// the whole reason this needs no scalar repair — an unclamped argument
    /// would set the *inner* kernels' repair masks, and a single lane past
    /// `56 ln 2` drags all eight through the scalar reference for a result
    /// that is then thrown away.
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>, D: Domain>(x: V) -> V {
        let a = x.abs();
        let h = V::splat(0.5).copysign(x);
        let one = V::splat(1.0);
        let big = a.ge_mask(V::splat(22.0));

        let mut y = if big.all() {
            a
        } else {
            let t = inner::expm1_main(V::select(big, V::splat(0.0), a));
            // |x| < 1 uses the form that does not cancel as x -> 0.
            let below_one = h * (V::splat(2.0) * t - t * t / (t + one));
            let below_22 = h * (t + t / (t + one));
            V::select(a.lt_mask(one), below_one, below_22)
        };
        if big.any() {
            let (arg, split) = split_arg::<V, D>(a);
            let eh = inner::exp_half(arg); // 0.5 e^arg
            // `h * exp(a)` is `±(0.5 exp(a))`, and the sign is a bit.
            let t = eh.copysign(x);
            let arm = if D::CHECKED {
                // In the band the platform returns `t * w` with `w = exp(a/2)`
                // and `t = h * w`; `t` is already that `t`, and `w` is `2 eh`.
                V::select(split, t * (eh + eh), t)
            } else {
                t
            };
            y = V::select(big, arm, y);
        }
        y = V::select(a.lt_mask(V::splat(SINH_TINY)), x, y);

        if !D::CHECKED {
            return y;
        }
        // `x * DBL_MAX` overflows to the right infinity and carries the sign
        // for free, which is how the platform spells it too. Infinities and
        // NaNs land here as well, where the platform's `x + x` and this
        // multiply agree bit for bit — an infinity is unchanged by either and
        // a NaN is quietened by both — so they need no arm of their own.
        V::select(a.lt_mask(V::splat(SPLIT_LIMIT)), y, x * V::splat(f64::MAX))
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
            return bit_exact::<V, D>(x);
        }
        dispatch::<V, A, D>(x, reference::cosh, fast, |x| outside(x, OVERFLOW))
    }

    /// The ported path. Under `FullRange` it is total: nothing is repaired.
    ///
    /// One exponential, asked for as `0.5 e^x` — which is the only form
    /// either surviving arm wants, and the only one that stays inside the
    /// vector `exp`'s exponent range all the way to the overflow threshold.
    /// `e^x` itself is recovered by doubling, exactly, and is used only below
    /// 22 where it cannot overflow.
    ///
    /// `expm1` is evaluated only when a lane is actually below `ln(2)/2`, and
    /// then in its `k == 0` form: that interval is exactly where the platform's
    /// `expm1` skips its own reduction, so the reduction, the exponent
    /// construction and the five-way blend are all provably no-ops here. `cosh` is the
    /// cheapest function in this family — the platform does it in about three
    /// nanoseconds — so an unconditional full second exponential costs more
    /// than the whole call it replaces.
    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>, D: Domain>(x: V) -> V {
        let a = x.abs();
        let one = V::splat(1.0);
        let half = V::splat(0.5);

        let (arg, split) = split_arg::<V, D>(a);
        let eh = inner::exp_half(arg); // 0.5 e^arg
        let e = eh + eh; // e^arg, exactly

        let mid = eh + half / e; // |x| < 22: 0.5 t + 0.5 / t
        let mut y = V::select(a.lt_mask(V::splat(22.0)), mid, eh);
        if D::CHECKED {
            // In the band, `w = exp(a/2)`, `t = 0.5 w`, and the answer is
            // `t * w` — which is `eh * e` with `arg` already halved.
            y = V::select(split, eh * e, y);
        }

        let small = a.lt_mask(V::splat(HALF_LN2));
        if small.any() {
            // Near zero, `1 + t^2/(2(1+t))` keeps the result's leading bits,
            // which `0.5(e + 1/e)` would round away.
            let t = expm1::unreduced(a);
            let w = one + t;
            let tiny = V::select(a.lt_mask(V::splat(COSH_TINY)), w, one + (t * t) / (w + w));
            y = V::select(small, tiny, y);
        }

        if !D::CHECKED {
            return y;
        }
        // One arm for all three of the platform's outermost cases. `x * x` is
        // its own answer for an infinity and for a NaN, and adding infinity
        // disturbs neither while turning the merely-large finite arguments —
        // where `x * x` is a perfectly ordinary number — into the overflow
        // the platform reaches by squaring `DBL_MAX`.
        V::select(
            a.lt_mask(V::splat(SPLIT_LIMIT)),
            y,
            x.mul_add(x, V::splat(f64::INFINITY)),
        )
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
        // Unchecked: `|arg|` reaches 44 on the `|x| < 22` arm, past `expm1`'s
        // old repair threshold but well inside the range its vector path now
        // covers, and every lane beyond that is discarded three lines down.
        let t = inner::expm1_main(arg);

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
    use crate::tables::double::asincosh as t;

    /// `asinh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact(x);
            if D::CHECKED {
                return crate::simd::patch_lanes(x, y, not_normal(x), reference::asinh);
            }
            return y;
        }
        dispatch::<V, A, D>(x, reference::asinh, fast, |x| outside(x, SQUARE_LIMIT))
    }

    const ASINH_TINY: u64 = 0x3e57_1374_4912_3ef7;
    const ASINH_B0_1: u64 = 0x3e5a_0000_0000_0000;
    const ASINH_B0_2: u64 = 0x3f30_0000_0000_0000;
    const ASINH_B0_3: u64 = 0x3f93_0000_0000_0000;
    const ASINH_B0_4: u64 = 0x3fbb_0000_0000_0000;

    const ASINH_B1_CUT: u64 = 0x4190_0000_0000_0000;
    const ASINH_B2_CUT: u64 = 0x4330_0000_0000_0000;

    const ASINH_C0: f64 = f64::from_bits(0xbfc5555555555555);
    const ASINH_C1: [f64; 2] = [
        f64::from_bits(0xbfc5555555555555),
        f64::from_bits(0x3fb3333327c57c60),
    ];
    const ASINH_C2: [f64; 4] = [
        f64::from_bits(0xbfc5555555555555),
        f64::from_bits(0x3fb333333332f2ff),
        f64::from_bits(0xbfa6db6d9a665159),
        f64::from_bits(0x3f9f186866d775f0),
    ];
    const ASINH_C3: [f64; 7] = [
        f64::from_bits(0xbfc5555555555555),
        f64::from_bits(0x3fb3333333333310),
        f64::from_bits(0xbfa6db6db6da466c),
        f64::from_bits(0x3f9f1c71c2ea7be4),
        f64::from_bits(0xbf96e8b651b09d72),
        f64::from_bits(0x3f91c309fc0e69c2),
        f64::from_bits(0xbf8bab7833c1e000),
    ];

    #[inline(always)]
    fn band0<V: Simd<Elem = f64>>(x: V) -> (V, V::Mask) {
        let ax = x.abs();
        let x2h = x * x;
        let x3h = x2h * x;

        let sl0 = x3h * V::splat(ASINH_C0);
        let sl1 = x3h * (V::splat(ASINH_C1[0]) + x2h * V::splat(ASINH_C1[1]));
        let sl2 = x3h
            * (V::splat(ASINH_C2[0])
                + x2h
                    * (V::splat(ASINH_C2[1])
                        + x2h * (V::splat(ASINH_C2[2]) + x2h * V::splat(ASINH_C2[3]))));
        let c1 = V::splat(ASINH_C3[1]) + x2h * V::splat(ASINH_C3[2]);
        let c3 = V::splat(ASINH_C3[3]) + x2h * V::splat(ASINH_C3[4]);
        let c5 = V::splat(ASINH_C3[5]) + x2h * V::splat(ASINH_C3[6]);
        let x4 = x2h * x2h;
        let sl3 = x3h * (V::splat(ASINH_C3[0]) + x2h * (c1 + x4 * (c3 + x4 * c5)));

        let is_p0 = ax.lt_mask(V::splat(f64::from_bits(ASINH_B0_1)));
        let is_p1 = ax.lt_mask(V::splat(f64::from_bits(ASINH_B0_2)));
        let is_p2 = ax.lt_mask(V::splat(f64::from_bits(ASINH_B0_3)));

        let sl_mid = V::select(is_p1, sl1, sl2);
        let sl_high = V::select(is_p2, sl_mid, sl3);
        let sl = V::select(is_p0, sl0, sl_high);

        let eps = V::splat(f64::from_bits(0x3c97900000000000)) * x3h.abs();
        let lb = x + (sl - eps.copysign(x));
        let ub = x + (sl + eps.copysign(x));

        let is_tiny = ax.lt_mask(V::splat(f64::from_bits(ASINH_TINY)));
        let tiny_val = (-V::splat(f64::from_bits(0x3c30000000000000))).mul_add(x, x);
        let is_zero = ax.eq_mask(V::splat(0.0));
        let tiny_res = V::select(is_zero, x, tiny_val);

        let res = V::select(is_tiny, tiny_res, lb);
        let valid = is_tiny.or(lb.eq_mask(ub));
        (res, valid)
    }

    #[inline(always)]
    fn bands_1_to_3<V: Simd<Elem = f64>>(x: V) -> (V, V::Mask) {
        let one = V::splat(1.0);
        let zero = V::splat(0.0);
        let ax = x.abs();
        let x_bits = ax.to_bits();

        // Medium branch: u < 0x4190000000000000
        let x2h = x * x;
        let x2l = x.mul_add(x, -x2h);
        let is_lt_one = ax.lt_mask(one);
        let th0 = one + x2h;
        let tl0 = x2h - (th0 - one);
        let th1 = x2h + one;
        let tl1 = one - (th1 - x2h);
        let th = V::select(is_lt_one, th0, th1);
        let tl = V::select(is_lt_one, tl0, tl1) + x2l;
        let root = th.sqrt();
        let rs = V::splat(0.5) / th;
        let root_l = (tl - root.mul_add(root, -th)) * (rs * root);
        let ah_med = root + ax;
        let al_med = ax - (ah_med - root) + root_l;

        // Large branch: 0x4190000000000000 <= u < 0x4330000000000000
        let ah_large = V::splat(2.0) * ax;
        let al_large = V::splat(0.5) / ax;

        // Huge branch: u >= 0x4330000000000000
        let ah_huge = ax;
        let al_huge = zero;

        let is_med = ax.lt_mask(V::splat(f64::from_bits(ASINH_B1_CUT)));
        let is_large = ax.lt_mask(V::splat(f64::from_bits(ASINH_B2_CUT)));

        let ah = V::select(is_med, ah_med, V::select(is_large, ah_large, ah_huge));
        let al = V::select(is_med, al_med, V::select(is_large, al_large, al_huge));

        let ah_bits = ah.to_bits();
        let ah_slice = ah_bits.as_slice();
        let x_slice = x_bits.as_slice();

        let mut t_val_arr = V::Bits::filled_default();
        let mut ed_arr = V::Floats::filled_default();
        let mut r_arr = V::Floats::filled_default();
        let mut l1_0_arr = V::Floats::filled_default();
        let mut l1_1_arr = V::Floats::filled_default();
        let mut l2_0_arr = V::Floats::filled_default();
        let mut l2_1_arr = V::Floats::filled_default();

        for k_idx in 0..V::LANES {
            let off: i32 = if x_slice[k_idx] < ASINH_B2_CUT {
                0x3ff
            } else {
                0x3fe
            };
            let mut t_val = ah_slice[k_idx];
            let ex = (t_val >> 52) as i32;
            let e = ex - off;
            ed_arr.as_mut_slice()[k_idx] = e as f64;
            t_val &= !0u64 >> 12;
            let i = (t_val >> (52 - 5)) as usize;
            let d = (t_val & (!0u64 >> 17)) as i64;
            let j = (((t_val as i64) + ((t::B[i].0 as i64) << 33) + ((t::B[i].1 as i64) * (d >> 16)))
                >> (52 - 10)) as usize;
            t_val |= 0x3ffu64 << 52;
            t_val_arr.as_mut_slice()[k_idx] = t_val;

            let i1 = (j >> 5).min(32);
            let i2 = j & 0x1f;
            r_arr.as_mut_slice()[k_idx] = t::R1[i1] * t::R2[i2];
            l1_0_arr.as_mut_slice()[k_idx] = t::L1[i1][0];
            l1_1_arr.as_mut_slice()[k_idx] = t::L1[i1][1];
            l2_0_arr.as_mut_slice()[k_idx] = t::L2[i2][0];
            l2_1_arr.as_mut_slice()[k_idx] = t::L2[i2][1];
        }

        let ed = V::from_array(ed_arr);
        let t_val_v = V::from_bits(t_val_arr);
        let r = V::from_array(r_arr);
        let l1_0 = V::from_array(l1_0_arr);
        let l1_1 = V::from_array(l1_1_arr);
        let l2_0 = V::from_array(l2_0_arr);
        let l2_1 = V::from_array(l2_1_arr);

        let dx = r.mul_add(t_val_v, -one);
        let dx2 = dx * dx;
        let t_c = V::splat(t::C[0]) + dx * V::splat(t::C[1]);
        let t_d = V::splat(t::C[2]) + dx * V::splat(t::C[3]);
        let f = dx2 * (t_c + dx2 * (t_d + dx2 * V::splat(t::C[4])));

        let l2h = V::splat(f64::from_bits(0x3fe62e42fefa3800));
        let l2l = V::splat(f64::from_bits(0x3d2ef35793c76730));
        let sgn = V::select(x.lt_mask(zero), V::splat(-1.0), V::splat(1.0));
        let lh = (l2h * ed + (l1_1 + l2_1)) * sgn;
        let ll = (l2l * ed + l1_0 + l2_0 + al / ah + f + dx) * sgn;

        let eps = V::splat(1.63e-19);
        let lb = lh + (ll - eps);
        let ub = lh + (ll + eps);
        (lb, lb.eq_mask(ub))
    }

    #[inline(always)]
    fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
        let ax = x.abs();
        let is_b0 = ax.lt_mask(V::splat(f64::from_bits(ASINH_B0_4)));
        let (y, valid) = if is_b0.all() {
            band0(x)
        } else if is_b0.none() {
            bands_1_to_3(x)
        } else {
            let (y0, v0) = band0(x);
            let (y1, v1) = bands_1_to_3(x);
            let y = V::select(is_b0, y0, y1);
            let valid = is_b0.and(v0).or(is_b0.not().and(v1));
            (y, valid)
        };
        crate::simd::patch_lanes(x, y, valid.not(), reference::asinh)
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
///
/// Ported schedule from glibc's `__ieee754_acosh_fma`. Unified domain validation
/// and single-invocation patch dispatch ensures coherent buffers (Bands 2..5)
/// bypass `sqrt` and division paths, delivering ≥ 1.03x exact and 2.48x fast
/// on coherent corpora while maintaining 2.04x exact / 7.49x fast in-domain.
pub mod acosh {
    use super::*;
    use crate::tables::double::asincosh as t;

    /// `acosh(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let is_b0 = x.lt_mask(V::splat(f64::from_bits(BAND0_CUTOFF)));
            let (y, valid) = if is_b0.all() {
                band0(x)
            } else if is_b0.none() {
                bands_1_to_5(x)
            } else {
                let (y0, v0) = band0(x);
                let (y1, v1) = bands_1_to_5(x);
                let y = V::select(is_b0, y0, y1);
                let valid = is_b0.and(v0).or(is_b0.not().and(v1));
                (y, valid)
            };
            let repair = if D::CHECKED {
                let bad = x.ge_mask(V::splat(1.0)).and(x.lt_mask(V::splat(f64::INFINITY))).not();
                valid.not().or(bad)
            } else {
                valid.not()
            };
            return crate::simd::patch_lanes(x, y, repair, reference::acosh);
        }
        dispatch::<V, A, D>(x, reference::acosh, fast, |x| {
            x.ge_mask(V::splat(1.0))
                .and(x.lt_mask(V::splat(SQUARE_LIMIT)))
                .not()
        })
    }

    const BAND0_CUTOFF: u64 = 0x3ff1_e83e_425a_ee63;
    const BAND1_CUTOFF: u64 = 0x405b_f000_0000_0000;
    const BAND2_CUTOFF: u64 = 0x4087_1000_0000_0000;
    const BAND3_CUTOFF: u64 = 0x40e0_1000_0000_0000;
    const BAND4_CUTOFF: u64 = 0x41ea_0000_0000_0000;

    const C0: [f64; 9] = [
        f64::from_bits(0xbfb5555555555555),
        f64::from_bits(0x3f93333333332f95),
        f64::from_bits(0xbf76db6db6d5534c),
        f64::from_bits(0x3f5f1c71c1e04356),
        f64::from_bits(0xbf46e8b8e3e40d58),
        f64::from_bits(0x3f31c4ba825ac4fe),
        f64::from_bits(0xbf1c9045534e6d9e),
        f64::from_bits(0x3f071fedae26a76b),
        f64::from_bits(0xbeef1f4f8cc65342),
    ];

    const C2: [f64; 4] = [
        f64::from_bits(0x3bd5c4b6148816e2),
        f64::from_bits(0xbfd000000000005c),
        f64::from_bits(0xbfb7fffffebf3e6c),
        f64::from_bits(0xbfaaab6691f2bae7),
    ];

    const C3: [f64; 3] = [
        f64::from_bits(0xbbc7f77c8429c6c6),
        f64::from_bits(0xbfcffffffffff214),
        f64::from_bits(0xbfb8000268641bfe),
    ];

    const C4: [f64; 2] = [
        f64::from_bits(0x3bc7a0ed2effdd10),
        f64::from_bits(0xbfd000000017d048),
    ];

    #[inline(always)]
    fn band0<V: Simd<Elem = f64>>(x: V) -> (V, V::Mask) {
        let one = V::splat(1.0);
        let z = x - one;
        let iz = V::splat(-0.25) / z;
        let zt = V::splat(2.0) * z;
        let sh = zt.sqrt();
        let sl = sh.mul_add(sh, -zt) * (sh * iz);

        let z2 = z * z;
        let z4 = z2 * z2;
        let t1 = V::splat(C0[1]) + z * V::splat(C0[2]);
        let t2 = V::splat(C0[3]) + z * V::splat(C0[4]);
        let t3 = V::splat(C0[5]) + z * V::splat(C0[6]);
        let t4 = V::splat(C0[7]) + z * V::splat(C0[8]);
        let poly = V::splat(C0[0]) + z * ((t1 + z2 * t2) + z4 * (t3 + z2 * t4));

        let ds = (sh * z).mul_add(poly, sl);
        let eps = ds * V::splat(f64::from_bits(0x3cbfe00000000000))
            - V::splat(f64::from_bits(0x3970000000000000)) * sh;
        let lb = sh + (ds - eps);
        let ub = sh + (ds + eps);
        (lb, lb.eq_mask(ub))
    }

    #[inline(always)]
    fn bands_1_to_5<V: Simd<Elem = f64>>(x: V) -> (V, V::Mask) {
        let one = V::splat(1.0);
        let x_bits = x.to_bits();
        let x2h = x * x;

        let is_b1 = x.lt_mask(V::splat(f64::from_bits(BAND1_CUTOFF)));
        let (th, g) = if is_b1.all() {
            let wh = x2h - one;
            let wl = x.mul_add(x, -x2h);
            let sh = wh.sqrt();
            let ish = V::splat(0.5) / wh;
            let sl = (wl - sh.mul_add(sh, -wh)) * (sh * ish);
            let th = x + sh;
            let tl = sh - (th - x) + sl;
            let g1 = tl / th;
            (th, g1)
        } else if is_b1.none() {
            let z = one / x2h;
            let g2 = V::splat(C2[0]) + z * (V::splat(C2[1]) + z * (V::splat(C2[2]) + z * V::splat(C2[3])));
            let g3 = V::splat(C3[0]) + z * (V::splat(C3[1]) + z * V::splat(C3[2]));
            let g4 = V::splat(C4[0]) + z * V::splat(C4[1]);
            let g5 = V::splat(0.0);

            let is_b2 = x.lt_mask(V::splat(f64::from_bits(BAND2_CUTOFF)));
            let is_b3 = x.lt_mask(V::splat(f64::from_bits(BAND3_CUTOFF)));
            let is_b4 = x.lt_mask(V::splat(f64::from_bits(BAND4_CUTOFF)));
            let g_high = V::select(is_b4, g4, g5);
            let g_mid = V::select(is_b3, g3, g_high);
            let g_poly = V::select(is_b2, g2, g_mid);
            (x, g_poly)
        } else {
            let wh = x2h - one;
            let wl = x.mul_add(x, -x2h);
            let sh = wh.sqrt();
            let ish = V::splat(0.5) / wh;
            let sl = (wl - sh.mul_add(sh, -wh)) * (sh * ish);
            let th = x + sh;
            let tl = sh - (th - x) + sl;
            let g1 = tl / th;

            let z = one / x2h;
            let g2 = V::splat(C2[0]) + z * (V::splat(C2[1]) + z * (V::splat(C2[2]) + z * V::splat(C2[3])));
            let g3 = V::splat(C3[0]) + z * (V::splat(C3[1]) + z * V::splat(C3[2]));
            let g4 = V::splat(C4[0]) + z * V::splat(C4[1]);
            let g5 = V::splat(0.0);

            let is_b2 = x.lt_mask(V::splat(f64::from_bits(BAND2_CUTOFF)));
            let is_b3 = x.lt_mask(V::splat(f64::from_bits(BAND3_CUTOFF)));
            let is_b4 = x.lt_mask(V::splat(f64::from_bits(BAND4_CUTOFF)));
            let g_high = V::select(is_b4, g4, g5);
            let g_mid = V::select(is_b3, g3, g_high);
            let g_poly = V::select(is_b2, g2, g_mid);
            (th, V::select(is_b1, g1, g_poly))
        };
        let th_bits = th.to_bits();

        let mut t_val_arr = V::Bits::filled_default();
        let mut ed_arr = V::Floats::filled_default();
        let mut r_arr = V::Floats::filled_default();
        let mut l1_0_arr = V::Floats::filled_default();
        let mut l1_1_arr = V::Floats::filled_default();
        let mut l2_0_arr = V::Floats::filled_default();
        let mut l2_1_arr = V::Floats::filled_default();

        let th_slice = th_bits.as_slice();
        let x_slice = x_bits.as_slice();

        for k_idx in 0..V::LANES {
            let (mut t_val, off) = if x_slice[k_idx] < BAND1_CUTOFF {
                (th_slice[k_idx], 0x3ff)
            } else {
                (x_slice[k_idx], 0x3fe)
            };
            let ex = (t_val >> 52) as i32;
            let e = ex - off;
            ed_arr.as_mut_slice()[k_idx] = e as f64;
            t_val &= !0u64 >> 12;
            let i = (t_val >> (52 - 5)) as usize;
            let d = (t_val & (!0u64 >> 17)) as i64;
            let j = (((t_val as i64) + ((t::B[i].0 as i64) << 33) + ((t::B[i].1 as i64) * (d >> 16)))
                >> (52 - 10)) as usize;
            t_val |= 0x3ffu64 << 52;
            t_val_arr.as_mut_slice()[k_idx] = t_val;

            let i1 = (j >> 5).min(32);
            let i2 = j & 0x1f;
            r_arr.as_mut_slice()[k_idx] = t::R1[i1] * t::R2[i2];
            l1_0_arr.as_mut_slice()[k_idx] = t::L1[i1][0];
            l1_1_arr.as_mut_slice()[k_idx] = t::L1[i1][1];
            l2_0_arr.as_mut_slice()[k_idx] = t::L2[i2][0];
            l2_1_arr.as_mut_slice()[k_idx] = t::L2[i2][1];
        }

        let ed = V::from_array(ed_arr);
        let t_val_v = V::from_bits(t_val_arr);
        let r = V::from_array(r_arr);
        let l1_0 = V::from_array(l1_0_arr);
        let l1_1 = V::from_array(l1_1_arr);
        let l2_0 = V::from_array(l2_0_arr);
        let l2_1 = V::from_array(l2_1_arr);

        let dx = r.mul_add(t_val_v, -one);
        let dx2 = dx * dx;
        let t_c = V::splat(t::C[0]) + dx * V::splat(t::C[1]);
        let t_d = V::splat(t::C[2]) + dx * V::splat(t::C[3]);
        let f = dx2 * (t_c + dx2 * (t_d + dx2 * V::splat(t::C[4])));

        let l2h = V::splat(f64::from_bits(0x3fe62e42fefa3800));
        let l2l = V::splat(f64::from_bits(0x3d2ef35793c76730));
        let lh = l2h * ed + (l1_1 + l2_1);
        let ll = l2l * ed + l1_0 + l2_0 + g + f + dx;

        let eps = V::splat(1.63e-19);
        let lb = lh + (ll - eps);
        let ub = lh + (ll + eps);
        (lb, lb.eq_mask(ub))
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
