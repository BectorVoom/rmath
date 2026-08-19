//! The Bessel functions of the first and second kind, orders 0, 1 and `n`.
//!
//! # What vectorises here, and what does not
//!
//! Below `|x| = 2` each of `j0`, `j1`, `y0`, `y1` is a rational function of
//! `x^2` — pure vector arithmetic. At or above 2 they are
//! `sqrt(2/(pi x)) (P cos(x0) -+ Q sin(x0))`, and that splits into two parts
//! with very different characters:
//!
//! * `P` and `Q` are rational functions of `1/x^2` with one of four coefficient
//!   sets chosen by interval — a gather and then vector arithmetic, exactly
//!   like every other table-driven kernel here.
//! * `sin x`, `cos x` and `cos 2x` are the platform's, and under
//!   [`crate::policy::BitExact`] the platform's trigonometry runs one lane at a
//!   time, for the reason set out in [`crate::reference`].
//!
//! So these kernels inherit the trigonometric family's policy: `BitExact`
//! vectorises everything except the three trigonometric calls, and
//! [`crate::policy::Fast`] vectorises those too by taking
//! [`crate::kernels::double::trig`]'s own fast path. Both branches are
//! evaluated and blended rather than one being deferred to a scalar fallback,
//! because neither is rare — a buffer of Bessel arguments can sit entirely on
//! either side of 2.
//!
//! # `jn` and `yn`
//!
//! Scalar, and honestly so. `jn` chooses between forward recurrence, backward
//! recurrence and a Taylor series on `n` against `x`, and the backward
//! recurrence runs a continued fraction whose *length is decided at run time*
//! by iterating until a convergent exceeds `1e9`. A vector of lanes would run
//! the longest lane's loop for every lane and blend three unrelated branches;
//! there is no version of that which is faster than calling the scalar routine
//! per lane. They are bit-exact, which is the guarantee that matters, and they
//! are at parity.

use crate::kernels::double::{ln, trig};
use crate::policy::{Accuracy, Domain};
use crate::reference::double::bessel as reference;
use crate::simd::{Lanes, Mask, Simd, map_lanes2, patch_lanes};
use crate::tables::double::bessel as t;

/// Above this the scalar routine stops correcting `ss`/`cc`, because `x + x`
/// would overflow. Those lanes go to the reference.
const MAIN_PATH_LIMIT: f64 = f64::from_bits(0x7fe0_0000_0000_0000);

/// Above this `P` is 1 and `Q` is 0 to the last bit, and the rational fits
/// are skipped entirely.
///
/// fdlibm writes the test as `ix > 0x48000000` on the *high word*, which is
/// not the same as `x > 2^129`: it is true from the first double whose high
/// word is `0x48000001`. Every threshold in this file is converted that way —
/// `ix > K` becomes `x >= bits((K + 1) << 32)` and `ix <= K` becomes
/// `x < bits((K + 1) << 32)` — because getting it wrong would move a branch
/// boundary by a millionth of a binade, which no sampled test would find.
const NO_PQ: f64 = f64::from_bits(0x4800_0001_0000_0000);

/// `2^28`, above which the `P`/`Q` rational fits collapse to their limits.
const PQ_LIMIT: f64 = f64::from_bits(0x41b0_0000_0000_0000);

/// Evaluate whichever of the two branches the lanes actually need.
///
/// # Why this is not just a blend
///
/// Both branches are correct wherever they are selected, so a vector could
/// compute both and blend. It should not. The asymptotic branch carries three
/// trigonometric calls and the series branch a logarithm and a nested order-0
/// evaluation; paying for the one the data does not want is what made an
/// earlier version of `y0` *slower* than the scalar routine it replaces.
///
/// # Why `BitExact` gives the asymptotic branch away entirely
///
/// Because vectorising it cannot win. Under [`crate::policy::BitExact`] the
/// three trigonometric calls run one lane at a time — glibc computes them with
/// the IBM Accurate Portable Math routines, which this crate does not
/// reproduce — and they are most of the cost. Worse, the scalar routine gets
/// `sin` and `cos` from a single `sincos` that shares one argument reduction,
/// while this crate has to ask for them separately. So a "vectorised"
/// bit-exact asymptotic branch does *more* scalar trigonometry than the call
/// it replaces and lands at about 0.8x: measurably slower, for no gain.
///
/// So `BitExact` takes the vector path only when every lane is below 2, where
/// there is no trigonometry at all and the win is large, and hands anything
/// else to the scalar routine — parity, which is the honest ceiling.
/// [`crate::policy::Fast`] vectorises both branches and is 2.4x to 3.8x.
#[inline(always)]
fn branch<V: Simd<Elem = f64>, A: Accuracy, B: Fn(V) -> V, S: Fn(V) -> V>(
    x: V,
    big: B,
    small: S,
    scalar: fn(f64) -> f64,
) -> V {
    let is_big = x.abs().ge_mask(V::splat(2.0));
    if is_big.none() {
        return small(x);
    }
    if A::BIT_EXACT {
        return crate::simd::map_lanes(x, scalar);
    }
    if is_big.all() {
        return big(x);
    }
    V::select(is_big, big(x), small(x))
}

/// `(sin x - cos x, sin x + cos x)`, each computed the way that does not
/// cancel.
///
/// `ss` is `sqrt(2) sin(x - pi/4)` and `cc` is `sqrt(2) cos(x - pi/4)`.
/// Whichever of the two is small is recovered from `-cos(2x)` divided by the
/// other, since their product is exactly `-cos(2x)` — computing both as
/// `sin +- cos` would lose every digit of the smaller one.
#[inline(always)]
fn ss_cc<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> (V, V) {
    let (s, c) = trig::sincos::eval::<V, A, D>(x);
    let ss = s - c;
    let cc = s + c;
    let z = -trig::cos::eval::<V, A, D>(x + x);
    let flip = (s * c).lt_mask(V::splat(0.0));
    (V::select(flip, ss, z / cc), V::select(flip, z / ss, cc))
}

/// Gather `n` coefficient vectors from a four-row table, by interval of `|x|`.
///
/// The four intervals are fdlibm's `[8, inf)`, `[4.5454, 8)`, `[2.8571, 4.547)`
/// and `[2, 2.8570)`; the thresholds are compared on the high word, which is
/// the same ordering as comparing the values for a positive `x`.
#[inline(always)]
fn gather<V: Simd<Elem = f64>, const N: usize>(x: &V, table: &[[f64; N]; 4]) -> [V; N] {
    let xs = x.to_array();
    let mut out = [V::Floats::filled_default(); N];
    for lane in 0..V::LANES {
        let ix = ((xs.as_slice()[lane].to_bits() >> 32) & 0x7fff_ffff) as u32;
        let i = if ix >= 0x4020_0000 {
            0
        } else if ix >= 0x4012_2e8b {
            1
        } else if ix >= 0x4006_db6d {
            2
        } else {
            3
        };
        for (k, o) in out.iter_mut().enumerate() {
            o.as_mut_slice()[lane] = table[i][k];
        }
    }
    out.map(V::from_array)
}

/// `1 + R(s)/S(s)` with `s = 1/x^2`, the shape both `P` factors share.
#[inline(always)]
fn rational_p<V: Simd<Elem = f64>>(x: V, p: &[V; 6], q: &[V; 5]) -> V {
    let z = V::splat(1.0) / (x * x);
    let r1 = p[0] + z * p[1];
    let z2 = z * z;
    let r2 = p[2] + z * p[3];
    let z4 = z2 * z2;
    let r3 = p[4] + z * p[5];
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = V::splat(1.0) + z * q[0];
    let s2 = q[1] + z * q[2];
    let s3 = q[3] + z * q[4];
    let s = s1 + z2 * s2 + z4 * s3;
    V::splat(1.0) + r / s
}

/// `R(s)/S(s)` with `s = 1/x^2`, the shape both `Q` factors share. One term
/// longer in the denominator than [`rational_p`].
#[inline(always)]
fn rational_q<V: Simd<Elem = f64>>(x: V, p: &[V; 6], q: &[V; 6]) -> V {
    let z = V::splat(1.0) / (x * x);
    let r1 = p[0] + z * p[1];
    let z2 = z * z;
    let r2 = p[2] + z * p[3];
    let z4 = z2 * z2;
    let r3 = p[4] + z * p[5];
    let z6 = z4 * z2;
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = V::splat(1.0) + z * q[0];
    let s2 = q[1] + z * q[2];
    let s3 = q[3] + z * q[4];
    let s = s1 + z2 * s2 + z4 * s3 + z6 * q[5];
    r / s
}

/// `(P(x), Q(x))` for order 0.
#[inline(always)]
fn pq0<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let big = x.ge_mask(V::splat(PQ_LIMIT));
    let p = rational_p(x, &gather(&x, &t::P0R), &gather(&x, &t::P0S));
    let q = (V::splat(-0.125) + rational_q(x, &gather(&x, &t::Q0R), &gather(&x, &t::Q0S))) / x;
    (
        V::select(big, V::splat(1.0), p),
        V::select(big, V::splat(-0.125) / x, q),
    )
}

/// `(P(x), Q(x))` for order 1.
#[inline(always)]
fn pq1<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let big = x.ge_mask(V::splat(PQ_LIMIT));
    let p = rational_p(x, &gather(&x, &t::P1R), &gather(&x, &t::P1S));
    let q = (V::splat(0.375) + rational_q(x, &gather(&x, &t::Q1R), &gather(&x, &t::Q1S))) / x;
    (
        V::select(big, V::splat(1.0), p),
        V::select(big, V::splat(0.375) / x, q),
    )
}

/// The Bessel function of the first kind, order 0.
pub mod j0 {
    use super::*;

    /// `j0(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let y = main::<V, A, D>(x);
        if D::CHECKED {
            patch_lanes(
                x,
                y,
                crate::kernels::outside(x, MAIN_PATH_LIMIT),
                reference::j0,
            )
        } else {
            y
        }
    }

    #[inline(always)]
    pub(super) fn main<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        branch::<V, A, _, _>(
            x.abs(),
            |z| asymptotic::<V, A, D>(z),
            series::<V>,
            reference::j0,
        )
    }

    /// `|x| >= 2`: the asymptotic form.
    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(z: V) -> V {
        let (ss, cc) = ss_cc::<V, A, D>(z);
        let (p, q) = pq0(z);
        let far = (V::splat(t::INVSQRTPI) * cc) / z.sqrt();
        let near = V::splat(t::INVSQRTPI) * (p * cc - q * ss) / z.sqrt();
        V::select(z.ge_mask(V::splat(NO_PQ)), far, near)
    }

    /// `|x| < 2`: a rational fit in `x^2`.
    #[inline(always)]
    fn series<V: Simd<Elem = f64>>(z: V) -> V {
        let zz = z * z;
        let r = super::j0_num(zz);
        let s = super::j0_den(zz);
        let u = V::splat(0.5) * z;
        // Two spellings of the same value: below 1 the `1 + z*(...)` form
        // keeps its digits, and above it the factored `(1+u)(1-u)` does.
        let sub1 = V::splat(1.0) + zz * (V::splat(-0.25) + (r / s));
        let ge1 = (V::splat(1.0) + u) * (V::splat(1.0) - u) + zz * (r / s);
        let small = V::select(z.lt_mask(V::splat(1.0)), sub1, ge1);
        // Below 2^-13 the fit collapses to `1 - x^2/4`, and below 2^-27 to 1.
        V::select(
            z.lt_mask(V::splat(f64::from_bits(0x3f20_0000_0000_0000))),
            V::select(
                z.lt_mask(V::splat(f64::from_bits(0x3e40_0000_0000_0000))),
                V::splat(1.0),
                V::splat(1.0) - V::splat(0.25) * z * z,
            ),
            small,
        )
    }
}

/// `j0`'s numerator on `[0, 2]`, in fdlibm's exact association.
#[inline(always)]
fn j0_num<V: Simd<Elem = f64>>(z: V) -> V {
    let r = &t::J0_R;
    let r1 = z * V::splat(r[2]);
    let z2 = z * z;
    let r2 = V::splat(r[3]) + z * V::splat(r[4]);
    let z4 = z2 * z2;
    r1 + z2 * r2 + z4 * V::splat(r[5])
}

/// `j0`'s denominator on `[0, 2]`.
#[inline(always)]
fn j0_den<V: Simd<Elem = f64>>(z: V) -> V {
    let s = &t::J0_S;
    let s1 = V::splat(1.0) + z * V::splat(s[1]);
    let z2 = z * z;
    let s2 = V::splat(s[2]) + z * V::splat(s[3]);
    let z4 = z2 * z2;
    s1 + z2 * s2 + z4 * V::splat(s[4])
}

/// The Bessel function of the second kind, order 0.
pub mod y0 {
    use super::*;

    /// `y0(x)` for a vector of lanes.
    ///
    /// The domain is the positive reals: `y0` is `-inf` at zero and NaN below
    /// it, and both go to the reference rather than being blended in, because
    /// the main path would have to produce them out of a logarithm that has
    /// already gone to `-inf`.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let y = main::<V, A, D>(x);
        if D::CHECKED {
            let special = x
                .gt_mask(V::splat(0.0))
                .and(x.lt_mask(V::splat(MAIN_PATH_LIMIT)))
                .not();
            patch_lanes(x, y, special, reference::y0)
        } else {
            y
        }
    }

    #[inline(always)]
    fn main<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        branch::<V, A, _, _>(
            x,
            |x| asymptotic::<V, A, D>(x),
            |x| series::<V, A, D>(x),
            reference::y0,
        )
    }

    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let (ss, cc) = ss_cc::<V, A, D>(x);
        let (p, q) = pq0(x);
        let far = (V::splat(t::INVSQRTPI) * ss) / x.sqrt();
        let near = V::splat(t::INVSQRTPI) * (p * ss + q * cc) / x.sqrt();
        V::select(x.ge_mask(V::splat(NO_PQ)), far, near)
    }

    #[inline(always)]
    fn series<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let lnx = ln::eval::<V, A, D>(x);
        let z = x * x;
        let u = &t::Y0_U;
        let v = &t::Y0_V;
        let u1 = V::splat(u[0]) + z * V::splat(u[1]);
        let z2 = z * z;
        let u2 = V::splat(u[2]) + z * V::splat(u[3]);
        let z4 = z2 * z2;
        let u3 = V::splat(u[4]) + z * V::splat(u[5]);
        let z6 = z4 * z2;
        let un = u1 + z2 * u2 + z4 * u3 + z6 * V::splat(u[6]);
        let v1 = V::splat(1.0) + z * V::splat(v[0]);
        let v2 = V::splat(v[1]) + z * V::splat(v[2]);
        let vd = v1 + z2 * v2 + z4 * V::splat(v[3]);
        let small = un / vd + V::splat(t::TPI) * (j0::main::<V, A, D>(x) * lnx);
        // Below 2^-27, U/V is U[0] and j0(x) is 1.
        V::select(
            x.lt_mask(V::splat(f64::from_bits(0x3e40_0001_0000_0000))),
            V::splat(u[0]) + V::splat(t::TPI) * lnx,
            small,
        )
    }
}

/// The Bessel function of the first kind, order 1.
pub mod j1 {
    use super::*;

    /// `j1(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let y = main::<V, A, D>(x);
        if D::CHECKED {
            patch_lanes(
                x,
                y,
                crate::kernels::outside(x, MAIN_PATH_LIMIT),
                reference::j1,
            )
        } else {
            y
        }
    }

    #[inline(always)]
    pub(super) fn main<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        branch::<V, A, _, _>(
            x,
            |x| asymptotic::<V, A, D>(x, x.abs()),
            |x| series::<V>(x, x.abs()),
            reference::j1,
        )
    }

    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V, y: V) -> V {
        // Order 1 puts `x0` at `x - 3pi/4` rather than `x - pi/4`, which
        // exchanges the roles and the signs of the two combinations.
        let (s, c) = trig::sincos::eval::<V, A, D>(y);
        let ss0 = -s - c;
        let cc0 = s - c;
        let z = trig::cos::eval::<V, A, D>(y + y);
        let flip = (s * c).gt_mask(V::splat(0.0));
        let ss = V::select(flip, ss0, z / cc0);
        let cc = V::select(flip, z / ss0, cc0);

        let (p, q) = pq1(y);
        let far = (V::splat(t::INVSQRTPI) * cc) / y.sqrt();
        let near = V::splat(t::INVSQRTPI) * (p * cc - q * ss) / y.sqrt();
        let big = V::select(y.ge_mask(V::splat(NO_PQ)), far, near);
        // `j1` is odd, and the branch above computed it at `|x|`.
        V::select(x.lt_mask(V::splat(0.0)), -big, big)
    }

    #[inline(always)]
    fn series<V: Simd<Elem = f64>>(x: V, y: V) -> V {
        let zz = x * x;
        let r = &t::J1_R;
        let s = &t::J1_S;
        let r1 = zz * V::splat(r[0]);
        let z2 = zz * zz;
        let r2 = V::splat(r[1]) + zz * V::splat(r[2]);
        let z4 = z2 * z2;
        let rn = (r1 + z2 * r2 + z4 * V::splat(r[3])) * x;
        let s1 = V::splat(1.0) + zz * V::splat(s[1]);
        let s2 = V::splat(s[2]) + zz * V::splat(s[3]);
        let s3 = V::splat(s[4]) + zz * V::splat(s[5]);
        let sd = s1 + z2 * s2 + z4 * s3;
        let small = x * V::splat(0.5) + rn / sd;
        // Below 2^-27, j1(x) is x/2 exactly.
        V::select(
            y.lt_mask(V::splat(f64::from_bits(0x3e40_0000_0000_0000))),
            V::splat(0.5) * x,
            small,
        )
    }
}

/// The Bessel function of the second kind, order 1.
pub mod y1 {
    use super::*;

    /// `y1(x)` for a vector of lanes. Positive reals only; see [`super::y0`].
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let y = main::<V, A, D>(x);
        if D::CHECKED {
            let special = x
                .gt_mask(V::splat(0.0))
                .and(x.lt_mask(V::splat(MAIN_PATH_LIMIT)))
                .not();
            patch_lanes(x, y, special, reference::y1)
        } else {
            y
        }
    }

    #[inline(always)]
    fn main<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        branch::<V, A, _, _>(
            x,
            |x| asymptotic::<V, A, D>(x),
            |x| series::<V, A, D>(x),
            reference::y1,
        )
    }

    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let (s, c) = trig::sincos::eval::<V, A, D>(x);
        let ss0 = -s - c;
        let cc0 = s - c;
        let z = trig::cos::eval::<V, A, D>(x + x);
        let flip = (s * c).gt_mask(V::splat(0.0));
        let ss = V::select(flip, ss0, z / cc0);
        let cc = V::select(flip, z / ss0, cc0);

        let (p, q) = pq1(x);
        let far = (V::splat(t::INVSQRTPI) * ss) / x.sqrt();
        let near = V::splat(t::INVSQRTPI) * (p * ss + q * cc) / x.sqrt();
        V::select(x.ge_mask(V::splat(NO_PQ)), far, near)
    }

    #[inline(always)]
    fn series<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        let zz = x * x;
        let u = &t::Y1_U;
        let v = &t::Y1_V;
        let u1 = V::splat(u[0]) + zz * V::splat(u[1]);
        let z2 = zz * zz;
        let u2 = V::splat(u[2]) + zz * V::splat(u[3]);
        let z4 = z2 * z2;
        let un = u1 + z2 * u2 + z4 * V::splat(u[4]);
        let v1 = V::splat(1.0) + zz * V::splat(v[0]);
        let v2 = V::splat(v[1]) + zz * V::splat(v[2]);
        let v3 = V::splat(v[3]) + zz * V::splat(v[4]);
        let vd = v1 + z2 * v2 + z4 * v3;
        let small = x * (un / vd)
            + V::splat(t::TPI)
                * (j1::main::<V, A, D>(x) * ln::eval::<V, A, D>(x) - V::splat(1.0) / x);
        // Below 2^-54 nothing but the pole survives.
        V::select(
            x.lt_mask(V::splat(f64::from_bits(0x3c90_0001_0000_0000))),
            -V::splat(t::TPI) / x,
            small,
        )
    }
}

/// The Bessel function of the first kind, order `n`.
pub mod jn {
    use super::*;

    /// `jn(n, x)` for vectors of lanes. Scalar per lane; see the module
    /// documentation for why.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(n: V, x: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        map_lanes2(n, x, |n, x| reference::jn(order(n), x))
    }
}

/// The Bessel function of the second kind, order `n`.
pub mod yn {
    use super::*;

    /// `yn(n, x)` for vectors of lanes. Scalar per lane.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(n: V, x: V) -> V {
        let _ = (A::BIT_EXACT, D::CHECKED);
        map_lanes2(n, x, |n, x| reference::yn(order(n), x))
    }
}

/// The order, carried in a float lane and read as C's `int` would read it.
///
/// Truncation towards zero and saturation at the ends, which is what a C
/// `(int)` cast of an out-of-range double does on every target this crate
/// builds for — and what Rust's `as` does by definition.
#[inline(always)]
fn order(n: f64) -> i32 {
    n.trunc() as i32
}
