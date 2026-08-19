//! The error function, single precision.
//!
//! * **`BitExact`**: both branches of the platform's `erff` are plain
//!   `double` arithmetic over a 56-interval table, so an `f32x8` widens to an
//!   `f64x8`, replays the schedule lane-parallel and narrows once. Correctly
//!   rounded, and therefore bit-exact to any correctly-rounded `erff` on any
//!   platform — see [`crate::reference::single::erf_parts`]. Unlike the
//!   double-precision kernel there is no rounding test and no accurate path:
//!   24 significand bits leave enough headroom that the polynomial settles
//!   the rounding on its own, for every one of the 2^32 inputs. `tests/glibc.rs`
//!   checks all of them.
//!
//! * **`Fast`**: native, no widening. Below [`p::ERF_SPLIT`], `erf(x) = x
//!   P(x^2)`, an odd minimax series (`ERF_NEAR`, shared with [`super::erfc`]).
//!   Above it, `erf(x) = 1 - exp(-x^2) Q(z)` with `z = (|x| - A0)/(|x| + B0)`
//!   — the platform's own compression of the tail into one polynomial, refit
//!   directly in single precision rather than borrowed from its `double`
//!   table — using this crate's own native `Fast` `exp2` for `exp(-x^2)`, not
//!   a table. See `fast` below for the measured bound.
//!
//! `Finite` means `|x| <= 0x1.f5a888p+1`. Above it `erf(x)` is `+-1`.

use crate::kernels::horner;
use crate::policy::{Accuracy, Domain, Fast, Finite};
use crate::reference::single::erf_parts as r;
use crate::simd::{Mask, Simd, patch_lanes};
use crate::tables::single::erf as t;
use crate::tables::single::poly as p;

/// `erf(x)` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { main(x) } else { fast(x) };
    if !D::CHECKED {
        return y;
    }
    // Above 0x1.f5a888p+1 the answer is the value just below 1, signed —
    // which is two vector operations, so it is computed here rather than
    // deferred to the scalar reference. That matters more than it looks: on a
    // corpus spanning [-6, 6] roughly two lanes in five are out of range, so
    // sending them out of the vector would cost most of the speedup.
    let os = V::splat(1.0).copysign(x);
    let saturated = os - V::splat(f32::from_bits(0x3300_0000)) * os; // 0x1p-25
    let out = x.abs().gt_mask(V::splat(f32::from_bits(r::ERF_MAX_BITS)));
    let y = V::select(out, saturated, y);
    // Only the infinities and NaN are left, and they are genuinely rare.
    patch_lanes(x, y, crate::kernels::outside(x, f32::INFINITY), r::erf)
}

/// `exp(-x^2)` for `erf`/`erfc`'s shared far branch, compensated.
///
/// A single `f32` rounding of `x^2 * log2(e)` is not accurate enough here:
/// `x` reaches `erfc`'s 10.05 domain bound, so `x^2` reaches ~101, and
/// rounding either `x^2` itself or the constant `log2(e)` to `f32` before
/// multiplying costs several ulp of the *exponentiated* result once that
/// error is scaled by `ln(2)` and carried through `2^t`'s derivative — the
/// gap this closed measured 119 ulp (`tests/ulp_scan.rs::scan_erf_single_precision`)
/// before it existed. Fixed the way [`crate::kernels::double::exp2`]'s own
/// `Fast` path fixed its analogous rounding: carry `x^2` and the product by
/// `LOG2E_HI`/`LOG2E_LO` through their FMA residuals (`a_mul`'s identity,
/// inlined rather than imported since only one product needs it here), call
/// the native `exp2` on the primary term, then correct its result by
/// `1 + argl * ln(2)` — valid because the residual `argl` left after
/// compensation is small enough for that linear approximation to be exact to
/// `f32`'s own rounding.
#[inline(always)]
pub(crate) fn exp_neg_x2<V: Simd<Elem = f32>>(ax: V) -> V {
    let x2 = ax * ax;
    let x2_lo = ax.mul_add(ax, -x2);
    let neg_hi = V::splat(-p::LOG2E_HI);
    let neg_lo = V::splat(-p::LOG2E_LO);
    let argh = x2 * neg_hi;
    let argh_res = x2.mul_add(neg_hi, -argh);
    let argl = (argh_res + x2 * neg_lo) + x2_lo * neg_hi;

    let e0 = super::exp2::eval::<V, Fast, Finite>(argh);
    e0.mul_add(argl * V::splat(core::f32::consts::LN_2), e0)
}

/// `erfc(x) / exp(-x^2)` for `erf`/`erfc`'s shared far branch, `x >=
/// ERF_SPLIT`. Two regions, not one — see `ERFC_FAR_LO`/`ERFC_FAR_HI`'s table
/// doc for why a single polynomial across the whole domain fit well
/// mathematically but did not evaluate well in `f32`.
#[inline(always)]
pub(crate) fn far_q<V: Simd<Elem = f32>>(ax: V) -> V {
    let is_lo = ax.lt_mask(V::splat(p::ERFC_MID));
    let zlo = (ax - V::splat(p::ERFC_A0_LO)) / (ax + V::splat(p::ERFC_B0_LO));
    let zhi = (ax - V::splat(p::ERFC_A0_HI)) / (ax + V::splat(p::ERFC_B0_HI));
    V::select(
        is_lo,
        horner(zlo, &p::ERFC_FAR_LO),
        horner(zhi, &p::ERFC_FAR_HI),
    )
}

/// The native path.
///
/// Both branches are evaluated and blended (`V::select`, not a whole-vector
/// branch): unlike `main`'s table gather, neither costs enough here to be
/// worth guarding, and `Fast` has no gather to avoid in the first place.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    let ax = x.abs();
    let x2 = ax * ax;
    let is_small = ax.lt_mask(V::splat(p::ERF_SPLIT));

    let near = ax * horner(x2, &p::ERF_NEAR);

    let e = exp_neg_x2::<V>(ax);
    let far = V::splat(1.0) - e * far_q::<V>(ax);

    V::select(is_small, near, far).copysign(x)
}

/// Whichever branch the lanes need, widened and narrowed once.
///
/// Guarded rather than blended. The table branch gathers eight coefficients
/// per lane, which is the expensive part of this kernel, and a vector of small
/// arguments has no business paying for it — nor a vector of large ones for
/// the series. A whole-vector test is a branch on data, which is right at this
/// granularity and wrong inside the arithmetic.
#[inline(always)]
fn main<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    let s = x.widen();
    let z = s.abs();

    // `|x| < 0.4375`, below which `floor(16|x|)` drops under the table's
    // first interval.
    let is_small = z.lt_mask(W::<V>::splat(f32::from_bits(r::ERF_SMALL_BITS) as f64));
    if is_small.all() {
        return V::narrow(small_series::<V>(s));
    }
    let tab = table::<V>(z, s);
    if is_small.none() {
        return V::narrow(tab);
    }
    V::narrow(W::<V>::select(is_small, small_series::<V>(s), tab))
}

/// The `|x| < 0.4375` odd series, by Estrin.
#[inline(always)]
fn small_series<V: Simd<Elem = f32>>(s: <V as Simd>::Wide) -> <V as Simd>::Wide {
    type W<V> = <V as Simd>::Wide;
    let c = &t::C_SMALL;
    let z2 = s * s;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let c0 = W::<V>::splat(c[0]) + z2 * W::<V>::splat(c[1]);
    let c2 = W::<V>::splat(c[2]) + z2 * W::<V>::splat(c[3]);
    let c4 = W::<V>::splat(c[4]) + z2 * W::<V>::splat(c[5]);
    let c6 = W::<V>::splat(c[6]) + z2 * W::<V>::splat(c[7]);
    let c0 = c0 + z4 * c2;
    let c4 = c4 + z4 * c6;
    s * (c0 + z8 * c4)
}

/// The table branch: one degree-7 polynomial per sixteenth of `|x|`.
#[inline(always)]
fn table<V: Simd<Elem = f32>>(z: <V as Simd>::Wide, s: <V as Simd>::Wide) -> <V as Simd>::Wide {
    use crate::simd::Lanes;
    type W<V> = <V as Simd>::Wide;

    let scaled = z * W::<V>::splat(16.0);
    let v = scaled.floor();

    // The gather: eight coefficients per lane. `i` is clamped because the
    // small-`|x|` lanes index below the table and the out-of-range lanes
    // index above it, and both are discarded after the blend.
    let idx = scaled.to_array();
    let mut cols = [<W<V> as Simd>::Floats::filled_default(); 8];
    for lane in 0..<W<V> as Simd>::LANES {
        let i = (idx.as_slice()[lane] as usize).clamp(7, t::C.len() + 6);
        let row = &t::C[i - 7];
        for (k, col) in cols.iter_mut().enumerate() {
            col.as_mut_slice()[lane] = row[k];
        }
    }
    let c = cols.map(<W<V> as Simd>::from_array);

    let z = (z - W::<V>::splat(0.03125)) - W::<V>::splat(0.0625) * v;
    let z2 = z * z;
    let z4 = z2 * z2;
    let c0 = c[0] + z * c[1];
    let c2 = c[2] + z * c[3];
    let c4 = c[4] + z * c[5];
    let c6 = c[6] + z * c[7];
    let c0 = c0 + z2 * c2;
    let c4 = c4 + z2 * c6;
    (c0 + z4 * c4).copysign(s)
}
