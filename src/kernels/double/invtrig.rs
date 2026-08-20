//! `asin`, `acos`, `atan` and `atan2`.
//!
//! `atan` and `atan2` have a genuine vector `BitExact` schedule now (this
//! module's `bit_exact` submodule — the `sin`/`cos`/`sincos` treatment
//! `src/kernels/double/trig.rs` got, applied here). `asin`/`acos` do not yet:
//! they still run one lane at a time through
//! [`crate::reference::double::invtrig`], which is a genuine port of the IBM
//! Accurate Portable Math Library routines glibc runs here (schedule read
//! from a disassembly, verified byte-exact against the platform), not a call
//! to the platform — so `BitExact` is bit-exact for all four, but only
//! `atan`/`atan2` are faster than the per-lane call they used to make; see
//! [`crate::kernels`] and `ROADMAP.md`'s A4 entry. Under `Fast` all four
//! share two polynomials and a handful of exact identities, and are entirely
//! branch-free.

use crate::kernels::{dispatch, dispatch2, horner, no_lanes, not_normal, outside};
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Mask, Simd, patch_lanes, patch_lanes2};
use crate::tables::double::poly as p;

/// `pi/2`, to the nearest double.
const PI_2: f64 = core::f64::consts::FRAC_PI_2;
/// `pi`, to the nearest double.
const PI: f64 = core::f64::consts::PI;
/// `pi/6`, the fold point of the `atan` reduction.
const PI_6: f64 = core::f64::consts::FRAC_PI_6;
/// `sqrt(3)`, the same reduction's multiplier.
const SQRT3: f64 = f64::from_bits(0x3ffbb67ae8584caa);

/// `asin(|t|)` for `t*t = s` with `|t| <= 1/2`, as `t * P(s)`.
#[inline(always)]
fn asin_core<V: Simd<Elem = f64>>(t: V, s: V) -> V {
    t * horner(s, &p::ASIN)
}

/// `atan` and `atan2`'s genuine vector replay of `__atan_fma`/
/// `__ieee754_atan2_fma`, mirroring [`crate::reference::double::invtrig`]
/// exactly — same bands, same fusions (see that module's doc for the
/// disassembly this was cross-checked against). `asin`/`acos` are not here
/// yet: their eight bands share a 2568-entry table and were judged a
/// separate follow-on rather than rushed alongside these two — see
/// `ROADMAP.md`'s A4 entry.
///
/// Every band below is evaluated for every lane, then blended with
/// `V::select`, the same whole-vector discipline `trig.rs` uses: `Simd` has
/// no scalar control flow to branch on. Table gathers (`cij_row`) go
/// through a per-lane loop exactly like `trig.rs`'s `TAB` lookup and
/// `ln.rs`'s exponent-field extraction, for the same reason — there is no
/// packed integer shift or index anywhere in this crate's `Simd` surface.
/// Lanes the reference reaches via an *early return* rather than the
/// regular arithmetic path (NaN; zero, subnormal or infinite arguments;
/// `atan2`'s extreme-exponent-difference short circuit; its extreme-
/// magnitude rescale bands) are not replayed here at all — they are
/// detected and handed to the scalar reference by `patch_lanes`/
/// `patch_lanes2`, per this crate's own rule that rare-lane repairs stay in
/// `reference/`, not in a vector main path.
mod bit_exact {
    use super::*;
    use crate::kernels::double::dd::{a_mul, two_sum};
    use crate::simd::Lanes;
    use crate::tables::double::atan2_data as at2;
    use crate::tables::double::atan_data as at;

    /// `2^52`, the round-to-nearest-integer trick constant `table_index`
    /// shares with the scalar reference.
    const TWO52: f64 = 4503599627370496.0;

    /// `((2^52 + 256w) - 2^52) - 16`, as a float — still an exact small
    /// integer for any `w` a real table band produces; `cij_row` clamps it
    /// for lanes where it is not (see that function's doc).
    #[inline(always)]
    fn table_index<V: Simd<Elem = f64>>(w: V) -> V {
        (V::splat(TWO52) + V::splat(256.0) * w) - V::splat(TWO52) - V::splat(16.0)
    }

    /// `at::CIJ` row `idx`, gathered per lane and clamped to `[0, 240]`.
    ///
    /// Every band that calls this runs unconditionally on every lane (the
    /// whole-vector-blend discipline), so a lane whose real band is
    /// something else entirely — say `u` in the billions, headed for the
    /// saturate band — still reaches this gather with whatever `table_index`
    /// produced for it, which for `u` that large is nowhere near `[0, 240]`.
    /// The clamp is defensive only, the same role `trig.rs`'s `% 110` plays:
    /// the value gathered for a lane like that is discarded by the blend
    /// that follows, never returned.
    #[inline(always)]
    fn cij_row<V: Simd<Elem = f64>>(idx: V) -> (V, V, V, V, V, V, V) {
        let idxa = idx.to_array();
        let mut x0 = V::Floats::filled_default();
        let mut t1 = V::Floats::filled_default();
        let mut c2 = V::Floats::filled_default();
        let mut c3 = V::Floats::filled_default();
        let mut c4 = V::Floats::filled_default();
        let mut c5 = V::Floats::filled_default();
        let mut c6 = V::Floats::filled_default();
        for i in 0..V::LANES {
            let raw = idxa.as_slice()[i] as i64;
            let row = (raw.clamp(0, 240) as usize) * 7;
            x0.as_mut_slice()[i] = f64::from_bits(at::CIJ[row]);
            t1.as_mut_slice()[i] = f64::from_bits(at::CIJ[row + 1]);
            c2.as_mut_slice()[i] = f64::from_bits(at::CIJ[row + 2]);
            c3.as_mut_slice()[i] = f64::from_bits(at::CIJ[row + 3]);
            c4.as_mut_slice()[i] = f64::from_bits(at::CIJ[row + 4]);
            c5.as_mut_slice()[i] = f64::from_bits(at::CIJ[row + 5]);
            c6.as_mut_slice()[i] = f64::from_bits(at::CIJ[row + 6]);
        }
        (
            V::from_array(x0),
            V::from_array(t1),
            V::from_array(c2),
            V::from_array(c3),
            V::from_array(c4),
            V::from_array(c5),
            V::from_array(c6),
        )
    }

    /// The `d3..d13` Horner chain both `atan` and `atan2`'s Taylor bands
    /// share.
    #[inline(always)]
    fn atan_taylor_poly<V: Simd<Elem = f64>>(v: V) -> V {
        let yy = v.mul_add(V::splat(at::D13), V::splat(at::D11));
        let yy = v.mul_add(yy, V::splat(at::D9));
        let yy = v.mul_add(yy, V::splat(at::D7));
        let yy = v.mul_add(yy, V::splat(at::D5));
        v.mul_add(yy, V::splat(at::D3))
    }

    /// `A <= u < B`: direct Taylor series.
    #[inline(always)]
    fn atan_taylor<V: Simd<Elem = f64>>(x: V, v: V) -> V {
        let poly = atan_taylor_poly(v);
        (x * v).mul_add(poly, x)
    }

    /// `B <= u < C`: direct table band.
    #[inline(always)]
    fn atan_table<V: Simd<Elem = f64>>(u: V) -> V {
        let (x0, t1, c2, c3, c4, c5, c6) = cij_row(table_index(u));
        let z = u - x0;
        let yy = z.mul_add(c6, c5);
        let yy = z.mul_add(yy, c4);
        let yy = z.mul_add(yy, c3);
        let yy = z.mul_add(yy, c2);
        z.mul_add(yy, t1)
    }

    /// `C <= u < D`: reciprocal fold plus table.
    #[inline(always)]
    fn atan_recip_table<V: Simd<Elem = f64>>(u: V) -> V {
        let w = V::splat(1.0) / u;
        let (t1p, t2p) = a_mul(w, u);
        let s = (V::splat(1.0) - t1p) - t2p;
        let (x0, t1, c2, c3, c4, c5, c6) = cij_row(table_index(w));
        let z = s.mul_add(w, w - x0);
        let yy = z.mul_add(c6, c5);
        let yy = z.mul_add(yy, c4);
        let yy = z.mul_add(yy, c3);
        let yy = z.mul_add(yy, c2);
        let yy = (-z).mul_add(yy, V::splat(at::HPI1));
        let t1f = V::splat(at::HPI) - t1;
        t1f + yy
    }

    /// `D <= u < E`: reciprocal fold plus Taylor series.
    #[inline(always)]
    fn atan_recip_taylor<V: Simd<Elem = f64>>(u: V) -> V {
        let w = V::splat(1.0) / u;
        let v = w * w;
        let (t1p, t2p) = a_mul(w, u);
        let poly = atan_taylor_poly(v);
        let t3 = V::splat(at::HPI) - w;
        let cor = (V::splat(at::HPI) - t3) - w;
        let s = (V::splat(1.0) - t1p) - t2p;
        let hpi1_cor = V::splat(at::HPI1) + cor;
        let inner = (-s).mul_add(w, hpi1_cor);
        let wv = w * v;
        let combined = (-wv).mul_add(poly, inner);
        t3 + combined
    }

    /// `atan(x)`, vectorised.
    #[inline(always)]
    pub(super) fn atan<V: Simd<Elem = f64>>(x: V) -> V {
        let u = x.abs();
        let taylor = atan_taylor(x, x * x);
        let table = atan_table(u).copysign(x);
        let recip_table = atan_recip_table(u).copysign(x);
        let recip_taylor = atan_recip_taylor(u).copysign(x);
        let sat = V::select(x.gt_mask(V::splat(0.0)), V::splat(at::HPI), V::splat(at::MHPI));

        let mut r = V::select(u.ge_mask(V::splat(at::E)), sat, recip_taylor);
        r = V::select(u.lt_mask(V::splat(at::D)), recip_table, r);
        r = V::select(u.lt_mask(V::splat(at::C)), table, r);
        r = V::select(u.lt_mask(V::splat(at::B)), taylor, r);
        V::select(u.lt_mask(V::splat(at::A)), x, r)
    }

    /// The one lane `atan`'s vector path does not already match the
    /// reference on: NaN. Every other input is real vector arithmetic
    /// replaying the reference's own schedule, so it matches by
    /// construction; NaN alone needs the reference's specific
    /// payload-preserving `x + x`, which nothing in the band arithmetic
    /// above reproduces.
    #[inline(always)]
    pub(super) fn atan_needs_repair<V: Simd<Elem = f64>>(x: V) -> V::Mask {
        x.is_nan()
    }

    // -----------------------------------------------------------------
    // atan2
    // -----------------------------------------------------------------

    /// Case (i): `x > 0`, `ay < ax`, `u < 1/16`.
    #[inline(always)]
    fn atan2_i_taylor<V: Simd<Elem = f64>>(u: V, du: V) -> V {
        let v = u * u;
        let poly = atan_taylor_poly(v);
        let zz = (u * v).mul_add(poly, du);
        u + zz
    }

    /// Case (i): `x > 0`, `ay < ax`, `u >= 1/16`.
    #[inline(always)]
    fn atan2_i_table<V: Simd<Elem = f64>>(u: V, du: V) -> V {
        let (x0, t1, t2, c3, c4, c5, c6) = cij_row(table_index(u));
        let t3 = u - x0;
        let (v, dv) = two_sum(t3, du);
        let poly = v.mul_add(c6, c5);
        let poly = v.mul_add(poly, c4);
        let poly = v.mul_add(poly, c3);
        let inner = (v * v).mul_add(poly, dv * t2);
        let zz = v.mul_add(t2, inner);
        t1 + zz
    }

    /// Cases (ii)/(iii)/(iv)'s table bands, `atan`'s own `atan_recip_table`
    /// shape with a caller-supplied base and sign (`add`, resolved at
    /// compile time — each call site is one fixed case, not a per-lane
    /// choice).
    #[inline(always)]
    fn atan2_table_shared<V: Simd<Elem = f64>>(u: V, du: V, base: V, base1: V, add: bool) -> V {
        let (x0, t1c, c2, c3, c4, c5, c6) = cij_row(table_index(u));
        let v = (u - x0) + du;
        let poly = v.mul_add(c6, c5);
        let poly = v.mul_add(poly, c4);
        let poly = v.mul_add(poly, c3);
        let poly = v.mul_add(poly, c2);
        let zz = if add { v.mul_add(poly, base1) } else { (-v).mul_add(poly, base1) };
        let t1 = if add { base + t1c } else { base - t1c };
        t1 + zz
    }

    /// Case (ii): `x > 0`, `ay >= ax`, `u < 1/16`.
    #[inline(always)]
    fn atan2_ii_taylor<V: Simd<Elem = f64>>(u: V, du: V) -> V {
        let v = u * u;
        let zz = (u * v) * atan_taylor_poly(v);
        let t2 = V::splat(at::HPI) - u;
        let cor = (V::splat(at::HPI) - t2) - u;
        let t3 = ((V::splat(at::HPI1) + cor) - du) - zz;
        t2 + t3
    }

    /// Case (iii): `x < 0`, `ax < ay`, `u < 1/16`.
    #[inline(always)]
    fn atan2_iii_taylor<V: Simd<Elem = f64>>(u: V, du: V) -> V {
        let v = u * u;
        let zz = (u * v) * atan_taylor_poly(v);
        let t2 = V::splat(at::HPI) + u;
        let cor = (V::splat(at::HPI) - t2) + u;
        let t3 = ((V::splat(at::HPI1) + cor) + du) + zz;
        t2 + t3
    }

    /// Case (iv): `x < 0`, `ax >= ay`, `u < 1/16`.
    #[inline(always)]
    fn atan2_iv_taylor<V: Simd<Elem = f64>>(u: V, du: V) -> V {
        let v = u * u;
        let zz = (u * v) * atan_taylor_poly(v);
        let t2 = V::splat(at2::OPI) - u;
        let cor = (V::splat(at2::OPI) - t2) - u;
        let t3 = ((V::splat(at2::OPI1) + cor) - du) - zz;
        t2 + t3
    }

    /// `atan2(y, x)`, vectorised, for the "regular" case only — the same
    /// case the scalar reference reaches after its own early returns. Every
    /// lane that took one of those early returns is caught by
    /// [`atan2_needs_repair`] before this is ever asked for an answer.
    #[inline(always)]
    pub(super) fn atan2<V: Simd<Elem = f64>>(y: V, x: V) -> V {
        let ax = x.abs();
        let ay = y.abs();
        let ay_lt_ax = ay.lt_mask(ax);

        let u_a = ay / ax;
        let (t1p_a, t2p_a) = a_mul(ax, u_a);
        let du_a = ((ay - t1p_a) - t2p_a) / ax;

        let u_b = ax / ay;
        let (t1p_b, t2p_b) = a_mul(ay, u_b);
        let du_b = ((ax - t1p_b) - t2p_b) / ay;

        let u = V::select(ay_lt_ax, u_a, u_b);
        let du = V::select(ay_lt_ax, du_a, du_b);
        let small = u.lt_mask(V::splat(at::B));

        let case_i = V::select(small, atan2_i_taylor(u, du), atan2_i_table(u, du));
        let case_ii = V::select(
            small,
            atan2_ii_taylor(u, du),
            atan2_table_shared(u, du, V::splat(at::HPI), V::splat(at::HPI1), false),
        );
        let case_iii = V::select(
            small,
            atan2_iii_taylor(u, du),
            atan2_table_shared(u, du, V::splat(at::HPI), V::splat(at::HPI1), true),
        );
        let case_iv = V::select(
            small,
            atan2_iv_taylor(u, du),
            atan2_table_shared(u, du, V::splat(at2::OPI), V::splat(at2::OPI1), false),
        );

        let pos_result = V::select(ay_lt_ax, case_i, case_ii);
        let neg_result = V::select(ax.lt_mask(ay), case_iii, case_iv);
        let z = V::select(x.gt_mask(V::splat(0.0)), pos_result, neg_result);
        z.copysign(y)
    }

    /// Lanes `atan2`'s vector path does not attempt: not-normal in either
    /// argument (zero, subnormal, infinite or NaN — every early
    /// special-case branch the scalar reference has), the
    /// extreme-exponent-difference short circuit (`de >= EP` or `<= -EP`
    /// in `e_atan2.c`), and the extreme-magnitude rescale bands. All three
    /// are read from the raw bits per lane, not approximated with a scaled
    /// float comparison: at the very magnitudes these exist to guard, a
    /// float comparison like `ay >= ax * 2^57` can itself overflow and
    /// silently under-count, which would be worse than the cost this
    /// avoids.
    #[inline(always)]
    pub(super) fn atan2_needs_repair<V: Simd<Elem = f64>>(y: V, x: V) -> V::Mask {
        let base = not_normal(y).or(not_normal(x));
        let yb = y.to_bits();
        let xb = x.to_bits();
        let mut extra = V::Floats::filled_default();
        for i in 0..V::LANES {
            let uy = ((yb.as_slice()[i] >> 32) as u32) & 0x7ff0_0000;
            let ux = ((xb.as_slice()[i] >> 32) as u32) & 0x7ff0_0000;
            let de = uy as i64 - ux as i64;
            let ax = f64::from_bits(xb.as_slice()[i] & 0x7fff_ffff_ffff_ffff);
            let ay = f64::from_bits(yb.as_slice()[i] & 0x7fff_ffff_ffff_ffff);
            let needs = de >= 59_768_832
                || de <= -59_768_832
                || ax < at2::TWOM500
                || ay < at2::TWOM500
                || ax > at2::TWO500
                || ay > at2::TWO500;
            extra.as_mut_slice()[i] = if needs { 1.0 } else { 0.0 };
        }
        base.or(V::from_array(extra).gt_mask(V::splat(0.5)))
    }
}

/// Arc sine.
pub mod asin {
    use super::*;

    /// `asin(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        // `outside(x, 1.0)` also catches |x| == 1 and sends it to the
        // reference. That is one extra scalar lane on an input that is rare in
        // bulk data, and it buys not having to reason about `sqrt(0)` in the
        // folded branch.
        dispatch::<V, A, D>(x, reference::asin, fast, |x| outside(x, 1.0))
    }

    /// Measured error: at most 3 ulp over `|x| < 1`, stably at the fold
    /// boundary below rather than growing with the sample count (checked to
    /// 100M samples in `tests/ulp_scan.rs`); asserted at 4.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let half = V::splat(0.5);
        let big = a.gt_mask(half);

        // Near 1, `asin` has infinite slope, so it is folded to a region where
        // it does not: asin(a) = pi/2 - 2 asin(sqrt((1-a)/2)).
        let t = (V::splat(1.0) - a) * half;
        let root = t.sqrt();
        let folded = V::splat(PI_2) - (asin_core(root, t) + asin_core(root, t));
        let direct = asin_core(a, a * a);
        V::select(big, folded, direct).copysign(x)
    }
}

/// Arc cosine.
pub mod acos {
    use super::*;

    /// `acos(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        dispatch::<V, A, D>(x, reference::acos, fast, |x| outside(x, 1.0))
    }

    /// Measured error: at most 2 ulp over `|x| < 1` (checked to 100M samples
    /// in `tests/ulp_scan.rs`); asserted at 4.
    ///
    /// Not written as `pi/2 - asin(x)`: near `x = 1`, `acos` is small and that
    /// subtraction cancels, losing exactly the digits the caller wanted. The
    /// folded form computes the small answer directly.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let half = V::splat(0.5);
        let big = a.gt_mask(half);

        let t = (V::splat(1.0) - a) * half;
        let root = t.sqrt();
        let two_asin = asin_core(root, t) + asin_core(root, t);
        // acos(a) = 2 asin(sqrt((1-a)/2)); acos(-a) = pi - that.
        let folded = V::select(x.lt_mask(V::splat(0.0)), V::splat(PI) - two_asin, two_asin);
        let direct = V::splat(PI_2) - asin_core(x, x * x);
        V::select(big, folded, direct)
    }
}

/// `atan(|t|)` for `|t| <= 1`, folded once more onto `|w| <= tan(pi/12)`.
#[inline(always)]
fn atan_unit<V: Simd<Elem = f64>>(z: V) -> V {
    let sqrt3 = V::splat(SQRT3);
    let hi = z.gt_mask(V::splat(f64::from_bits(0x3fd126145e9ecd56))); // tan(pi/12)
    // atan(z) = pi/6 + atan((z sqrt3 - 1)/(sqrt3 + z)), which maps
    // [tan(pi/12), 1] onto [-tan(pi/12), tan(pi/12)].
    let w = z.mul_add(sqrt3, V::splat(-1.0)) / (sqrt3 + z);
    let u = V::select(hi, w, z);
    let r = u * horner(u * u, &p::ATAN);
    r + V::select(hi, V::splat(PI_6), V::splat(0.0))
}

/// Arc tangent.
pub mod atan {
    use super::*;

    /// `atan(x)` for a vector of lanes.
    ///
    /// Under `Fast`, needs no special-lane repair: the reciprocal fold turns
    /// an infinite argument into `1/inf == 0`, so `atan(±inf)` falls out as
    /// `±pi/2` without a test, and NaN propagates through every operation
    /// (that only needs to land within the measured ulp bound, not match a
    /// specific bit pattern). Under `BitExact` the vector path is a genuine
    /// schedule replay (`bit_exact::atan`) rather than a per-lane scalar
    /// call, but it does need a NaN repair — see that function's doc.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact::atan(x);
            return if D::CHECKED {
                patch_lanes(x, y, bit_exact::atan_needs_repair(x), reference::atan)
            } else {
                y
            };
        }
        dispatch::<V, A, D>(x, reference::atan, fast, no_lanes)
    }

    /// Measured error: below 2 ulp over the whole real line.
    #[inline(always)]
    pub(super) fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let a = x.abs();
        let inv = a.gt_mask(V::splat(1.0));
        let z = V::select(inv, V::splat(1.0) / a, a);
        let r = atan_unit(z);
        V::select(inv, V::splat(PI_2) - r, r).copysign(x)
    }
}

/// Two-argument arc tangent: the angle of the point `(x, y)`.
pub mod atan2 {
    use super::*;

    /// `atan2(y, x)` for vectors of lanes.
    ///
    /// Argument order follows C and Rust: `y` first. Under `BitExact` the
    /// vector path is a genuine schedule replay (`bit_exact::atan2`); see
    /// `bit_exact::atan2_needs_repair`'s doc for exactly which lanes it does
    /// not attempt and hands to the reference instead.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(y: V, x: V) -> V {
        if A::BIT_EXACT {
            let z = bit_exact::atan2(y, x);
            return if D::CHECKED {
                patch_lanes2(y, x, z, bit_exact::atan2_needs_repair(y, x), reference::atan2)
            } else {
                z
            };
        }
        dispatch2::<V, A, D>(y, x, reference::atan2, fast, |y, x| {
            // The quotient is what the vector path is built on, so anything
            // that makes it meaningless — a zero or non-finite in either
            // argument — goes to the reference. That is also where all the
            // signed-zero rules live, which are not worth restating in vector
            // form to save a handful of lanes.
            not_normal(y).or(not_normal(x))
        })
    }

    /// Measured error: below 2 ulp for normal, non-zero arguments.
    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(y: V, x: V) -> V {
        let base = atan::fast(y / x);
        let zero = V::splat(0.0);
        // x < 0 puts the point in the second or third quadrant, which the
        // quotient cannot distinguish; shift by pi with the sign of y.
        let shift = V::splat(PI).copysign(y);
        V::select(x.lt_mask(zero), base + shift, base)
    }
}
