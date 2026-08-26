//! `asin`, `acos`, `atan` and `atan2`.
//!
//! All four have a genuine vector `BitExact` schedule now — this module's
//! `bit_exact` submodule, the `sin`/`cos`/`sincos` treatment
//! `src/kernels/double/trig.rs` got, applied to the whole invtrig family.
//! Each replays the IBM Accurate Portable Math Library routine glibc runs
//! here (schedule read from a disassembly, verified byte-exact against the
//! platform), not a call to the platform — so `BitExact` is bit-exact for all
//! four, and all four are faster than the per-lane call they used to make;
//! see [`crate::kernels`] and `ROADMAP.md`'s A4 entry. Under `Fast` all four
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

/// The whole family's genuine vector replay of `__atan_fma`/
/// `__ieee754_atan2_fma`/`__ieee754_asin_fma`/`__ieee754_acos_fma`,
/// mirroring [`crate::reference::double::invtrig`] exactly — same bands, same
/// fusions (see that module's doc for the disassembly this was cross-checked
/// against). `asin`/`acos` landed as the follow-on `ROADMAP.md`'s A4 entry
/// planned: their eight bands share the `asncs.x` table, repacked at one
/// stride by [`crate::tables::double::asincos_packed`] and read one whole row
/// per lane by [`Simd::gather_run`].
///
/// Every band below is evaluated for every lane, then blended with
/// `V::select`, the same whole-vector discipline `trig.rs` uses: `Simd` has
/// no scalar control flow to branch on — except where a band is expensive
/// *and* rare, in which case it is skipped by an `any()` test on the mask
/// that would have selected it (`asin`/`acos`'s near-one band). `atan`'s
/// table gather (`cij_row`) still goes through a per-lane loop exactly like
/// `trig.rs`'s `TAB` lookup, because it reads a single value per lane rather
/// than a run; there is no packed integer shift or index anywhere in this
/// crate's `Simd` surface.
///
/// Lanes the reference reaches via an *early return* rather than the regular
/// arithmetic path are handled two different ways here. `atan`/`atan2` hand
/// theirs (NaN; zero, subnormal or infinite arguments; `atan2`'s
/// extreme-exponent-difference short circuit and its extreme-magnitude
/// rescale bands) to the scalar reference through `patch_lanes`/
/// `patch_lanes2`, per this crate's rule that rare-lane repairs stay in
/// `reference/` rather than in a vector main path. `asin`/`acos` have none
/// left to hand over: their only such case is a NaN argument, whose
/// payload-preserving `x + x` is what a hardware add already does to a NaN
/// lane, so it is two instructions in the main path instead of a repair.
mod bit_exact {
    use super::*;
    use crate::kernels::double::dd::{a_mul, two_sum};
    use crate::simd::Lanes;
    use crate::tables::double::asincos_data as ac;
    use crate::tables::double::asincos_packed as pk;
    use crate::tables::double::atan_data as at;
    use crate::tables::double::atan2_data as at2;

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
        let idx0 = idxa.as_slice()[0] as i64;
        let mut all_same = true;
        for i in 1..V::LANES {
            if idxa.as_slice()[i] as i64 != idx0 {
                all_same = false;
                break;
            }
        }
        if all_same {
            let row = (idx0.clamp(0, 240) as usize) * 7;
            return (
                V::splat(f64::from_bits(at::CIJ[row])),
                V::splat(f64::from_bits(at::CIJ[row + 1])),
                V::splat(f64::from_bits(at::CIJ[row + 2])),
                V::splat(f64::from_bits(at::CIJ[row + 3])),
                V::splat(f64::from_bits(at::CIJ[row + 4])),
                V::splat(f64::from_bits(at::CIJ[row + 5])),
                V::splat(f64::from_bits(at::CIJ[row + 6])),
            );
        }

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

        let is_taylor = u.lt_mask(V::splat(at::B));
        if is_taylor.all() {
            let taylor = atan_taylor(x, x * x);
            return V::select(u.lt_mask(V::splat(at::A)), x, taylor);
        }
        let is_table = u.ge_mask(V::splat(at::B)).and(u.lt_mask(V::splat(at::C)));
        if is_table.all() {
            return atan_table(u).copysign(x);
        }
        let is_recip_table = u.ge_mask(V::splat(at::C)).and(u.lt_mask(V::splat(at::D)));
        if is_recip_table.all() {
            return atan_recip_table(u).copysign(x);
        }
        let is_recip_taylor = u.ge_mask(V::splat(at::D)).and(u.lt_mask(V::splat(at::E)));
        if is_recip_taylor.all() {
            return atan_recip_taylor(u).copysign(x);
        }
        let is_sat = u.ge_mask(V::splat(at::E));
        if is_sat.all() {
            return V::select(
                x.gt_mask(V::splat(0.0)),
                V::splat(at::HPI),
                V::splat(at::MHPI),
            );
        }

        let taylor = atan_taylor(x, x * x);
        let table = atan_table(u).copysign(x);
        let recip_table = atan_recip_table(u).copysign(x);
        let recip_taylor = atan_recip_taylor(u).copysign(x);
        let sat = V::select(
            x.gt_mask(V::splat(0.0)),
            V::splat(at::HPI),
            V::splat(at::MHPI),
        );

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
        let zz = if add {
            v.mul_add(poly, base1)
        } else {
            (-v).mul_add(poly, base1)
        };
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

    // -----------------------------------------------------------------
    // asin / acos
    // -----------------------------------------------------------------
    //
    // Both replay `__ieee754_asin_fma`/`__ieee754_acos_fma`'s shared schedule:
    // a direct-Taylor band below `0.125`, the `asncs.x` table bands from
    // `0.125` up to `0.96875` (six index formulas, five polynomial degrees),
    // and the near-1 band with its shared reciprocal-square-root refinement.
    // The two functions differ only in the linear combination applied to the
    // table polynomial and in the `t24`/`t27` split of the near-1 band's `c`
    // (see `crate::reference::double::invtrig`'s module doc for the
    // disassembly this was cross-checked against, and `ROADMAP.md`'s A4 entry
    // for the `t24`-vs-`t27` bug the fix records).

    /// The band thresholds, as the exact `f64` values the scalar reference's
    /// `k = high32(|x|)` comparisons are testing for.
    ///
    /// `k` is the top 32 bits of a non-negative double, which is a monotone
    /// function of the value, so `k < K` and `|x| < f64::from_bits(K << 32)`
    /// select the same lanes — and the float form costs one compare against
    /// a constant instead of a per-lane shift out of the bit pattern. The
    /// bounds are all exactly representable (they *are* doubles with a zero
    /// low word), so nothing is lost at the boundary.
    const TINY: f64 = f64::from_bits(0x3e50_0000_0000_0000); // 2^-26, asin's A band
    const ACOS_TINY: f64 = f64::from_bits(0x3c88_0000_0000_0000); // 1.5*2^-55, acos's A band
    const TAYLOR_HI: f64 = 0.125; // 0x3fc00000
    const TABLE_HI: f64 = 0.96875; // 0x3fef0000

    /// `2^52 - 32 * STRIDE`, the constant that turns `floor(256*|x|)` into a
    /// packed *slot offset* inside a double's mantissa — see [`asncs_key`].
    const KEY_BIAS: f64 = 4503599627370496.0 - 32.0 * pk::STRIDE as f64;

    /// The packed table's row index and band centre, from one `floor`.
    ///
    /// `e_asin.c` picks a row with six different band formulas over
    /// `k = high32(|x|)`; every one of them is `floor(256*|x|) - 32` in
    /// disguise (see [`crate::tables::double::asincos_packed`], and
    /// `tests/bit_exact.rs` for the proof over every `k` in the band). So the
    /// whole per-lane index computation — the shift out of the bit pattern,
    /// the four range tests, the six mask/shift/multiply/add pairs —
    /// collapses to a multiply, a `floor` and two FMAs, on the whole vector at
    /// once.
    ///
    /// Returns `(xx, key)`:
    ///
    /// * `xx = |x| - x0` is the reduced argument, where the band centre
    ///   `x0 = (floor(256*|x|) + 0.5) / 256` is upstream's slot 0 exactly
    ///   (checked for all 216 rows in `tests/bit_exact.rs`). It is spelled
    ///   `(f - m)/256 - 1/512` with `f = 256*|x|` so that the whole reduction
    ///   is one subtraction and one FMA, and every step of it is *exact*:
    ///   `f` is a power-of-two scaling, `f - m` is Sterbenz-exact and a
    ///   multiple of `2^-47`, and dividing by 256 and shifting by `1/512`
    ///   keep it inside 53 bits. Exact at every step is what makes this
    ///   reproduce the reference's own `|x| - row[0]` bit for bit rather than
    ///   merely closely.
    ///
    /// * `key` is `2^52 + row * STRIDE`, so that `key`'s low bits *are* the
    ///   offset of the row's first slot: the gather reads it with one mask and
    ///   never converts a float to an integer, never scales an index and never
    ///   clamps one. The scaling rides along in the `floor`'s own FMA, which
    ///   is exact — every value involved is a small integer. A lane outside
    ///   the table band lands on some arbitrary offset in `[0, 0xfff]`,
    ///   including a NaN or infinite lane whose bit pattern masks to something
    ///   harmless, and the table is sized past `0xfff` precisely so that
    ///   offset is in bounds; the blend discards its result.
    #[inline(always)]
    fn asncs_key<V: Simd<Elem = f64>>(u: V) -> (V, V) {
        let f = V::splat(256.0) * u;
        let m = f.floor();
        (
            (f - m).mul_add(V::splat(1.0 / 256.0), V::splat(-1.0 / 512.0)),
            m.mul_add(V::splat(pk::STRIDE as f64), V::splat(KEY_BIAS)),
        )
    }

    /// One packed row's twelve slots, as twelve vectors.
    ///
    /// The row is `[t1, c0..c8, outer, final]` at a fixed stride, which is
    /// what lets this be one [`Simd::gather_run`] — a contiguous run per lane,
    /// so the backend can read whole rows and transpose them in register
    /// rather than issue `LANES * 12` scalar loads (see that method's doc, and
    /// the AVX-512 override in `src/simd/wide_backend.rs`). The coefficient
    /// slots past a lane's own degree are `+0.0` in the table itself, which is
    /// what lets one degree-9 fold serve all five degrees: `fma(xx, 0, c) == c`
    /// exactly.
    #[allow(unsafe_code)]
    #[inline(always)]
    fn asncs_row<V: Simd<Elem = f64>>(key: V) -> (V, [V; 9], V, V) {
        let kb = key.to_bits();
        let mut idx = V::Bits::filled_default();
        for i in 0..V::LANES {
            idx.as_mut_slice()[i] = kb.as_slice()[i] & 0xfff;
        }
        // SAFETY: every index is masked to `0xfff`, and `pk::PACKED` is sized
        // to `0x1000 + 16` precisely so that a run of `pk::USED` slots from
        // any such index — rounded up to the eight-slot block a hardware
        // override reads whether or not every slot in it is wanted — is in
        // bounds. See `pk::SLOTS`.
        let r: [V; pk::USED] = unsafe { V::gather_run(&pk::PACKED.0, idx) };
        (
            r[0],
            [r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]],
            r[10],
            r[11],
        )
    }

    /// One `asncs.x` band polynomial: `p = xx^2*horner(xx, c) + outer` and
    /// `t = xx*t1 + p`, each one fused FMA exactly as the reference's
    /// `asncs_band` compiles, then the unfused trailing combine the caller
    /// applies. The fold runs all nine coefficient slots on every lane; the
    /// slots beyond a lane's own degree are `+0.0` in the table, so `val` is
    /// identically zero until the fold reaches that lane's real leading
    /// coefficient and the chain is bit-for-bit the reference's shorter one.
    /// `final` is returned separately because `asin`'s bands finish with the
    /// unfused `final + t` while `acos`'s apply their own `HP0`/`HP1`
    /// combination on top. Returns `(t_plus_p, final)`.
    #[inline(always)]
    fn asncs_poly<V: Simd<Elem = f64>>(xx: V, t1: V, c: [V; 9], outer: V, fin: V) -> (V, V) {
        let mut val = c[8];
        for &ci in c[..8].iter().rev() {
            val = xx.mul_add(val, ci);
        }
        let p = (xx * xx).mul_add(val, outer);
        let t_plus_p = xx.mul_add(t1, p);
        (t_plus_p, fin)
    }

    /// `1/sqrt(z)`'s seed and Newton refinement — the scalar reference's
    /// `near_one_root`, including its two-rounding `r = 1 - t*t*z`.
    #[inline(always)]
    fn near_one_root<V: Simd<Elem = f64>>(z: V) -> V {
        let bits = z.to_bits();
        let mut seeds = V::Floats::filled_default();
        for i in 0..V::LANES {
            let k = (bits.as_slice()[i] >> 32) as u32;
            let ir = ac::INROOT[((k & 0x001fffff) >> 14) as usize];
            // `511 - (k >> 21)` is always in `[0, 27]` for a real near-1 lane
            // (`z >= 2^-54`); the clamp is defensive for the lanes whose
            // result the blend discards (`z` zero, negative or NaN).
            let p = (511i32 - (k >> 21) as i32).clamp(0, 27) as usize;
            seeds.as_mut_slice()[i] = ir * ac::POWTWO[p];
        }
        let seed = V::from_array(seeds);
        let tt = seed * seed;
        let r = (-tt).mul_add(z, V::splat(1.0));
        let poly = r.mul_add(V::splat(ac::RT3), V::splat(ac::RT2));
        let poly = r.mul_add(poly, V::splat(ac::RT1));
        let poly = r.mul_add(poly, V::splat(ac::RT0));
        seed * poly
    }

    /// The near-1 band's shared front half — `z`, `c`, `inner` and `p` — the
    /// parts `asin` and `acos` compute identically before diverging in how
    /// they split `c` (the `t24`/`t27` Dekker constant) and combine the tail.
    #[inline(always)]
    fn near1_common<V: Simd<Elem = f64>>(u: V) -> (V, V, V, V) {
        let z = V::splat(0.5) * (V::splat(1.0) - u);
        let t = near_one_root(z);
        let c = t * z;
        let inner = (t * V::splat(0.5)).mul_add(-c, V::splat(1.5));
        let p = z.mul_add(V::splat(ac::F6), V::splat(ac::F5));
        let p = z.mul_add(p, V::splat(ac::F4));
        let p = z.mul_add(p, V::splat(ac::F3));
        let p = z.mul_add(p, V::splat(ac::F2));
        let p = z.mul_add(p, V::splat(ac::F1));
        (z, c, inner, p * z)
    }

    /// `asin`'s near-1 band on the magnitude; the caller applies the sign.
    #[inline(always)]
    fn asin_near1<V: Simd<Elem = f64>>(u: V) -> V {
        let (z, c, inner, p) = near1_common(u);
        let y = (c + V::splat(ac::T24)) - V::splat(ac::T24);
        let t_plus_y = inner.mul_add(c, y);
        let z_minus_y2 = y.mul_add(-y, z);
        let cc = z_minus_y2 / t_plus_y;
        let hp1_minus_2cc = cc.mul_add(V::splat(-2.0), V::splat(ac::HP1));
        let res1 = y.mul_add(V::splat(-2.0), V::splat(ac::HP0));
        let y_plus_cc_x2 = (y + cc) + (y + cc);
        let cor = y_plus_cc_x2.mul_add(-p, hp1_minus_2cc);
        res1 + cor
    }

    /// `acos`'s near-1 band, sign-aware: the positive- and negative-`x` arms
    /// differ only in the final combine, and both end in the unfused
    /// `res + res`. Uses the `t27` split in both arms — `e_asin.c` never uses
    /// `t24` in `acos` (see the fix note in `crate::reference::double::
    /// invtrig`).
    #[inline(always)]
    fn acos_near1<V: Simd<Elem = f64>>(x: V) -> V {
        let u = x.abs();
        let (z, c, inner, p) = near1_common(u);
        let y = V::splat(ac::T27).mul_add(c, c) - V::splat(ac::T27) * c;
        let t_plus_y = inner.mul_add(c, y);
        let z_minus_y2 = y.mul_add(-y, z);
        let cc = z_minus_y2 / t_plus_y;
        let neg_cor = (V::splat(ac::HP1) - cc) - (y + cc) * p;
        let neg_res = (V::splat(ac::HP0) - y) + neg_cor;
        let pos_cor = cc + p * (y + cc);
        let pos_res = y + pos_cor;
        let r = V::select(x.lt_mask(V::splat(0.0)), neg_res, pos_res);
        r + r
    }

    /// `asin`'s Taylor band, `|x| < 0.125`.
    #[inline(always)]
    fn asin_taylor<V: Simd<Elem = f64>>(x: V) -> V {
        let x2 = x * x;
        let t = x2.mul_add(V::splat(ac::F6), V::splat(ac::F5));
        let t = x2.mul_add(t, V::splat(ac::F4));
        let t = x2.mul_add(t, V::splat(ac::F3));
        let t = x2.mul_add(t, V::splat(ac::F2));
        let t = x2.mul_add(t, V::splat(ac::F1));
        (x2 * x).mul_add(t, x)
    }

    /// `acos`'s Taylor band, `|x| < 0.125`: `r = HP0 - x` plus a fused
    /// correction, the `HP1`-compensated shape that keeps the small answer
    /// near `x = 1` from cancelling.
    #[inline(always)]
    fn acos_taylor<V: Simd<Elem = f64>>(x: V) -> V {
        let x2 = x * x;
        let t = x2.mul_add(V::splat(ac::F6), V::splat(ac::F5));
        let t = x2.mul_add(t, V::splat(ac::F4));
        let t = x2.mul_add(t, V::splat(ac::F3));
        let t = x2.mul_add(t, V::splat(ac::F2));
        let t = x2.mul_add(t, V::splat(ac::F1));
        let r = V::splat(ac::HP0) - x;
        let inner = (V::splat(ac::HP0) - r) - x + V::splat(ac::HP1);
        let cor = (x2 * x).mul_add(-t, inner);
        r + cor
    }

    /// `acos`'s table bands' trailing combine: `y = HP0 ∓ final`,
    /// `t = HP1 ∓ t_plus_p`, `y + t` — `e_asin.c`'s `acos` bands never add
    /// `row[3+degree]` themselves (`asin`'s do).
    ///
    /// The `∓` is `x`'s sign bit XORed into both operands rather than a pair
    /// of blends: `HP0 - (fin ^ sign(x))` is `HP0 - fin` when `x > 0` and
    /// `HP0 + fin` when `x < 0`, exactly — negation is a bit flip, and the
    /// subtraction that follows rounds once either way. `x` is never zero in
    /// a table band, so there is no signed-zero case to get wrong.
    #[inline(always)]
    fn acos_table_combine<V: Simd<Elem = f64>>(x: V, fin: V, t: V) -> V {
        let sign = x.and_bits(V::splat(-0.0));
        let y = V::splat(ac::HP0) - fin.xor_bits(sign);
        y + (V::splat(ac::HP1) - t.xor_bits(sign))
    }

    /// `asin`'s table bands (`0.125 <= |x| < 0.96875`): one gather and one
    /// degree-9 fold for every lane, whatever band each lane is really in —
    /// the whole-vector-blend discipline this module uses throughout.
    #[inline(always)]
    fn asin_table<V: Simd<Elem = f64>>(u: V) -> V {
        let (xx, key) = asncs_key(u);
        let (t1, c, outer, fin) = asncs_row(key);
        let (t_plus_p, fin) = asncs_poly(xx, t1, c, outer, fin);
        fin + t_plus_p
    }

    /// `acos`'s table bands: the same gather and polynomial as `asin`'s,
    /// combined by `acos_table_combine` instead of the plain unfused
    /// `final + t`.
    #[inline(always)]
    fn acos_table<V: Simd<Elem = f64>>(x: V, u: V) -> V {
        let (xx, key) = asncs_key(u);
        let (t1, c, outer, fin) = asncs_row(key);
        let (t_plus_p, fin) = asncs_poly(xx, t1, c, outer, fin);
        acos_table_combine(x, fin, t_plus_p)
    }

    /// `asin(x)`, vectorised — a genuine replay of `__ieee754_asin_fma`'s
    /// schedule, not a per-lane call, and one that answers *every* input in
    /// vector form. `|x| == 1` is a real band (`+-pi/2`, computed by select),
    /// `|x| > 1` is the canonical NaN the reference's wrapper returns, and a
    /// NaN input gets the reference's own payload-preserving `x + x` from a
    /// hardware add — so there is no scalar repair here and `FullRange` costs
    /// nothing over `Finite`. The sign is applied *before* the out-of-domain
    /// overwrite, so only the real bands take the input's sign
    /// (`asin(-inf)` must stay the positive canonical NaN, not a negative
    /// one).
    ///
    /// # Which bands actually run
    ///
    /// The table band is the one every real workload spends its time in, and
    /// it is also the only band whose cost is a gather, so it runs
    /// unconditionally: its index arithmetic is safe on every input
    /// ([`asncs_key`]) and blending it away costs nothing. The Taylor band is
    /// six FMAs and likewise not worth a branch. The near-one band is the
    /// expensive one — a second table, a divide and a Newton refinement — and
    /// it is the one band that is genuinely rare in bulk data, so it alone is
    /// guarded by an `any()` test. That is the whole dispatch: one predictable
    /// branch instead of the five `all()` tests this used to open with, which
    /// only paid off when an entire vector landed in one band.
    #[inline(always)]
    pub(super) fn asin<V: Simd<Elem = f64>>(x: V) -> V {
        let u = x.abs();
        let below = u.lt_mask(V::splat(TAYLOR_HI));
        let above = u.ge_mask(V::splat(TABLE_HI));

        let mut r = asin_table(u);
        if below.any() {
            r = V::select(below, asin_taylor(u), r);
        }
        if above.any() {
            let near1 = V::select(u.eq_mask(V::splat(1.0)), V::splat(ac::HP0), asin_near1(u));
            r = V::select(above, near1, r);
        }
        // Every band above returns a magnitude, so the sign goes on with an
        // XOR rather than `copysign`'s abs-and-or: `r` is known non-negative.
        r = V::select(u.lt_mask(V::splat(TINY)), u, r).xor_bits(x.and_bits(V::splat(-0.0)));
        // Out of domain *or* NaN in one test: `!(|x| <= 1)` is true for both,
        // since every comparison against a NaN is false. The canonical NaN
        // goes in first and the NaN lanes are then refined to the reference's
        // own payload-preserving `x + x`, which is exactly what a hardware
        // add does to a NaN lane — so this function needs no scalar repair on
        // any input at all, and `FullRange` costs nothing over `Finite`.
        let r = V::select(u.le_mask(V::splat(1.0)).not(), V::splat(f64::NAN), r);
        V::select(x.is_nan(), x + x, r)
    }

    /// `acos(x)`, vectorised — shares `asin`'s bands and differs only in the
    /// linear combination applied to each (see [`acos_table`] and
    /// [`acos_near1`]), and in dispatching on the signed `x` rather than on
    /// the magnitude. Same band-selection reasoning as [`asin`]'s.
    #[inline(always)]
    pub(super) fn acos<V: Simd<Elem = f64>>(x: V) -> V {
        let u = x.abs();
        let below = u.lt_mask(V::splat(TAYLOR_HI));
        let above = u.ge_mask(V::splat(TABLE_HI));

        let mut r = acos_table(x, u);
        if below.any() {
            r = V::select(below, acos_taylor(x), r);
        }
        if above.any() {
            let at_one = V::select(
                x.lt_mask(V::splat(0.0)),
                V::splat(2.0 * ac::HP0),
                V::splat(0.0),
            );
            let near1 = V::select(u.eq_mask(V::splat(1.0)), at_one, acos_near1(x));
            r = V::select(above, near1, r);
        }
        r = V::select(u.lt_mask(V::splat(ACOS_TINY)), V::splat(ac::HP0), r);
        // See `asin`'s identical tail: no scalar repair on any input.
        let r = V::select(u.le_mask(V::splat(1.0)).not(), V::splat(f64::NAN), r);
        V::select(x.is_nan(), x + x, r)
    }
}

/// Arc sine.
pub mod asin {
    use super::*;

    /// `asin(x)` for a vector of lanes.
    ///
    /// Under `BitExact` the vector path is a genuine schedule replay
    /// (`bit_exact::asin`) with no scalar repair on any input: `|x| >= 1` is
    /// handled in-vector (`+-pi/2` at exactly 1, the canonical NaN beyond),
    /// and so is a NaN argument, whose payload-preserving `x + x` is what a
    /// hardware add already does to a NaN lane. Under
    /// `Fast`, `outside(x, 1.0)` also catches `|x| == 1` and sends it to the
    /// reference — one extra scalar lane on an input that is rare in bulk
    /// data, buying not having to reason about `sqrt(0)` in the folded
    /// branch.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let _ = D::CHECKED;
            return bit_exact::asin(x);
        }
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
    ///
    /// Under `BitExact` the vector path is a genuine schedule replay
    /// (`bit_exact::acos`) with no scalar repair on any input, as for
    /// [`asin::eval`]; `|x| >= 1` is handled in-vector (`0` or `2*pi` at
    /// exactly 1, the canonical NaN beyond). Under `Fast`, `outside(x, 1.0)` also catches `|x| == 1` and
    /// sends it to the reference, as in [`asin::eval`].
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let _ = D::CHECKED;
            return bit_exact::acos(x);
        }
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
                patch_lanes2(
                    y,
                    x,
                    z,
                    bit_exact::atan2_needs_repair(y, x),
                    reference::atan2,
                )
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
