//! `e^x`.
//!
//! * [`crate::policy::BitExact`] — glibc's `__ieee754_exp_fma` schedule:
//!   argument reduction to `|r| < ln(2)/256`, a 128-entry `2^(k/128)` table,
//!   and a degree-4 correction. Bit-identical to the scalar `exp`.
//! * [`crate::policy::Fast`] — no table. Reduces to `|r| <= ln(2)/2` and
//!   takes a degree-13 series, evaluated by Estrin's scheme to keep the
//!   dependency chain short. The point is not fewer operations — it is more
//!   arithmetic and *no gather*, and the gather is the part that does not
//!   vectorise. The series carries no leading 1 — see `fast` below — so the
//!   final combine with `scale` is one true FMA rather than a separate
//!   rounding.
//!
//! `Finite` means `|x| < 512`. Outside that, results are wrong, not merely
//! imprecise; note that `exp` overflows at 709.78, so the safe range is set
//! by the reduction, not by the function's own domain.

use crate::kernels::outside;
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Lanes, Simd, patch_lanes};
use crate::tables::double::exp as t;

/// Widest `|x|` the vector main path is valid for.
const MAIN_PATH_LIMIT: f64 = 512.0;

/// `e^x` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    let y = if A::BIT_EXACT { bit_exact(x) } else { fast(x) };
    if D::CHECKED {
        patch_lanes(x, y, outside(x, MAIN_PATH_LIMIT), reference::exp)
    } else {
        y
    }
}

/// `t::TAB[idx[i]]` per lane. The one safe wrapper this crate's sole
/// `unsafe` call site (`Simd::gather_bits`) goes through here — kept tiny and
/// local so the safety argument stays next to the index arithmetic that
/// makes it true, rather than travelling to a call site far from it.
///
/// # Panics
/// Never, given how the caller builds `idx` — but this is not a `debug_assert`
/// away from being one: every `idx[i]` genuinely must be `< t::TAB.len()`.
#[allow(unsafe_code)]
#[inline(always)]
fn gather_tab<V: Simd<Elem = f64>>(idx: V::Bits) -> V::Bits {
    const { assert!(t::TAB.len() == 256) };
    // SAFETY: every caller in this module builds `idx` as `(k & 127) * 2`
    // or that plus one, which is always even (or one more than even) and
    // `< 256` — checked against `t::TAB`'s actual length just above, at
    // compile time, so a future resize of the table cannot silently
    // invalidate this.
    unsafe { V::gather_bits(&t::TAB, idx) }
}

/// The table path, lane-for-lane identical to [`reference::exp`]'s main body.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>>(x: V) -> V {
    let shift = V::splat(t::SHIFT);
    let kd_s = x.mul_add(V::splat(t::INVLN2N), shift);
    let ki = kd_s.to_bits();
    let kd = kd_s - shift;
    let r = kd.mul_add(
        V::splat(t::NEGLN2LON),
        kd.mul_add(V::splat(t::NEGLN2HIN), x),
    );

    // The gather. Index arithmetic and the post-gather `wrapping_add` are
    // still per-lane -- `Simd::Bits` has no shift or add of its own, see
    // `crate::kernels::exact`'s module doc for the fuller account of why --
    // but the two table reads themselves go through `Simd::gather_bits`,
    // which is a real hardware gather instruction on `f64x4`/`f64x8` when
    // the target has `avx2`/`avx512f` (see `src/simd/wide_backend.rs`), and
    // the same checked per-lane loop as before otherwise. This is the A5
    // prototype (`ROADMAP.md`): landed here first because `exp` measures the
    // win most directly, one gather, no other per-lane work sharing the loop.
    let mut idx0 = V::Bits::filled_default();
    let mut idx1 = V::Bits::filled_default();
    let mut kshift = V::Bits::filled_default();
    for i in 0..V::LANES {
        let k = ki.as_slice()[i];
        let ix = (k & 127) * 2;
        idx0.as_mut_slice()[i] = ix;
        idx1.as_mut_slice()[i] = ix + 1;
        kshift.as_mut_slice()[i] = k << 45;
    }
    let tail_bits = gather_tab::<V>(idx0);
    let scale_raw = gather_tab::<V>(idx1);
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        scale_bits.as_mut_slice()[i] = scale_raw.as_slice()[i].wrapping_add(kshift.as_slice()[i]);
    }
    let tail = V::from_bits(tail_bits);
    let scale = V::from_bits(scale_bits);

    let p12 = r.mul_add(V::splat(t::C3), V::splat(t::C2));
    let t3 = tail + r;
    let r2 = r * r;
    let p45 = r.mul_add(V::splat(t::C5), V::splat(t::C4));
    let s1 = r2.mul_add(p12, t3);
    let r4 = r2 * r2;
    let tmp = r4.mul_add(p45, s1);
    scale.mul_add(tmp, scale)
}

/// `1/ln(2)`, for the table-free reduction.
const LOG2E: f64 = core::f64::consts::LOG2_E;
/// `ln(2)`, split so that `kd * LN2HI` is exact.
///
/// LN2HI is exactly representable in 33 bits, which makes that product exact
/// for every `kd` the reduction produces; LN2LO carries the remainder. The
/// digits are not stray precision — rounding either to the shortest
/// round-tripping form, as `clippy::excessive_precision` suggests, destroys
/// that property and costs accuracy in the reduction.
#[allow(clippy::excessive_precision)]
const LN2HI: f64 = 6.93147180369123816490e-01;

/// `ln(2)`, low part. See [`LN2HI`].
#[allow(clippy::excessive_precision)]
const LN2LO: f64 = 1.90821492927058770002e-10;

/// `1/k!` for `k` in `2..=13`. Truncating the series here leaves a relative
/// error of `r^14 / 14!`, which at the reduction's worst case
/// (`|r| = ln(2)/2`) is about `6.5e-18` — a fifth of an ulp, so the error
/// that matters is the accumulated rounding, not the truncation.
const F: [f64; 12] = [
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
];

/// The table-free path.
///
/// The series is evaluated *without* its leading 1 — `poly` below is
/// `e^r - 1`, not `e^r` — for the same reason as [`super::exp2`]'s fast path:
/// it lets the final combine with `scale` be one true FMA,
/// `scale.mul_add(poly, scale)`, instead of rounding `1 + poly` and then
/// rounding `scale * (that)` separately.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V) -> V {
    let shift = V::splat(t::SHIFT);
    let kd_s = x.mul_add(V::splat(LOG2E), shift);
    let kd = kd_s - shift;
    // Cody-Waite: subtract `kd * ln(2)` in two exactly-representable pieces.
    let r = kd.mul_add(V::splat(-LN2LO), kd.mul_add(V::splat(-LN2HI), x));

    // Estrin. Horner would be 13 dependent FMAs — about 52 cycles of latency
    // that no amount of lane parallelism hides, since every lane waits on the
    // same chain. This is the same operation count at depth 6.
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;

    let c23 = r.mul_add(V::splat(F[1]), V::splat(F[0]));
    let c45 = r.mul_add(V::splat(F[3]), V::splat(F[2]));
    let c67 = r.mul_add(V::splat(F[5]), V::splat(F[4]));
    let c89 = r.mul_add(V::splat(F[7]), V::splat(F[6]));
    let cab = r.mul_add(V::splat(F[9]), V::splat(F[8]));
    let ccd = r.mul_add(V::splat(F[11]), V::splat(F[10]));

    let lo = r2.mul_add(c23, r); // r itself is the degree-1 term (c0 = c1 = 1)
    let mid = r2.mul_add(c67, c45);
    let hi = r2.mul_add(cab, c89);
    let lo = r4.mul_add(mid, lo);
    let hi = r4.mul_add(ccd, hi);
    let poly = r8.mul_add(hi, lo);

    // 2^kd, built straight into the exponent field. Pure integer arithmetic,
    // no memory: this is what the table path cannot do.
    let ki = kd_s.to_bits();
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        // The shift constant puts `k + 2^51` in the low 52 bits.
        let k = (ki.as_slice()[i] & 0x000f_ffff_ffff_ffff).wrapping_sub(1u64 << 51);
        scale_bits.as_mut_slice()[i] = k.wrapping_add(1023) << 52;
    }
    let scale = V::from_bits(scale_bits);
    scale.mul_add(poly, scale)
}
