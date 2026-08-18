//! `x^y`.
//!
//! A port: the vector code replays glibc's `__ieee754_pow_fma` schedule
//! lane-parallel, so `BitExact` is both exact and faster than the call it
//! replaces.
//!
//! `pow` is the hardest function here, and the reason is one line of algebra:
//! `x^y = e^(y log x)`, so `y` multiplies the *error* in `log x` along with
//! its value. A logarithm good to half an ulp is not good enough — at `|y|`
//! near 1000 it would give up ten bits of the answer. So the logarithm is
//! carried as an unevaluated `hi + lo` pair throughout, over a table of its
//! own whose `log(c)` entries are themselves stored in two pieces. That is
//! also why this kernel gathers twice: once for the logarithm's table and once
//! for the exponential's.
//!
//! Under `Fast` there is no table at all — the point of that policy is to
//! avoid the gather — and the price is the accuracy this function is most
//! sensitive to.

use crate::kernels::double::exp2;
use crate::kernels::{dispatch2, log_poly, log_split, not_normal, not_positive_normal};
use crate::policy::{Accuracy, Domain, Fast, Finite};
use crate::reference::double::{self as reference, POW_OFF};
use crate::simd::{Lanes, Mask, Simd, patch_lanes2};
use crate::tables::double::exp as et;
use crate::tables::double::poly as p;
use crate::tables::double::pow as pt;

/// Widest `|y log2 x|` the `Fast` exponential path stays valid for.
///
/// Not 1024. The exponential builds `2^k` straight into the exponent field,
/// which cannot represent a subnormal, so a result below `2^-1022` comes back
/// as zero rather than as the small number it should be. Stopping short of
/// that hands the whole subnormal tail to the reference, where it is right.
const EXPONENT_LIMIT: f64 = 1020.0;

/// `x^y` for vectors of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V, y: V) -> V {
    if A::BIT_EXACT {
        let (z, off) = bit_exact(x, y);
        if D::CHECKED {
            return patch_lanes2(x, y, z, off, reference::pow);
        }
        return z;
    }
    dispatch2::<V, A, D>(x, y, reference::pow, fast, |x, y| {
        let over = exponent(x, y).abs().lt_mask(V::splat(EXPONENT_LIMIT)).not();
        not_positive_normal(x).or(not_normal(y)).or(over)
    })
}

/// `|v|` outside `[lo, hi)` — including NaN, which fails both comparisons.
#[inline(always)]
fn outside_band<V: Simd<Elem = f64>>(v: V, lo: f64, hi: f64) -> V::Mask {
    let a = v.abs();
    a.ge_mask(V::splat(lo)).and(a.lt_mask(V::splat(hi))).not()
}

/// The ported path, and the lanes it must not be trusted with.
///
/// The mask is returned rather than applied here because it is only known
/// once `ehi` exists, which is halfway through the computation — and computing
/// the whole thing for every lane and repairing afterwards is cheaper than
/// branching, which is the pattern throughout this crate.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>>(x: V, y: V) -> (V, V::Mask) {
    let (lhi, llo) = pow_log(x);

    // y * (hi + lo), as an unevaluated pair; both products fused.
    let ehi = y * lhi;
    let elo = y.mul_add(llo, y.mul_add(lhi, -ehi));

    let z = pow_exp(ehi, elo);

    // The vector path is for a positive normal base, an exponent in
    // [2^-65, 2^63), and a product the exponential's main path can take.
    // Everything else — negative bases with integer exponents, the signed
    // zeros, `pow(1, NaN) == 1`, overflow, underflow — goes to the reference,
    // where those rules are the platform's rather than restated here.
    let bad = not_positive_normal(x)
        .or(outside_band(
            y,
            f64::from_bits(0x3be0000000000000),
            f64::from_bits(0x43e0000000000000),
        ))
        .or(outside_band(ehi, f64::from_bits(0x3c90000000000000), 512.0));
    (z, bad)
}

/// `log(x)` to more than double precision, as an unevaluated `hi + lo`.
#[inline(always)]
fn pow_log<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let ix = x.to_bits();

    let mut invc_a = V::Floats::filled_default();
    let mut logc_a = V::Floats::filled_default();
    let mut tail_a = V::Floats::filled_default();
    let mut z_bits = V::Bits::filled_default();
    let mut kd_a = V::Floats::filled_default();
    for i in 0..V::LANES {
        let b = ix.as_slice()[i];
        let tmp = b.wrapping_sub(POW_OFF);
        let idx = ((tmp >> 45) & 127) as usize;
        z_bits.as_mut_slice()[i] = b.wrapping_sub(tmp & (0xfffu64 << 52));
        invc_a.as_mut_slice()[i] = pt::TAB[3 * idx];
        logc_a.as_mut_slice()[i] = pt::TAB[3 * idx + 1];
        tail_a.as_mut_slice()[i] = pt::TAB[3 * idx + 2];
        kd_a.as_mut_slice()[i] = ((tmp as i64) >> 52) as f64;
    }
    let z = V::from_bits(z_bits);
    let kd = V::from_array(kd_a);
    let logc = V::from_array(logc_a);
    let logctail = V::from_array(tail_a);

    let r = z.mul_add(V::from_array(invc_a), V::splat(-1.0));

    let t1 = kd.mul_add(V::splat(pt::LN2HI), logc);
    let t2 = t1 + r;
    let lo1 = kd.mul_add(V::splat(pt::LN2LO), logctail);
    let lo2 = t1 - t2 + r;

    let ar = V::splat(pt::A0) * r;
    let ar2 = r * ar;
    let ar3 = r * ar2;
    let hi = t2 + ar2;
    let lo3 = ar.mul_add(r, -ar2);
    let lo4 = t2 - hi + ar2;

    let c21 = V::splat(pt::A2).mul_add(r, V::splat(pt::A1));
    let c43 = V::splat(pt::A4).mul_add(r, V::splat(pt::A3));
    let c65 = V::splat(pt::A6).mul_add(r, V::splat(pt::A5));
    let inner = ar2.mul_add(c65.mul_add(ar2, c43), c21);
    let lo = ar3.mul_add(inner, lo1 + lo2 + lo3 + lo4);

    let y = hi + lo;
    (y, hi - y + lo)
}

/// `exp(x + xtail)`, for the main path only.
///
/// The sign-bias argument the scalar reference carries is absent: a negative
/// base is a special lane, so the vector path never needs to negate.
#[inline(always)]
fn pow_exp<V: Simd<Elem = f64>>(x: V, xtail: V) -> V {
    let shift = V::splat(et::SHIFT);
    let kd_s = x.mul_add(V::splat(et::INVLN2N), shift);
    let ki = kd_s.to_bits();
    let kd = kd_s - shift;
    let r = kd.mul_add(
        V::splat(et::NEGLN2LON),
        kd.mul_add(V::splat(et::NEGLN2HIN), x),
    ) + xtail;

    let mut tail_bits = V::Bits::filled_default();
    let mut scale_bits = V::Bits::filled_default();
    for i in 0..V::LANES {
        let k = ki.as_slice()[i];
        let idx = ((k % 128) * 2) as usize;
        tail_bits.as_mut_slice()[i] = et::TAB[idx];
        scale_bits.as_mut_slice()[i] = et::TAB[idx + 1].wrapping_add(k << 45);
    }
    let tail = V::from_bits(tail_bits);
    let scale = V::from_bits(scale_bits);

    let r2 = r * r;
    let p23 = V::splat(et::C3).mul_add(r, V::splat(et::C2));
    let s = tail + r;
    let p45 = r.mul_add(V::splat(et::C5), V::splat(et::C4));
    let t = p23.mul_add(r2, s);
    let tmp = p45.mul_add(r2 * r2, t);
    scale.mul_add(tmp, scale)
}

/// `y * log2(x)`, to double precision, for the `Fast` range test.
#[inline(always)]
fn exponent<V: Simd<Elem = f64>>(x: V, y: V) -> V {
    let (hi, _) = log2_extended(x);
    y * hi
}

/// `log2(x)` as an unevaluated `hi + lo`, without a table.
///
/// The split is exact: `e` is an integer and `|lm| <= 1/2`, so `hi = e + lm`
/// and `lo = (e - hi) + lm` is the Fast2Sum identity with its precondition met.
#[inline(always)]
fn log2_extended<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let (e, s) = log_split(x);
    let lm = log_poly(s) * V::splat(p::LOG2E);
    let hi = e + lm;
    (hi, (e - hi) + lm)
}

/// Measured error: below 16 ulp for `|y log2 x| < 32`, below 40 ulp over the
/// whole vector domain.
///
/// Looser than every other kernel here, and structurally so: `y` multiplies
/// the error in `log2 x` as well as its value. The pair below removes the
/// error of *representing* the logarithm but not of computing it, so the
/// residue grows with `|y log2 x|`. Closing that is exactly what the
/// `BitExact` path's table does — at the cost of the gather this policy exists
/// to avoid.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V, y: V) -> V {
    let (lhi, llo) = log2_extended(x);

    let ph = y * lhi;
    let pl = y.mul_add(lhi, -ph) + y * llo;

    // 2^(ph + pl) = 2^ph * (1 + pl ln2 + ...). `|pl|` is below an ulp of `ph`,
    // so the first-order term is enough to carry the low part back.
    let s = exp2::eval::<V, Fast, Finite>(ph);
    s.mul_add(pl * V::splat(core::f64::consts::LN_2), s)
}
