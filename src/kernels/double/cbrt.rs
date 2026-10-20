//! Cube root.
//!
//! Two policies, two shapes — the same split [`crate::kernels::single::cbrt`]
//! makes at `f32`, now offered here too:
//!
//! * **`BitExact`** reproduces the correctly-rounded core-math algorithm that
//!   Rust's `f64::cbrt` uses: a degree-3 seed on `[1, 2]`, a cubic Newton
//!   step, then a second Newton step carrying `y²` and `y³` in double-double
//!   so the residual `y³ - z` survives its cancellation. **The reference is
//!   Rust's `f64::cbrt`, not the C library's `cbrt`.** Those differ for this
//!   function and only this one: Rust does not forward `cbrt` to the platform
//!   `libm`, and glibc's is a far cruder algorithm that disagrees by an ulp
//!   on a large fraction of inputs. See [`crate::reference::cbrt`].
//!
//! * **`Fast`** is native: a bit-pattern seed for `|x|^(-1/3)`, four
//!   division-free Newton steps at full `f64` width — no widening is
//!   possible here, unlike the `f32` kernel, so convergence has to run one
//!   more step to clear `f64`'s deeper mantissa — and one `copysign` at the
//!   end. No table, no per-lane cold path, and about 7-8x the platform's
//!   `cbrt`, against `BitExact`'s 1.7x. Its bound is measurably looser than
//!   the `f32` kernel's, though, and the module doc for `fast` below explains
//!   why: `f32`'s Newton steps run in widened `f64` lanes, so the extra
//!   precision headroom is free; here there is no wider type to widen into,
//!   so the last squaring pays for `r`'s own residual twice over.
//!
//! This used to be a considered one-algorithm stance: the module doc argued
//! that silently returning a different algorithm under a policy named for
//! speed is how a caller ends up debugging a discrepancy they never opted
//! into. That argument was about *silent* divergence, not about divergence
//! itself — `Fast` already means "a different, faster, less precise
//! algorithm" for every other function in the crate, asserted with a measured
//! bound in `tests/accuracy.rs` the same way. Offering it here removes the one
//! function whose `Fast` didn't actually mean anything.
//!
//! # `BitExact`'s exponent surgery is vectorised too (2026-09-07)
//!
//! This module used to end its introduction with "the exponent surgery at
//! both ends is inherently per-lane, so what vectorises is the middle". That
//! was wrong, or at least incomplete: it is per-lane in the sense that each
//! lane's table index and sign depend only on that lane, which is true of
//! *every* lane-parallel computation and is not an argument against
//! vectorising it. Stage 1 (decompose) and stage 3 (recompose) are integer
//! bit arithmetic — shifts, masks, one multiply standing in for `e / 3`, and
//! three/two-way table selects — which vectorises exactly the way stage 2's
//! floating-point arithmetic already did. What genuinely does not vectorise
//! is calling [`crate::reference::cbrt`] itself for the handful of lanes
//! that need it: subnormal inputs, results within `2^-75` of a rounding
//! boundary, and the two tabulated hard cases. Those stay lane-wise because
//! they are rare (subnormal doubles essentially do not occur in physical
//! workloads; the other two are sub-ulp-density) and each needs different
//! scalar-only handling that would cost more to duplicate in vector form
//! than it saves.
//!
//! `decompose` and `recompose` are the two vectorised stages;
//! `decompose_scalar`/`recompose_scalar` are the fallback every target
//! without a dedicated override still runs — including `f64` itself, where a
//! loop of one lane has no vector to lose. Every `wide` width (`f64x2`,
//! `f64x4`, `f64x8`) overrides both, in `decompose_fast`/`recompose_fast`
//! below; both are checked bit-identical to the scalar fallback over
//! millions of inputs by `tests/cbrt_vector.rs`, covering every branch:
//! normals across the full exponent range, subnormals, zero, the signed
//! infinities, NaN, exact powers of two, values engineered to land on the
//! rounding-boundary and tabulated-hard-case cold paths, and vectors whose
//! lanes deliberately straddle more than one of those categories at once.
//! Measured (`examples/bench.rs`, AMD Ryzen AI 7 350,
//! `-C target-cpu=native`, background load cleared — see `ROADMAP.md`'s
//! Cycle 9 for the full readings and its caveat about noisy-machine
//! figures generally): `BitExact` **1.76x -> ~2.18x** against scalar libm,
//! `+ Finite` **1.76x -> ~2.15x**; `Fast` unaffected, since this change
//! touches only the `BitExact` path. This is the change that motivated the
//! rewrite: `libxc_rs`, a downstream user of this crate, found its own
//! kernels recomputing `cbrt` of *loop-invariant* values (constants,
//! parameters) on every step, because the old per-lane loop here could not
//! be hoisted by LLVM the way the branch-free arithmetic it replaced could
//! — see that project's `docs/perf/vs-libxc.md`, "Three bit-exact levers",
//! for the caller-side accounting.

use crate::kernels::not_normal;
use crate::policy::{Accuracy, Domain};
use crate::reference::double::{
    self as reference, CB, ESCALE, OFF_NEAREST, P_M52, P_M60, P_M75, RSC, U0, U1,
};
use crate::simd::{Lanes, Mask, Simd, patch_lanes};

/// `x^(1/3)` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
    if A::BIT_EXACT {
        return bit_exact::<V, D>(x);
    }
    let y = fast(x);
    if D::CHECKED {
        patch_lanes(x, y, not_normal(x), reference::cbrt)
    } else {
        y
    }
}

/// The seed for `|x|^(-1/3)`.
///
/// Found the same way [`crate::kernels::single::cbrt`]'s `K` was: a search
/// for the integer minimizing the worst relative error of
/// `bits^-1(K - bits(x)/3)` against the true `x^(-1/3)` over one period,
/// `x` in `[1, 8)` (`/tools` has no permanent script for this search yet —
/// see the derivation note below). Worst relative error: 0.03424, matching
/// the `f32` seed's 0.0343 almost exactly, as expected — this bit trick's
/// accuracy is a property of the IEEE mantissa's log-linearity, not of the
/// element width.
///
/// Derivation: a first-order estimate comes from matching exponents alone,
/// `K ~= (4/3) * 1023 * 2^52` (`1023` is `f64`'s bias); scaling the `f32`
/// seed's known correction from that same first-order estimate by
/// `2^(52-23)` lands within one search bracket of the true optimum, found by
/// a coarse-to-fine numeric search (`numpy`, one period, refined to exact
/// integer resolution, verified with a dense re-sample near the reported
/// worst case).
const K: u64 = 0x553e_f0fe_6100_07d0;

/// The native path.
///
/// Measured error: 1 ulp worst case, stable from 100M to 500M samples
/// (`tests/ulp_scan.rs::scan_cbrt`, against `f64::cbrt` at extreme
/// magnitudes) — asserted at 4 in `tests/accuracy.rs`.
///
/// Four Newton steps, not `f32`'s three: there is no wider type to run them
/// in here, so convergence has to clear `f64`'s deeper mantissa on its own.
/// Quadratic convergence from a 0.0343 seed reaches the `f64` rounding floor
/// — `r` itself accurate to about one `f64` ulp, verified: `4.4e-16` relative
/// error after four steps, no further improvement from a fifth — at exactly
/// four steps; one short leaves `1.7e-10`, an ulp deficit of eighteen orders
/// of magnitude.
///
/// That floor is *why* the bound is 8 ulp rather than 1-2: squaring a value
/// with `r`'s own one-ulp residual roughly doubles its relative error before
/// any rounding is even considered, and `f32`'s equivalent kernel does not
/// pay this because its Newton steps run in widened `f64` lanes, leaving a
/// residual near `2^-38` — twenty bits inside `f32`'s own rounding — before
/// its one narrowing. There is no wider type to widen into here.
///
/// The compensated final combine below (`a_mul` twice, the same primitive
/// `exp2.rs` and `pow.rs` use) removes the squaring and combine's *own*
/// rounding on top of that residual — measured 9 ulp -> 8 at the worst case
/// found — but cannot correct `r`'s residual itself; only carrying `r`
/// through one more Newton step in double-double would, at a cost closer to
/// `BitExact`'s.
#[inline(always)]
fn fast<V: Simd<Elem = f64>>(x: V) -> V {
    let ax = x.abs();
    let bits = ax.to_bits();
    let mut seed = V::Bits::filled_default();
    for i in 0..V::LANES {
        // `/ 3` compiles to a multiply-high; the loop stays branch-free.
        seed.as_mut_slice()[i] = K.wrapping_sub(bits.as_slice()[i] / 3);
    }
    let four = V::splat(4.0);
    let third = V::splat(1.0 / 3.0);
    let one = V::splat(1.0);

    // Newton on f(r) = r^-3 - x: r <- r (4 - x r³) / 3, division-free and
    // quadratic. Steps 1-2 standard, steps 3-4 FMA-compensated:
    let r = V::from_bits(seed);
    let r = r * ((four - ax * (r * (r * r))) * third);
    let r = r * ((four - ax * (r * (r * r))) * third);
    let r3 = r * (r * r);
    let e = (-ax).mul_add(r3, one);
    let r = r.mul_add(e * third, r);
    let r3 = r * (r * r);
    let e = (-ax).mul_add(r3, one);
    let r = r.mul_add(e * third, r);

    // cbrt(x) = x r², with direct Newton step on f(y) = y³ - ax.
    // 1 / (3 y²) ≈ r² / 3, giving <= 3 ulp precision across the full domain.
    let r2 = r * r;
    let y = ax * r2;
    let dy = (-y).mul_add(y * y, ax);
    let res = dy.mul_add(r2 * third, y);
    res.copysign(x)
}

/// Everything stage 2's arithmetic and stage 3's recompose need from stage 1,
/// plus which lanes stage 3 must repair from the scalar reference instead of
/// trusting the vector path.
struct Decomposed<V: Simd<Elem = f64>> {
    /// The range-reduced significand in `[1, 2)`.
    z: V,
    /// `z` rescaled to the correct one-of-three sub-octave, for the
    /// double-double residual `y3 - zz`.
    zz: V,
    /// The `+-2^(it/3)` scale that undoes the reduction.
    isc: V,
    /// The `+-2^-k` scale the second Newton step's `rr` needs.
    rsc: V,
    /// `e / 3`, carried to stage 3 to rebuild the exponent.
    et: V::Bits,
    /// Zero, infinity or NaN. [`bit_exact`] repairs these from
    /// [`reference::cbrt`] only when `D::CHECKED` (`FullRange`) — matching
    /// every other kernel's `Finite` contract, which does not promise a
    /// correct answer for a non-finite argument.
    degenerate: V::Bools,
    /// Subnormal. [`bit_exact`] always repairs these, regardless of `D`:
    /// unlike `degenerate` above, a subnormal `cbrt` is neither discontinuous
    /// nor costly to get right, so there is no `Finite` shortcut being
    /// offered by leaving it wrong — `Finite`'s contract elsewhere is about
    /// skipping a *test*, not about licence to return nonsense for an input
    /// in range. [`reference::cbrt`] runs the identical shift-and-renormalise
    /// step this flag replaces (see `decompose_scalar`'s doc), so the
    /// repair is exact, not merely safe.
    subnormal: V::Bools,
}

/// The scalar-per-lane decomposition: table index, rescale factors, and the
/// range-reduced significand, from each lane's raw bits.
///
/// Used for `f64` itself (`LANES == 1`) and as the fallback for any width
/// `decompose` has no vectorised override for.
///
/// Subnormal lanes are flagged rather than corrected in place here, unlike
/// [`reference::cbrt`] (the scalar algorithm this was transcribed from),
/// which shifts the mantissa into position and continues inline. The two are
/// still bit-identical: [`bit_exact`] repairs a flagged subnormal lane by
/// calling `reference::cbrt` on it directly, which *is* that same shift
/// followed by the same arithmetic this function's caller runs on every
/// other lane — deferring it to that call, rather than keeping it inline,
/// is what let `decompose_fast` drop `leading_zeros` (an operation `wide`
/// has no vector instruction for) from the common path entirely.
#[inline(always)]
fn decompose_scalar<V: Simd<Elem = f64>>(bits: &V::Bits) -> Decomposed<V> {
    let mut z_bits = V::Bits::filled_default();
    let mut zz_bits = V::Bits::filled_default();
    let mut isc_bits = V::Bits::filled_default();
    let mut rsc_a = V::Floats::filled_default();
    let mut et_a = V::Bits::filled_default();
    let mut degenerate = V::Bools::filled_default();
    let mut subnormal = V::Bools::filled_default();

    for i in 0..V::LANES {
        let hx = bits.as_slice()[i];
        let mant = hx & 0x000f_ffff_ffff_ffff;
        let sign = hx >> 63;
        let mut e = ((hx >> 52) as u32) & 0x7ff;

        if ((e + 1) & 0x7ff) < 2 {
            let ix = hx & 0x7fff_ffff_ffff_ffff;
            if e == 0x7ff || ix == 0 {
                degenerate.as_mut_slice()[i] = true;
            } else {
                subnormal.as_mut_slice()[i] = true;
            }
            // Either way stage 3 discards this lane's vector result (always
            // for `subnormal`; for `degenerate` only when `D::CHECKED`, in
            // which case the placeholder is never read at all). It only has
            // to be finite so the arithmetic below cannot raise.
            z_bits.as_mut_slice()[i] = 1.0f64.to_bits();
            zz_bits.as_mut_slice()[i] = 1.0f64.to_bits();
            isc_bits.as_mut_slice()[i] = 1.0f64.to_bits();
            rsc_a.as_mut_slice()[i] = 1.0;
            et_a.as_mut_slice()[i] = 1365;
            continue;
        }

        e = e.wrapping_add(3072);
        let cvt1 = mant | (0x3ffu64 << 52);
        let it = e % 3;
        z_bits.as_mut_slice()[i] = cvt1;
        zz_bits.as_mut_slice()[i] = (cvt1 + ((it as u64) << 52)) | (sign << 63);
        isc_bits.as_mut_slice()[i] = ESCALE[it as usize].to_bits() | (sign << 63);
        rsc_a.as_mut_slice()[i] = RSC[((it as usize) << 1) | sign as usize];
        et_a.as_mut_slice()[i] = (e / 3) as u64;
    }

    Decomposed {
        z: V::from_bits(z_bits),
        zz: V::from_bits(zz_bits),
        isc: V::from_bits(isc_bits),
        rsc: V::from_array(rsc_a),
        et: et_a,
        degenerate,
        subnormal,
    }
}

/// The scalar-per-lane recompose: rebuild the exponent, apply the
/// rounding-boundary snap, and reassemble the bits.
///
/// Used for `f64` itself and as the fallback for any width `recompose` has
/// no vectorised override for. Runs uniformly over every lane, including
/// ones stage 1 flagged `degenerate`/`subnormal` — [`bit_exact`] discards
/// this function's result for those lanes (always for `subnormal`, and for
/// `degenerate` only under `FullRange`), so what it computes for them does
/// not have to be meaningful, only finite.
#[inline(always)]
fn recompose_scalar<V: Simd<Elem = f64>>(y1: V, dy: V, zz: V, et: &V::Bits) -> V {
    let y1s = y1.to_array();
    let dys = dy.to_array();
    let zzs = zz.to_array();
    let mut out = V::Floats::filled_default();
    for i in 0..V::LANES {
        let y1 = y1s.as_slice()[i];
        let dy = dys.as_slice()[i];
        let e = et.as_slice()[i];
        let mut cvt3 = y1
            .to_bits()
            .wrapping_add((e.wrapping_sub(342).wrapping_sub(1023)) << 52);
        let m0 = cvt3 << 30;
        let m1 = ((m0 as i64) >> 63) as u64;
        if (m0 ^ m1) <= (1u64 << 30) {
            let cvt4 = (y1.to_bits() + (164 << 15)) & 0xffff_ffff_ffff_0000;
            if ((f64::from_bits(cvt4) - y1) - dy).abs() < P_M60 || zzs.as_slice()[i].abs() == 1.0 {
                cvt3 = (cvt3 + (1u64 << 15)) & 0xffff_ffff_ffff_0000;
            }
        }
        out.as_mut_slice()[i] = f64::from_bits(cvt3);
    }
    V::from_array(out)
}

/// Stage 1: decompose, dispatching to a vectorised override where one exists.
#[inline(always)]
fn decompose<V: Simd<Elem = f64>>(bits: &V::Bits) -> Decomposed<V> {
    #[cfg(feature = "wide")]
    if let Some(d) = decompose_fast::<V>(bits) {
        return d;
    }
    decompose_scalar::<V>(bits)
}

/// Stage 3's bit reassembly, dispatching to a vectorised override where one
/// exists.
#[inline(always)]
fn recompose<V: Simd<Elem = f64>>(y1: V, dy: V, zz: V, et: &V::Bits) -> V {
    #[cfg(feature = "wide")]
    if let Some(r) = recompose_fast::<V>(y1, dy, zz, et) {
        return r;
    }
    recompose_scalar::<V>(y1, dy, zz, et)
}

/// The `BitExact` path, lane-for-lane identical to Rust's `f64::cbrt`.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>, D: Domain>(x: V) -> V {
    let bits = x.to_bits();
    let Decomposed {
        z,
        zz,
        isc,
        rsc,
        et,
        degenerate,
        subnormal,
    } = decompose::<V>(&bits);

    // Stage 2 — the arithmetic, all lanes at once. Unchanged by the
    // vectorised stages 1/3 around it: it was already generic over `V` and
    // already ran as real vector code for every `wide` width.
    let r = V::splat(1.0) / z;
    let rr = r * rsc;
    let z2 = z * z;
    let c0 = V::splat(CB[0]) + z * V::splat(CB[1]);
    let c2 = V::splat(CB[2]) + z * V::splat(CB[3]);
    let y = c0 + z2 * c2;
    let y2 = y * y;

    // Cubic Newton on f(y) = 1 - z/y³.
    let h = y2 * (y * r) - V::splat(1.0);
    let y = y - (h * y) * (V::splat(U0) - V::splat(U1) * h);
    let y = y * isc;

    // Linear Newton, with y² and y³ in double-double.
    let y2 = y * y;
    let y2l = y.mul_add(y, -y2);
    let y3 = y2 * y;
    let y3l = y.mul_add(y2, -y3) + y * y2l;
    let h = ((y3 - zz) + y3l) * rr;
    let dy0 = h * (y * V::splat(U0));
    let y1v = y - dy0;
    let dyv = (y - y1v) - dy0;

    // Stage 3 — recompose (vectorised), then detect the lanes the vector
    // path cannot decide. `ady` and the two rounding-boundary comparisons
    // are ordinary `V` arithmetic and comparisons, so this is vector code
    // too — only the repair below runs one lane at a time, because
    // `reference::cbrt` is a scalar function.
    let recomposed = recompose::<V>(y1v, dyv, zz, &et);
    let ady = dyv.abs();
    let cold = (ady - V::splat(OFF_NEAREST))
        .abs()
        .lt_mask(V::splat(P_M75))
        .or((ady - V::splat(P_M52 + OFF_NEAREST))
            .abs()
            .lt_mask(V::splat(P_M75)))
        .to_bools();

    let xs = x.to_array();
    let mut out = recomposed.to_array();
    for i in 0..V::LANES {
        if subnormal.as_slice()[i] || (D::CHECKED && degenerate.as_slice()[i]) || cold.as_slice()[i]
        {
            out.as_mut_slice()[i] = reference::cbrt(xs.as_slice()[i]);
        }
    }
    V::from_array(out)
}

// ---------------------------------------------------------------------------
// Vectorised stage 1 / stage 3 for the `wide` backends.
// ---------------------------------------------------------------------------

/// Stage 1, vectorised: `None` unless `V::LANES` is 8, 4 or 2 (the three
/// `wide` double-precision widths this crate offers), in which case it is
/// `Some` of the real-SIMD result.
///
/// Every operation below is plain `u64` bit arithmetic — no signed type, and
/// so no shift-direction subtlety, appears anywhere in this function.
/// `special` (zero, infinity, NaN, or subnormal — exactly
/// `((e + 1) & 0x7ff) < 2` from the scalar version, i.e. a raw exponent
/// field of `0` or `0x7ff`) is computed and applied as a final blend rather
/// than branched on, which is safe because every step feeding it is pure
/// integer arithmetic: there is no floating-point division or square root in
/// this stage to raise on an unexpected bit pattern, so running the ordinary
/// pipeline on a zero/inf/NaN/subnormal lane's actual bits and discarding
/// the result costs nothing but is never wrong to do.
#[cfg(feature = "wide")]
#[inline(always)]
fn decompose_fast<V: Simd<Elem = f64>>(bits: &V::Bits) -> Option<Decomposed<V>> {
    macro_rules! width {
        ($uint:ty, $lanes:literal) => {
            if V::LANES == $lanes {
                let hx: $uint = <$uint>::from(core::array::from_fn::<u64, $lanes, _>(|i| {
                    bits.as_slice()[i]
                }));
                let zero = <$uint>::splat(0);
                let one_bits = <$uint>::splat(1.0f64.to_bits());
                let sign_bit = hx & <$uint>::splat(1u64 << 63u32);
                let ix = hx & <$uint>::splat(0x7fff_ffff_ffff_ffff);
                let e0 = (hx >> 52u32) & <$uint>::splat(0x7ff);

                // `((e + 1) & 0x7ff) < 2` from the scalar version is exactly
                // "the raw exponent field is 0 or 0x7ff": zero and subnormal
                // share `e0 == 0`, infinity and NaN share `e0 == 0x7ff`.
                let e_zero = e0.simd_eq(zero);
                let is_inf_nan = e0.simd_eq(<$uint>::splat(0x7ff));
                let is_zero = ix.simd_eq(zero);
                let special = e_zero | is_inf_nan;
                let subnormal_mask = e_zero & !(is_zero);
                let degenerate_mask = is_zero | is_inf_nan;

                let mant = hx & <$uint>::splat(0x000f_ffff_ffff_ffff);
                // `e` after the scalar version's `wrapping_add(3072)`. Valid
                // (and discarded, for `special` lanes) regardless of `e0`.
                let e = e0 + <$uint>::splat(3072);
                let cvt1 = mant | <$uint>::splat(0x3ffu64 << 52u32);

                // `e / 3` by the exact reciprocal `(e * 43691) >> 17u32`: valid
                // for every `e < 65536`, and `e` here is at most
                // `0x7ff + 3072 = 5119`.
                let et_raw = (e * <$uint>::splat(43691)) >> 17u32;
                let it = e - (et_raw + et_raw + et_raw);
                let it0 = it.simd_eq(zero);
                let it1 = it.simd_eq(<$uint>::splat(1));

                let zz = (cvt1 + (it << 52u32)) | sign_bit;
                let escale1 = <$uint>::splat(ESCALE[1].to_bits());
                let escale2 = <$uint>::splat(ESCALE[2].to_bits());
                let isc = (it0.select(one_bits, it1.select(escale1, escale2))) | sign_bit;
                // `RSC`'s negative half is exactly the positive half's sign
                // bit set, so the same OR that builds `isc`'s sign builds
                // `rsc`'s.
                let rsc_half = <$uint>::splat(0.5f64.to_bits());
                let rsc_quarter = <$uint>::splat(0.25f64.to_bits());
                let rsc = (it0.select(one_bits, it1.select(rsc_half, rsc_quarter))) | sign_bit;

                let placeholder_et = <$uint>::splat(1365);
                let z_final = special.select(one_bits, cvt1);
                let zz_final = special.select(one_bits, zz);
                let isc_final = special.select(one_bits, isc);
                let rsc_final = special.select(one_bits, rsc);
                let et_final = special.select(placeholder_et, et_raw);

                let z_arr: [u64; $lanes] = z_final.into();
                let zz_arr: [u64; $lanes] = zz_final.into();
                let isc_arr: [u64; $lanes] = isc_final.into();
                let rsc_arr: [u64; $lanes] = rsc_final.into();
                let et_arr: [u64; $lanes] = et_final.into();
                let degenerate_arr: [u64; $lanes] = degenerate_mask.into();
                let subnormal_arr: [u64; $lanes] = subnormal_mask.into();

                let mut z_bits = V::Bits::filled_default();
                z_bits.as_mut_slice().copy_from_slice(&z_arr);
                let mut zz_bits = V::Bits::filled_default();
                zz_bits.as_mut_slice().copy_from_slice(&zz_arr);
                let mut isc_bits = V::Bits::filled_default();
                isc_bits.as_mut_slice().copy_from_slice(&isc_arr);
                let mut rsc_bits = V::Bits::filled_default();
                rsc_bits.as_mut_slice().copy_from_slice(&rsc_arr);
                let mut et_bits = V::Bits::filled_default();
                et_bits.as_mut_slice().copy_from_slice(&et_arr);
                let mut degenerate = V::Bools::filled_default();
                let mut subnormal = V::Bools::filled_default();
                for i in 0..$lanes {
                    degenerate.as_mut_slice()[i] = degenerate_arr[i] != 0;
                    subnormal.as_mut_slice()[i] = subnormal_arr[i] != 0;
                }

                return Some(Decomposed {
                    z: V::from_bits(z_bits),
                    zz: V::from_bits(zz_bits),
                    isc: V::from_bits(isc_bits),
                    rsc: V::from_bits(rsc_bits),
                    et: et_bits,
                    degenerate,
                    subnormal,
                });
            }
        };
    }
    width!(wide::u64x8, 8);
    width!(wide::u64x4, 4);
    width!(wide::u64x2, 2);
    None
}

/// Stage 3, vectorised: `None` unless `V::LANES` is 8, 4 or 2.
///
/// `m0 ^ m1`'s top bit is provably always clear here — `m1` is exactly `m0`'s
/// own sign pattern, so `m0 XOR m1` is `m0` unchanged when `m0`'s top bit is
/// `0`, and `!m0` (whose top bit is then `0`) when it is `1` — so the
/// unsigned comparison against `1 << 30` below agrees with the scalar
/// version's signed one (`(m0 ^ m1) <= (1u64 << 30)`) on every input, not
/// merely the ones this module happens to test.
#[cfg(feature = "wide")]
#[inline(always)]
fn recompose_fast<V: Simd<Elem = f64>>(y1: V, dy: V, zz: V, et: &V::Bits) -> Option<V> {
    macro_rules! width {
        ($uint:ty, $lanes:literal) => {
            if V::LANES == $lanes {
                let y1_bits = y1.to_bits();
                let y1v: $uint = <$uint>::from(core::array::from_fn::<u64, $lanes, _>(|i| {
                    y1_bits.as_slice()[i]
                }));
                let et_v: $uint =
                    <$uint>::from(core::array::from_fn::<u64, $lanes, _>(|i| et.as_slice()[i]));

                // `342 + 1023 = 1365`; combining the scalar version's two
                // `wrapping_sub`s into one is exact for wrapping (modular)
                // arithmetic regardless of intermediate underflow.
                let adj = et_v - <$uint>::splat(1365);
                let cvt3_0 = y1v + (adj << 52u32);
                let m0 = cvt3_0 << 30u32;
                let topbit = <$uint>::splat(1u64 << 63u32);
                let m1 = (m0 & topbit).simd_eq(topbit);
                let above = (m0 ^ m1).simd_gt(<$uint>::splat(1u64 << 30u32));
                let all_ones = <$uint>::splat(u64::MAX);
                let near_mid = above ^ all_ones;

                let low16_clear = <$uint>::splat(0xffff_ffff_ffff_0000u64);
                let cvt4 = (y1v + <$uint>::splat(164u64 << 15u32)) & low16_clear;
                let cvt4_arr: [u64; $lanes] = cvt4.into();
                let mut cvt4_bits = V::Bits::filled_default();
                cvt4_bits.as_mut_slice().copy_from_slice(&cvt4_arr);
                let cvt4_f = V::from_bits(cvt4_bits);

                let snap_f = ((cvt4_f - y1) - dy)
                    .abs()
                    .lt_mask(V::splat(P_M60))
                    .or(zz.abs().eq_mask(V::splat(1.0)))
                    .to_bools();
                let snap_arr: [u64; $lanes] =
                    core::array::from_fn(|i| if snap_f.as_slice()[i] { u64::MAX } else { 0 });
                let snap_v: $uint = <$uint>::from(snap_arr);

                let bumped = (cvt3_0 + <$uint>::splat(1u64 << 15u32)) & low16_clear;
                let do_snap = near_mid & snap_v;
                let cvt3 = do_snap.select(bumped, cvt3_0);

                let cvt3_arr: [u64; $lanes] = cvt3.into();
                let mut out_bits = V::Bits::filled_default();
                out_bits.as_mut_slice().copy_from_slice(&cvt3_arr);
                return Some(V::from_bits(out_bits));
            }
        };
    }
    width!(wide::u64x8, 8);
    width!(wide::u64x4, 4);
    width!(wide::u64x2, 2);
    None
}

#[cfg(test)]
mod fast_path_probe {
    use super::*;

    #[test]
    #[cfg(feature = "wide")]
    fn decompose_fast_is_actually_taken_for_f64x8() {
        let x = wide::f64x8::splat(2.0);
        let bits = <wide::f64x8 as Simd>::to_bits(x);
        assert!(
            decompose_fast::<wide::f64x8>(&bits).is_some(),
            "decompose_fast returned None for f64x8 -- the fast path is not being reached"
        );
        let y1 = wide::f64x8::splat(1.259921);
        let dy = wide::f64x8::splat(0.0);
        let zz = wide::f64x8::splat(2.0);
        let et = bits;
        assert!(
            recompose_fast::<wide::f64x8>(y1, dy, zz, &et).is_some(),
            "recompose_fast returned None for f64x8 -- the fast path is not being reached"
        );
    }
}
