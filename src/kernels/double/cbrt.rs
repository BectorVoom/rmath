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
//! Shape of the `BitExact` implementation: the exponent surgery at both ends
//! is inherently per-lane, so what vectorises is the middle — the reciprocal,
//! the seed, and the two Newton steps, which is where the arithmetic is. Two
//! data-dependent cold paths (a result too close to a rounding boundary to
//! decide, and two tabulated hard cases) fall back to the scalar reference for
//! the affected lanes only.

use crate::kernels::double::dd::a_mul;
use crate::kernels::not_normal;
use crate::policy::{Accuracy, Domain};
use crate::reference::double::{
    self as reference, CB, ESCALE, OFF_NEAREST, P_M52, P_M60, P_M75, RSC, U0, U1,
};
use crate::simd::{Lanes, Simd, patch_lanes};

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
/// Measured error: 8 ulp worst case, stable from 100M to 500M samples
/// (`tests/ulp_scan.rs::scan_cbrt`, against `f64::cbrt` at extreme
/// magnitudes) — asserted at 16 in `tests/accuracy.rs` for headroom.
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

    // Newton on f(r) = r^-3 - x: r <- r (4 - x r³) / 3, division-free and
    // quadratic. Seed error 0.0343 -> 2.4e-3 -> 1.1e-5 -> 2.6e-10 -> 4.4e-16.
    let r = V::from_bits(seed);
    let r = r * ((four - ax * (r * (r * r))) * third);
    let r = r * ((four - ax * (r * (r * r))) * third);
    let r = r * ((four - ax * (r * (r * r))) * third);
    let r = r * ((four - ax * (r * (r * r))) * third);

    // cbrt(x) = x r², with r² and the final product compensated. `r` itself
    // is accurate to about a `f64` ulp at this point — the Newton iteration
    // above cannot refine it further in plain `f64` arithmetic — but squaring
    // a value that carries even one ulp of relative error roughly doubles it,
    // and a plain `ax * (r * r)` adds two more roundings on top of that. This
    // does not fix `r`'s own residual (only carrying `r` itself in
    // double-double through one more Newton step would), but it removes the
    // squaring and combine's own contribution, which measurably tightens the
    // worst case (see the module doc's measured bound).
    let (r2h, r2l) = a_mul(r, r);
    let (yh, yl) = a_mul(ax, r2h);
    (yh + (yl + ax * r2l)).copysign(x)
}

/// The `BitExact` path, lane-for-lane identical to Rust's `f64::cbrt`.
#[inline(always)]
fn bit_exact<V: Simd<Elem = f64>, D: Domain>(x: V) -> V {
    let bits = x.to_bits();

    // Stage 1 — decompose. Per-lane by necessity: a table index, a leading-zero
    // count for subnormals, and a `% 3` on the exponent.
    let mut z_bits = V::Bits::filled_default();
    let mut zz_bits = V::Bits::filled_default();
    let mut isc_bits = V::Bits::filled_default();
    let mut rsc_a = V::Floats::filled_default();
    let mut et_a = V::Bits::filled_default();
    let mut degenerate = V::Bools::filled_default();

    for i in 0..V::LANES {
        let hx = bits.as_slice()[i];
        let mut mant = hx & 0x000f_ffff_ffff_ffff;
        let sign = hx >> 63;
        let mut e = ((hx >> 52) as u32) & 0x7ff;

        if ((e + 1) & 0x7ff) < 2 {
            let ix = hx & 0x7fff_ffff_ffff_ffff;
            if e == 0x7ff || ix == 0 {
                // Zero, infinity, NaN. Flag the lane and give the vector maths
                // a harmless value so it cannot raise on data we discard.
                degenerate.as_mut_slice()[i] = true;
                z_bits.as_mut_slice()[i] = 1.0f64.to_bits();
                zz_bits.as_mut_slice()[i] = 1.0f64.to_bits();
                isc_bits.as_mut_slice()[i] = 1.0f64.to_bits();
                rsc_a.as_mut_slice()[i] = 1.0;
                et_a.as_mut_slice()[i] = 1365;
                continue;
            }
            // Subnormal: shift the leading one into place. Handled here rather
            // than deferred to the reference — it is pure integer work that
            // costs nothing extra inside a loop already doing integer work.
            let nz = ix.leading_zeros() - 11;
            mant <<= nz;
            mant &= 0x000f_ffff_ffff_ffff;
            e = e.wrapping_sub(nz - 1);
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

    // Stage 2 — the arithmetic, all lanes at once.
    let z = V::from_bits(z_bits);
    let zz = V::from_bits(zz_bits);
    let isc = V::from_bits(isc_bits);
    let rsc = V::from_array(rsc_a);

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

    // Stage 3 — recompose, and detect the lanes the vector path cannot decide.
    let xs = x.to_array();
    let y1s = y1v.to_array();
    let dys = dyv.to_array();
    let zzs = zz.to_array();
    let mut out = V::Floats::filled_default();

    for i in 0..V::LANES {
        let xi = xs.as_slice()[i];
        if D::CHECKED && degenerate.as_slice()[i] {
            out.as_mut_slice()[i] = reference::cbrt(xi);
            continue;
        }
        let y1 = y1s.as_slice()[i];
        let dy = dys.as_slice()[i];

        // Within 2^-75 of a rounding boundary: the reference takes another
        // Newton step and consults two tabulated hard cases. Rare enough that
        // duplicating it in vector form would be all cost and no benefit.
        let ady = dy.abs();
        if (ady - OFF_NEAREST).abs() < P_M75 || (ady - (P_M52 + OFF_NEAREST)).abs() < P_M75 {
            out.as_mut_slice()[i] = reference::cbrt(xi);
            continue;
        }

        let et = et_a.as_slice()[i];
        let mut cvt3 = y1
            .to_bits()
            .wrapping_add((et.wrapping_sub(342).wrapping_sub(1023)) << 52);
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
