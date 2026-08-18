//! Cube root.
//!
//! Reproduces the correctly-rounded core-math algorithm that Rust's
//! `f64::cbrt` uses: a degree-3 seed on `[1, 2]`, a cubic Newton step, then a
//! second Newton step carrying `y²` and `y³` in double-double so the residual
//! `y³ - z` survives its cancellation.
//!
//! **The reference is Rust's `f64::cbrt`, not the C library's `cbrt`.** Those
//! differ for this function and only this one: Rust does not forward `cbrt` to
//! the platform `libm`, and glibc's is a far cruder algorithm that disagrees by
//! an ulp on a large fraction of inputs. See [`crate::reference::cbrt`].
//!
//! **The accuracy axis is a no-op here.** `Fast` and `BitExact` run the same
//! code. A cheaper cube root is certainly possible — drop the double-double
//! step and accept an ulp — but it would no longer be *this* function, and
//! silently returning a different algorithm under a policy named for speed is
//! how a caller ends up debugging a discrepancy they never opted into. The
//! axis is accepted rather than rejected so generic code can configure every
//! function uniformly.
//!
//! Shape of the vector implementation: the exponent surgery at both ends is
//! inherently per-lane, so what vectorises is the middle — the reciprocal, the
//! seed, and the two Newton steps, which is where the arithmetic is. Two
//! data-dependent cold paths (a result too close to a rounding boundary to
//! decide, and two tabulated hard cases) fall back to the scalar reference for
//! the affected lanes only.

use crate::policy::{Accuracy, Domain};
use crate::reference::{self, CB, ESCALE, OFF_NEAREST, P_M52, P_M60, P_M75, RSC, U0, U1};
use crate::simd::{Lanes, Simd};

/// `x^(1/3)` for a vector of lanes.
#[inline(always)]
pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V {
    let _ = A::BIT_EXACT; // see the module docs: one algorithm, deliberately
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
