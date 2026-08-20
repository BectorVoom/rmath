//! `log1p(x) = ln(1 + x)`, accurate for small `x`.
//!
//! A port of glibc's `__log1p_fma` (`sysdeps/ieee754/dbl-64/s_log1p.c`,
//! compiled with `-mfma` — the source is untouched between the two builds;
//! only the compiler's fusion choices differ). Branchy near zero, where the
//! naive `1.0 + x` would round away the low bits of a small `x`: a
//! reduction recovers exactly what got lost (`c = (u - 1) - x`, exact by
//! Sterbenz wherever it matters) and folds it back in as a first-order
//! correction, `ln(u) - c/u`.

// Threshold/coefficient constants are quoted at the precision `s_log1p.c`
// itself quotes them, matching `tests/bit_exact.rs`'s convention.
#![allow(clippy::excessive_precision)]

/// High part of `ln(2)`; `n * LN2_HI` is exact for every `|n| < 2000`.
const LN2_HI: f64 = f64::from_bits(0x3fe62e42fee00000);
/// Low part of `ln(2)`. See [`LN2_HI`].
const LN2_LO: f64 = f64::from_bits(0x3dea39ef35793c76);
/// `2^54`, both the underflow-detection scale and the `-1.0` overflow numerator.
const TWO54: f64 = f64::from_bits(0x4350000000000000);
/// `Lp[1..=7]`: the odd-series minimax coefficients for `R(z)` on `s in
/// [0, 0.1716]`, `s = f/(2+f)`. `s_log1p.c`'s `Lp[0]` is the unused `0.0`
/// placeholder for 1-based indexing in the source; dropped here.
const LP: [f64; 7] = [
    f64::from_bits(0x3FE5555555555593),
    f64::from_bits(0x3FD999999997FA04),
    f64::from_bits(0x3FD2492494229359),
    f64::from_bits(0x3FCC71C51D8E78AF),
    f64::from_bits(0x3FC7466496CB03DE),
    f64::from_bits(0x3FC39A09D078C69F),
    f64::from_bits(0x3FC2F112DF3E5244),
];

/// `log1p(x)`, the platform's.
#[allow(unused_assignments)] // matches `s_log1p.c`'s own dead `k = 1` init
pub fn log1p(x: f64) -> f64 {
    if x.is_nan() {
        // Verified against the live platform, not assumed: a NaN input
        // (either sign) comes back with its sign and payload untouched but
        // the quiet bit forced on — standard NaN-quieting through an
        // arithmetic op, which this crate's own hardware does *not* give
        // for `x + x`/`x * 1.0`/`x - 0.0` on an identical-operand NaN (those
        // collapse to the canonical `0x7ff8...0`, losing the payload). The
        // C source's own `(x - x) / (x - x)` in the `x <= -1.0` domain-error
        // branch only reads as "return a NaN" when `x` is itself finite;
        // for a NaN `x` the real function does not take that branch at all.
        return f64::from_bits(x.to_bits() | (1u64 << 51));
    }
    let bits = x.to_bits();
    let hx = (bits >> 32) as i32;
    let ax = (hx & 0x7fffffff) as u32;

    let mut k = 1i32;
    let mut c = 0.0f64;
    let mut hu: u32;

    if hx < 0x3FDA827A {
        // x < 0.41422
        if ax >= 0x3ff00000 {
            // x <= -1.0 (or a NaN with the sign/exponent bits that implies)
            return if x == -1.0 {
                -TWO54 / 0.0
            } else {
                // `s_log1p.c`'s own `(x - x) / (x - x)`: a genuine runtime
                // `0.0/0.0`, which x86-64's `divsd` resolves to its hardware
                // default NaN, `0xfff8000000000000` — sign bit *set*, unlike
                // Rust's own `f64::NAN` constant (`0x7ff8...`, unset). The
                // original port wrote the literal `(x - x) / (x - x)` and
                // got this right by construction (a real division at
                // runtime); simplifying it to `f64::NAN` to silence
                // clippy's `eq_op` lint substituted the wrong constant —
                // caught by `tests/delegating.rs`'s corpus, not the
                // brute-force sweep above (whose own `want` came from a
                // real `divsd` too, so it never exercised the bug).
                f64::from_bits(0xfff8000000000000)
            };
        }
        if ax < 0x3e200000 {
            // |x| < 2^-29
            if ax < 0x3c900000 {
                // |x| < 2^-54
                return x;
            }
            // Fused on the disassembly (`vfnmadd231sd`): `x - (x*x)*0.5` as
            // one rounding, not `x*x*0.5` computed first.
            return (x * x).mul_add(-0.5, x);
        }
        if hx > 0 || hx <= 0xbfd2bec3u32 as i32 {
            // -0.2929 < x < 0.41422: direct, no reduction needed.
            k = 0;
            hu = 1;
            let f0 = x;
            return log1p_tail(f0, hu, k, c);
        }
    } else if ax >= 0x7ff00000 {
        return x + x;
    }

    // k != 0: reduce `1 + x` to `2^k * (1 + f)`.
    let u: f64;
    if hx < 0x43400000 {
        let uu = 1.0 + x;
        let huw = (uu.to_bits() >> 32) as i32;
        k = (huw >> 20) - 1023;
        c = if k > 0 {
            1.0 - (uu - x)
        } else {
            x - (uu - 1.0)
        };
        c /= uu;
        u = uu;
        hu = huw as u32;
    } else {
        u = x;
        hu = (u.to_bits() >> 32) as u32;
        k = (hu as i32 >> 20) - 1023;
        c = 0.0;
    }
    hu &= 0x000fffff;
    let low32 = u.to_bits() & 0xffff_ffff;
    let u_norm;
    if hu < 0x6a09e {
        u_norm = f64::from_bits(((hu | 0x3ff00000) as u64) << 32 | low32);
    } else {
        k += 1;
        u_norm = f64::from_bits(((hu | 0x3fe00000) as u64) << 32 | low32);
        hu = (0x00100000u32.wrapping_sub(hu)) >> 2;
    }
    let f0 = u_norm - 1.0;
    log1p_tail(f0, hu, k, c)
}

/// The tail shared by every reduction path: `hfsq`, the tiny-`|f|` shortcut,
/// and the `Lp[]` polynomial.
#[inline]
fn log1p_tail(f: f64, hu: u32, k: i32, c: f64) -> f64 {
    let kf = k as f64;
    let hfsq = 0.5 * f * f;
    if hu == 0 {
        // |f| < 2^-20. `(1.0 - 2/3 * f)` is one fused `vfnmadd231sd` on the
        // live disassembly, not two roundings.
        if f == 0.0 {
            return if k == 0 {
                0.0
            } else {
                kf.mul_add(LN2_HI, c + kf * LN2_LO)
            };
        }
        let r = f.mul_add(-0.66666666666666666, 1.0) * hfsq;
        return if k == 0 {
            f - r
        } else {
            kf * LN2_HI - ((r - (kf * LN2_LO + c)) - f)
        };
    }
    // The `Lp[]` evaluation is a genuine FMA chain on the disassembly, not
    // the flat sum `e_log1p.c`'s own grouping shows: `R2`/`R3`/`R4` are each
    // one `fma`, `z2*R2` is a plain product formed before the chain starts,
    // and `R1 + z2*R2 + z4*R3 + z6*R4` is folded as
    // `fma(z6, R4, fma(z4, R3, fma(z, Lp1, z2*R2)))` — algebraically the
    // same polynomial (Horner in `z2`), but not the same roundings.
    let s = f / (2.0 + f);
    let z = s * s;
    let r2 = z.mul_add(LP[2], LP[1]);
    let z2 = z * z;
    let r2z2 = r2 * z2;
    let z4 = z2 * z2;
    let r3 = z.mul_add(LP[4], LP[3]);
    let z6 = z4 * z2;
    let r4 = z.mul_add(LP[6], LP[5]);
    let acc0 = z.mul_add(LP[0], r2z2);
    let acc1 = z4.mul_add(r3, acc0);
    let r = z6.mul_add(r4, acc1);
    if k == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        // `kf*LN2_HI - (...)` is one fused `vfmsub231sd`; `kf*LN2_LO + c` is
        // one fused `vfmadd231sd`. Both matter: this is the branch that
        // dominates large-`|x|` inputs.
        kf.mul_add(
            LN2_HI,
            -((hfsq - (s * (hfsq + r) + kf.mul_add(LN2_LO, c))) - f),
        )
    }
}
