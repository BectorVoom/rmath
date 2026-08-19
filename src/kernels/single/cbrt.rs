//! Cube root, single precision.
//!
//! Two policies, two shapes:
//!
//! * **`BitExact`** delegates to the platform's `cbrtf` one lane at a time,
//!   exactly as before this kernel existed: the platform routine is cheap
//!   enough that widening into the double-precision `cbrt` kernel measured at
//!   half its speed, so parity is the honest ceiling there.
//!
//! * **`Fast`** is native: a bit-pattern seed for `|x|^(-1/3)`, three Newton
//!   steps run in widened `f64` lanes — where quadratic convergence is not
//!   blunted by `f32` rounding — and one narrowing at the end. No table, no
//!   division, and the sign is restored with a `copysign` since the cube root
//!   is odd.
//!
//! The seed constant is the classic exponent-field trick: for
//! `r = x^(-1/3)`, `bits(r) ≈ K - bits(x)/3` with
//! `K ≈ (4/3)(127 - σ) 2^23`. The value below was found by exhaustive search
//! of the worst relative error over one period `x ∈ [1, 8)`, giving a seed
//! within 0.0343 everywhere; three Newton steps square that to about `2^-38`,
//! two orders below the narrowing rounding.
//!
//! `Finite` means a *normal* `x` (of either sign): zero, subnormals,
//! infinities and NaN are wrong under that policy, not merely imprecise.

use crate::kernels::not_normal;
use crate::policy::{Accuracy, Domain};
use crate::reference::single as reference;
use crate::simd::{Lanes, Simd, map_lanes, patch_lanes};

/// `cbrt(x)` for a vector of `f32` lanes.
#[inline(always)]
pub fn eval<V: Simd<Elem = f32>, A: Accuracy, D: Domain>(x: V) -> V {
    if A::BIT_EXACT {
        return map_lanes(x, reference::cbrt);
    }
    let y = fast(x);
    if D::CHECKED {
        patch_lanes(x, y, not_normal(x), reference::cbrt)
    } else {
        y
    }
}

/// The native path.
///
/// Measured error: below 1 ulp over the normals — the three `f64` Newton
/// steps leave a residual near `2^-38`, so what remains is essentially the
/// single narrowing at the end.
#[inline(always)]
fn fast<V: Simd<Elem = f32>>(x: V) -> V {
    type W<V> = <V as Simd>::Wide;
    /// The seed for `|x|^(-1/3)`; see the module docs for its derivation.
    const K: u32 = 0x54a2_329d;

    let ax = x.abs();
    let bits = ax.to_bits();
    let mut seed = V::Bits::filled_default();
    for i in 0..V::LANES {
        // `/ 3` compiles to a multiply-high; the loop stays branch-free.
        seed.as_mut_slice()[i] = K.wrapping_sub(bits.as_slice()[i] / 3);
    }
    let xd = ax.widen();
    let four = W::<V>::splat(4.0);
    let third = W::<V>::splat(1.0 / 3.0);

    // Newton on f(r) = r^-3 - x: r <- r (4 - x r³) / 3, division-free and
    // quadratic. Seed error 0.0343 → 1.2e-3 → 1.5e-6 → 2.2e-12.
    let r = V::from_bits(seed).widen();
    let r = r * ((four - xd * (r * (r * r))) * third);
    let r = r * ((four - xd * (r * (r * r))) * third);
    let r = r * ((four - xd * (r * (r * r))) * third);

    // cbrt(x) = x r², narrowed once, with the argument's sign restored.
    V::narrow(xd * (r * r)).copysign(x)
}
