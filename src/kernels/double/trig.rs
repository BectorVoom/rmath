//! Sine, cosine and tangent.
//!
//! # `BitExact`
//!
//! A genuine vector replay of glibc's `__sin_fma`/`__cos_fma`/`__sincos`/
//! `__tan_fma` (the IBM Accurate Mathematical Library), for `|x| <
//! 105414350` (`tan`: `|x| <= 1e8`) — see this module's `bit_exact`
//! submodule for the schedule, which was read out of a disassembly
//! (`objdump -d`, glibc 2.43, x86-64), not inferred from the C source:
//! several steps fuse (or specifically do *not* fuse) in a way the source's
//! own expression grouping does not predict, verified against the platform
//! over tens of millions of inputs before this was vectorised (see
//! [`crate::reference::double::trig`]). Past that magnitude — glibc's own
//! `__branred` Payne-Hanek reduction, not ported — `FullRange` repairs the
//! lanes via `patch_lanes`; `Finite` returns a wrong number for them, the
//! same contract every other kernel here makes.
//!
//! # `Fast`
//!
//! A genuine vector path, and the reason to reach for this configuration:
//! Cody-Waite reduction against a three-part `pi/2`, one degree-7 polynomial
//! in `r^2` for each of sine and cosine, and quadrant selection done with
//! arithmetic rather than a branch. No table, no gather, no per-lane work at
//! all.
//!
//! Both polynomials are always evaluated, even when only one result is wanted.
//! That is deliberate: the quadrant decides *which* of the two is the answer,
//! so selecting between them costs one blend, whereas branching on the
//! quadrant would serialise the lanes and lose the whole point.
//!
//! `Finite` means `|x| < TRIG_LIMIT` (about 1.6e6). Past that the quadrant
//! count no longer multiplies exactly against the split `pi/2` and the reduced
//! argument loses its leading bits; `FullRange` sends those lanes to the
//! reference, `Finite` returns a wrong number.

use crate::kernels::{dispatch, horner, outside};
use crate::policy::{Accuracy, Domain};
use crate::reference::double as reference;
use crate::simd::{Lanes, Mask, Simd};
use crate::tables::double::poly as p;

/// Lanes the Cody-Waite reduction must not be trusted with.
#[inline(always)]
fn too_large<V: Simd<Elem = f64>>(x: V) -> V::Mask {
    outside(x, p::TRIG_LIMIT)
}

/// `x` reduced to `(r, q)`: `x = r + q * pi/2` with `|r| <= pi/4` and
/// `q` the quadrant index in `0..4`, as a float.
///
/// `q` is computed as `n - floor(n/4) * 4` rather than by converting to an
/// integer and masking. Both are exact for every `n` in range, but this one is
/// three vector operations and no round trip through the lane array — and the
/// round trip is what the whole `Fast` policy exists to avoid.
#[inline(always)]
fn reduce<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let n = (x * V::splat(p::TWO_OVER_PI)).round_ties_even();
    // Three fused steps: each `n * PIO2[i]` is exact, so the only error is the
    // truncation of pi/2 itself, some 2^-150.
    let r = n.mul_add(
        V::splat(-p::PIO2[2]),
        n.mul_add(V::splat(-p::PIO2[1]), n.mul_add(V::splat(-p::PIO2[0]), x)),
    );
    let q = n - (n * V::splat(0.25)).floor() * V::splat(4.0);
    (r, q)
}

/// `(sin(r), cos(r))` on `|r| <= pi/4`.
#[inline(always)]
fn kernels<V: Simd<Elem = f64>>(r: V) -> (V, V) {
    let s = r * r;
    (r * horner(s, &p::SIN), horner(s, &p::COS))
}

/// Apply the quadrant to `(sin r, cos r)`, giving `(sin x, cos x)`.
#[inline(always)]
fn place<V: Simd<Elem = f64>>(q: V, sr: V, cr: V) -> (V, V) {
    let one = V::splat(1.0);
    let two = V::splat(2.0);
    // Odd quadrants swap the two kernels; that is the same test for both.
    let swap = q.eq_mask(one).or(q.eq_mask(V::splat(3.0)));
    // sin is negative in quadrants 2 and 3; cos in 1 and 2.
    let sin_neg = q.ge_mask(two);
    let cos_neg = q.eq_mask(one).or(q.eq_mask(two));

    let sin = V::select(swap, cr, sr);
    let cos = V::select(swap, sr, cr);
    let flip = |v: V, m| V::select(m, -v, v);
    (flip(sin, sin_neg), flip(cos, cos_neg))
}

/// `sin(x)` and `cos(x)` together, vectorised.
#[inline(always)]
fn both<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let (r, q) = reduce(x);
    let (sr, cr) = kernels(r);
    place(q, sr, cr)
}

/// A genuine vector replay of glibc's `__sin_fma`/`__cos_fma`/`__sincos`, for
/// `|x| < 105414350`.
///
/// Mirrors [`crate::reference::double::trig`] exactly — same functions, same
/// FMA placement, same three-separate-entry-points shape (see that module's
/// doc for why `sin`'s and `sincos`'s mid-band are not the same computation).
/// The one thing that is genuinely per-lane here is the table gather
/// (`sincos_table_lookup`) and the quadrant/band bookkeeping that needs a
/// bit pattern's high word, exactly the shape every other `BitExact` table
/// kernel in this crate already has; the polynomials, the reduction and the
/// four-band blend are real packed arithmetic.
mod bit_exact {
    use super::*;
    use crate::tables::double::trig as t;

    /// The four table entries `[sn, ssn, cs, ccs]` per lane, for the rounded
    /// index carried in each lane's own bits.
    #[inline(always)]
    fn sincos_table_lookup<V: Simd<Elem = f64>>(u: V) -> (V, V, V, V) {
        let bits = u.to_bits();
        let mut sn = V::Bits::filled_default();
        let mut ssn = V::Bits::filled_default();
        let mut cs = V::Bits::filled_default();
        let mut ccs = V::Bits::filled_default();
        for i in 0..V::LANES {
            let low32 = bits.as_slice()[i] as u32 as i32;
            // For every lane this function is meant to answer for (`|a| <=
            // pi/4` reduced arguments, or `|x| < 0.855469` in the poly
            // band), this index is already small. But every band's formula
            // runs on every lane unconditionally — that is the whole point
            // of blending instead of branching — so a lane whose *result*
            // will be discarded (the tiny/mid/table blend, or a lane bound
            // for `patch_lanes`) can still carry an adversarial `x` here
            // (NaN, infinity, far outside `TABLE_LIMIT`) and must not panic
            // even though its answer is thrown away. `% 110` is defensive
            // for exactly that case, not part of the algorithm.
            let k = ((low32 as u32 as usize) % 110) * 4;
            sn.as_mut_slice()[i] = t::TAB[k];
            ssn.as_mut_slice()[i] = t::TAB[k + 1];
            cs.as_mut_slice()[i] = t::TAB[k + 2];
            ccs.as_mut_slice()[i] = t::TAB[k + 3];
        }
        (
            V::from_bits(sn),
            V::from_bits(ssn),
            V::from_bits(cs),
            V::from_bits(ccs),
        )
    }

    #[inline(always)]
    fn cos_inner<V: Simd<Elem = f64>>(xx: V) -> V {
        let inner = xx.mul_add(V::splat(t::CS6), V::splat(t::CS4));
        xx.mul_add(inner, V::splat(t::CS2))
    }

    #[inline(always)]
    fn sin_inner<V: Simd<Elem = f64>>(xx: V) -> V {
        xx.mul_add(V::splat(t::SN5), V::splat(t::SN3))
    }

    /// `TAYLOR_SIN`, vectorised — see the scalar reference's doc for the
    /// exact fusion. Unconditional here: `do_sin` blends this against the
    /// main path rather than branching, so both are always evaluated.
    #[inline(always)]
    fn taylor_sin<V: Simd<Elem = f64>>(xx: V, x: V, dx: V) -> V {
        let poly = xx.mul_add(V::splat(t::S5), V::splat(t::S4));
        let poly = xx.mul_add(poly, V::splat(t::S3));
        let poly = xx.mul_add(poly, V::splat(t::S2));
        let poly = xx.mul_add(poly, V::splat(t::S1));
        let half_dx = dx * V::splat(0.5);
        let inner = poly.mul_add(x, -half_dx);
        let t_val = xx.mul_add(inner, dx);
        x + t_val
    }

    /// `do_cos`, vectorised. See the scalar reference's doc for the fusion.
    #[inline(always)]
    fn do_cos<V: Simd<Elem = f64>>(x: V, dx: V) -> V {
        let dx = V::select(x.lt_mask(V::splat(0.0)), -dx, dx);
        let u = V::splat(t::BIG) + x.abs();
        let x = x.abs() - (u - V::splat(t::BIG)) + dx;

        let xx = x * x;
        let s = (x * xx).mul_add(sin_inner(xx), x);
        let c = xx * cos_inner(xx);
        let (sn, ssn, cs, ccs) = sincos_table_lookup(u);
        let step1 = s.mul_add(-ssn, ccs);
        let step2 = c.mul_add(-cs, step1);
        let cor = s.mul_add(-sn, step2);
        cs + cor
    }

    /// `do_sin`, vectorised: the `|x| < 0.126` `TAYLOR_SIN` branch and the
    /// main table path are both evaluated and blended, matching every other
    /// dual-path kernel in this crate.
    #[inline(always)]
    fn do_sin<V: Simd<Elem = f64>>(x: V, dx: V) -> V {
        let xold = x;
        let is_small = x.abs().lt_mask(V::splat(0.126));

        let dx_main = V::select(x.le_mask(V::splat(0.0)), -dx, dx);
        let u = V::splat(t::BIG) + x.abs();
        let xr = x.abs() - (u - V::splat(t::BIG));

        let xxr = xr * xr;
        let s = xr + (xr * xxr).mul_add(sin_inner(xxr), dx_main);
        let c = xr.mul_add(dx_main, xxr * cos_inner(xxr));
        let (sn, ssn, cs, ccs) = sincos_table_lookup(u);
        let step1 = s.mul_add(ccs, ssn);
        let step2 = c.mul_add(-sn, step1);
        let cor = s.mul_add(cs, step2);
        let main = (sn + cor).copysign(xold);

        let small = taylor_sin(x * x, x, dx);
        V::select(is_small, small, main)
    }

    /// `reduce_sincos`, vectorised: `(a, da, n_bit0, n_bit1)` with the last
    /// two as `0.0`/`1.0` flags (`Simd` has no packed integer arithmetic —
    /// see `crate::kernels::exact`'s module doc — so quadrant bits become
    /// float flags a caller turns into masks with `eq_mask`, rather than
    /// staying an integer this trait cannot carry).
    #[inline(always)]
    fn reduce_sincos<V: Simd<Elem = f64>>(x: V) -> (V, V, V, V) {
        let t_val = x.mul_add(V::splat(t::HPINV), V::splat(t::TOINT));
        let xn = t_val - V::splat(t::TOINT);
        let y = xn.mul_add(-V::splat(t::MP1), x);
        let y = xn.mul_add(-V::splat(t::MP2), y);

        let bits = t_val.to_bits();
        let mut bit0 = V::Floats::filled_default();
        let mut bit1 = V::Floats::filled_default();
        for i in 0..V::LANES {
            let n = bits.as_slice()[i] as u32 as i32 & 3;
            bit0.as_mut_slice()[i] = (n & 1) as f64;
            bit1.as_mut_slice()[i] = ((n >> 1) & 1) as f64;
        }
        let n_bit0 = V::from_array(bit0);
        let n_bit1 = V::from_array(bit1);

        let t2 = xn.mul_add(-V::splat(t::PP3), y);
        let w1 = y - t2;
        let db = xn.mul_add(-V::splat(t::PP3), w1);

        let b = xn.mul_add(-V::splat(t::PP4), t2);
        let w2 = t2 - b;
        let tail = xn.mul_add(-V::splat(t::PP4), w2);
        let da = tail + db;

        (b, da, n_bit0, n_bit1)
    }

    /// Lanes past `__branred`'s domain: `|x| >= 105414350`.
    #[inline(always)]
    pub(super) fn needs_branred<V: Simd<Elem = f64>>(x: V) -> V::Mask {
        outside(x, TABLE_LIMIT)
    }

    /// `0x1.9921fb0000000p26`: the exact value whose high 32 bits are
    /// `0x419921FB`, matching `__sin`/`__cos`'s bit-level (not floating-point)
    /// boundary test — see `crate::reference::double::trig`'s `k` comparisons.
    const TABLE_LIMIT: f64 = f64::from_bits(0x419921fb00000000);

    /// Lanes past `__tan`'s second reduction's domain: `|x| >= 1e8`. glibc's
    /// own test is `w > g5`, so the `|x| == 1e8` lane is patched too — a
    /// no-op in value, since there the reference and this kernel run the same
    /// schedule — and `outside`'s NaN-catching negation makes the patch the
    /// specials handler as well, exactly as it is for `sin`/`cos`.
    #[inline(always)]
    pub(super) fn tan_needs_branred<V: Simd<Elem = f64>>(x: V) -> V::Mask {
        outside(x, t::G5)
    }

    /// The fused table index `(256*w - 15.5) as i32`, per lane, truncated
    /// exactly as `cvttsd2si` does, and clamped `% 186` defensively — a lane
    /// whose answer is blended away or patched can still carry an adversarial
    /// `x` here, and the gather must not index out of bounds even though its
    /// result is thrown away (the same defence as `sincos_table_lookup`'s
    /// `% 110`).
    #[inline(always)]
    fn xfg_index<V: Simd<Elem = f64>>(w: V) -> V::Bits {
        let f = w.mul_add(V::splat(256.0), V::splat(t::MFFTNHF));
        let bits = f.to_bits();
        let mut idx = V::Bits::filled_default();
        for i in 0..V::LANES {
            let j = f64::from_bits(bits.as_slice()[i]) as i32;
            idx.as_mut_slice()[i] = (j as usize % 186) as u64;
        }
        idx
    }

    /// `xfg[i][0..3] = [xi, Fi, Gi]`, gathered per lane.
    #[inline(always)]
    fn xfg_lookup<V: Simd<Elem = f64>>(idx: V::Bits) -> (V, V, V) {
        let mut xi = V::Bits::filled_default();
        let mut fi = V::Bits::filled_default();
        let mut gi = V::Bits::filled_default();
        for i in 0..V::LANES {
            let j = (idx.as_slice()[i] as usize) * 3;
            xi.as_mut_slice()[i] = t::XFG[j];
            fi.as_mut_slice()[i] = t::XFG[j + 1];
            gi.as_mut_slice()[i] = t::XFG[j + 2];
        }
        (
            V::from_bits(xi),
            V::from_bits(fi),
            V::from_bits(gi),
        )
    }

    /// `__tan`'s (IV)/(V) common tail, vectorised: `tan` (or `-cot`) of the
    /// reduced argument `a + da`, with quadrant parity `n` as a `0.0`/`1.0`
    /// flag. Every step runs on every lane and is blended by the parity and
    /// `gy2` masks, exactly as the scalar reference's branch flow does —
    /// including the odd polynomial's use of the *signed* `a`/`da` and
    /// `sy`'s `+-1.0` only in the table bands.
    #[inline(always)]
    fn tan_sub<V: Simd<Elem = f64>>(n: V, a: V, da: V) -> V {
        let neg = a.lt_mask(V::splat(0.0));
        let ya = V::select(neg, -a, a);
        let yya = V::select(neg, -da, da);
        let sy = V::select(neg, V::splat(-1.0), V::splat(1.0));
        let is_odd = n.eq_mask(V::splat(1.0));

        let a2 = a * a;
        let t2 = a2.mul_add(V::splat(t::D11), V::splat(t::D9));
        let t2 = a2.mul_add(t2, V::splat(t::D7));
        let t2 = a2.mul_add(t2, V::splat(t::D5));
        let t2 = a2.mul_add(t2, V::splat(t::D3));
        let t2 = (a * a2).mul_add(t2, da);
        let y = a + t2;
        let b = y;
        // EADD(a, t2, b, db): the strict `>` selects the first branch, the
        // equal/unordered case the second — the same mapping as the
        // disassembly's `jbe`.
        let db = V::select(
            a.abs().gt_mask(t2.abs()),
            (a - b) + t2,
            (t2 - b) + a,
        );
        // DIV2(1.0, 0.0, b, db, ...) — `-cot`. The `+ 0.0` is the `xx` term.
        let c = V::splat(1.0) / b;
        let u = c * b;
        let uu = c.mul_add(b, -u);
        let t3 = (V::splat(1.0) - u - uu) + V::splat(0.0);
        let cc = (-db).mul_add(c, t3) / b;
        let z = c + cc;
        let zz = (c - z) + cc;
        let cot_val = -(z + zz);

        let (xi, fi, gi) = xfg_lookup(xfg_index(ya));
        let z = (ya - xi) + yya;
        let z2 = z * z;
        let pz = (z * z2).mul_add(z2.mul_add(V::splat(t::E1), V::splat(t::E0)), z);
        let cot_tab = {
            let t2 = pz * (fi + gi) / (fi + pz);
            -sy * (gi - t2)
        };
        let tan_tab = {
            let t2 = pz * (gi + fi) / (gi - pz);
            sy * (fi + t2)
        };
        let table_val = V::select(is_odd, cot_tab, tan_tab);

        let poly_val = V::select(is_odd, cot_val, y);
        V::select(ya.le_mask(V::splat(t::GY2)), poly_val, table_val)
    }

    /// The (IV) reduction — algorithm i, the three-part `mp` split.
    #[inline(always)]
    fn reduce_mp<V: Simd<Elem = f64>>(x: V) -> (V, V, V) {
        let t = x.mul_add(V::splat(t::HPINV), V::splat(t::TOINT));
        let xn = t - V::splat(t::TOINT);
        let t1 = xn.mul_add(-V::splat(t::MP1), x);
        let t1 = xn.mul_add(-V::splat(t::MP2), t1);
        let a = xn.mul_add(-V::splat(t::MP3), t1);
        let da = xn.mul_add(-V::splat(t::MP3), t1 - a);
        (t, a, da)
    }

    /// The (V) reduction — algorithm ii, the four-part `pp` split.
    #[inline(always)]
    fn reduce_pp<V: Simd<Elem = f64>>(x: V) -> (V, V, V) {
        let t = x.mul_add(V::splat(t::HPINV), V::splat(t::TOINT));
        let xn = t - V::splat(t::TOINT);
        let t1 = xn.mul_add(-V::splat(t::MP1), x);
        let t1 = xn.mul_add(-V::splat(t::MP2), t1);
        let a = xn.mul_add(-V::splat(t::PP3), t1);
        let da = xn.mul_add(-V::splat(t::PP3), t1 - a);
        let b = xn.mul_add(-V::splat(t::PP4), a);
        let db = xn.mul_add(-V::splat(t::PP4), a - b);
        let da = db + da;
        let sum = b + da;
        let cc = V::select(
            b.abs().gt_mask(da.abs()),
            (b - sum) + da,
            (da - sum) + b,
        );
        (t, sum, cc)
    }

    /// `n = t.i[LOW_HALF] & 1` as a `0.0`/`1.0` float flag.
    #[inline(always)]
    fn tan_parity<V: Simd<Elem = f64>>(t: V) -> V {
        let bits = t.to_bits();
        let mut flag = V::Floats::filled_default();
        for i in 0..V::LANES {
            flag.as_mut_slice()[i] = (bits.as_slice()[i] & 1) as f64;
        }
        V::from_array(flag)
    }

    /// `tan(x)` for `|x| <= 1e8`; callers repair the rest.
    #[inline(always)]
    pub(super) fn tan<V: Simd<Elem = f64>>(x: V) -> V {
        let w = x.abs();
        let band1 = w.le_mask(V::splat(t::G1));
        let band2 = w.le_mask(V::splat(t::G2));
        let band3 = w.le_mask(V::splat(t::G3));
        let band4 = w.le_mask(V::splat(t::G4));

        // (II): polynomial I, the direct Taylor series in `x`.
        let poly_val = {
            let x2 = x * x;
            let t2 = x2.mul_add(V::splat(t::D11), V::splat(t::D9));
            let t2 = x2.mul_add(t2, V::splat(t::D7));
            let t2 = x2.mul_add(t2, V::splat(t::D5));
            let t2 = x2.mul_add(t2, V::splat(t::D3));
            (x * x2).mul_add(t2, x)
        };

        // (III): the `w`-indexed table.
        let table_w = {
            let (xi, fi, gi) = xfg_lookup(xfg_index(w));
            let z = w - xi;
            let z2 = z * z;
            let sy = V::select(x.lt_mask(V::splat(0.0)), V::splat(-1.0), V::splat(1.0));
            let pz = (z * z2).mul_add(z2.mul_add(V::splat(t::E1), V::splat(t::E0)), z);
            let t2 = pz * (gi + fi) / (gi - pz);
            sy * (fi + t2)
        };

        // (IV)/(V): the two reductions, then the shared tail. For a lane in
        // the wrong band either value is garbage — the blend discards it.
        let (t4, a4, da4) = reduce_mp(x);
        let reduced_val = tan_sub(tan_parity(t4), a4, da4);
        let (t5, a5, da5) = reduce_pp(x);
        let reduced_pp_val = tan_sub(tan_parity(t5), a5, da5);

        let r = V::select(band4, reduced_val, reduced_pp_val);
        let r = V::select(band3, table_w, r);
        let r = V::select(band2, poly_val, r);
        V::select(band1, x, r)
    }

    /// `sin(x)` for `|x| < 105414350`; callers repair the rest.
    #[inline(always)]
    pub(super) fn sin<V: Simd<Elem = f64>>(x: V) -> V {
        let tiny = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x3e500000_00000000)));
        let poly = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x3feb6000_00000000)));
        let mid = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x400368fd_00000000)));

        let poly_val = do_sin(x, V::splat(0.0));

        let t = V::splat(t::HP0) - x.abs();
        let mid_val = do_cos(t, V::splat(t::HP1)).copysign(x);

        let (a, da, n_bit0, n_bit1) = reduce_sincos(x);
        let table_val = {
            let is_odd = n_bit0.eq_mask(V::splat(1.0));
            let flip = n_bit1.eq_mask(V::splat(1.0));
            let r = V::select(is_odd, do_cos(a, da), do_sin(a, da));
            V::select(flip, -r, r)
        };

        let r = V::select(mid, mid_val, table_val);
        let r = V::select(poly, poly_val, r);
        V::select(tiny, x, r)
    }

    /// `cos(x)` for `|x| < 105414350`; callers repair the rest.
    #[inline(always)]
    pub(super) fn cos<V: Simd<Elem = f64>>(x: V) -> V {
        let tiny = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x3e400000_00000000)));
        let poly = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x3feb6000_00000000)));
        let mid = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x400368fd_00000000)));

        let poly_val = do_cos(x, V::splat(0.0));

        let y = V::splat(t::HP0) - x.abs();
        let a = y + V::splat(t::HP1);
        let da = (y - a) + V::splat(t::HP1);
        let mid_val = do_sin(a, da);

        let (a, da, n_bit0, n_bit1) = reduce_sincos(x);
        let table_val = {
            // `n + 1`: bit0 flips, bit1 flips only when bit0 was already set
            // (the carry out of `+1` on a 2-bit field) — i.e. `n_bit1 ^= n_bit0`.
            let new_bit1 = V::select(
                n_bit0.eq_mask(V::splat(1.0)),
                V::splat(1.0) - n_bit1,
                n_bit1,
            );
            let new_bit0 = V::splat(1.0) - n_bit0;
            let is_odd = new_bit0.eq_mask(V::splat(1.0));
            let flip = new_bit1.eq_mask(V::splat(1.0));
            let r = V::select(is_odd, do_cos(a, da), do_sin(a, da));
            V::select(flip, -r, r)
        };

        let r = V::select(mid, mid_val, table_val);
        let r = V::select(poly, poly_val, r);
        V::select(tiny, V::splat(1.0), r)
    }

    /// `(sin(x), cos(x))` for `|x| < 105414350`; callers repair the rest.
    #[inline(always)]
    pub(super) fn sincos<V: Simd<Elem = f64>>(x: V) -> (V, V) {
        let tiny = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x3e400000_00000000)));
        let poly = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x3feb6000_00000000)));
        let mid = x
            .abs()
            .lt_mask(V::splat(f64::from_bits(0x400368fd_00000000)));

        let poly_sin = do_sin(x, V::splat(0.0));
        let poly_cos = do_cos(x, V::splat(0.0));

        let y = V::splat(t::HP0) - x.abs();
        let a_mid = y + V::splat(t::HP1);
        let da_mid = (y - a_mid) + V::splat(t::HP1);
        let mid_sin = do_cos(a_mid, da_mid).copysign(x);
        let mid_cos = do_sin(a_mid, da_mid);

        let (a, da, n_bit0, n_bit1) = reduce_sincos(x);
        // `if (n & 3) == 1 || (n & 3) == 2 { a = -a; da = -da; }`
        let neg_ad = n_bit0
            .eq_mask(V::splat(1.0))
            .and(n_bit1.eq_mask(V::splat(0.0)))
            .or(n_bit0
                .eq_mask(V::splat(0.0))
                .and(n_bit1.eq_mask(V::splat(1.0))));
        let a = V::select(neg_ad, -a, a);
        let da = V::select(neg_ad, -da, da);
        let sinx = do_sin(a, da);
        let xx = do_cos(a, da);
        let cosx = V::select(n_bit1.eq_mask(V::splat(1.0)), -xx, xx);
        let is_odd = n_bit0.eq_mask(V::splat(1.0));
        let table_sin = V::select(is_odd, cosx, sinx);
        let table_cos = V::select(is_odd, sinx, cosx);

        let rs = V::select(mid, mid_sin, table_sin);
        let rs = V::select(poly, poly_sin, rs);
        let rs = V::select(tiny, x, rs);

        let rc = V::select(mid, mid_cos, table_cos);
        let rc = V::select(poly, poly_cos, rc);
        let rc = V::select(tiny, V::splat(1.0), rc);

        (rs, rc)
    }
}

/// Sine.
pub mod sin {
    use super::*;

    /// `sin(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact::sin(x);
            return if D::CHECKED {
                crate::simd::patch_lanes(x, y, bit_exact::needs_branred(x), reference::sin)
            } else {
                y
            };
        }
        dispatch::<V, A, D>(x, reference::sin, |x| both(x).0, too_large)
    }
}

/// Cosine.
pub mod cos {
    use super::*;

    /// `cos(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact::cos(x);
            return if D::CHECKED {
                crate::simd::patch_lanes(x, y, bit_exact::needs_branred(x), reference::cos)
            } else {
                y
            };
        }
        dispatch::<V, A, D>(x, reference::cos, |x| both(x).1, too_large)
    }
}

/// Sine and cosine of the same argument.
///
/// The reason this exists as its own function rather than two calls: the
/// argument reduction and both polynomials are already shared, so the pair
/// costs one blend more than either alone. Calling `sin` and `cos` separately
/// does all of that work twice.
pub mod sincos {
    use super::*;

    /// `(sin(x), cos(x))` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> (V, V) {
        if A::BIT_EXACT {
            let (s, c) = bit_exact::sincos(x);
            if !D::CHECKED {
                return (s, c);
            }
            let bad = bit_exact::needs_branred(x);
            return crate::simd::patch_lanes_pair(x, (s, c), bad, reference::sincos);
        }
        let (s, c) = both(x);
        if !D::CHECKED {
            return (s, c);
        }
        let bad = too_large(x);
        crate::simd::patch_lanes_pair(x, (s, c), bad, reference::sincos)
    }
}

/// Tangent.
///
/// Two paths, one per policy:
///
/// - `BitExact` replays glibc's `__tan_fma` directly (its own six bands and
///   the `xfg` table — see [`crate::reference::double::trig`]'s `tan`), for
///   `|x| <= 1e8`; past that — glibc's `__branred` Payne-Hanek reduction, not
///   ported — `FullRange` repairs the lanes via `patch_lanes`.
/// - `Fast` forms `tan` as a ratio of the same two kernels as `sin`/`cos`,
///   which costs a division and is still the better trade: a direct `tan`
///   polynomial needs a much higher degree to hold its accuracy as `r`
///   approaches `pi/4`, where `tan` is steep, and the ratio keeps the
///   relative error bounded across the whole quadrant including near the
///   pole, which is where a `tan` caller usually is.
pub mod tan {
    use super::*;

    /// `tan(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if A::BIT_EXACT {
            let y = bit_exact::tan(x);
            return if D::CHECKED {
                crate::simd::patch_lanes(x, y, bit_exact::tan_needs_branred(x), reference::tan)
            } else {
                y
            };
        }
        dispatch::<V, A, D>(x, reference::tan, fast, too_large)
    }

    #[inline(always)]
    fn fast<V: Simd<Elem = f64>>(x: V) -> V {
        let (r, q) = reduce(x);
        let (sr, cr) = kernels(r);
        // Odd quadrants give the cotangent, negated.
        let odd = q.eq_mask(V::splat(1.0)).or(q.eq_mask(V::splat(3.0)));
        let num = V::select(odd, -cr, sr);
        let den = V::select(odd, sr, cr);
        num / den
    }
}
