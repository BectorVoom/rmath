//! The Bessel functions of the first and second kind, orders 0, 1 and `n`.
//!
//! # What vectorises here
//!
//! All of it, under both policies — which was not true until
//! recently, and the two reasons it was not are worth keeping.
//!
//! Below `|x| = 2` each of `j0`, `j1`, `y0`, `y1` is a rational function of
//! `x^2` — pure vector arithmetic. At or above 2 they are
//! `sqrt(2/(pi x)) (P cos(x0) -+ Q sin(x0))`, and that splits into two parts
//! with very different characters:
//!
//! * `P` and `Q` are rational functions of `1/x^2` with one of four
//!   coefficient sets chosen by interval. Four rows is few enough that there
//!   is nothing to gather: `pick` is a three-`select` blend ladder against
//!   broadcast constants, which beats a per-lane gather of the twenty-three
//!   coefficients `pq0` and `pq1` want between them.
//! * `sin x`, `cos x` and `cos 2x` are the platform's, and are the whole cost
//!   of that branch — fdlibm's own routine is about ninety per cent
//!   trigonometry for `|x| >= 2`.
//!
//! Both are vector under both policies. That is a change: while glibc's
//! `__sin`/`__cos` were only available here one lane at a time, `BitExact`
//! handed any vector with a lane at or above 2 to the scalar routine and took
//! parity, because three scalar trigonometric calls per lane is *more* scalar
//! trigonometry than the routine being replaced. Since that schedule was
//! ported and vectorised ([`trig`]), the three calls cost one shared argument
//! reduction and two kernel evaluations plus a second reduction for
//! `cos 2x` — and the branch that used to be given away is now the one
//! carrying the speed-up.
//!
//! Both branches are evaluated and blended rather than one being deferred to
//! a scalar fallback, because neither is rare — a buffer of Bessel arguments
//! can sit entirely on either side of 2 — but each is handed a stand-in
//! argument for the lanes it is not answering for, so a mixed vector does not
//! pay for narrow-band work nobody asked for. See `branch`.
//!
//! # Where the vector stops
//!
//! At `|x| = 52707175`, which is [`trig::safe_limit`] halved (`cos(x + x)`
//! needs the half). Above it the trigonometry would need glibc's `__branred`,
//! which this crate does not reproduce, so the *whole lane* goes to
//! [`crate::reference`] — one `reference::j0` call does the three scalar
//! trigonometric calls that three separate `patch_lanes` would have done, and
//! none of the vector work around them. Under [`crate::policy::Finite`] the
//! caller has vouched for that bound and no test is made.
//!
//! # `jn` and `yn`
//!
//! Vector too, and the run-time trip counts that kept them scalar turn out
//! not to be the obstacle they look like. `jn` chooses between forward
//! recurrence, backward recurrence and a Taylor series on `n` against `x`,
//! and the backward recurrence runs a continued fraction whose *length is
//! decided at run time* by iterating until a convergent exceeds `1e9`. The
//! reading that says a vector must therefore "run the longest lane's loop for
//! every lane" is right about the loop and wrong about the comparison: the
//! scalar routine runs the *sum* of the lanes' trip counts, so the longest is
//! already the cheaper of the two, and it is eight times cheaper on the
//! ordinary case of one order swept across a buffer. A lane whose loop has
//! finished freezes under a mask, the loop stops once every lane has frozen,
//! and each lane's counter is carried per lane so that a short lane costs
//! nothing after it stops. See [`jn`] for the per-branch account.
//!
//! # No fused multiply-adds, and that is deliberate
//!
//! `rational_p`, `rational_q`, `j0_num` and `j0_den` below use no
//! `mul_add` at all — every rational fit
//! is separate `*` and `+`. Confirmed against `__j0_finite` in `libm.so.6`
//! (`objdump -d`, glibc 2.43): zero `vfmadd`/`vfnmadd`/`vfmsub` instructions
//! anywhere in the function, near-origin rational and far asymptotic branch
//! alike. Same reason as [`super::exp2`]: glibc ships no `_fma` ifunc variant
//! of the Bessel functions, so what actually runs on x86-64 is the baseline
//! build, and fusing here would be *more* accurate and would stop matching.

use crate::kernels::double::{ln, trig};
use crate::kernels::mostly_repaired;
use crate::policy::{Accuracy, Domain};
use crate::reference::double::bessel as reference;
use crate::simd::{Lanes, Mask, Simd, map_lanes2, patch_lanes, patch_lanes2};
use crate::tables::double::bessel as t;

/// Above this `P` is 1 and `Q` is 0 to the last bit, and the rational fits
/// are skipped entirely.
///
/// fdlibm writes the test as `ix > 0x48000000` on the *high word*, which is
/// not the same as `x > 2^129`: it is true from the first double whose high
/// word is `0x48000001`. Every threshold in this file is converted that way —
/// `ix > K` becomes `x >= bits((K + 1) << 32)` and `ix <= K` becomes
/// `x < bits((K + 1) << 32)` — because getting it wrong would move a branch
/// boundary by a millionth of a binade, which no sampled test would find.
const NO_PQ: f64 = f64::from_bits(0x4800_0001_0000_0000);

/// `2^28`, above which the `P`/`Q` rational fits collapse to their limits.
const PQ_LIMIT: f64 = f64::from_bits(0x41b0_0000_0000_0000);

/// The largest `|x|` the vector asymptotic branch answers for.
///
/// Every asymptotic branch here wants `sin x`, `cos x` *and* `cos 2x`, so the
/// binding constraint is [`trig::safe_limit`] halved — exact, and the halving
/// is what `cos(x + x)` needs. Above it these kernels hand the *whole lane*
/// to [`crate::reference`] rather than letting three separate `patch_lanes`
/// repair three thirds of an answer: one `reference::j0` call does the same
/// three scalar trigonometric calls the repairs would have done, and none of
/// the vector work around them.
///
/// It is also why [`sc2`] can ask the trigonometry for `Finite`: under
/// `FullRange` no lane that survives to be read is outside this bound, and
/// under `Finite` the caller has vouched for exactly that. It replaces the
/// It replaces an earlier outer test at `2^1022` — the point where fdlibm
/// stops correcting `ss`/`cc` because `x + x` would overflow — which is far
/// past the point where the vector trigonometry stops answering.
#[inline(always)]
fn trig_safe<A: Accuracy>() -> f64 {
    trig::safe_limit::<A>() * 0.5
}

/// Evaluate whichever of the two branches the lanes actually need.
///
/// # Why this is not just a blend
///
/// Both branches are correct wherever they are selected, so a vector could
/// compute both and blend. It should not. The asymptotic branch carries three
/// trigonometric calls and the series branch a logarithm and a nested order-0
/// evaluation; paying for the one the data does not want is what made an
/// earlier version of `y0` *slower* than the scalar routine it replaces.
///
/// # The stand-in arguments
///
/// When the vector really is mixed there is no choice but to run both and
/// blend. The lanes that are about to be discarded still go *through* the
/// other branch, and what they carry there matters, in both directions:
///
/// * A lane below 2 inside the asymptotic branch opens the bit-exact
///   trigonometry's three narrow bands (see [`trig`]'s `bit_exact::sin`) for
///   the whole vector, on an answer nobody will read. It is handed `+-4`
///   instead — outside every guard, same sign.
/// * A lane at or above 2 inside the *series* branch is worse, because
///   `y0` and `y1`'s series contains a nested `j0`/`j1`, which would then
///   take its own asymptotic branch and evaluate the trigonometry a second
///   time. It is handed `+-1`.
///
/// Neither stand-in reaches a result: the blend keeps the branch each lane
/// asked for, exactly as before.
///
/// # Both policies vectorise both branches
///
/// This was not always true and the reason it changed is worth recording.
/// Under [`crate::policy::BitExact`] the asymptotic branch needs `sin x`,
/// `cos x` and `cos 2x` bit-identical to glibc's, which used to mean three
/// scalar calls per lane — *more* scalar trigonometry than the routine being
/// replaced, since fdlibm gets `sin` and `cos` from one shared reduction. So
/// `BitExact` handed any vector with a lane at or above 2 to the scalar
/// routine and settled for parity. Since the IBM Accurate Portable Math
/// `__sin`/`__cos` schedule was ported and vectorised ([`trig`]), that
/// reasoning is obsolete: the three calls are now one shared argument
/// reduction and two kernel evaluations
/// ([`trig::sincos_split`]) plus a second reduction for `cos 2x`, all packed.
/// Both policies take the vector path.
#[inline(always)]
fn branch<V: Simd<Elem = f64>, B: Fn(V) -> V, S: Fn(V) -> V>(x: V, big: B, small: S) -> V {
    let is_big = x.abs().ge_mask(V::splat(2.0));
    if is_big.none() {
        return small(x);
    }
    if is_big.all() {
        return big(x);
    }
    let for_big = V::select(is_big, x, V::splat(4.0).copysign(x));
    let for_small = V::select(is_big, V::splat(1.0).copysign(x), x);
    V::select(is_big, big(for_big), small(for_small))
}

/// `(sin x, cos x, cos 2x)` — the three values every asymptotic branch here
/// is built from, and essentially its whole cost.
///
/// `sin` and `cos` come from [`trig::sincos_split`] rather than
/// [`trig::sincos`] because fdlibm's Bessel routines are bit-exact against
/// the *separate* entry points; see that module for why the two are not the
/// same computation.
#[inline(always)]
fn sc2<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> (V, V, V) {
    use crate::policy::Finite;
    let (s, c) = trig::sincos_split::eval::<V, A, Finite>(x);
    (s, c, trig::cos::eval::<V, A, Finite>(x + x))
}

/// Order 0's `(sin x - cos x, sin x + cos x)`, each computed the way that
/// does not cancel.
///
/// `ss` is `sqrt(2) sin(x - pi/4)` and `cc` is `sqrt(2) cos(x - pi/4)`.
/// Whichever of the two is small is recovered from `-cos(2x)` divided by the
/// other, since their product is exactly `-cos(2x)` — computing both as
/// `sin +- cos` would lose every digit of the smaller one.
#[inline(always)]
fn ss_cc0<V: Simd<Elem = f64>>(s: V, c: V, c2: V) -> (V, V) {
    let ss = s - c;
    let cc = s + c;
    let z = -c2;
    let flip = (s * c).lt_mask(V::splat(0.0));
    (V::select(flip, ss, z / cc), V::select(flip, z / ss, cc))
}

/// Order 1's pair, from the same three values. `x0` is `x - 3pi/4` rather
/// than `x - pi/4`, which exchanges the roles and the signs of the two
/// combinations and leaves `cos 2x` un-negated.
#[inline(always)]
fn ss_cc1<V: Simd<Elem = f64>>(s: V, c: V, c2: V) -> (V, V) {
    let ss0 = -s - c;
    let cc0 = s - c;
    let flip = (s * c).gt_mask(V::splat(0.0));
    (
        V::select(flip, ss0, c2 / cc0),
        V::select(flip, c2 / ss0, cc0),
    )
}

/// `(j0(y), j1(y))` for `y >= 0`, sharing everything the two have in common.
///
/// [`jn`] wants both — as orders 0 and 1 themselves, as the forward
/// recurrence's two seeds, and as the backward recurrence's normaliser — and
/// asking `j0::main` and `j1::main` separately computes `sin y`, `cos y` and
/// `cos 2y` twice over. Those three *are* the asymptotic branch's cost, so
/// sharing them is close to halving it. Identical, lane for lane, to the two
/// separate calls: the `2.0` split, the stand-in argument and both rational
/// fits are the same, only evaluated once.
#[inline(always)]
fn j01<V: Simd<Elem = f64>, A: Accuracy>(y: V) -> (V, V) {
    let is_big = y.ge_mask(V::splat(2.0));
    if is_big.none() {
        return (j0::series(y), j1::series(y, y));
    }
    let all = is_big.all();
    let arg = if all {
        y
    } else {
        V::select(is_big, y, V::splat(4.0))
    };
    let (s, c, c2) = sc2::<V, A>(arg);
    let (ss0, cc0) = ss_cc0(s, c, c2);
    let (ss1, cc1) = ss_cc1(s, c, c2);
    let a = j0::asym(arg, ss0, cc0);
    let b = j1::asym(arg, ss1, cc1);
    if all {
        return (a, b);
    }
    (
        V::select(is_big, a, j0::series(y)),
        V::select(is_big, b, j1::series(y, y)),
    )
}

/// `(y0(x), y1(x))` for `x > 0`, sharing what [`j01`] shares and the
/// logarithm and nested `j0`/`j1` of the series branch besides.
#[inline(always)]
fn y01<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> (V, V) {
    let is_big = x.ge_mask(V::splat(2.0));
    let series = |x: V| {
        let (jj0, jj1) = j01::<V, A>(x);
        let lnx = ln::eval::<V, A, crate::policy::Finite>(x);
        (y0::series_from(x, jj0, lnx), y1::series_from(x, jj1, lnx))
    };
    if is_big.none() {
        return series(x);
    }
    let all = is_big.all();
    let arg = if all {
        x
    } else {
        V::select(is_big, x, V::splat(4.0))
    };
    let (s, c, c2) = sc2::<V, A>(arg);
    let (ss0, cc0) = ss_cc0(s, c, c2);
    let (ss1, cc1) = ss_cc1(s, c, c2);
    let a = y0::asym(arg, ss0, cc0);
    let b = y1::asym(arg, ss1, cc1);
    if all {
        return (a, b);
    }
    // The series branch nests a whole `j0`/`j1`; a lane already answered by
    // the asymptotic branch is given `1.0` so that nesting stays in its own
    // series branch instead of paying for the trigonometry twice. Same
    // stand-in idea as `branch`.
    let (sa, sb) = series(V::select(is_big, V::splat(1.0), x));
    (V::select(is_big, a, sa), V::select(is_big, b, sb))
}

/// Which of the four rational fits each lane wants, as three nested masks.
///
/// The four intervals are fdlibm's `[8, inf)`, `[4.5454, 8)`, `[2.8571, 4.547)`
/// and `[2, 2.8570)`. fdlibm tests the high word, `ix >= 0x40200000` and so
/// on; for a positive `x` that is exactly `x >= bits(0x4020000000000000)`,
/// because a high word compares greater iff the value does and the low half
/// only breaks the tie upwards. Every threshold here is the same rewrite the
/// rest of this file uses, so the interval boundaries land on the same doubles
/// fdlibm's do.
///
/// A NaN lane answers "false" to all three and lands in interval 3 where
/// fdlibm's masked high word would have put it in 0. Both give NaN, and the
/// lane is patched from the reference regardless — [`trig_safe`]'s `outside`
/// test catches it.
type Bands<V> = (<V as Simd>::Mask, <V as Simd>::Mask, <V as Simd>::Mask);

#[inline(always)]
fn bands<V: Simd<Elem = f64>>(x: V) -> Bands<V> {
    (
        x.ge_mask(V::splat(f64::from_bits(0x4020_0000_0000_0000))),
        x.ge_mask(V::splat(f64::from_bits(0x4012_2e8b_0000_0000))),
        x.ge_mask(V::splat(f64::from_bits(0x4006_db6d_0000_0000))),
    )
}

/// The `N` coefficients of whichever of the four rows each lane wants.
///
/// A blend ladder, not a gather. There are only four rows, so the choice is
/// three `select`s against four broadcast constants — loop-invariant, and on
/// x86-64 each one folds into the blend as a memory operand. The per-lane
/// gather it replaces did `N` scalar loads *and* `N` inserts for every lane,
/// and `pq0`/`pq1` ask for twenty-three coefficients between them: that was
/// forty-six scalar operations per element against a packed three per
/// coefficient per vector.
#[inline(always)]
fn pick<V: Simd<Elem = f64>, const N: usize>(m: &Bands<V>, table: &[[f64; N]; 4]) -> [V; N] {
    core::array::from_fn(|k| {
        let r = V::select(m.2, V::splat(table[2][k]), V::splat(table[3][k]));
        let r = V::select(m.1, V::splat(table[1][k]), r);
        V::select(m.0, V::splat(table[0][k]), r)
    })
}

/// `1 + R(s)/S(s)` with `s = 1/x^2`, the shape both `P` factors share.
#[inline(always)]
fn rational_p<V: Simd<Elem = f64>>(x: V, p: &[V; 6], q: &[V; 5]) -> V {
    let z = V::splat(1.0) / (x * x);
    let r1 = p[0] + z * p[1];
    let z2 = z * z;
    let r2 = p[2] + z * p[3];
    let z4 = z2 * z2;
    let r3 = p[4] + z * p[5];
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = V::splat(1.0) + z * q[0];
    let s2 = q[1] + z * q[2];
    let s3 = q[3] + z * q[4];
    let s = s1 + z2 * s2 + z4 * s3;
    V::splat(1.0) + r / s
}

/// `R(s)/S(s)` with `s = 1/x^2`, the shape both `Q` factors share. One term
/// longer in the denominator than [`rational_p`].
#[inline(always)]
fn rational_q<V: Simd<Elem = f64>>(x: V, p: &[V; 6], q: &[V; 6]) -> V {
    let z = V::splat(1.0) / (x * x);
    let r1 = p[0] + z * p[1];
    let z2 = z * z;
    let r2 = p[2] + z * p[3];
    let z4 = z2 * z2;
    let r3 = p[4] + z * p[5];
    let z6 = z4 * z2;
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = V::splat(1.0) + z * q[0];
    let s2 = q[1] + z * q[2];
    let s3 = q[3] + z * q[4];
    let s = s1 + z2 * s2 + z4 * s3 + z6 * q[5];
    r / s
}

/// `(P(x), Q(x))` for order 0.
#[inline(always)]
fn pq0<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let big = x.ge_mask(V::splat(PQ_LIMIT));
    let m = bands(x);
    let p = rational_p(x, &pick(&m, &t::P0R), &pick(&m, &t::P0S));
    let q = (V::splat(-0.125) + rational_q(x, &pick(&m, &t::Q0R), &pick(&m, &t::Q0S))) / x;
    (
        V::select(big, V::splat(1.0), p),
        V::select(big, V::splat(-0.125) / x, q),
    )
}

/// `(P(x), Q(x))` for order 1.
#[inline(always)]
fn pq1<V: Simd<Elem = f64>>(x: V) -> (V, V) {
    let big = x.ge_mask(V::splat(PQ_LIMIT));
    let m = bands(x);
    let p = rational_p(x, &pick(&m, &t::P1R), &pick(&m, &t::P1S));
    let q = (V::splat(0.375) + rational_q(x, &pick(&m, &t::Q1R), &pick(&m, &t::Q1S))) / x;
    (
        V::select(big, V::splat(1.0), p),
        V::select(big, V::splat(0.375) / x, q),
    )
}

/// The Bessel function of the first kind, order 0.
pub mod j0 {
    use super::*;

    /// `j0(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if !D::CHECKED {
            return main::<V, A>(x);
        }
        let special = crate::kernels::outside(x, trig_safe::<A>());
        if mostly_repaired(special) {
            return crate::simd::map_lanes(x, reference::j0);
        }
        patch_lanes(x, main::<V, A>(x), special, reference::j0)
    }

    #[inline(always)]
    pub(super) fn main<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        branch(x.abs(), |z| asymptotic::<V, A>(z), series::<V>)
    }

    /// `|x| >= 2`: the asymptotic form.
    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy>(z: V) -> V {
        let (s, c, c2) = sc2::<V, A>(z);
        let (ss, cc) = ss_cc0(s, c, c2);
        asym(z, ss, cc)
    }

    /// The asymptotic form once `(ss, cc)` are in hand, so that [`super::j01`]
    /// can supply a pair it computed once.
    #[inline(always)]
    pub(super) fn asym<V: Simd<Elem = f64>>(z: V, ss: V, cc: V) -> V {
        let (p, q) = pq0(z);
        let root = z.sqrt();
        let far = (V::splat(t::INVSQRTPI) * cc) / root;
        let near = V::splat(t::INVSQRTPI) * (p * cc - q * ss) / root;
        V::select(z.ge_mask(V::splat(NO_PQ)), far, near)
    }

    /// `|x| < 2`: a rational fit in `x^2`.
    #[inline(always)]
    pub(super) fn series<V: Simd<Elem = f64>>(z: V) -> V {
        let zz = z * z;
        let r = super::j0_num(zz);
        let s = super::j0_den(zz);
        let u = V::splat(0.5) * z;
        // Two spellings of the same value: below 1 the `1 + z*(...)` form
        // keeps its digits, and above it the factored `(1+u)(1-u)` does.
        let sub1 = V::splat(1.0) + zz * (V::splat(-0.25) + (r / s));
        let ge1 = (V::splat(1.0) + u) * (V::splat(1.0) - u) + zz * (r / s);
        let small = V::select(z.lt_mask(V::splat(1.0)), sub1, ge1);
        // Below 2^-13 the fit collapses to `1 - x^2/4`, and below 2^-27 to 1.
        V::select(
            z.lt_mask(V::splat(f64::from_bits(0x3f20_0000_0000_0000))),
            V::select(
                z.lt_mask(V::splat(f64::from_bits(0x3e40_0000_0000_0000))),
                V::splat(1.0),
                V::splat(1.0) - V::splat(0.25) * z * z,
            ),
            small,
        )
    }
}

/// `j0`'s numerator on `[0, 2]`, in fdlibm's exact association.
#[inline(always)]
fn j0_num<V: Simd<Elem = f64>>(z: V) -> V {
    let r = &t::J0_R;
    let r1 = z * V::splat(r[2]);
    let z2 = z * z;
    let r2 = V::splat(r[3]) + z * V::splat(r[4]);
    let z4 = z2 * z2;
    r1 + z2 * r2 + z4 * V::splat(r[5])
}

/// `j0`'s denominator on `[0, 2]`.
#[inline(always)]
fn j0_den<V: Simd<Elem = f64>>(z: V) -> V {
    let s = &t::J0_S;
    let s1 = V::splat(1.0) + z * V::splat(s[1]);
    let z2 = z * z;
    let s2 = V::splat(s[2]) + z * V::splat(s[3]);
    let z4 = z2 * z2;
    s1 + z2 * s2 + z4 * V::splat(s[4])
}

/// The Bessel function of the second kind, order 0.
pub mod y0 {
    use super::*;

    /// `y0(x)` for a vector of lanes.
    ///
    /// The domain is the positive reals: `y0` is `-inf` at zero and NaN below
    /// it, and both go to the reference rather than being blended in, because
    /// the main path would have to produce them out of a logarithm that has
    /// already gone to `-inf`.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if D::CHECKED {
            // A *normal* positive, not merely a positive: the series branch
            // takes a logarithm, and `ln`'s own vector path does not answer
            // for subnormals. Sending them to the reference is both correct
            // and cheaper than making every lane pay for `ln`'s repair test.
            let special = x
                .ge_mask(V::splat(f64::MIN_POSITIVE))
                .and(x.lt_mask(V::splat(trig_safe::<A>())))
                .not();
            if mostly_repaired(special) {
                return crate::simd::map_lanes(x, reference::y0);
            }
            let y = main::<V, A>(x);
            patch_lanes(x, y, special, reference::y0)
        } else {
            main::<V, A>(x)
        }
    }

    #[inline(always)]
    pub(super) fn main<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        branch(x, |x| asymptotic::<V, A>(x), |x| series::<V, A>(x))
    }

    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        let (s, c, c2) = sc2::<V, A>(x);
        let (ss, cc) = ss_cc0(s, c, c2);
        asym(x, ss, cc)
    }

    /// The asymptotic form once `(ss, cc)` are in hand. See [`super::y01`].
    #[inline(always)]
    pub(super) fn asym<V: Simd<Elem = f64>>(x: V, ss: V, cc: V) -> V {
        let (p, q) = pq0(x);
        let root = x.sqrt();
        let far = (V::splat(t::INVSQRTPI) * ss) / root;
        let near = V::splat(t::INVSQRTPI) * (p * ss + q * cc) / root;
        V::select(x.ge_mask(V::splat(NO_PQ)), far, near)
    }

    #[inline(always)]
    fn series<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        series_from(
            x,
            j0::main::<V, A>(x),
            ln::eval::<V, A, crate::policy::Finite>(x),
        )
    }

    /// The series branch with its two delegated pieces supplied, so that
    /// [`super::y01`] can share them with `y1`.
    #[inline(always)]
    pub(super) fn series_from<V: Simd<Elem = f64>>(x: V, j0x: V, lnx: V) -> V {
        let z = x * x;
        let u = &t::Y0_U;
        let v = &t::Y0_V;
        let u1 = V::splat(u[0]) + z * V::splat(u[1]);
        let z2 = z * z;
        let u2 = V::splat(u[2]) + z * V::splat(u[3]);
        let z4 = z2 * z2;
        let u3 = V::splat(u[4]) + z * V::splat(u[5]);
        let z6 = z4 * z2;
        let un = u1 + z2 * u2 + z4 * u3 + z6 * V::splat(u[6]);
        let v1 = V::splat(1.0) + z * V::splat(v[0]);
        let v2 = V::splat(v[1]) + z * V::splat(v[2]);
        let vd = v1 + z2 * v2 + z4 * V::splat(v[3]);
        let small = un / vd + V::splat(t::TPI) * (j0x * lnx);
        // Below 2^-27, U/V is U[0] and j0(x) is 1.
        V::select(
            x.lt_mask(V::splat(f64::from_bits(0x3e40_0001_0000_0000))),
            V::splat(u[0]) + V::splat(t::TPI) * lnx,
            small,
        )
    }
}

/// The Bessel function of the first kind, order 1.
pub mod j1 {
    use super::*;

    /// `j1(x)` for a vector of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if !D::CHECKED {
            return main::<V, A>(x);
        }
        let special = crate::kernels::outside(x, trig_safe::<A>());
        if mostly_repaired(special) {
            return crate::simd::map_lanes(x, reference::j1);
        }
        patch_lanes(x, main::<V, A>(x), special, reference::j1)
    }

    #[inline(always)]
    pub(super) fn main<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        branch(
            x,
            |x| asymptotic::<V, A>(x, x.abs()),
            |x| series::<V>(x, x.abs()),
        )
    }

    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy>(x: V, y: V) -> V {
        let (s, c, c2) = sc2::<V, A>(y);
        let (ss, cc) = ss_cc1(s, c, c2);
        let big = asym(y, ss, cc);
        // `j1` is odd, and the branch above computed it at `|x|`.
        V::select(x.lt_mask(V::splat(0.0)), -big, big)
    }

    /// The asymptotic form at `|x|`, once `(ss, cc)` are in hand — the sign
    /// is the caller's. See [`super::j01`].
    #[inline(always)]
    pub(super) fn asym<V: Simd<Elem = f64>>(y: V, ss: V, cc: V) -> V {
        let (p, q) = pq1(y);
        let root = y.sqrt();
        let far = (V::splat(t::INVSQRTPI) * cc) / root;
        let near = V::splat(t::INVSQRTPI) * (p * cc - q * ss) / root;
        V::select(y.ge_mask(V::splat(NO_PQ)), far, near)
    }

    #[inline(always)]
    pub(super) fn series<V: Simd<Elem = f64>>(x: V, y: V) -> V {
        let zz = x * x;
        let r = &t::J1_R;
        let s = &t::J1_S;
        let r1 = zz * V::splat(r[0]);
        let z2 = zz * zz;
        let r2 = V::splat(r[1]) + zz * V::splat(r[2]);
        let z4 = z2 * z2;
        let rn = (r1 + z2 * r2 + z4 * V::splat(r[3])) * x;
        let s1 = V::splat(1.0) + zz * V::splat(s[1]);
        let s2 = V::splat(s[2]) + zz * V::splat(s[3]);
        let s3 = V::splat(s[4]) + zz * V::splat(s[5]);
        let sd = s1 + z2 * s2 + z4 * s3;
        let small = x * V::splat(0.5) + rn / sd;
        // Below 2^-27, j1(x) is x/2 exactly.
        V::select(
            y.lt_mask(V::splat(f64::from_bits(0x3e40_0000_0000_0000))),
            V::splat(0.5) * x,
            small,
        )
    }
}

/// The Bessel function of the second kind, order 1.
pub mod y1 {
    use super::*;

    /// `y1(x)` for a vector of lanes. Positive reals only; see [`super::y0`].
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V {
        if D::CHECKED {
            // A *normal* positive, not merely a positive: the series branch
            // takes a logarithm, and `ln`'s own vector path does not answer
            // for subnormals. Sending them to the reference is both correct
            // and cheaper than making every lane pay for `ln`'s repair test.
            let special = x
                .ge_mask(V::splat(f64::MIN_POSITIVE))
                .and(x.lt_mask(V::splat(trig_safe::<A>())))
                .not();
            if mostly_repaired(special) {
                return crate::simd::map_lanes(x, reference::y1);
            }
            let y = main::<V, A>(x);
            patch_lanes(x, y, special, reference::y1)
        } else {
            main::<V, A>(x)
        }
    }

    #[inline(always)]
    pub(super) fn main<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        branch(x, |x| asymptotic::<V, A>(x), |x| series::<V, A>(x))
    }

    #[inline(always)]
    fn asymptotic<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        let (s, c, c2) = sc2::<V, A>(x);
        let (ss, cc) = ss_cc1(s, c, c2);
        asym(x, ss, cc)
    }

    /// The asymptotic form once `(ss, cc)` are in hand. See [`super::y01`].
    #[inline(always)]
    pub(super) fn asym<V: Simd<Elem = f64>>(x: V, ss: V, cc: V) -> V {
        let (p, q) = pq1(x);
        let root = x.sqrt();
        let far = (V::splat(t::INVSQRTPI) * ss) / root;
        let near = V::splat(t::INVSQRTPI) * (p * ss + q * cc) / root;
        V::select(x.ge_mask(V::splat(NO_PQ)), far, near)
    }

    #[inline(always)]
    fn series<V: Simd<Elem = f64>, A: Accuracy>(x: V) -> V {
        series_from(
            x,
            j1::main::<V, A>(x),
            ln::eval::<V, A, crate::policy::Finite>(x),
        )
    }

    /// The series branch with its two delegated pieces supplied, so that
    /// [`super::y01`] can share them with `y0`.
    #[inline(always)]
    pub(super) fn series_from<V: Simd<Elem = f64>>(x: V, j1x: V, lnx: V) -> V {
        let zz = x * x;
        let u = &t::Y1_U;
        let v = &t::Y1_V;
        let u1 = V::splat(u[0]) + zz * V::splat(u[1]);
        let z2 = zz * zz;
        let u2 = V::splat(u[2]) + zz * V::splat(u[3]);
        let z4 = z2 * z2;
        let un = u1 + z2 * u2 + z4 * V::splat(u[4]);
        let v1 = V::splat(1.0) + zz * V::splat(v[0]);
        let v2 = V::splat(v[1]) + zz * V::splat(v[2]);
        let v3 = V::splat(v[3]) + zz * V::splat(v[4]);
        let vd = v1 + z2 * v2 + z4 * v3;
        let small = x * (un / vd) + V::splat(t::TPI) * (j1x * lnx - V::splat(1.0) / x);
        // Below 2^-54 nothing but the pole survives.
        V::select(
            x.lt_mask(V::splat(f64::from_bits(0x3c90_0001_0000_0000))),
            -V::splat(t::TPI) / x,
            small,
        )
    }
}

/// The largest `|n|` the vector recurrences take.
///
/// Past `2^24` two things stop being true at once: fdlibm's `2 * (n + k)` is
/// `int` arithmetic that the reference reproduces and a float counter would
/// not, and the recurrence is long enough that nothing is lost by running it
/// one lane at a time anyway — `jn(2^24, x)` is sixteen million iterations
/// whichever way it is spelled.
const N_LIMIT: f64 = 16_777_216.0;

/// `x < 2^-30`, where `jn` collapses to the leading Taylor term.
const JN_TINY: f64 = f64::from_bits(0x3e10_0000_0000_0000);

/// `ln(DBL_MAX)`, spelled as fdlibm spells it — see the reference's `jn`.
#[allow(clippy::excessive_precision)]
const LN_MAX: f64 = 7.09782712893383973096e+02;

/// Is this (non-negative, integral) order odd? `m` is at most `N_LIMIT`, so
/// the halving and doubling are both exact and this is three vector
/// operations rather than a round trip through the lane array.
#[inline(always)]
fn is_odd<V: Simd<Elem = f64>>(m: V) -> V::Mask {
    let half = (m * V::splat(0.5)).floor();
    (m - (half + half)).eq_mask(V::splat(1.0))
}

/// The largest lane of `v` among the lanes `mask` selects, or `0.0` if none.
///
/// Every recurrence below runs to this and no further. It is what makes the
/// vector version cheaper rather than merely wider: the scalar routine pays
/// the *sum* of the lanes' trip counts and this pays the *maximum*, and each
/// lane's own loop is aligned to start on the same iteration so that maximum
/// is the real bound and not an artefact of where the counters began.
#[inline(always)]
fn masked_max<V: Simd<Elem = f64>>(v: V, mask: V::Mask) -> f64 {
    let vs = v.to_array();
    let bits = mask.to_bitmask();
    let mut out = 0.0f64;
    for i in 0..V::LANES {
        if bits & (1 << i) != 0 && vs.as_slice()[i] > out {
            out = vs.as_slice()[i];
        }
    }
    out
}

/// The Bessel function of the first kind, order `n`.
///
/// # What is vectorised, and what the loops cost
///
/// All three of fdlibm's branches, each as its own masked recurrence, and
/// they are blended only where a vector actually straddles two of them:
///
/// * `n <= x` — forward recurrence from `j0` and `j1`, `n - 1` steps.
/// * `x < 2^-30` — the leading Taylor term, at most thirty-two steps.
/// * otherwise — backward recurrence: a continued fraction whose length `k`
///   *is decided at run time*, then a downward pass of `k + 1` steps, then
///   `n - 1` more, and a renormalising variant of that last pass for the
///   orders where it would otherwise overflow.
///
/// The run-time trip counts are the reason this was left scalar for a long
/// time, and they are not actually an obstacle: a lane whose loop has
/// finished is *frozen* by a mask rather than exited, and the loop stops when
/// every lane is frozen. That makes the vector's cost the largest lane's trip
/// count where the scalar routine's is the sum of all of them — never worse,
/// and eight times better on the ordinary case of one order across a buffer.
/// Each lane's counter starts on the same iteration as every other lane's
/// (`i = 1`, `j = 0`, `di = 2(n-1)` held per lane rather than shared), so a
/// short lane costs nothing after it freezes.
///
/// The branch is chosen per lane and evaluated per vector: a whole-vector
/// `any()` skips a branch no lane wants, which is the common case — a buffer
/// of one order is usually entirely inside one of the three.
pub mod jn {
    use super::*;

    /// `jn(n, x)` for vectors of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(n: V, x: V) -> V {
        let nt = n.trunc();
        let m = nt.abs();
        // `J(-n, x) = J(n, -x)`: fold the order positive and the sign into x.
        let xf = V::select(nt.lt_mask(V::splat(0.0)), -x, x);
        let y = xf.abs();
        // The lanes the recurrences answer for. Zero, infinity, NaN, the
        // 2^302 asymptotic form and orders past `N_LIMIT` are the
        // reference's, one lane at a time.
        let ok = y
            .gt_mask(V::splat(0.0))
            .and(y.lt_mask(V::splat(trig_safe::<A>())))
            .and(m.le_mask(V::splat(N_LIMIT)));
        if mostly_repaired(ok.not()) {
            return map_lanes2(n, x, |n, x| reference::jn(order(n), x));
        }
        // Clamping the excluded lanes' order to zero keeps them out of every
        // `live` mask below, so they can neither extend a loop nor divide by
        // something they should not have reached.
        let m = V::select(ok, m, V::splat(0.0));
        let b = main::<V, A>(m, y, ok);
        // Even orders are even in `x`, odd orders odd.
        let b = V::select(is_odd(m).and(xf.lt_mask(V::splat(0.0))), -b, b);
        patch_lanes2(n, x, b, ok.not(), |n, x| reference::jn(order(n), x))
    }

    #[inline(always)]
    fn main<V: Simd<Elem = f64>, A: Accuracy>(m: V, y: V, ok: V::Mask) -> V {
        let one = V::splat(1.0);
        let two = V::splat(2.0);
        let forward = m.le_mask(y).and(ok);
        let tiny = m.le_mask(y).not().and(y.lt_mask(V::splat(JN_TINY))).and(ok);
        let backward = forward.or(tiny).not().and(ok);

        // `j0` and `j1` are three things at once: orders 0 and 1 themselves,
        // the forward recurrence's two seeds, and the backward recurrence's
        // normaliser. One evaluation of each serves all three — and orders 0
        // and 1 are why the test is not just `forward | backward`: a vector
        // of `jn(1, 1e-30)` is entirely inside the Taylor branch and still
        // answers from `j1`.
        let (j0v, j1v) = if forward.or(backward).or(m.le_mask(one).and(ok)).any() {
            j01::<V, A>(y)
        } else {
            (one, one)
        };

        let mut out = one;

        if forward.any() {
            let mut a = j0v;
            let mut b = j1v;
            let mut i = one;
            let stop = masked_max(m, forward);
            let mut step = 1.0f64;
            while step < stop {
                let live = i.lt_mask(m).and(forward);
                if live.none() {
                    break;
                }
                // The division rather than a multiply by `1/x` is fdlibm's,
                // and it is what keeps the recurrence from underflowing.
                let next = b * ((i + i) / y) - a;
                a = V::select(live, b, a);
                b = V::select(live, next, b);
                i = i + one;
                step += 1.0;
            }
            out = b;
        }

        if tiny.any() {
            // `J(n, x) = (x/2)^n / n!`, and above n = 33 that underflows.
            let half = y * V::splat(0.5);
            let mut num = half;
            let mut den = one;
            let mut i = two;
            let stop = masked_max(m, tiny).min(33.0);
            let mut step = 2.0f64;
            while step <= stop {
                let live = i.le_mask(m).and(tiny);
                if live.none() {
                    break;
                }
                den = V::select(live, den * i, den);
                num = V::select(live, num * half, num);
                i = i + one;
                step += 1.0;
            }
            let val = V::select(m.gt_mask(V::splat(33.0)), V::splat(0.0), num / den);
            out = V::select(tiny, val, out);
        }

        if backward.any() {
            out = V::select(backward, back::<V>(m, y, backward, j0v, j1v), out);
        }

        // Orders 0 and 1 answer before any branch does.
        let out = V::select(m.eq_mask(V::splat(1.0)), j1v, out);
        V::select(m.eq_mask(V::splat(0.0)), j0v, out)
    }

    /// The backward recurrence, for `x < n`, where forward is unstable.
    #[inline(always)]
    fn back<V: Simd<Elem = f64>>(m: V, y: V, live0: V::Mask, j0v: V, j1v: V) -> V {
        let one = V::splat(1.0);
        let two = V::splat(2.0);

        // (1) The continued fraction for `J(n,x)/J(n-1,x)`, run until a
        // convergent exceeds 1e9 — fdlibm's stopping rule, and the loop whose
        // length is decided at run time. A lane that has passed the threshold
        // freezes; the loop ends when all of them have.
        let h = two / y;
        let w = (m + m) / y;
        let mut q0 = w;
        let mut z = w + h;
        let mut q1 = w * z - one;
        let mut k = one;
        loop {
            let live = q1.lt_mask(V::splat(1.0e9)).and(live0);
            if live.none() {
                break;
            }
            let zn = z + h;
            let next = zn * q1 - q0;
            z = V::select(live, zn, z);
            q0 = V::select(live, q1, q0);
            q1 = V::select(live, next, q1);
            k = V::select(live, k + one, k);
        }

        // (2) `tt = 1/(i/x - tt)` from `i = 2(n+k)` down to `2n`, which is
        // `k + 1` steps. Each lane carries its own `i` so that every lane's
        // step 0 is its own first one and the vector runs `max(k) + 1` steps
        // rather than the span of the lanes' `i` ranges.
        let mut iv = (m + k) * two;
        let mut tt = V::splat(0.0);
        let kmax = masked_max(k, live0);
        let mut j = 0.0f64;
        while j <= kmax {
            let live = V::splat(j).le_mask(k).and(live0);
            if live.none() {
                break;
            }
            tt = V::select(live, one / (iv / y - tt), tt);
            iv = iv - two;
            j += 1.0;
        }

        // (3) The downward pass, `n - 1` steps, with `di` from `2(n-1)`.
        // fdlibm splits this in two on `n ln(2n/x)` against `ln(DBL_MAX)`,
        // because the test is once and not per iteration; the renormalising
        // variant is guarded here for the same reason — its two extra
        // divisions are on the critical path and almost no vector needs them.
        let renorm = (m * ln::eval::<V, crate::policy::BitExact, crate::policy::Finite>(
            (two / y * m).abs(),
        ))
        .lt_mask(V::splat(LN_MAX))
        .not()
        .and(live0);
        let any_renorm = renorm.any();
        let mut a = tt;
        let mut b = one;
        let mut di = (m - one) * two;
        let stop = masked_max(m, live0);
        let mut step = 1.0f64;
        while step < stop {
            let live = di.ge_mask(two).and(live0);
            if live.none() {
                break;
            }
            let next = (b * di) / y - a;
            a = V::select(live, b, a);
            b = V::select(live, next, b);
            if any_renorm {
                let hit = renorm.and(b.gt_mask(V::splat(1.0e100))).and(live);
                a = V::select(hit, a / b, a);
                tt = V::select(hit, tt / b, tt);
                b = V::select(hit, one, b);
            }
            di = di - two;
            step += 1.0;
        }

        // Normalise against `j0` or `j1` — whichever is further from zero,
        // since their zeros never coincide.
        V::select(j0v.abs().ge_mask(j1v.abs()), tt * j0v / b, tt * j1v / a)
    }
}

/// The Bessel function of the second kind, order `n`.
///
/// One branch, not three: forward recurrence from `y0` and `y1` is the
/// *stable* direction for `y`, the opposite of `j`. What it needs instead is
/// fdlibm's early exit — the recurrence reaches `-inf` for moderate `n` at
/// small `x`, and carrying on from there would produce `-inf - -inf` and turn
/// a correct infinity into a NaN. A lane at `-inf` freezes, and the loop ends
/// when every lane has either frozen or finished its own `n` steps.
pub mod yn {
    use super::*;

    /// `yn(n, x)` for vectors of lanes.
    #[inline(always)]
    pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(n: V, x: V) -> V {
        let nt = n.trunc();
        let m = nt.abs();
        let ok = x
            .ge_mask(V::splat(f64::MIN_POSITIVE))
            .and(x.lt_mask(V::splat(trig_safe::<A>())))
            .and(m.le_mask(V::splat(N_LIMIT)));
        if mostly_repaired(ok.not()) {
            return map_lanes2(n, x, |n, x| reference::yn(order(n), x));
        }
        let m = V::select(ok, m, V::splat(0.0));

        let one = V::splat(1.0);
        let (a0, b0) = y01::<V, A>(x);
        let mut a = a0;
        let mut b = b0;
        let mut i = one;
        let stop = masked_max(m, ok);
        let mut step = 1.0f64;
        while step < stop {
            // fdlibm's own guard is `high != 0xfff00000` on the running
            // value, which for anything this recurrence can produce is
            // exactly "not `-inf`": the only other bit pattern with that high
            // word is a NaN, and the one place a NaN could come from is the
            // very step this test prevents.
            let live = i
                .lt_mask(m)
                .and(b.eq_mask(V::splat(f64::NEG_INFINITY)).not());
            if live.none() {
                break;
            }
            let next = ((i + i) / x) * b - a;
            a = V::select(live, b, a);
            b = V::select(live, next, b);
            i = i + one;
            step += 1.0;
        }
        let b = V::select(m.eq_mask(V::splat(0.0)), a0, b);
        // `Y(-n, x) = (-1)^n Y(n, x)`.
        let b = V::select(nt.lt_mask(V::splat(0.0)).and(is_odd(m)), -b, b);
        patch_lanes2(n, x, b, ok.not(), |n, x| reference::yn(order(n), x))
    }
}

/// The order, carried in a float lane and read as C's `int` would read it.
///
/// Truncation towards zero and saturation at the ends, which is what a C
/// `(int)` cast of an out-of-range double does on every target this crate
/// builds for — and what Rust's `as` does by definition.
#[inline(always)]
fn order(n: f64) -> i32 {
    n.trunc() as i32
}
