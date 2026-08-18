#!/usr/bin/env python3
"""Generate `src/tables/*/poly.rs`: the coefficients the `Fast` kernels use.

    python3 tools/gen_poly.py && cargo fmt

Why generate rather than transcribe
-----------------------------------

The `BitExact` kernels get their constants from `gen_tables.py`, which pins
them to an upstream C source. The `Fast` kernels have no upstream — they are
this crate's own approximations — so the honest way to publish their
coefficients is the recipe that produced them. Everything below is derived
here, at 200-bit precision, from the function definitions themselves: change a
degree or an interval and re-run, and the error bound in the documentation is
recomputed rather than re-guessed.

Method: Chebyshev interpolation on the fitting interval, then a Remez exchange
to equioscillation. Remez is worth the extra code because the gap matters at
these degrees — Chebyshev truncation is typically 1.5-2x the minimax error,
which is the difference between a 1-ulp and a 2-ulp kernel.

Every emitted array is accompanied by the measured maximum relative error of
the polynomial alone, which is what `tests/policy.rs` holds the kernels to.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from mpmath import mp, mpf, chebyt, cos as mcos, pi as mpi
    import mpmath as M
except ImportError:  # pragma: no cover
    sys.exit("gen_poly.py needs mpmath: pip install mpmath")

mp.prec = 200


# --------------------------------------------------------------------------
# Minimax fitting
# --------------------------------------------------------------------------

def chebyshev_nodes(a, b, n):
    """`n` Chebyshev points of the second kind on [a, b]."""
    return [
        (a + b) / 2 + (b - a) / 2 * mcos(mpi * mpf(k) / (n - 1))
        for k in range(n)
    ]


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting, at working precision."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[piv][col] == 0:
            raise ZeroDivisionError("singular system")
        a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def remez(f, a, b, degree, weight=None, iterations=40):
    """Minimax polynomial of `degree` for `f` on [a, b].

    `weight(x)` scales the error, so passing `1/f` fits *relative* error --
    which is what matters for a floating-point kernel, since an absolute-error
    fit wastes accuracy where the function is small.
    """
    if weight is None:
        weight = lambda x: mpf(1)

    n = degree + 1
    nodes = chebyshev_nodes(a, b, n + 1)

    coeffs = None
    for _ in range(iterations):
        # Solve for coefficients and the equioscillating error E.
        rows, rhs = [], []
        for i, x in enumerate(nodes):
            row = [x ** k for k in range(n)]
            row.append(mpf(-1) ** i / weight(x))
            rows.append(row)
            rhs.append(f(x))
        sol = solve(rows, rhs)
        coeffs, err = sol[:n], sol[n]

        # Exchange: move each node to the local extremum of the weighted error.
        def werr(x):
            p = sum(c * x ** k for k, c in enumerate(coeffs))
            return (p - f(x)) * weight(x)

        fresh = []
        grid = 40
        pts = [a + (b - a) * mpf(i) / (grid * n) for i in range(grid * n + 1)]
        vals = [werr(x) for x in pts]
        i = 0
        while i < len(pts):
            j = i
            sign = 1 if vals[i] > 0 else -1
            while j + 1 < len(pts) and (1 if vals[j + 1] > 0 else -1) == sign:
                j += 1
            seg = max(range(i, j + 1), key=lambda k: abs(vals[k]))
            fresh.append(pts[seg])
            i = j + 1
        if len(fresh) < n + 1:
            break
        # Keep the n+1 largest-|error| extrema, in order.
        fresh.sort(key=lambda x: -abs(werr(x)))
        fresh = sorted(fresh[: n + 1])
        if fresh == nodes:
            break
        nodes = fresh

    # Measure the achieved error on a fine grid.
    def werr(x):
        p = sum(c * x ** k for k, c in enumerate(coeffs))
        return (p - f(x)) * weight(x)

    worst = max(
        abs(werr(a + (b - a) * mpf(i) / 20000)) for i in range(20001)
    )
    return coeffs, worst


def truncate_bits(x, keep, single):
    """`x` rounded to `keep` significand bits, towards zero.

    Clearing the low bits is what makes `n * part` exact: the product of a
    `keep`-bit value and an integer below `2^(prec - keep)` still fits the
    significand.
    """
    import math
    if x == 0:
        return mpf(0)
    e = int(M.floor(M.log(abs(x), 2)))
    scale = mpf(2) ** (keep - 1 - e)
    return mpf(int(x * scale)) / scale


def relative(f):
    """A weight that turns an absolute fit into a relative one."""
    def w(x):
        v = f(x)
        return mpf(1) / abs(v) if v != 0 else mpf(1)
    return w


# --------------------------------------------------------------------------
# Emitting
# --------------------------------------------------------------------------

def bits(x, single):
    import struct
    if single:
        return struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def rust_lit(x, single):
    ty = "f32" if single else "f64"
    width = 8 if single else 16
    return f"{ty}::from_bits(0x{bits(x, single):0{width}x})"


def emit_array(name, coeffs, doc, err, single):
    ty = "f32" if single else "f64"
    lines = [f"/// {line}" for line in doc.strip().splitlines()]
    lines.append("///")
    lines.append(f"/// Maximum relative error of the polynomial alone: {float(err):.3e}")
    lines.append(f"pub const {name}: [{ty}; {len(coeffs)}] = [")
    for c in coeffs:
        lines.append(f"    {rust_lit(c, single)}, // {float(c):+.17e}")
    lines.append("];")
    return "\n".join(lines)


HEADER = '''\
//! Coefficients for the {prec}-precision [`crate::policy::Fast`] kernels.
//!
//! GENERATED by `tools/gen_poly.py` — do not edit by hand.
//!
//! These are this crate's own minimax approximations, not a port of anyone
//! else's: `Fast` has no upstream to be bit-exact to. Each array carries the
//! measured maximum relative error of the polynomial in isolation, which is
//! the budget the kernel then spends on argument reduction and reconstruction.
//! `tests/policy.rs` holds the finished kernels to the resulting bounds.
//!
//! Regenerate with:
//!
//! ```text
//! python3 tools/gen_poly.py
//! ```

'''


def build(single: bool) -> str:
    """Every `Fast` coefficient array for one precision."""
    out = [HEADER.format(prec="single" if single else "double")]
    w = out.append

    # Degrees are chosen so the polynomial error sits comfortably under half an
    # ulp of the target format, leaving the rest of the budget to reduction.
    if single:
        d_sin, d_cos, d_asin, d_atan, d_expm1, d_atanh = 3, 4, 7, 6, 6, 3
    else:
        # `asin` needs the highest degree of the set: `asin(x)/x` has its
        # nearest singularity at x = 1, only just outside the fitting interval,
        # so it converges far more slowly than the others.
        d_sin, d_cos, d_asin, d_atan, d_expm1, d_atanh = 7, 7, 17, 11, 10, 8

    quarter = (mpi / 4) ** 2

    # sin(r)/r as a polynomial in s = r^2, so the odd symmetry is exact and the
    # leading 1 costs no accuracy.
    f = lambda s: (M.sin(M.sqrt(s)) / M.sqrt(s)) if s > 0 else mpf(1)
    c, e = remez(f, mpf(0), quarter, d_sin, relative(f))
    w(emit_array("SIN", c, "`sin(r)/r` as a polynomial in `r^2`, on `|r| <= pi/4`.", e, single))
    w("")

    # cos(r) as a polynomial in s = r^2.
    f = lambda s: M.cos(M.sqrt(s))
    c, e = remez(f, mpf(0), quarter, d_cos, relative(f))
    w(emit_array("COS", c, "`cos(r)` as a polynomial in `r^2`, on `|r| <= pi/4`.", e, single))
    w("")

    # asin(x)/x on |x| <= 1/2, as a polynomial in x^2.
    f = lambda s: (M.asin(M.sqrt(s)) / M.sqrt(s)) if s > 0 else mpf(1)
    c, e = remez(f, mpf(0), mpf(0.25), d_asin, relative(f))
    w(emit_array("ASIN", c, "`asin(x)/x` as a polynomial in `x^2`, on `|x| <= 1/2`.", e, single))
    w("")

    # atan(x)/x on |x| <= tan(pi/12) ~ 0.2679, as a polynomial in x^2. The
    # kernel folds any argument into that range with two identities.
    lim = M.tan(mpi / 12) ** 2
    f = lambda s: (M.atan(M.sqrt(s)) / M.sqrt(s)) if s > 0 else mpf(1)
    c, e = remez(f, mpf(0), lim, d_atan, relative(f))
    w(emit_array("ATAN", c, "`atan(x)/x` as a polynomial in `x^2`, on `|x| <= tan(pi/12)`.", e, single))
    w("")

    # (exp(r)-1)/r on |r| <= ln(2)/2, as a polynomial in r.
    half_ln2 = M.log(2) / 2
    f = lambda r: (M.expm1(r) / r) if r != 0 else mpf(1)
    c, e = remez(f, -half_ln2, half_ln2, d_expm1, relative(f))
    w(emit_array("EXPM1", c, "`(exp(r) - 1)/r` as a polynomial in `r`, on `|r| <= ln(2)/2`.", e, single))
    w("")

    # atanh(s)/s as a polynomial in s^2. Three callers reduce to this:
    # log1p(x) = 2 atanh(x/(2+x)), atanh itself, and the logarithm kernels,
    # which fold the significand to [1/sqrt2, sqrt2) and take s = (m-1)/(m+1).
    # That last one is what sets the interval: it reaches |s| = 0.17157, so a
    # fit that stopped at 1/6 would be extrapolating over part of its range.
    s_max = (M.sqrt(2) - 1) / (M.sqrt(2) + 1)
    c, e = remez(
        lambda t: (M.atanh(M.sqrt(t)) / M.sqrt(t)) if t > 0 else mpf(1),
        mpf(0), s_max ** 2 * mpf(1.02), d_atanh,
        relative(lambda t: (M.atanh(M.sqrt(t)) / M.sqrt(t)) if t > 0 else mpf(1)),
    )
    w(emit_array("ATANH", c,
                 "`atanh(s)/s` as a polynomial in `s^2`, on `|s| <= (sqrt2-1)/(sqrt2+1)`.",
                 e, single))
    w("")

    # Gamma on [1, 2], where both Gamma functions are reduced to.
    #
    # `lgamma` is fitted divided by `t(t-1)`, which is exactly its two zeros at
    # x = 1 and x = 2. Factoring them out is the whole point: `lgamma` is tiny
    # there while the Stirling-plus-recurrence route computes it as a
    # difference of numbers near 12, so that route loses every significant
    # digit exactly where scientific code most often evaluates it.
    # High degrees: both functions have a pole at t = -1, just outside the
    # unit fitting interval, so the series converges slowly and the degree is
    # what buys the last few digits.
    d_lg, d_g = (24, 24) if not single else (10, 10)
    f = lambda t: M.loggamma(1 + t) / (t * (t - 1)) if 0 < t < 1 else (
        M.euler if t == 0 else mpf(1) - M.euler)
    c, e = remez(f, mpf(0), mpf(1), d_lg, relative(f))
    w(emit_array("LGAMMA", c,
                 "`lgamma(1 + t) / (t(t - 1))` on `t` in `[0, 1]`.\n"
                 "The two zeros are factored out, so the kernel reproduces them exactly.",
                 e, single))
    w("")

    f = lambda t: M.gamma(1 + t)
    c, e = remez(f, mpf(0), mpf(1), d_g, relative(f))
    w(emit_array("GAMMA", c, "`Gamma(1 + t)` on `t` in `[0, 1]`.", e, single))
    w("")

    # Argument-reduction constants. `PIO2` splits pi/2 across three doubles
    # whose low `trailing` significand bits are zero, so `n * PIO2[i]` is exact
    # for every integer `|n| < 2^trailing` -- that exactness is the whole point
    # of a Cody-Waite reduction, and it is what bounds the kernel's domain.
    trailing = 20 if not single else 12
    keep = (53 - trailing) if not single else (24 - trailing)
    pio2 = []
    rest = mpi / 2
    for _ in range(3):
        part = truncate_bits(rest, keep, single)
        pio2.append(part)
        rest = rest - part
    ty = "f32" if single else "f64"
    w(f"/// `pi/2`, split so that `n * PIO2[i]` is exact for `|n| < 2^{trailing}`.")
    w("///")
    w("/// Each part carries only its leading significand bits, with the low")
    w(f"/// {trailing} cleared; together they give `pi/2` to about {keep * 3} bits, which")
    w("/// is what keeps the reduced argument accurate when the quadrant count")
    w("/// is large.")
    w(f"pub const PIO2: [{ty}; 3] = [")
    for c in pio2:
        w(f"    {rust_lit(c, single)}, // {float(c):+.17e}")
    w("];")
    w("")
    w("/// `2/pi`, for choosing the quadrant.")
    w(f"pub const TWO_OVER_PI: {ty} = {rust_lit(2 / mpi, single)};")
    w("")
    w("/// Largest `|x|` the Cody-Waite reduction above stays exact for.")
    w("///")
    w("/// Beyond it the quadrant count exceeds what the split can multiply")
    w("/// exactly, so the kernels hand those lanes to the scalar reference")
    w("/// under `FullRange` -- and are simply wrong under `Finite`.")
    w(f"pub const TRIG_LIMIT: {ty} = {rust_lit(mpf(2) ** trailing * mpi / 2, single)};")
    w("")
    w("/// `ln(2)`, split so that `k * LN2[i]` is exact for any `k` a reduction")
    w("/// to `|r| <= ln(2)/2` can produce.")
    ln2 = []
    rest = M.log(2)
    for _ in range(2):
        part = truncate_bits(rest, keep, single)
        ln2.append(part)
        rest = rest - part
    w(f"pub const LN2: [{ty}; 2] = [")
    for c in ln2:
        w(f"    {rust_lit(c, single)}, // {float(c):+.17e}")
    w("];")
    w("")
    w("/// `log2(e)`, for the exponent reduction.")
    w(f"pub const LOG2E: {ty} = {rust_lit(1 / M.log(2), single)};")
    w("")

    # Stirling's series for `lgamma`, used above the cutoff below. The terms
    # are `B(2n) / (2n (2n-1))`, exact rationals, so they are computed rather
    # than fitted -- there is nothing to approximate.
    terms = 7 if not single else 4
    stirling = [
        M.bernoulli(2 * k) / (mpf(2 * k) * mpf(2 * k - 1)) for k in range(1, terms + 1)
    ]
    ty = "f32" if single else "f64"
    w("/// Stirling series coefficients for `lgamma`: `B(2n) / (2n(2n-1))`.")
    w("///")
    w("/// `lgamma(y) ~ (y - 1/2) ln y - y + ln(2 pi)/2 + sum_n S[n] / y^(2n-1)`,")
    w("/// for `y` past [`LGAMMA_CUTOFF`]. Exact rationals, computed rather than")
    w("/// fitted; the series is asymptotic, so the term count and the cutoff")
    w("/// are chosen together.")
    w(f"pub const STIRLING: [{ty}; {len(stirling)}] = [")
    for c in stirling:
        w(f"    {rust_lit(c, single)}, // {float(c):+.17e}")
    w("];")
    w("")
    w("/// Where the Stirling series above is truncated accurately enough.")
    w("///")
    w("/// Below this, `lgamma` walks up to it with the recurrence")
    w("/// `lgamma(y) = lgamma(y + 1) - ln(y)` and subtracts the logarithms of")
    w("/// the terms it stepped over.")
    cutoff = mpf(8) if not single else mpf(8)
    w(f"pub const LGAMMA_CUTOFF: {ty} = {rust_lit(cutoff, single)};")
    w("")
    w("/// `ln(2 pi) / 2`, the constant term of Stirling's series.")
    w(f"pub const HALF_LN_2PI: {ty} = {rust_lit(M.log(2 * mpi) / 2, single)};")
    return "\n".join(out) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "src" / "tables"
    for single, sub in ((False, "double"), (True, "single")):
        dst = root / sub / "poly.rs"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(build(single))
        print(f"wrote {dst} ({len(dst.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
