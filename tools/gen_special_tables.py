#!/usr/bin/env python3
"""Generate `src/tables/{double,single}/{erf,erfc}.rs` from glibc's C data.

    python3 tools/gen_special_tables.py            # download, regenerate
    python3 tools/gen_special_tables.py --src DIR  # use downloaded sources

Run `cargo fmt` afterwards.

Why generate rather than hand-copy
----------------------------------

`erf` and `erfc` are correctly rounded in glibc, which is the strongest form
this crate's `BitExact` policy can take: any correctly-rounded implementation
agrees with the platform on every input, on every platform. Getting there
means reproducing CORE-MATH's two-step scheme exactly, and that rests on some
five thousand exact bit patterns spread over eight tables. Transcribing those
by hand is a silent-corruption risk with no upside.

Every value is emitted as `f64::from_bits(0x...)` with the original hex-float
spelling in a trailing comment, because Rust has no hex-float literals and a
decimal spelling would be a lossy round-trip.

Upstream: glibc `sysdeps/ieee754/{dbl-64,flt-32}/s_erf*.c`, itself copied from
the CORE-MATH project. Copyright (c) 2022-2025 Alexei Sibidanov, Paul
Zimmermann, Tom Hubrecht and Claude-Pierre Jeannerod, MIT licence.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/bminor/glibc/master/sysdeps/ieee754"

FILES = {
    "s_erf_data.c": "dbl-64",
    "s_erfc_data.c": "dbl-64",
    "s_erff.c": "flt-32",
    "s_erfcf.c": "flt-32",
    "e_j0.c": "dbl-64",
    "e_j1.c": "dbl-64",
    "e_j0f.c": "flt-32",
    "e_j1f.c": "flt-32",
    "s_sincosf_data.c": "flt-32",
}

ROOT = Path(__file__).resolve().parent.parent


def fetch(src: Path | None, name: str) -> str:
    if src is not None:
        return (src / name).read_text()
    url = f"{BASE}/{FILES[name]}/{name}"
    print(f"fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def array_body(text: str, decl: str) -> str:
    """The braced initialiser of the C declaration whose text starts `decl`."""
    i = text.index(decl)
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j : k + 1]
    raise ValueError(f"unterminated initialiser for {decl!r}")


# Hex-float first, so that the `0` of an `0x...` literal is never taken as a
# decimal on its own. fdlibm writes its double constants with 21 significant
# digits and its float constants with 11, both of which round-trip, so parsing
# the decimal is exact -- but the CORE-MATH tables use hex floats, and the
# Bessel zero tables mix the two.
NUM = re.compile(
    r"[-+]?0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*[pP][-+]?\d+"
    r"|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def numbers(body: str) -> list[tuple[str, float]]:
    """Every float literal in a braced initialiser, in order."""
    out = []
    for m in NUM.finditer(body):
        s = m.group(0)
        v = float.fromhex(s) if "x" in s or "X" in s else float(s)
        out.append((s, v))
    return out


def bits64(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def bits32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def cell(v: float, w: int) -> str:
    """One `fN::from_bits(0x..)` literal."""
    if w == 32:
        return f"f32::from_bits(0x{bits32(v):08x}),"
    return f"f64::from_bits(0x{bits64(v):016x}),"


def emit_flat(
    name: str, doc: str, vals: list[tuple[str, float]], per_row: int, w: int = 64
) -> str:
    out = [doc, f"pub static {name}: [f{w}; {len(vals)}] = ["]
    for i in range(0, len(vals), per_row):
        chunk = vals[i : i + per_row]
        out.append("    " + " ".join(cell(v, w) for _, v in chunk))
    out.append("];")
    return "\n".join(out)


def emit_const(name: str, doc: str, v: float, w: int = 64) -> str:
    return f"{doc}\npub const {name}: f{w} = {cell(v, w)[:-1]};"


def emit_rows(name: str, doc: str, groups: list[list[tuple[str, float]]], w: int = 64) -> str:
    """Emit a `[[fN; cols]; n]` from already-grouped rows of equal length."""
    cols = len(groups[0])
    assert all(len(g) == cols for g in groups), (name, [len(g) for g in groups])
    out = [doc, f"pub static {name}: [[f{w}; {cols}]; {len(groups)}] = ["]
    for g in groups:
        out.append("    [" + " ".join(cell(v, w) for _, v in g) + "],")
    out.append("];")
    return "\n".join(out)


def rows(body: str) -> list[str]:
    """The depth-1 `{...}` groups of a braced initialiser, in order."""
    out, depth, start = [], 0, None
    for k, ch in enumerate(body):
        if ch == "{":
            depth += 1
            if depth == 2:
                start = k
        elif ch == "}":
            if depth == 2:
                out.append(body[start : k + 1])
            depth -= 1
    return out


def emit_2d(name: str, doc: str, body: str, cols: int, w: int = 64) -> str:
    """Emit a `[[f64; cols]; n]`, zero-padding short rows.

    C zero-fills a short initialiser and several of these tables rely on it:
    `Tacc`'s polynomials rise in degree, so its low-order rows genuinely do end
    in zeros. Splitting on the row braces rather than trusting a flat count is
    what stops one ragged row from shifting every row after it.
    """
    rs = rows(body)
    out = [doc, f"pub static {name}: [[f{w}; {cols}]; {len(rs)}] = ["]
    for r in rs:
        vals = numbers(r)
        assert len(vals) <= cols, (name, len(vals), cols)
        cells = [cell(v, w) for _, v in vals]
        cells += ["0.0,"] * (cols - len(vals))
        out.append(f"    [{' '.join(cells)}],")
    out.append("];")
    return "\n".join(out)


HEADER = """//! {title}
//!
//! GENERATED by `tools/gen_special_tables.py` — do not edit by hand.
//!
//! Transcribed from glibc `{origin}`, itself taken from the CORE-MATH project.
//! Copyright (c) 2022-2025 Alexei Sibidanov, Paul Zimmermann, Tom Hubrecht and
//! Claude-Pierre Jeannerod, MIT licence.
//!
//! Values are exact bit patterns: Rust has no hex-float literals, and the
//! decimal spellings do not round-trip.
"""


def gen_erf(src: Path | None) -> str:
    text = strip_comments(fetch(src, "s_erf_data.c"))
    parts = [
        HEADER.format(
            title="`erf` data: the fast path's table, and the accurate path's.",
            origin="sysdeps/ieee754/dbl-64/s_erf_data.c",
        )
    ]
    c0 = numbers(array_body(text, "__erf_data_c0"))
    parts.append(
        emit_flat(
            "C0",
            "/// The `z < 1/16` polynomial, degrees 1..=11, the first two\n"
            "/// coefficients as double-double pairs.",
            c0,
            2,
        )
    )
    c = array_body(text, "__erf_data_C[94][13]")
    parts.append(
        emit_2d(
            "C",
            "/// `C[i-1]` is a degree-10 minimax polynomial for `erf(i/16 + 1/32 + z)`\n"
            "/// on `|z| <= 1/32`, for `1 <= i < 95`. Degrees 0 and 1 are\n"
            "/// double-double — `p[0]+p[1]` and `p[2]+p[3]` — and degrees 2..=10 are\n"
            "/// `p[4]..p[12]`.",
            c,
            13,
        )
    )
    c2 = array_body(text, "__erf_data_C2[47][27]")
    parts.append(
        emit_2d(
            "C2",
            "/// The accurate path's table: `C2[i-1]` is a degree-18 polynomial for\n"
            "/// `erf(i/8 + 1/16 + z)` on `|z| <= 1/16`, with the low eight\n"
            "/// coefficients as double-double pairs.",
            c2,
            27,
        )
    )
    p = numbers(array_body(text, "__erf_data_p["))
    parts.append(
        emit_flat(
            "P",
            "/// The accurate path's `|z| < 1/8` series, odd degrees 1..=21, the low\n"
            "/// four coefficients as double-double pairs.",
            p,
            2,
        )
    )
    exc = array_body(text, "__erf_data_exceptions[")
    parts.append(
        emit_2d(
            "EXCEPTIONS",
            "/// Arguments whose correctly-rounded result the accurate polynomial\n"
            "/// still cannot resolve: `(z, h, l)` triples, consulted first.",
            exc,
            3,
        )
    )
    exct = array_body(text, "__erf_data_exceptions_tiny[")
    parts.append(
        emit_2d(
            "EXCEPTIONS_TINY",
            "/// The same for `|z| < 1/8`, sorted by increasing `z` so the lookup can\n"
            "/// bisect.",
            exct,
            3,
        )
    )
    return "\n\n".join(parts) + "\n"


def gen_erfc(src: Path | None) -> str:
    text = strip_comments(fetch(src, "s_erfc_data.c"))
    erf = strip_comments(fetch(src, "s_erf_data.c"))
    parts = [
        HEADER.format(
            title="`erfc` data: the asymptotic table, and the double-double `exp`.",
            origin="sysdeps/ieee754/dbl-64/s_erfc_data.c",
        )
    ]
    t = array_body(text, "__erfc_data_T[6][13]")
    parts.append(
        emit_2d(
            "T",
            "/// `T[i]` is a degree-23 even polynomial in `1/x` approximating\n"
            "/// `erfc(x) exp(x^2) x`, one per interval of `1/x`. Degree 1 is the\n"
            "/// double-double `p[0]+p[1]`; the rest are `p[2]..p[12]`.",
            t,
            13,
        )
    )
    e2 = numbers(array_body(text, "__erfc_data_E2["))
    parts.append(
        emit_flat(
            "E2",
            "/// The accurate path's `exp` table.",
            e2,
            2,
        )
    )
    tacc = array_body(text, "__erfc_data_Tacc[10][30]")
    parts.append(
        emit_2d(
            "TACC",
            "/// The accurate path's asymptotic table: ten polynomials of rising\n"
            "/// degree, one per interval of `1/x`.",
            tacc,
            30,
        )
    )
    for cname, rname, doc in [
        (
            "__erfc_data_exceptions[22][3]",
            "EXCEPTIONS",
            "/// Fast-path exceptions: `(x, h, l)` triples.",
        ),
        (
            "__erfc_data_exceptions_accurate[17][3]",
            "EXCEPTIONS_ACCURATE",
            "/// Accurate-path exceptions.",
        ),
        (
            "__erfc_data_exceptions_accurate_2[29][3]",
            "EXCEPTIONS_ACCURATE_2",
            "/// Accurate-path exceptions for the asymptotic branch.",
        ),
    ]:
        parts.append(emit_2d(rname, doc, array_body(text, cname), 3))

    # The double-double exp that `erfc`'s asymptotic branch runs lives in the
    # erf data file, because erf's accurate path uses it too.
    t1 = array_body(erf, "__erf_data_T1[][2]")
    parts.append(
        emit_2d(
            "T1",
            "/// `2^(i/64)` as a double-double, for the `exp(-x^2)` the asymptotic\n"
            "/// branch needs to 74 bits.",
            t1,
            2,
        )
    )
    t2 = array_body(erf, "__erf_data_T2[][2]")
    parts.append(
        emit_2d("T2", "/// `2^(i/4096)` as a double-double. See [`T1`].", t2, 2)
    )
    q1 = numbers(array_body(erf, "__erf_data_Q_1["))
    parts.append(
        emit_flat(
            "Q1",
            "/// The degree-4 correction that same `exp` takes on `|z| < 2^-12.88`.",
            q1,
            5,
        )
    )
    return "\n\n".join(parts) + "\n"


def gen_single(src: Path | None) -> str:
    erff = strip_comments(fetch(src, "s_erff.c"))
    erfcf = strip_comments(fetch(src, "s_erfcf.c"))
    parts = [
        HEADER.format(
            title="`erff` / `erfcf` data.\n//!\n"
            "//! Every constant is an `f64`: both routines evaluate in double\n"
            "//! precision over a small table and round once, which is what lets\n"
            "//! [`crate::kernels::single`] replay them lane-parallel.",
            origin="sysdeps/ieee754/flt-32/s_erff.c and s_erfcf.c",
        )
    ]
    c = array_body(erff, "C[56][8]")
    parts.append(
        emit_2d(
            "C",
            "/// `C[i-7]` is a degree-7 polynomial for `erf(i/16 + 1/32 + z)` on\n"
            "/// `|z| <= 1/32`, for `7 <= i < 63`.",
            c,
            8,
        )
    )
    small = numbers(array_body(erff[erff.index("0x3ee00000") :], "c[]"))
    parts.append(
        emit_flat(
            "C_SMALL",
            "/// The `|x| < 0.4375` branch: a degree-15 odd series in `x`, given as\n"
            "/// its eight even coefficients in `x^2`.",
            small,
            2,
        )
    )
    e = numbers(array_body(erfcf, "static const double E[]"))
    parts.append(
        emit_flat(
            "E",
            "/// `2^(i/128)` for `i` in `0..128`, the scale `erfcf` reads for\n"
            "/// `exp(-x^2)`.",
            e,
            3,
        )
    )
    tail = erfcf[erfcf.index("__erfcf") if "__erfcf" in erfcf else 0 :]
    cn = numbers(array_body(tail[tail.index("0x3db80000") :], "c[]"))
    parts.append(
        emit_flat(
            "CN",
            "/// The `|x| <= 0x1.7p-4` branch, where `erfc` is `1` minus an odd\n"
            "/// series: five even coefficients in `x^2`.",
            cn,
            3,
        )
    )
    ch = numbers(array_body(tail, "ch[]"))
    parts.append(
        emit_flat(
            "CH",
            "/// The degree-4 correction `erfcf` uses for `exp(-x^2)` after reducing\n"
            "/// against `ln(2)/128`.",
            ch,
            4,
        )
    )
    ct = array_body(tail, "ct[][16]")
    parts.append(
        emit_2d(
            "CT",
            "/// Two degree-12 Chebyshev fits of `erfc(x) exp(x^2)` in the variable\n"
            "/// `z = (|x| - ct[i][0]) / (|x| + ct[i][1])`, selected by\n"
            "/// `|x| > 0x1.0a2p+1`. `ct[i][2]` is the constant term and\n"
            "/// `ct[i][3..]` the coefficients of `z`.",
            ct,
            16,
        )
    )
    return "\n\n".join(parts) + "\n"


BESSEL_HEADER = """//! {title}
//!
//! GENERATED by `tools/gen_special_tables.py` — do not edit by hand.
//!
//! Transcribed from glibc `{origin}`, which is Sun Microsystems' fdlibm —
//! Copyright (c) 1993 Sun Microsystems, Inc., permission to use granted
//! provided this notice is preserved. glibc still runs this code for the
//! Bessel family, so reproducing its schedule is what makes
//! [`crate::policy::BitExact`] bit-exact here.
//!
//! Values are exact bit patterns; fdlibm's own decimal spellings carry enough
//! digits to round-trip, and the generator converts them rather than trusting
//! Rust's parser to agree.
"""


def scalars(text: str, names: list[str], w: int) -> dict[str, float]:
    """Pull `NAME = <literal>` definitions out of a C source."""
    out = {}
    for n in names:
        m = re.search(rf"\b{re.escape(n)}\s*=\s*({NUM.pattern})", text)
        if m is None:
            raise ValueError(f"no definition of {n}")
        lit = m.group(1)
        out[n] = float.fromhex(lit) if "x" in lit or "X" in lit else float(lit)
    return out


def four(text: str, stem: str, w: int, doc: str, rust: str) -> str:
    """Group fdlibm's four per-interval arrays (`...8`, `...5`, `...3`, `...2`)
    into one 2-D table, so the kernels index rather than branch."""
    groups = [numbers(array_body(text, f"{stem}{k}[")) for k in ("8", "5", "3", "2")]
    return emit_rows(rust, doc, groups, w)


def gen_bessel_double(src: Path | None) -> str:
    j0 = strip_comments(fetch(src, "e_j0.c"))
    j1 = strip_comments(fetch(src, "e_j1.c"))
    parts = [
        BESSEL_HEADER.format(
            title="`j0` / `j1` / `y0` / `y1` data.",
            origin="sysdeps/ieee754/dbl-64/e_j0.c and e_j1.c",
        )
    ]
    c = scalars(j0, ["invsqrtpi", "tpi"], 64)
    parts.append(emit_const("INVSQRTPI", "/// `1/sqrt(pi)`.", c["invsqrtpi"]))
    parts.append(emit_const("TPI", "/// `2/pi`.", c["tpi"]))

    parts.append(
        emit_flat(
            "J0_R",
            "/// `j0`'s numerator on `[0, 2]`, with two leading zeros so the\n"
            "/// indices match fdlibm's `R02..R05`.",
            numbers(array_body(j0, "R[] =")),
            3,
        )
    )
    parts.append(
        emit_flat(
            "J0_S",
            "/// `j0`'s denominator on `[0, 2]`, leading zero as above.",
            numbers(array_body(j0, "S[] =")),
            3,
        )
    )
    parts.append(
        emit_flat("Y0_U", "/// `y0`'s numerator on `[0, 2]`.", numbers(array_body(j0, "U[] =")), 3)
    )
    parts.append(
        emit_flat("Y0_V", "/// `y0`'s denominator on `[0, 2]`.", numbers(array_body(j0, "V[] =")), 3)
    )
    for stem, rust, doc in [
        ("pR", "P0R", "/// `pzero`'s numerator, one row per interval of `1/x`:\n/// `[inf,8]`, `[8,4.5454]`, `[4.547,2.8571]`, `[2.8570,2]`."),
        ("pS", "P0S", "/// `pzero`'s denominator. See [`P0R`]."),
        ("qR", "Q0R", "/// `qzero`'s numerator. See [`P0R`]."),
        ("qS", "Q0S", "/// `qzero`'s denominator. See [`P0R`]."),
    ]:
        parts.append(four(j0, stem, 64, doc, rust))

    parts.append(
        emit_flat("J1_R", "/// `j1`'s numerator on `[0, 2]`.", numbers(array_body(j1, "R[] =")), 3)
    )
    parts.append(
        emit_flat(
            "J1_S",
            "/// `j1`'s denominator on `[0, 2]`, leading zero as for [`J0_S`].",
            numbers(array_body(j1, "S[] =")),
            3,
        )
    )
    parts.append(
        emit_flat("Y1_U", "/// `y1`'s numerator on `[0, 2]`.", numbers(array_body(j1, "U0[5]")), 3)
    )
    parts.append(
        emit_flat("Y1_V", "/// `y1`'s denominator on `[0, 2]`.", numbers(array_body(j1, "V0[5]")), 3)
    )
    for stem, rust, doc in [
        ("pr", "P1R", "/// `pone`'s numerator, one row per interval of `1/x`. See [`P0R`]."),
        ("ps", "P1S", "/// `pone`'s denominator. See [`P0R`]."),
        ("qr", "Q1R", "/// `qone`'s numerator. See [`P0R`]."),
        ("qs", "Q1S", "/// `qone`'s denominator. See [`P0R`]."),
    ]:
        parts.append(four(j1, stem, 64, doc, rust))
    return "\n\n".join(parts) + "\n"


def gen_bessel_single(src: Path | None) -> str:
    j0 = strip_comments(fetch(src, "e_j0f.c"))
    j1 = strip_comments(fetch(src, "e_j1f.c"))
    parts = [
        BESSEL_HEADER.format(
            title="`j0f` / `j1f` / `y0f` / `y1f` data.\n//!\n"
            "//! Two kinds of table. The rational fits are fdlibm's, transcribed\n"
            "//! as they stand. The `*_ZEROS` tables are glibc's addition: near a\n"
            "//! zero of the function the rational fit loses every significant\n"
            "//! digit to cancellation, so glibc substitutes a degree-3 polynomial\n"
            "//! fitted to that zero. Each row is `[x0, xmid, x1, p0, p1, p2, p3]`.",
            origin="sysdeps/ieee754/flt-32/e_j0f.c and e_j1f.c",
        )
    ]
    c = scalars(j0, ["invsqrtpi", "tpi"], 32)
    parts.append(emit_const("INVSQRTPI", "/// `1/sqrt(pi)`.", c["invsqrtpi"], 32))
    parts.append(emit_const("TPI", "/// `2/pi`.", c["tpi"], 32))

    r0 = scalars(j0, ["R02", "R03", "R04", "R05", "S01", "S02", "S03", "S04"], 32)
    parts.append(
        emit_flat(
            "J0_R",
            "/// `j0f`'s numerator on `[0, 2]`, `R02..R05`.",
            [("", r0[k]) for k in ("R02", "R03", "R04", "R05")],
            4,
            32,
        )
    )
    parts.append(
        emit_flat(
            "J0_S",
            "/// `j0f`'s denominator on `[0, 2]`, `S01..S04`.",
            [("", r0[k]) for k in ("S01", "S02", "S03", "S04")],
            4,
            32,
        )
    )
    u0 = scalars(j0, ["u00", "u01", "u02", "u03", "u04", "u05", "u06"], 32)
    parts.append(
        emit_flat(
            "Y0_U",
            "/// `y0f`'s numerator on `[0, 2]`.",
            [("", u0[f"u0{i}"]) for i in range(7)],
            4,
            32,
        )
    )
    v0 = scalars(j0, ["v01", "v02", "v03", "v04"], 32)
    parts.append(
        emit_flat(
            "Y0_V",
            "/// `y0f`'s denominator on `[0, 2]`.",
            [("", v0[f"v0{i}"]) for i in range(1, 5)],
            4,
            32,
        )
    )
    parts.append(
        emit_2d(
            "J0_ZEROS",
            "/// Degree-3 fits around the first 64 zeros of `j0`.",
            array_body(j0, "Pj[SMALL_SIZE][7]"),
            7,
            32,
        )
    )
    parts.append(
        emit_2d(
            "Y0_ZEROS",
            "/// Degree-3 fits around the first 64 zeros of `y0`.",
            array_body(j0, "Py[SMALL_SIZE][7]"),
            7,
            32,
        )
    )
    for stem, rust, doc in [
        ("pR", "P0R", "/// `pzerof`'s numerator, one row per interval of `1/x`."),
        ("pS", "P0S", "/// `pzerof`'s denominator."),
        ("qR", "Q0R", "/// `qzerof`'s numerator."),
        ("qS", "Q0S", "/// `qzerof`'s denominator."),
    ]:
        parts.append(four(j0, stem, 32, doc, rust))

    r1 = scalars(j1, ["r00", "r01", "r02", "r03", "s01", "s02", "s03", "s04", "s05"], 32)
    parts.append(
        emit_flat(
            "J1_R",
            "/// `j1f`'s numerator on `[0, 2]`.",
            [("", r1[f"r0{i}"]) for i in range(4)],
            4,
            32,
        )
    )
    parts.append(
        emit_flat(
            "J1_S",
            "/// `j1f`'s denominator on `[0, 2]`.",
            [("", r1[f"s0{i}"]) for i in range(1, 6)],
            4,
            32,
        )
    )
    parts.append(
        emit_flat("Y1_U", "/// `y1f`'s numerator on `[0, 2]`.", numbers(array_body(j1, "U0[5]")), 4, 32)
    )
    parts.append(
        emit_flat("Y1_V", "/// `y1f`'s denominator on `[0, 2]`.", numbers(array_body(j1, "V0[5]")), 4, 32)
    )
    parts.append(
        emit_2d(
            "J1_ZEROS",
            "/// Degree-3 fits around the first 64 positive zeros of `j1`.",
            array_body(j1, "Pj[SMALL_SIZE][7]"),
            7,
            32,
        )
    )
    parts.append(
        emit_2d(
            "Y1_ZEROS",
            "/// Degree-3 fits around the first 64 zeros of `y1`.",
            array_body(j1, "Py[SMALL_SIZE][7]"),
            7,
            32,
        )
    )
    sc = strip_comments(fetch(src, "s_sincosf_data.c"))
    words = re.findall(r"0x[0-9a-fA-F]+", array_body(sc, "__inv_pio4[24]"))
    parts.append(
        "/// `4/pi` to 192 bits, as six overlapping 32-bit windows per shift.\n"
        "///\n"
        "/// Payne-Hanek reduction for the asymptotic branches: above `x = 2^7`\n"
        "/// or so the argument of the cosine is known to far fewer bits than the\n"
        "/// answer needs, and only an exact `4/pi` recovers them. Shared with\n"
        "/// glibc's own `sinf`/`cosf`, which is where these words come from.\n"
        f"pub static INV_PIO4: [u32; {len(words)}] = [\n"
        + "\n".join(
            "    " + " ".join(f"{w}," for w in words[i : i + 4])
            for i in range(0, len(words), 4)
        )
        + "\n];"
    )
    for stem, rust, doc in [
        ("pr", "P1R", "/// `ponef`'s numerator, one row per interval of `1/x`."),
        ("ps", "P1S", "/// `ponef`'s denominator."),
        ("qr", "Q1R", "/// `qonef`'s numerator."),
        ("qs", "Q1S", "/// `qonef`'s denominator."),
    ]:
        parts.append(four(j1, stem, 32, doc, rust))
    return "\n\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None)
    args = ap.parse_args()

    for path, gen in [
        (ROOT / "src/tables/double/erf.rs", gen_erf),
        (ROOT / "src/tables/double/erfc.rs", gen_erfc),
        (ROOT / "src/tables/single/erf.rs", gen_single),
        (ROOT / "src/tables/double/bessel.rs", gen_bessel_double),
        (ROOT / "src/tables/single/bessel.rs", gen_bessel_single),
    ]:
        path.write_text(gen(args.src))
        print(f"wrote {path.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
