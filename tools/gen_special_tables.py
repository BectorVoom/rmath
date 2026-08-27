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

GLIBC_COMMIT = "04e750e75b73957cf1c791535a3f4319534a52fc"
BASE = f"https://raw.githubusercontent.com/bminor/glibc/{GLIBC_COMMIT}/sysdeps/ieee754"

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
    "s_asincosf_data.c": "flt-32",
    "s_asincoshf_data.c": "flt-32",
    "e_sincoshf_data.c": "flt-32",
    "s_asinh.c": "dbl-64",
    "e_acosh.c": "dbl-64",
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
    for stem, rust, doc in [
        ("pr", "P1R", "/// `ponef`'s numerator, one row per interval of `1/x`."),
        ("ps", "P1S", "/// `ponef`'s denominator."),
        ("qr", "Q1R", "/// `qonef`'s numerator."),
        ("qs", "Q1S", "/// `qonef`'s denominator."),
    ]:
        parts.append(four(j1, stem, 32, doc, rust))
    return "\n\n".join(parts) + "\n"



def gen_single_trig(src: Path | None) -> str:
    """`src/tables/single/trig.rs`: `sinf`/`cosf`/`sincosf`/`tanf` data.

    A different upstream from the CORE-MATH families above -- these are ARM's
    optimized-routines, which glibc took wholesale for the binary32 sine
    family -- so the header names that origin rather than CORE-MATH's.
    """
    sc = strip_comments(fetch(src, "s_sincosf_data.c"))
    parts = [
        HEADER.format(
            title="`sinf`/`cosf`/`sincosf`/`tanf` data: the shared quadrant "
                  "polynomials and the 192-bit `4/pi`.",
            origin="sysdeps/ieee754/flt-32/s_sincosf_data.c",
        ).replace(
            "itself taken from the CORE-MATH project.\n"
            "//! Copyright (c) 2022-2025 Alexei Sibidanov, Paul Zimmermann, Tom Hubrecht and\n"
            "//! Claude-Pierre Jeannerod, MIT licence.",
            "itself taken from ARM's optimized-routines\n"
            "//! project -- a different upstream from the CORE-MATH families in the\n"
            "//! sibling table modules. Copyright (c) 2018-2026 Arm Limited and\n"
            "//! Free Software Foundation, Inc., MIT licence.",
        )
    ]

    # `__sincosf_table[2]`: two `sincos_t` records, the second holding the
    # cosine polynomial negated so that quadrants 2 and 3 come out free. The
    # C is `#if TOINT_INTRINSICS`-conditional in exactly one field, `hpi_inv`;
    # x86-64 does not define it, so the `p+23` prescaled spelling is the one
    # that compiles -- asserted rather than assumed.
    body = array_body(sc, "__sincosf_table[2]")
    recs = re.findall(r"\{\s*\{[^}]*\}(?:[^{}]*)\}", body, re.S)
    assert len(recs) == 2, f"expected 2 sincos_t records, found {len(recs)}"

    fields = ["SIGN", "HPI_INV", "HPI", "C0", "C1", "C2", "C3", "C4", "S1", "S2", "S3"]
    rows = []
    for rec in recs:
        vals = numbers(rec)
        # Four `sign` entries, then `hpi_inv` twice (both arms of the #if),
        # then the nine remaining scalars.
        assert len(vals) == 4 + 2 + 9, f"sincos_t has {len(vals)} literals"
        sign, both_hpi_inv, rest = vals[:4], vals[4:6], vals[6:]
        toint, scaled = both_hpi_inv
        assert scaled[1] == toint[1] * 2.0 ** 24, (
            "hpi_inv's two spellings are not the same value scaled by 2^24"
        )
        rows.append((sign, scaled, rest))
    assert len(fields) == 11

    parts.append(
        "/// The sign of `sin` in quadrants 0..3. Identical in both records,\n"
        "/// asserted by the generator rather than emitted twice."
    )
    assert rows[0][0] == rows[1][0], "the two records' sign tables differ"
    parts[-1] += "\npub static SIGN: [f64; 4] = [\n    " + " ".join(
        cell(v, 64) for _, v in rows[0][0]
    ) + "\n];"

    parts.append(
        "/// `2/pi`, prescaled by `2^24` so that `reduce_fast`'s quadrant lands\n"
        "/// in bits 24..31 of the `int32` conversion.\n"
        "///\n"
        "/// The `TOINT_INTRINSICS` arm of `s_sincosf_data.c` spells this\n"
        "/// `p-1` instead and rounds with a dedicated instruction. x86-64 does\n"
        "/// not define that macro, so this is the constant that compiles.\n"
        f"pub const HPI_INV: f64 = {cell(rows[0][1][1], 64)[:-1]};"
    )
    parts.append(
        "/// `pi/2`, `reduce_fast`'s single subtraction constant.\n"
        f"pub const HPI: f64 = {cell(rows[0][2][0][1], 64)[:-1]};"
    )

    docs = {
        "C0": "/// Cosine polynomial, constant term. `POLY[0]` is `cos`; `POLY[1]` is\n"
              "/// the same polynomial negated, which is how quadrants 2 and 3 get\n"
              "/// their sign for free.",
        "C1": "/// Cosine polynomial, `x^2` coefficient.",
        "C2": "/// Cosine polynomial, `x^4` coefficient.",
        "C3": "/// Cosine polynomial, `x^6` coefficient.",
        "C4": "/// Cosine polynomial, `x^8` coefficient.",
        "S1": "/// Sine polynomial, `x^3` coefficient. Not negated in the second\n"
              "/// record -- only the cosine half is.",
        "S2": "/// Sine polynomial, `x^5` coefficient.",
        "S3": "/// Sine polynomial, `x^7` coefficient.",
    }
    names = ["C0", "C1", "C2", "C3", "C4", "S1", "S2", "S3"]
    parts.append(
        "/// `POLY[k][i]`: the two `sincos_t` records' polynomial coefficients,\n"
        "/// in the order " + ", ".join(f"`{n}`" for n in names) + ".\n"
        "///\n"
        "/// One flat pair of rows rather than a struct, because the only thing\n"
        "/// the algorithm does with the record is index it by `n & 2`.\n"
        "pub static POLY: [[f64; 8]; 2] = [\n"
        + "".join(
            "    [\n        " + " ".join(cell(v, 64) for _, v in row[2][1:]) + "\n    ],\n"
            for row in rows
        )
        + "];"
    )
    for i, n in enumerate(names):
        parts.append(f"{docs[n]}\npub const {n}: usize = {i};")

    words = re.findall(r"0x[0-9a-fA-F]+", array_body(sc, "__inv_pio4[24]"))
    parts.append(
        "/// `4/pi` to 192 bits, as six overlapping 32-bit windows per shift.\n"
        "///\n"
        "/// Payne-Hanek reduction: past `|x| = 120` the argument of the sine is\n"
        "/// known to far fewer bits than the answer needs, and only an exact\n"
        "/// `4/pi` recovers them. Shared with the `f32` Bessel functions, whose\n"
        "/// asymptotic branches reduce the same way.\n"
        f"pub static INV_PIO4: [u32; {len(words)}] = [\n"
        + "\n".join(
            "    " + " ".join(f"{w}," for w in words[i : i + 4])
            for i in range(0, len(words), 4)
        )
        + "\n];"
    )
    return "\n\n".join(parts) + "\n"



def gen_single_invtrig(src: Path | None) -> str:
    """`src/tables/single/invtrig.rs`: `asinf`/`acosf`'s shared accurate path."""
    text = strip_comments(fetch(src, "s_asincosf_data.c"))
    parts = [
        HEADER.format(
            title="`asinf`/`acosf` data: the two degree-11 polynomials their "
                  "accurate paths share.",
            origin="sysdeps/ieee754/flt-32/s_asincosf_data.c",
        )
    ]
    parts.append(
        emit_flat(
            "C0",
            "/// `asin(z)/z - 1` as a degree-11 polynomial in `z^2`, for the\n"
            "/// small-argument accurate path both functions fall back to.",
            numbers(array_body(text, "__asincosf_c0[]")),
            3,
        )
    )
    parts.append(
        emit_flat(
            "C1",
            "/// The near-1 accurate path's polynomial in `1 - |x|`, multiplied\n"
            "/// by `sqrt(1 - |x|)` at the call site — which is where the\n"
            "/// leading `sqrt(2)` in `C1[0]` comes from.",
            numbers(array_body(text, "__asincosf_c1[]")),
            3,
        )
    )
    return "\n\n".join(parts) + "\n"


def gen_single_asincosh(src: Path | None) -> str:
    """`src/tables/single/asincosh.rs`: `asinhf`/`acoshf`'s shared logarithm."""
    text = strip_comments(fetch(src, "s_asincoshf_data.c"))
    parts = [
        HEADER.format(
            title="`asinhf`/`acoshf` data: the 129-entry reciprocal / logarithm "
                  "table and its two polynomials.",
            origin="sysdeps/ieee754/flt-32/s_asincoshf_data.c",
        )
    ]
    ix = numbers(array_body(text, "__asincoshf_ix[]"))
    lix = numbers(array_body(text, "__asincoshf_lix[]"))
    assert len(ix) == 129 and len(lix) == 129, (len(ix), len(lix))
    parts.append(
        emit_flat(
            "IX",
            "/// `IX[j] ~= 1/c_j` for the 128 subintervals of `[1, 2)`, so that\n"
            "/// `w * IX[j] - 1` is the small residual the polynomial sees.",
            ix,
            4,
        )
    )
    parts.append(
        emit_flat(
            "LIX",
            "/// `LIX[j] = log(c_j)`, with `LIX[128] = log(2)` doing double duty\n"
            "/// as the per-exponent scale.",
            lix,
            4,
        )
    )
    parts.append(
        emit_flat(
            "C",
            "/// The fast path's degree-3 correction, `log(1 + z) ~= z * C`.",
            numbers(array_body(text, "__asincoshf_c[]")),
            3,
        )
    )
    parts.append(
        emit_flat(
            "CP",
            "/// The accurate path's degree-6 correction, reached only when the\n"
            "/// fast path's error bound cannot settle the last bit.",
            numbers(array_body(text, "__asincoshf_cp[]")),
            2,
        )
    )
    return "\n\n".join(parts) + "\n"


def gen_single_sincosh(src: Path | None) -> str:
    """`src/tables/single/sincosh.rs`: `sinhf`/`coshf`'s shared exponential."""
    text = strip_comments(fetch(src, "e_sincoshf_data.c"))
    parts = [
        HEADER.format(
            title="`sinhf`/`coshf` data: the 32-entry `2^(k/32)` table and the "
                  "two exponential polynomials.",
            origin="sysdeps/ieee754/flt-32/e_sincoshf_data.c",
        )
    ]
    parts.append(
        emit_flat(
            "C",
            "/// The fast path's degree-3 `2^z` polynomial on `|z| <= 1/64`.",
            numbers(array_body(text, "__sincoshf_c[]")),
            4,
        )
    )
    parts.append(
        emit_flat(
            "CH",
            "/// The accurate path's degree-6 `2^z` polynomial, same interval.",
            numbers(array_body(text, "__sincoshf_ch[]")),
            3,
        )
    )
    words = re.findall(r"0x[0-9a-fA-F]+", array_body(text, "__sincoshf_tb[]"))
    assert len(words) == 32, f"tb has {len(words)} entries"
    parts.append(
        "/// `TB[k] = 2^(k/32 - 1)` as an exact bit pattern: the table half of\n"
        "/// the exponential's `2^(k/32) * 2^z` split. Stored as `u64` because\n"
        "/// that is how the C stores it -- the exponent is added into the bits\n"
        "/// directly rather than by a multiply.\n"
        f"pub static TB: [u64; {len(words)}] = [\n"
        + "\n".join(
            "    " + " ".join(f"{w}," for w in words[i : i + 3])
            for i in range(0, len(words), 3)
        )
        + "\n];"
    )
    return "\n\n".join(parts) + "\n"



def decl_body(text: str, decl: str, occurrence: int = 0) -> str:
    """The braced initialiser of the `occurrence`-th `decl ... = {...}`.

    `array_body` above anchors on the next `{` after the declaration, which
    is enough for a data file but not for a source file: `s_asinh.c` contains
    both a `static const double cl[3] = {...}` and expressions that index
    `cl[3]`, and the naive search lands on a *code* block. Anchoring on the
    `=` between the declaration and the brace rules that out.
    """
    pos = -1
    for _ in range(occurrence + 1):
        pos = text.index(decl, pos + 1)
    eq = text.index("=", pos + len(decl) - 1)
    j = text.index("{", eq)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j : k + 1]
    raise ValueError(f"unterminated initialiser for {decl!r}")


def gen_asincosh_double(src: Path | None) -> str:
    """`src/tables/double/asincosh_acc.rs`: `asinh`/`acosh`'s accurate paths.

    The fast paths' tables live in the sibling `asincosh.rs`. These are the
    ones the *second* step needs — the four-way logarithm table, its
    triple-double logarithms, the two double-double polynomials, and the
    short lists of arguments where even that cannot place the last bit.
    """
    a = strip_comments(fetch(src, "s_asinh.c"))
    c = strip_comments(fetch(src, "e_acosh.c"))
    parts = [
        HEADER.format(
            title="`asinh`/`acosh` accurate-path data: the four-way logarithm "
                  "table, its triple-double logarithms, and the exception lists.",
            origin="sysdeps/ieee754/dbl-64/{s_asinh.c,e_acosh.c}",
        )
    ]

    # `t1`..`t4` and `LL` are byte-identical between the two files -- they are
    # one logarithm shared by both functions. Asserted, then emitted once.
    for name in ["t1[]", "t2[]", "t3[]", "t4[]", "LL[4][17][3]"]:
        va = numbers(decl_body(a, "static const double " + name))
        vc = numbers(decl_body(c, "static const double " + name))
        assert [v for _, v in va] == [v for _, v in vc], (
            f"{name} differs between s_asinh.c and e_acosh.c"
        )

    sizes = {"t1[]": 17, "t2[]": 16, "t3[]": 16, "t4[]": 16}
    docs = {
        "t1[]": "/// `T1[i]` is `2^(-i/16)` to 28 bits: the first of four table\n"
                "/// stages, together reducing the argument to within `2^-16` of 1.",
        "t2[]": "/// The second stage, `2^(-i/256)`.",
        "t3[]": "/// The third stage, `2^(-i/4096)`.",
        "t4[]": "/// The fourth stage, `2^(-i/65536)`.",
    }
    for name, rust in [("t1[]", "T1"), ("t2[]", "T2"), ("t3[]", "T3"), ("t4[]", "T4")]:
        v = numbers(decl_body(a, "static const double " + name))
        assert len(v) == sizes[name], (name, len(v))
        parts.append(emit_flat(rust, docs[name], v, 4))

    ll = numbers(decl_body(a, "static const double LL[4][17][3]"))
    assert len(ll) == 4 * 17 * 3, len(ll)
    parts.append(
        "/// `LL[s][i]` is `-log(T{s+1}[i])` as an unevaluated triple-double.\n"
        "///\n"
        "/// Three limbs, not two: the accurate path exists precisely because\n"
        "/// two are not enough near the argument where the logarithm cancels.\n"
        "/// The four stages' entries are summed, which is what makes one\n"
        "/// 65-thousand-entry logarithm out of four sixteen-entry ones."
        + "\npub static LL: [[[f64; 3]; 17]; 4] = [\n"
        + "".join(
            "    [\n"
            + "".join(
                "        ["
                + " ".join(cell(v, 64) for _, v in ll[(s * 17 + i) * 3 : (s * 17 + i) * 3 + 3])
                + "],\n"
                for i in range(17)
            )
            + "    ],\n"
            for s in range(4)
        )
        + "];"
    )

    for label, text, zero_n, zero_name, zero_doc in [
        ("ASINH", a, 12, "ZERO",
         "the series `asinh(x)/x - 1` in `x^2` that `as_asinh_zero` needs where\n"
         "/// the argument is small enough that the logarithm has nothing left"),
        ("ACOSH", c, 10, "ONE",
         "the series in `x - 1` that `as_acosh_one` needs just above 1, where\n"
         "/// `sqrt(x^2 - 1)` cancels"),
    ]:
        ch = numbers(decl_body(text, "static const double ch[][2]", 0))
        assert len(ch) == 2 * zero_n, (label, len(ch))
        parts.append(
            emit_rows(
                f"{label}_{zero_name}_CH",
                f"/// The double-double half of {zero_doc}.",
                [ch[i : i + 2] for i in range(0, len(ch), 2)],
            )
        )
        cl = numbers(decl_body(text, "static const double cl[]", 0))
        parts.append(
            emit_flat(
                f"{label}_{zero_name}_CL",
                "/// Its single-double tail, evaluated separately and folded in\n"
                "/// as `polydd`'s incoming low word.",
                cl,
                3,
            )
        )
        rch = numbers(decl_body(text, "static const double ch[][2]", 1))
        assert len(rch) == 6, (label, len(rch))
        parts.append(
            emit_rows(
                f"{label}_REFINE_CH",
                "/// The refinement's double-double `log(1 + x)` series.",
                [rch[i : i + 2] for i in range(0, len(rch), 2)],
            )
        )
        rcl = numbers(decl_body(text, "static const double cl[3]"))
        assert len(rcl) == 3, (label, len(rcl))
        parts.append(
            emit_flat(
                f"{label}_REFINE_CL",
                "/// Its single-double tail.",
                rcl,
                3,
            )
        )
        db = numbers(decl_body(text, "static const double db[][3]"))
        assert len(db) % 3 == 0
        parts.append(
            emit_rows(
                f"{label}_DB",
                "/// `(argument, high, low)` for the arguments where even the\n"
                "/// refinement lands on a halfway point. Sorted, and searched\n"
                "/// by bisection; `high + low` is the correctly-rounded answer.",
                [db[i : i + 3] for i in range(0, len(db), 3)],
            )
        )
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
        (ROOT / "src/tables/single/trig.rs", gen_single_trig),
        (ROOT / "src/tables/single/invtrig.rs", gen_single_invtrig),
        (ROOT / "src/tables/single/asincosh.rs", gen_single_asincosh),
        (ROOT / "src/tables/single/sincosh.rs", gen_single_sincosh),
        (ROOT / "src/tables/double/asincosh_acc.rs", gen_asincosh_double),
    ]:
        path.write_text(gen(args.src))
        print(f"wrote {path.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
