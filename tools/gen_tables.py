#!/usr/bin/env python3
"""Generate `src/tables/*.rs` from ARM optimized-routines' C data files.

    python3 tools/gen_tables.py            # download the sources, regenerate
    python3 tools/gen_tables.py --src DIR  # use already-downloaded sources

Run `cargo fmt` afterwards: the emitted files are valid Rust but not
rustfmt-normalised, and leaving them unformatted makes the next `cargo fmt`
look like a source change.

Why generate rather than hand-copy
----------------------------------

The tables are several hundred exact bit patterns. Transcribing them by hand
is a silent-corruption risk with no upside, and it hides *which* upstream
variant a table came from. This script pins that: it evaluates the same
`#if` conditions the C build would (N = 128 for both exp and log), so the
constants it emits are provably the ones glibc compiles into
`__ieee754_exp_fma` / `__ieee754_log_fma` — which is what makes rmath's
`BitExact` policy bit-exact rather than merely accurate.

Every value is emitted as `f64::from_bits(0x...)` with the original hex-float
spelling in a trailing comment. Rust has no hex-float literals, and a decimal
spelling would be a lossy round-trip.

Upstream: https://github.com/ARM-software/optimized-routines (MIT OR
Apache-2.0 WITH LLVM-exception). The same code, by the same author (Szabolcs
Nagy), is what glibc uses; verified byte-identical against glibc's
`sysdeps/ieee754/dbl-64/e_{exp,log}_data.c`.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/ARM-software/optimized-routines/master/math"

# The `#if` environment the C build would have for the double-precision,
# N = 128 configuration. Both double tables are selected by these.
DEFS = {
    "N": 128,
    "EXP_TABLE_BITS": 7,
    "EXP_POLY_ORDER": 5,
    "EXP2_POLY_ORDER": 5,
    "LOG_TABLE_BITS": 7,
    "LOG_POLY_ORDER": 6,
    "LOG_POLY1_ORDER": 12,
    "WANT_ROUNDING": 1,
}

# The single-precision configurations. Each file selects its own N, so they
# are kept apart rather than merged into one environment -- `exp2f_data.c`
# and `logf_data.c` both test `N` and mean different things by it.
DEFS_EXP2F = {"N": 32, "EXP2F_TABLE_BITS": 5, "EXP2F_POLY_ORDER": 3, "WANT_ROUNDING": 1}
DEFS_LOGF = {"N": 16, "LOGF_TABLE_BITS": 4, "LOGF_POLY_ORDER": 4, "WANT_ROUNDING": 1}
DEFS_LOG2F = {"N": 16, "LOG2F_TABLE_BITS": 4, "LOG2F_POLY_ORDER": 4, "WANT_ROUNDING": 1}
DEFS_LOG2 = {
    "N": 64,
    "LOG2_TABLE_BITS": 6,
    "LOG2_POLY_ORDER": 7,
    "LOG2_POLY1_ORDER": 11,
    # The FMA build. glibc selects `__log2_fma` on any FMA-capable x86-64, and
    # that build drops `tab2` entirely -- the extra table only exists to
    # compensate for not having a fused multiply-add.
    "HAVE_FAST_FMA": 1,
    "WANT_ROUNDING": 1,
}
DEFS_POW_LOG = {
    "N": 128,
    "POW_LOG_TABLE_BITS": 7,
    "POW_LOG_POLY_ORDER": 8,
    "HAVE_FAST_FMA": 1,
    "WANT_ROUNDING": 1,
}
DEFS_POWF = {
    "N": 16,
    "POWF_LOG2_TABLE_BITS": 4,
    "POWF_LOG2_POLY_ORDER": 5,
    "POWF_SCALE_BITS": 0,
    "TOINT_INTRINSICS": 0,
    "WANT_ROUNDING": 1,
}


# --------------------------------------------------------------------------
# A very small C preprocessor: enough for these two files, and no more.
# --------------------------------------------------------------------------

def c_cond(expr: str, defs: dict) -> bool:
    e = expr.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"(?<![=!<>])!(?!=)", " not ", e)
    e = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)", lambda m: str(int(m.group(1) in defs)), e)
    e = re.sub(
        r"[A-Za-z_][A-Za-z_0-9]*",
        lambda m: m.group(0) if m.group(0) in ("and", "or", "not") else str(defs.get(m.group(0), 0)),
        e,
    )
    return bool(eval(e))


def preprocess(text: str, defs: dict) -> str:
    out: list[str] = []
    stack: list[bool] = [True]
    for line in text.splitlines():
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif|include|define|error)\b(.*)", line.strip())
        if m:
            directive, rest = m.group(1), m.group(2).strip()
            if directive == "if":
                stack.append(all(stack) and c_cond(rest, defs))
            elif directive == "ifdef":
                stack.append(all(stack) and rest in defs)
            elif directive == "ifndef":
                stack.append(all(stack) and rest not in defs)
            elif directive == "elif":
                prev = stack.pop()
                stack.append(all(stack) and not prev and c_cond(rest, defs))
            elif directive == "else":
                prev = stack.pop()
                stack.append(all(stack) and not prev)
            elif directive == "endif":
                stack.pop()
            continue
        if all(stack):
            out.append(line)
    return "\n".join(out)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def field(text: str, name: str) -> str:
    """The initialiser text of `.name = ...`, brace-balanced."""
    i = text.index(f".{name} =")
    j = text.index("=", i) + 1
    depth, k = 0, j
    while True:
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                k += 1
                break
        elif c == "," and depth == 0:
            break
        k += 1
    return text[j:k]


# --------------------------------------------------------------------------
# Value formatting
# --------------------------------------------------------------------------

def hexfloat(tok: str) -> float:
    tok = tok.strip()
    if tok.startswith("-"):
        return -hexfloat(tok[1:])
    return float.fromhex(tok) if tok.lower().startswith("0x") else float(tok)


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def rust_f64(x: float) -> str:
    return f"f64::from_bits(0x{bits(x):016x})"


def rust_const(name: str, x: float, note: str, doc: str | None = None) -> str:
    """A documented `pub const`.

    The upstream spelling becomes the doc comment rather than a trailing
    comment: it is the most useful thing to say about a bare coefficient, and
    it means the generated file satisfies `#![deny(missing_docs)]` without the
    generator having to invent prose for 30 polynomial terms.
    """
    lines = [f"/// {doc}"] if doc else []
    lines.append(f"/// Upstream: `{note}`.")
    lines.append(f"pub const {name}: f64 = {rust_f64(x)};")
    return "\n".join(lines)


def nums(text: str) -> list[str]:
    """Bare numeric literals, for fields that are only literals.

    Guarded: an arithmetic operator anywhere in the field means the literals
    are *not* the values, and scraping them would silently emit the wrong
    constants. That is not hypothetical -- `pow_log_data.c` writes its
    coefficients as `0x1.555555555556p-2 * -2` with the comment "Coefficients
    are scaled to match the scaling during evaluation", and dropping the `* -2`
    produced a `pow` that looked plausible and was wrong in the ninth digit.
    Use `values()` for anything that might carry a factor.
    """
    if re.search(r"[*/]", text):
        raise ValueError(
            "field carries arithmetic; use values() rather than nums():\n" + text.strip()
        )
    return re.findall(r"-?0x[0-9a-fA-F.]+(?:p[+-]?\d+)?|-?\d+\.\d+(?:e-?\d+)?", text)


def elements(text: str) -> list[str]:
    """Split a brace initialiser into its top-level comma-separated elements."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth <= 1:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [e.strip(" \t\n{}") for e in out if e.strip(" \t\n{}")]


def values(text: str, defs: dict) -> list[float]:
    """Evaluate every element of an initialiser as a C constant expression.

    Handles the two forms upstream uses to fold evaluation scaling into the
    stored constants: `x / N / N / N` in `exp2f_data.c` and `x * -2` in
    `pow_log_data.c`. Anything it cannot account for raises rather than being
    silently dropped -- that guard is the whole point of the function.
    """
    out = []
    for element in elements(text):
        e = re.sub(
            r"-?0x[0-9a-fA-F]*\.?[0-9a-fA-F]*(?:p[+-]?\d+)?",
            lambda m: repr(float.fromhex(m.group(0))),
            element,
        )
        e = re.sub(r"\bN\b", str(defs["N"]), e)
        if not re.fullmatch(r"[-+*/(). \t\n0-9e]+", e):
            raise ValueError(f"unparsed initialiser element: {element!r}")
        out.append(float(eval(e)))
    return out


def scalar(text: str, name: str, defs: dict) -> tuple[float, str]:
    """A `.name = <expr>` field, allowing the two `* N` / `/ N` forms upstream uses."""
    raw = " ".join(field(text, name).split())
    m = re.fullmatch(r"(\S+)\s*([*/])\s*N", raw)
    if m:
        v = hexfloat(m.group(1))
        v = v * defs["N"] if m.group(2) == "*" else v / defs["N"]
        return v, raw
    return hexfloat(raw), raw


# --------------------------------------------------------------------------
# Emitters
# --------------------------------------------------------------------------

HEADER = """\
//! {title}
//!
//! GENERATED by `tools/gen_tables.py` — do not edit by hand.
//!
//! Transcribed from ARM optimized-routines `math/{src}` (N = {n}),
//! Copyright (c) 2018-2023 Arm Limited, SPDX-License-Identifier:
//! `MIT OR Apache-2.0 WITH LLVM-exception`. Byte-identical to the table glibc
//! compiles into {glibc}, which is what lets
//! [`crate::policy::BitExact`] mean *bit*-exact and not merely accurate.
//!
//! Values are exact bit patterns; the trailing comment is the upstream
//! hex-float spelling.

"""


def emit_exp(text: str, defs: dict) -> str:
    poly = values(field(text, "poly"), defs)
    exp2_poly = values(field(text, "exp2_poly"), defs)
    tab = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", field(text, "tab"))]
    assert len(tab) == 2 * defs["N"], f"exp tab has {len(tab)} entries"
    assert len(poly) == 4 and len(exp2_poly) == 5

    out = [HEADER.format(
        title="Shared `exp` / `exp2` data: the 128-entry 2^(k/128) table and polynomials.",
        src="exp_data.c", n=defs["N"], glibc="`__ieee754_exp_fma` / `__ieee754_exp2`",
    )]
    w = out.append

    for name, doc in [
        ("invln2N", "N/ln(2). Argument reduction scale: `k = round(x * INVLN2N)`."),
        ("negln2hiN", "-ln(2)/N, high part. Exact in the reduction `x + kd*NEGLN2HIN`."),
        ("negln2loN", "-ln(2)/N, low part."),
        ("shift", "The round-to-nearest-integer trick constant, 0x1.8p52."),
    ]:
        v, raw = scalar(text, name, defs)
        w(rust_const(name.upper(), v, raw, doc) + "\n")

    w("/// `exp` minimax coefficients for `exp(r) - 1` on |r| < ln(2)/256.\n"
      "///\n"
      "/// Named `C2..C5` to match glibc's `e_exp.c`, where `C2 = poly[0]`.")
    for i, c in enumerate(poly):
        w(rust_const(f"C{i + 2}", c, c.hex()))
    w("")

    v, raw = scalar(text, "exp2_shift", defs)
    w(rust_const("EXP2_SHIFT", v, raw, "`exp2`'s rounding constant, `0x1.8p52 / N`.") + "\n")
    w("/// `exp2` minimax coefficients, `C1..C5` as in glibc's `e_exp2.c`.")
    for i, c in enumerate(exp2_poly):
        w(rust_const(f"EXP2_C{i + 1}", c, c.hex()))
    w("")

    w("/// `2^(k/128) ~= H[k] * (1 + T[k])` for `k` in `0..128`, stored as\n"
      "/// `TAB[2k] = bits(T[k])` and `TAB[2k+1] = bits(H[k]) - ((k << 52) / 128)`.\n"
      "///\n"
      "/// The odd entries carry a pre-subtracted exponent so that adding\n"
      "/// `ki << (52 - 7)` reconstructs the scale with one integer add.")
    w(f"pub static TAB: [u64; {len(tab)}] = [")
    for i in range(0, len(tab), 4):
        w("    " + " ".join(f"0x{v:016x}," for v in tab[i:i + 4]))
    w("];")
    return "\n".join(out) + "\n"


def emit_log(text: str, defs: dict) -> str:
    poly = values(field(text, "poly"), defs)
    poly1 = values(field(text, "poly1"), defs)
    tab = [hexfloat(t) for t in nums(field(text, "tab"))]
    assert len(tab) == 2 * defs["N"], f"log tab has {len(tab)} entries"
    assert len(poly) == 5 and len(poly1) == 11

    out = [HEADER.format(
        title="`ln` data: the 128-subinterval `1/c` / `log(c)` table and polynomials.",
        src="log_data.c", n=defs["N"], glibc="`__ieee754_log_fma`",
    )]
    w = out.append

    for name, const, doc in [
        ("ln2hi", "LN2HI", "ln(2), high part."),
        ("ln2lo", "LN2LO", "ln(2), low part."),
    ]:
        v, raw = scalar(text, name, defs)
        w(rust_const(const, v, raw, doc) + "\n")

    w("/// Main-path coefficients for `log1p(r) - r`, |r| < 1/256.")
    for i, c in enumerate(poly):
        w(rust_const(f"A{i}", c, c.hex()))
    w("")
    w("/// Near-1.0 coefficients, used on `0.9375 <= x < 1 + 0x1.09p-4`,\n"
      "/// where the main path's cancellation would cost too much accuracy.")
    for i, c in enumerate(poly1):
        w(rust_const(f"B{i}", c, c.hex()))
    w("")

    w("/// `[invc, logc]` interleaved for each of the 128 subintervals:\n"
      "/// `TAB[2i] = 1/c` and `TAB[2i+1] = log(c)`, with `c` near the centre\n"
      "/// of subinterval `i`.")
    w(f"pub static TAB: [f64; {len(tab)}] = [")
    for i in range(0, len(tab), 2):
        w(f"    {rust_f64(tab[i])}, {rust_f64(tab[i + 1])}, // {i // 2}")
    w("];")
    return "\n".join(out) + "\n"



# --------------------------------------------------------------------------
# Single-precision emitters
#
# The single-precision routines do their arithmetic in `double` and round once
# at the end, so every constant below is an `f64` even though the function it
# serves takes and returns `f32`. That is not an artefact of this generator --
# it is the algorithm, and it is what lets rmath's `f32` kernels be bit-exact
# *and* vectorised: they widen to `f64xN`, replay the schedule, and narrow.
# --------------------------------------------------------------------------

HEADER_F32 = """\
//! {title}
//!
//! GENERATED by `tools/gen_tables.py` — do not edit by hand.
//!
//! Transcribed from ARM optimized-routines `math/{src}` (N = {n}),
//! Copyright (c) 2017-2024 Arm Limited, SPDX-License-Identifier:
//! `MIT OR Apache-2.0 WITH LLVM-exception`. Byte-identical to the table glibc
//! compiles into {glibc}.
//!
//! Every constant here is an `f64`, and deliberately so: the single-precision
//! routines evaluate in double precision and round once at the end. See
//! [`crate::kernels::single`].
//!
//! Values are exact bit patterns; the trailing comment is the upstream
//! hex-float spelling.

"""


def emit_exp2f(text: str, defs: dict) -> str:
    n = defs["N"]
    tab = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", field(text, "tab"))]
    poly = values(field(text, "poly"), defs)
    poly_scaled = values(field(text, "poly_scaled"), defs)
    assert len(tab) == n, f"exp2f tab has {len(tab)} entries, expected {n}"
    assert len(poly) == 3 and len(poly_scaled) == 3

    out = [HEADER_F32.format(
        title="Shared `expf` / `exp2f` / `powf` data: the 32-entry 2^(i/32) table.",
        src="exp2f_data.c", n=n, glibc="`__expf` / `__exp2f` / `__powf`",
    )]
    w = out.append

    for name, const, doc in [
        ("shift_scaled", "SHIFT_SCALED", "`exp2f`'s rounding constant, `0x1.8p52 / N`."),
        ("shift", "SHIFT", "The round-to-nearest-integer trick constant, `0x1.8p52`."),
        ("invln2_scaled", "INVLN2_SCALED", "`N / ln(2)`, `expf`'s reduction scale."),
    ]:
        v, raw = scalar(text, name, defs)
        w(rust_const(const, v, raw, doc) + "\n")

    w("/// `exp2f` coefficients for `2^r - 1` on `|r| < 1/64`, in the order the\n"
      "/// kernel consumes them: `C[0] r^3 + C[1] r^2 + C[2] r + 1`.")
    for i, c in enumerate(poly):
        w(rust_const(f"C{i}", c, c.hex()))
    w("")
    w("/// The same coefficients pre-divided by powers of `N`, for `expf`,\n"
      "/// whose reduced argument is `N` times larger.")
    for i, c in enumerate(poly_scaled):
        w(rust_const(f"CS{i}", c, c.hex()))
    w("")

    w("/// `2^(i/32)` for `i` in `0..32`, with the exponent pre-subtracted so\n"
      "/// that adding `ki << (52 - 5)` reconstructs the scale with one\n"
      "/// integer add.")
    w(f"pub static TAB: [u64; {len(tab)}] = [")
    for i in range(0, len(tab), 4):
        w("    " + " ".join(f"0x{v:016x}," for v in tab[i:i + 4]))
    w("];")
    return "\n".join(out) + "\n"


def emit_logf_like(kind: str):
    """Build an emitter for the `{invc, logc}` table shape.

    `logf_data.c`, `log2f_data.c` and `powf_log2_data.c` all have it, and
    differ only in which scalars accompany it.
    """

    def emit(text: str, defs: dict) -> str:
        n = defs["N"]
        tab = [hexfloat(t) for t in nums(field(text, "tab"))]
        poly = values(field(text, "poly"), defs)
        assert len(tab) == 2 * n, f"{kind} tab has {len(tab)} entries, expected {2 * n}"

        title, src, glibc = {
            "logf": ("`logf` / `log10f` data: the 16-subinterval table.", "logf_data.c", "`__logf`"),
            "log2f": ("`log2f` data: the 16-subinterval table.", "log2f_data.c", "`__log2f`"),
            "powf": ("`powf` data: the 16-subinterval log2 table.", "powf_log2_data.c", "`__powf`"),
        }[kind]
        out = [HEADER_F32.format(title=title, src=src, n=n, glibc=glibc)]
        w = out.append

        if kind == "logf":
            for name, const, doc in [
                ("ln2", "LN2", "ln(2)."),
                ("invln10", "INVLN10", "1 / ln(10), for `log10f`."),
            ]:
                v, raw = scalar(text, name, defs)
                w(rust_const(const, v, raw, doc) + "\n")

        w("/// Polynomial coefficients, in the order the kernel consumes them.")
        for i, c in enumerate(poly):
            w(rust_const(f"A{i}", c, c.hex()))
        w("")

        w("/// `[invc, logc]` interleaved for each subinterval: `TAB[2i] = 1/c`\n"
          "/// and `TAB[2i+1] = log(c)` (log2 for the `log2f` and `powf` tables),\n"
          "/// with `c` near the centre of subinterval `i`.")
        w(f"pub static TAB: [f64; {len(tab)}] = [")
        for i in range(0, len(tab), 2):
            w(f"    {rust_f64(tab[i])}, {rust_f64(tab[i + 1])}, // {i // 2}")
        w("];")
        return "\n".join(out) + "\n"

    return emit



def emit_log2(text: str, defs: dict) -> str:
    n = defs["N"]
    poly = values(field(text, "poly"), defs)
    poly1 = values(field(text, "poly1"), defs)
    tab = [hexfloat(t) for t in nums(field(text, "tab"))]
    assert len(tab) == 2 * n, f"log2 tab has {len(tab)} entries, expected {2 * n}"
    assert len(poly) == 6 and len(poly1) == 10, (len(poly), len(poly1))
    assert ".tab2" not in text, "tab2 should be preprocessed away under HAVE_FAST_FMA"

    out = [HEADER.format(
        title="`log2` data: the 64-subinterval `1/c` / `log2(c)` table and polynomials.",
        src="log2_data.c", n=n, glibc="`__ieee754_log2_fma`",
    )]
    w = out.append

    for name, const, doc in [
        ("invln2hi", "INVLN2HI", "1/ln(2), high part. Exactly representable in 33 bits."),
        ("invln2lo", "INVLN2LO", "1/ln(2), low part."),
    ]:
        v, raw = scalar(text, name, defs)
        w(rust_const(const, v, raw, doc) + "\n")

    w("/// Main-path coefficients, for `log2(1 + r) - r/ln(2)` on |r| < 1/128.")
    for i, c in enumerate(poly):
        w(rust_const(f"A{i}", c, c.hex()))
    w("")
    w("/// Near-1.0 coefficients, used where |log2(x)| < 0x1p-4 and the main\n"
      "/// path's cancellation would cost too much accuracy.")
    for i, c in enumerate(poly1):
        w(rust_const(f"B{i}", c, c.hex()))
    w("")

    w("/// `[invc, logc]` interleaved for each of the 64 subintervals:\n"
      "/// `TAB[2i] = 1/c` and `TAB[2i+1] = log2(c)`, with `c` near the centre\n"
      "/// of subinterval `i`.")
    w(f"pub static TAB: [f64; {len(tab)}] = [")
    for i in range(0, len(tab), 2):
        w(f"    {rust_f64(tab[i])}, {rust_f64(tab[i + 1])}, // {i // 2}")
    w("];")
    return "\n".join(out) + "\n"


def emit_pow_log(text: str, defs: dict) -> str:
    n = defs["N"]
    poly = values(field(text, "poly"), defs)
    # Entries are written through `A(invc, logc, logctail)`, which expands to
    # `{invc, 0, logc, logctail}` -- the zero is an unused padding field that
    # only exists to make the C indexing arithmetic cheaper, so it is dropped
    # here rather than carried into Rust.
    tab = [hexfloat(t) for t in nums(field(text, "tab"))]
    assert len(tab) == 3 * n, f"pow_log tab has {len(tab)} values, expected {3 * n}"
    assert len(poly) == 7, len(poly)

    out = [HEADER.format(
        title="`pow` data: the 128-subinterval log table, carried in double-double.",
        src="pow_log_data.c", n=n, glibc="`__ieee754_pow_fma`",
    )]
    w = out.append

    for name, const, doc in [
        ("ln2hi", "LN2HI", "ln(2), high part."),
        ("ln2lo", "LN2LO", "ln(2), low part."),
    ]:
        v, raw = scalar(text, name, defs)
        w(rust_const(const, v, raw, doc) + "\n")

    w("/// Coefficients for `log(1 + r) - r`; the first-order term is 1 and is\n"
      "/// not stored.")
    for i, c in enumerate(poly):
        w(rust_const(f"A{i}", c, c.hex()))
    w("")

    w("/// `[invc, logc, logctail]` interleaved for each of the 128\n"
      "/// subintervals. `pow` needs the logarithm to more than double\n"
      "/// precision, so `log(c)` is carried as an unevaluated sum of two\n"
      "/// doubles rather than one.\n"
      "///\n"
      "/// Upstream stores a fourth, unused padding field per entry to make the\n"
      "/// C index arithmetic a shift; there is no such benefit here, so it is\n"
      "/// dropped and the stride is 3.")
    w(f"pub static TAB: [f64; {len(tab)}] = [")
    for i in range(0, len(tab), 3):
        w(f"    {rust_f64(tab[i])}, {rust_f64(tab[i + 1])}, {rust_f64(tab[i + 2])}, // {i // 3}")
    w("];")
    return "\n".join(out) + "\n"


# (source file, preprocessor environment, output path relative to `--out`, emitter)
TARGETS = [
    ("exp_data.c", DEFS, "double/exp.rs", emit_exp),
    ("log_data.c", DEFS, "double/log.rs", emit_log),
    ("log2_data.c", DEFS_LOG2, "double/log2.rs", emit_log2),
    ("pow_log_data.c", DEFS_POW_LOG, "double/pow.rs", emit_pow_log),
    ("exp2f_data.c", DEFS_EXP2F, "single/exp.rs", emit_exp2f),
    ("logf_data.c", DEFS_LOGF, "single/log.rs", emit_logf_like("logf")),
    ("log2f_data.c", DEFS_LOG2F, "single/log2.rs", emit_logf_like("log2f")),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, help="directory holding the upstream .c files")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "src" / "tables")
    args = ap.parse_args()

    for src, defs, name, emit in TARGETS:
        if args.src:
            raw = (args.src / src).read_text()
        else:
            url = f"{BASE}/{src}"
            print(f"fetching {url}")
            with urllib.request.urlopen(url) as fh:
                raw = fh.read().decode()
        text = strip_comments(preprocess(raw, defs))
        dst = args.out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(emit(text, defs))
        print(f"wrote {dst} ({len(dst.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
