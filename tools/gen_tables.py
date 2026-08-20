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


# --------------------------------------------------------------------------
# The trig family: a different upstream (glibc's own dbl-64 sources, the IBM
# Accurate Mathematical Library port, not ARM-optimized-routines -- glibc's
# trig implementation predates that project and was never migrated), a
# different C struct shape (`mynumber` unions storing each double as two
# `int4` halves, rather than designated-initialiser doubles or hex-float
# tokens), and no `#if` configuration to preprocess: these files have no
# `N`-style build-time parameter, unlike `exp_data.c`/`log_data.c`.
# --------------------------------------------------------------------------

GLIBC_BASE = "https://raw.githubusercontent.com/bminor/glibc/master/sysdeps/ieee754/dbl-64"

TRIG_HEADER = """\
//! {title}
//!
//! GENERATED by `tools/gen_tables.py` — do not edit by hand.
//!
//! Transcribed from glibc's own `sysdeps/ieee754/dbl-64/{src}` (the IBM
//! Accurate Mathematical Library port -- a different upstream from
//! `exp`/`log`/`pow`'s ARM-optimized-routines, and read directly since it is
//! itself the readable C source, not a binary to reverse: glibc's trig
//! implementation predates the ARM project and was never migrated to it).
//! Copyright (C) 2001-2026 Free Software Foundation, Inc.,
//! SPDX-License-Identifier: `LGPL-2.1-or-later`. FMA placement is *not*
//! visible here -- see `src/kernels/double/trig.rs` for the disassembly this
//! was cross-checked against.
//!
//! Values are exact bit patterns; the trailing comment is the upstream
//! hex-float spelling where upstream spells one, otherwise the decimal value.

"""


def strip_c_comments_only(text: str) -> str:
    """Like `strip_comments`, but keeps `#include`/`#ifdef` lines intact.

    The trig sources are read directly rather than run through `preprocess`
    (no build-time `#if` configuration selects among variants here), but they
    still guard `BIG_ENDI`/`LITTLE_ENDI` with `#ifdef`, which this generator
    picks apart directly by locating the `LITTLE_ENDI` span as text.
    """
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def little_endi_span(text: str) -> str:
    """The `#ifdef LITTLE_ENDI ... #else`/`#endif` span, as text."""
    start = text.index("#ifdef LITTLE_ENDI")
    start = text.index("\n", start) + 1
    depth = 1
    i = start
    while depth > 0:
        nl = text.index("\n", i)
        line = text[i:nl].strip()
        if line.startswith("#if") or line.startswith("#ifdef") or line.startswith("#ifndef"):
            depth += 1
        elif line.startswith("#endif"):
            depth -= 1
            if depth == 0:
                return text[start:i]
        elif line.startswith("#else") and depth == 1:
            return text[start:i]
        i = nl + 1
    raise ValueError("unterminated #ifdef LITTLE_ENDI")


def mynumber_bits(text: str, name: str) -> int:
    """A `name = {{lo_hex, hi_hex}}` `mynumber` initialiser's bit pattern.

    `mynumber`'s `int4 i[2]` stores the low 32 bits at index 0 and the high
    32 at index 1 for `LITTLE_ENDI` (matching x86's actual word order), so
    the double's bits are `(hi << 32) | lo`.
    """
    m = re.search(
        rf"\b{re.escape(name)}\s*=\s*\{{\{{\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*\}}\s*\}}",
        text,
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"mynumber initialiser not found: {name}")
    lo, hi = int(m.group(1), 16), int(m.group(2), 16)
    return (hi << 32) | lo


def double_const(text: str, name: str) -> tuple[float, str]:
    """A `static const double name = <hexfloat or decimal>;` field."""
    m = re.search(rf"\b{re.escape(name)}\s*=\s*([^;]+);", text)
    if not m:
        raise ValueError(f"double constant not found: {name}")
    raw = m.group(1).strip()
    return hexfloat(raw), raw


def chained_double_const(text: str, name: str) -> tuple[float, str]:
    """One name from a `static const double a = ..., b = ..., c = ...;` chain.

    `s_sin.c`'s `sn3`/`sn5`/`cs2`/`cs4`/`cs6` are declared this way, as
    decimal literals rather than the hex-float spelling every other constant
    here uses -- `double_const` would over-capture up to the chain's final
    `;` instead of just this one field, so this stops at the next comma too.
    """
    m = re.search(rf"\b{re.escape(name)}\s*=\s*([^;,]+)[;,]", text)
    if not m:
        raise ValueError(f"chained double constant not found: {name}")
    raw = m.group(1).strip()
    return hexfloat(raw), raw


def fetch_glibc(name: str, src_dir: Path | None) -> str:
    if src_dir:
        return (src_dir / name).read_text()
    url = f"{GLIBC_BASE}/{name}"
    print(f"fetching {url}")
    with urllib.request.urlopen(url) as fh:
        return fh.read().decode()


def emit_trig_tables(src_dir: Path | None) -> str:
    usncs = strip_c_comments_only(fetch_glibc("usncs.h", src_dir))
    branred_h = strip_c_comments_only(fetch_glibc("branred.h", src_dir))
    branred_le = little_endi_span(branred_h)
    sincostab = strip_c_comments_only(fetch_glibc("sincostab.c", src_dir))
    sincostab_le = little_endi_span(sincostab)
    s_sin = strip_c_comments_only(fetch_glibc("s_sin.c", src_dir))

    out = [TRIG_HEADER.format(
        title="`sin`/`cos`/`sincos`/`tan` data: the IBM Accurate Math Library's "
              "reduction constants, polynomial coefficients and the 440-entry "
              "`sincostab` / 186-row `xfg` tables.",
        src="usncs.h / branred.h / sincostab.c / utan.h / utan.tbl",
    )]
    w = out.append

    # usncs.h: the Taylor-series polynomial coefficients and the main
    # reduction's constants. All plain `static const double`, so a direct
    # regex read -- no union unpacking needed here.
    for name, const, doc in [
        ("s1", "S1", "`-1/3!`, the linear-in-`x`-times coefficient of `TAYLOR_SIN`."),
        ("s2", "S2", "`1/5!`."),
        ("s3", "S3", "`-1/7!`."),
        ("s4", "S4", "`1/9!`."),
        ("s5", "S5", "`-1/11!`, `TAYLOR_SIN`'s highest-order term."),
        ("big", "BIG", "`1.5 * 2^45`: the round-to-45-bits trick constant `do_sin`/`do_cos` use to split `x` into a leading part exact to the table's index and a residual."),
        ("hp0", "HP0", "`pi/2`, high part -- used to fold the `(0.855469, 2.426265)` band onto `do_cos`."),
        ("hp1", "HP1", "`pi/2`, low part."),
        ("mp1", "MP1", "`pi/2`, high part for `reduce_sincos`'s multiply -- a different split from `HP0`, chosen so `xn * MP1` is exact for the `n` that reduction produces."),
        ("mp2", "MP2", "`pi/2` residue, second part of `reduce_sincos`'s three-part split."),
        ("pp3", "PP3", "`pi/2` residue, `reduce_sincos`'s third part -- carries the reduction to 136 bits total."),
        ("pp4", "PP4", "`pi/2` residue, fourth and final part."),
        ("hpinv", "HPINV", "`2/pi`, for `reduce_sincos`'s quadrant count `t = x*HPINV + TOINT`."),
        ("toint", "TOINT", "The round-to-nearest-integer trick constant, `1.5 * 2^52`."),
    ]:
        v, raw = double_const(usncs, name)
        w(rust_const(const, v, raw, doc) + "\n")

    # s_sin.c: `do_sin`/`do_cos`'s own Taylor coefficients, declared in the
    # kernel file itself rather than `usncs.h`, and as decimal literals
    # rather than hex-float -- the one place this generator does not have an
    # exact-bit-pattern upstream spelling to quote, since upstream itself
    # wrote these in decimal.
    for name, const, doc in [
        ("sn3", "SN3", "`do_sin`/`do_cos`'s cubic coefficient, `-1/3!` to full precision."),
        ("sn5", "SN5", "Quintic coefficient, `1/5!` to full precision."),
        ("cs2", "CS2", "Quadratic coefficient, `1/2!` to full precision."),
        ("cs4", "CS4", "Quartic coefficient, `-1/4!` to full precision."),
        ("cs6", "CS6", "Sextic coefficient, `1/6!` to full precision."),
    ]:
        v, raw = chained_double_const(s_sin, name)
        w(rust_const(const, v, raw, doc) + "\n")

    # branred.h / branred.c: the Payne-Hanek-style reduction for |x| past the
    # table band, `mynumber` unions rather than plain doubles.
    for name, const, doc in [
        ("t576", "T576", "`2^576`, the top scale `__branred` walks `toverp` down from."),
        ("tm600", "TM600", "`2^-600`, pre-scales `x` before splitting it in two."),
        ("tm24", "TM24", "`2^-24`, the per-step scale `__branred`'s loop divides `T576` by."),
        ("big", "BRANRED_BIG", "`1.5 * 2^52` (`= TOINT`), `__branred`'s own round-to-integer constant -- named separately from `usncs.h`'s `BIG` since the two are different magnitudes."),
        ("big1", "BRANRED_BIG1", "`1.5 * 2^54`, four times `BRANRED_BIG`."),
    ]:
        bits = mynumber_bits(branred_le, name)
        w(f"/// {doc}")
        w(f"pub const {const}: f64 = f64::from_bits(0x{bits:016x});\n")

    w("/// `2/pi`, base `2^24`, as 75 24-bit digits -- `__branred`'s arbitrary-\n"
      "/// precision multiplier for the Payne-Hanek reduction. Each entry is an\n"
      "/// exact small integer, stored as `f64` because that is what `__branred`\n"
      "/// multiplies with.")
    toverp_m = re.search(r"toverp\[75\]\s*=\s*\{([^}]*)\}", branred_h, re.S)
    if not toverp_m:
        raise ValueError("toverp[75] not found")
    toverp = [float(x) for x in re.findall(r"-?\d+\.0", toverp_m.group(1))]
    assert len(toverp) == 75, f"toverp has {len(toverp)} entries"
    w("pub const TOVERP: [f64; 75] = [")
    for i in range(0, len(toverp), 5):
        w("    " + " ".join(f"{v:.1f}," for v in toverp[i:i + 5]))
    w("];")
    w("")
    w("/// `2^27 + 1`: splits a double into a high part exact to 27 bits and a\n"
      "/// low residual, exactly (`dla.h`'s `CN`, the classic Dekker split\n"
      "/// constant for `f64`'s 53-bit mantissa: `ceil(53/2) = 27`).")
    w("pub const SPLIT: f64 = 134217729.0;\n")

    # sincostab.c: 440 doubles, sn/ssn/cs/ccs interleaved four at a time, one
    # quadruple per table index `k = 0..110`.
    ints = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", sincostab_le)]
    assert len(ints) == 880, f"sincostab has {len(ints)} u32 halves, expected 880"
    tab = [(ints[i + 1] << 32) | ints[i] for i in range(0, 880, 2)]
    assert len(tab) == 440
    w("/// `SINCOSTAB[4k..4k+4] = [sn, ssn, cs, ccs]` for `k` in `0..110`: the\n"
      "/// sine and cosine of `k/128`, each split into a leading double and a\n"
      "/// residual (`sn + ssn ~= sin(k/128)`, `cs + ccs ~= cos(k/128)`), so that\n"
      "/// `SINCOS_TABLE_LOOKUP`'s reconstruction keeps precision the single\n"
      "/// nearest double would lose. Indexed by the top bits of the reduced\n"
      "/// argument's mantissa (`u.i[LOW_HALF] << 2` in `do_sin`/`do_cos`).")
    w(f"pub static TAB: [u64; {len(tab)}] = [")
    for i in range(0, len(tab), 4):
        w("    " + " ".join(f"0x{v:016x}," for v in tab[i:i + 4]))
    w("];")

    # utan.h / utan.tbl: `tan`'s own data. Most of the reduction constants
    # (`MP1`/`MP2`/`PP3`/`PP4`/`HPINV`/`TOINT`) are bit-identical to
    # `usncs.h`'s, emitted above -- asserted rather than re-emitted. New here
    # are `tan`'s two polynomial sets (`D3..D11`, `E0`/`E1`), the band
    # thresholds, the second reduction's extra `MP3`, and the 186-row `xfg`
    # table.
    utan = strip_c_comments_only(fetch_glibc("utan.h", src_dir))
    utan_le = little_endi_span(utan)
    utan_tbl = strip_c_comments_only(fetch_glibc("utan.tbl", src_dir))
    utan_tbl_le = little_endi_span(utan_tbl)

    for name, const in [
        ("mp1", "MP1"),
        ("mp2", "MP2"),
        ("pp3", "PP3"),
        ("pp4", "PP4"),
        ("hpinv", "HPINV"),
        ("toint", "TOINT"),
    ]:
        shared = struct.unpack("<Q", struct.pack("<d", double_const(usncs, name)[0]))[0]
        assert mynumber_bits(utan_le, name) == shared, f"{const} differs between usncs.h and utan.h"

    for name, const, doc in [
        ("d3", "D3", "`tan(x)`'s direct Taylor series (`polynomial I`), cubic coefficient -- the small-argument and reduced bands' odd polynomial."),
        ("d5", "D5", "Quintic coefficient."),
        ("d7", "D7", "Septic coefficient."),
        ("d9", "D9", "Nonic coefficient."),
        ("d11", "D11", "11th-order coefficient, `polynomial I`'s highest-order term."),
        ("e0", "E0", "`polynomial III`: `e0 + e1*z^2`, the table bands' interpolation polynomial for `tan`."),
        ("e1", "E1", "Quadratic coefficient of `polynomial III`."),
        ("mfftnhf", "MFFTNHF", "`-15.5`, the table index's `256*w - 15.5` offset."),
        ("g1", "G1", "`1.259e-8`: below this, `tan(x)` returns `x`."),
        ("g2", "G2", "`0.0608`, boundary between the direct-Taylor and first-reduction bands."),
        ("g3", "G3", "`0.787`, boundary between the `w`-indexed table band and the quadrant reduction."),
        ("g4", "G4", "`25.0`, the first (three-part `mp`) reduction's ceiling."),
        ("g5", "G5", "`1e8`, the second (four-part `pp`) reduction's ceiling; past this, `__branred`."),
        ("gy2", "GY2", "`0.0608`, the reduced-argument sub-band boundary: below it the odd polynomial, above it the table."),
        ("mp3", "MP3", "`pi/2` residue, the first (`mp`) reduction's third part."),
    ]:
        v = mynumber_bits(utan_le, name)
        w(f"/// {doc}")
        w(f"pub const {const}: f64 = f64::from_bits(0x{v:016x});\n")

    # utan.tbl: 186 rows of four doubles; the fourth column (FFi) is unused by
    # `__tan`'s schedule, so it is dropped and the emitted stride is 3.
    ints = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", utan_tbl_le)]
    assert len(ints) == 186 * 4 * 2, f"utan.tbl has {len(ints)} u32 halves, expected {186 * 4 * 2}"
    flat = [(ints[i + 1] << 32) | ints[i] for i in range(0, len(ints), 2)]
    assert len(flat) == 186 * 4
    xfg = [v for i, v in enumerate(flat) if i % 4 != 3]
    assert len(xfg) == 186 * 3
    w("/// `XFG[3i..3i+3] = [xi, Fi, Gi]` for `i` in `0..186`: `__tan`'s\n"
      "/// per-interval table (indexed by `256*w - 15.5`, `i` in `16..201`).\n"
      "/// `xi` is the interval's base point, and `Fi + Gi ~= tan(xi)` is\n"
      "/// carried as an unevaluated sum so the interpolation keeps accuracy a\n"
      "/// single nearest double would lose. Upstream stores a fourth, unused\n"
      "/// `FFi` column per row; it is dropped here (stride 3, not 4).")
    w(f"pub static XFG: [u64; {len(xfg)}] = [")
    for i in range(0, len(xfg), 3):
        w("    " + " ".join(f"0x{v:016x}," for v in xfg[i:i + 3]))
    w("];")
    return "\n".join(out) + "\n"


def emit_atan_tables(src_dir: Path | None) -> str:
    atnat = strip_c_comments_only(fetch_glibc("atnat.h", src_dir))
    atnat_le = little_endi_span(atnat)
    uatan = strip_c_comments_only(fetch_glibc("uatan.tbl", src_dir))
    uatan_le = little_endi_span(uatan)

    out = [TRIG_HEADER.format(
        title="`atan` data: the IBM Accurate Math Library's reduction "
              "constants, Taylor coefficients and 241-row `cij` table.",
        src="atnat.h / uatan.tbl",
    )]
    w = out.append

    # atnat.h: thresholds and the two Taylor-series coefficient sets, all
    # `mynumber` unions.
    for name, const, doc in [
        ("d3", "D3", "`atan(x)`'s direct Taylor series, cubic coefficient (`A <= u < B` band)."),
        ("d5", "D5", "Quintic coefficient."),
        ("d7", "D7", "Septic coefficient."),
        ("d9", "D9", "Nonic coefficient."),
        ("d11", "D11", "11th-order coefficient."),
        ("d13", "D13", "13th-order coefficient, the series' highest order term."),
        ("a", "A", "Lower threshold: below this, `atan(x)` underflows to `x`."),
        ("b", "B", "`1/16`, boundary between the direct-Taylor and table bands."),
        ("c", "C", "`1`, boundary between the table and reciprocal-table bands."),
        ("d", "D", "`16`, boundary between the reciprocal-table and reciprocal-Taylor bands."),
        ("e", "E", "`5.805e15`, above which `atan(x)` saturates to `+-pi/2`."),
        ("hpi", "HPI", "`pi/2`."),
        ("mhpi", "MHPI", "`-pi/2`."),
        ("hpi1", "HPI1", "`pi/2 - HPI`, the low part of `pi/2`."),
    ]:
        bits = mynumber_bits(atnat_le, name)
        w(f"/// {doc}")
        w(f"pub const {const}: f64 = f64::from_bits(0x{bits:016x});\n")

    # uatan.tbl: cij[241][7] -- for row i, [0] = x0 (the table's base point),
    # [1] = atan(x0)'s leading part, [2..6] = the 5 polynomial coefficients
    # `__atan`/`__atan2` evaluate at `z = u - x0` (direct band) or
    # `z = (w - x0) + ww` (reciprocal band, `w = 1/u`).
    ints = [int(t, 16) for t in re.findall(r"0[xX][0-9a-fA-F]+", uatan_le)]
    assert len(ints) == 241 * 7 * 2, f"uatan.tbl has {len(ints)} u32 halves, expected {241 * 7 * 2}"
    flat = [(ints[i + 1] << 32) | ints[i] for i in range(0, len(ints), 2)]
    assert len(flat) == 241 * 7
    w("/// `CIJ[7i..7i+7] = [x0, t1, c2, c3, c4, c5, c6]` for row `i` in `0..241`:\n"
      "/// `__atan`/`__atan2`'s per-interval table, indexed by\n"
      "/// `((TWO52 + 256*w) - TWO52) - 16` where `w` is `u` (direct band) or\n"
      "/// `1/u` (reciprocal band).")
    w(f"pub static CIJ: [u64; {len(flat)}] = [")
    for i in range(0, len(flat), 4):
        w("    " + " ".join(f"0x{v:016x}," for v in flat[i:i + 4]))
    w("];")
    return "\n".join(out) + "\n"


def emit_asincos_tables(src_dir: Path | None) -> str:
    uasncs = strip_c_comments_only(fetch_glibc("uasncs.h", src_dir))
    uasncs_le = little_endi_span(uasncs)
    asincos = strip_c_comments_only(fetch_glibc("asincos.tbl", src_dir))
    asincos_le = little_endi_span(asincos)
    root = strip_c_comments_only(fetch_glibc("root.tbl", src_dir))
    powtwo_src = strip_c_comments_only(fetch_glibc("powtwo.tbl", src_dir))

    out = [TRIG_HEADER.format(
        title="`asin`/`acos` data: the IBM Accurate Math Library's Taylor\n"
              "//! coefficients, reduction constants, 2568-entry band table and the\n"
              "//! shared reciprocal-square-root seed tables.",
        src="uasncs.h / asincos.tbl / root.tbl",
    )]
    w = out.append

    for name, const, doc in [
        ("hp0", "HP0", "`pi/2`, high part."),
        ("hp1", "HP1", "`pi/2`, low part (`pi/2 - HP0`)."),
    ]:
        bits = mynumber_bits(uasncs_le, name)
        w(f"/// {doc}")
        w(f"pub const {const}: f64 = f64::from_bits(0x{bits:016x});\n")

    for name, const, doc in [
        ("f1", "F1", "`asin`/`acos`'s shared Taylor series, `x^1` coefficient (`|x| < 1/8` band)."),
        ("f2", "F2", "`x^3` coefficient."),
        ("f3", "F3", "`x^5` coefficient."),
        ("f4", "F4", "`x^7` coefficient."),
        ("f5", "F5", "`x^9` coefficient."),
        ("f6", "F6", "`x^11` coefficient, the series' highest order term."),
        ("big", "BIG", "`103079215104.0` (`= 1.5 * 2^36`), unused by the `_fma` build's schedule but kept for provenance."),
        ("t24", "T24", "`2^24`, the round-to-24-bits trick constant the `0.96875 <= |x| < 1` band uses to split `c`."),
        ("t27", "T27", "`2^27`, the same trick at 27 bits, for `acos`'s equivalent split."),
        ("rt0", "RT0", "Newton-step seed refinement constant for `1/sqrt(z)`."),
        ("rt1", "RT1", "Second refinement constant."),
        ("rt2", "RT2", "Third refinement constant."),
        ("rt3", "RT3", "Fourth (last) refinement constant."),
    ]:
        v, raw = chained_double_const(uasncs, name)
        w(rust_const(const, v, raw, doc) + "\n")

    # root.tbl: `inroot[128]`, `1/sqrt(z)`'s seed, plain decimal.
    m = re.search(r"\binroot\s*\[128\]\s*=\s*\{([^}]*)\}", root, re.S)
    if not m:
        raise ValueError("inroot[128] not found")
    vals = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(1))]
    assert len(vals) == 128, f"inroot has {len(vals)} entries"
    w("/// `1/sqrt(z)`'s seed, indexed by the top 7 bits of `z`'s mantissa.")
    w("pub const INROOT: [f64; 128] = [")
    for i in range(0, 128, 4):
        w("    " + " ".join(f"{rust_f64(v)}," for v in vals[i:i + 4]))
    w("];")
    w("")

    # powtwo.tbl: `powtwo[]`, 28 entries (2^0 .. 2^27, exact), plain decimal --
    # a different, much smaller table from `pow_log_data.c`'s (unrelated) own
    # naming, and not sized in its own declaration, so the count comes from
    # what upstream actually wrote rather than an assumed literal.
    m = re.search(r"\bpowtwo\s*\[\s*\]\s*=\s*\{([^}]*)\}", powtwo_src, re.S)
    if not m:
        raise ValueError("powtwo[] not found")
    vals = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(1))]
    w(f"/// The matching power-of-two scale, indexed by `511 - (k>>21)`; upstream\n"
      f"/// spells only `2^0..2^{len(vals) - 1}` ({len(vals)} entries), which is enough\n"
      f"/// range for the exponents `asin`/`acos`'s near-1 band produces.")
    w(f"pub const POWTWO: [f64; {len(vals)}] = [")
    for i in range(0, len(vals), 4):
        w("    " + " ".join(f"{rust_f64(v)}," for v in vals[i:i + 4]))
    w("];")
    w("")

    # asincos.tbl: a flat 2568-entry `double` table (`union { int4 i[5136]; double x[2568]; }`),
    # indexed by band-specific offsets computed from the input's mantissa bits.
    ints = [int(t, 16) for t in re.findall(r"0[xX][0-9a-fA-F]+", asincos_le)]
    assert len(ints) == 2568 * 2, f"asincos.tbl has {len(ints)} u32 halves, expected {2568 * 2}"
    tab = [(ints[i + 1] << 32) | ints[i] for i in range(0, len(ints), 2)]
    assert len(tab) == 2568
    w("/// `asncs.x`: the shared per-interval table for all six of `asin`/`acos`'s\n"
      "/// table bands (`0.125 <= |x| < 0.96875`). Each band indexes a different\n"
      "/// stride/offset into the same flat array -- see `src/reference/double/\n"
      "/// invtrig.rs` for the per-band index arithmetic, transcribed from\n"
      "/// `e_asin.c`.")
    w(f"pub static ASNCS: [u64; {len(tab)}] = [")
    for i in range(0, len(tab), 4):
        w("    " + " ".join(f"0x{v:016x}," for v in tab[i:i + 4]))
    w("];")
    return "\n".join(out) + "\n"


def emit_atan2_tables(src_dir: Path | None) -> str:
    atnat2 = strip_c_comments_only(fetch_glibc("atnat2.h", src_dir))
    atnat2_le = little_endi_span(atnat2)

    out = [TRIG_HEADER.format(
        title="`atan2` data: the constants `__ieee754_atan2` needs beyond\n"
              "//! what `atan` (`atan_data.rs`) already provides -- the shared `d3..d13`\n"
              "//! Taylor coefficients, `HPI`/`HPI1`/`MHPI` and the `1/16` band edge are\n"
              "//! bit-identical between the two tables and are not repeated here.",
        src="atnat2.h",
    )]
    w = out.append

    for name, const, doc in [
        ("opi", "OPI", "`pi`."),
        ("opi1", "OPI1", "`pi - OPI`, the low part of `pi`."),
        ("mopi", "MOPI", "`-pi`."),
        ("qpi", "QPI", "`pi/4`, returned for `atan2(+-inf, +-inf)`."),
        ("mqpi", "MQPI", "`-pi/4`."),
        ("tqpi", "TQPI", "`3*pi/4`."),
        ("mtqpi", "MTQPI", "`-3*pi/4`."),
        ("two500", "TWO500", "`2^500`, the far-apart-exponents rescale for `x`/`y`."),
        ("twom500", "TWOM500", "`2^-500`."),
    ]:
        bits = mynumber_bits(atnat2_le, name)
        w(f"/// {doc}")
        w(f"pub const {const}: f64 = f64::from_bits(0x{bits:016x});\n")

    return "\n".join(out) + "\n"


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

    dst = args.out / "double" / "trig.rs"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(emit_trig_tables(args.src))
    print(f"wrote {dst} ({len(dst.read_text().splitlines())} lines)")

    dst = args.out / "double" / "atan_data.rs"
    dst.write_text(emit_atan_tables(args.src))
    print(f"wrote {dst} ({len(dst.read_text().splitlines())} lines)")

    dst = args.out / "double" / "asincos_data.rs"
    dst.write_text(emit_asincos_tables(args.src))
    print(f"wrote {dst} ({len(dst.read_text().splitlines())} lines)")

    dst = args.out / "double" / "atan2_data.rs"
    dst.write_text(emit_atan2_tables(args.src))
    print(f"wrote {dst} ({len(dst.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
