#!/usr/bin/env python3
"""Generate `src/tables/*.rs` from ARM optimized-routines' C data files.

    python3 tools/gen_tables.py            # download the sources, regenerate
    python3 tools/gen_tables.py --src DIR  # use already-downloaded sources

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
SOURCES = ["exp_data.c", "log_data.c"]

# The `#if` environment the C build would have for the double-precision,
# N = 128 configuration. Both tables are selected by these.
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
    return re.findall(r"-?0x[0-9a-fA-F.]+(?:p[+-]?\d+)?|-?\d+\.\d+(?:e-?\d+)?", text)


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
    poly = [hexfloat(t) for t in nums(field(text, "poly"))]
    exp2_poly = [hexfloat(t) for t in nums(field(text, "exp2_poly"))]
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
    poly = [hexfloat(t) for t in nums(field(text, "poly"))]
    poly1 = [hexfloat(t) for t in nums(field(text, "poly1"))]
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


EMITTERS = {"exp_data.c": ("exp.rs", emit_exp), "log_data.c": ("log.rs", emit_log)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, help="directory holding the upstream .c files")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "src" / "tables")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        if args.src:
            raw = (args.src / src).read_text()
        else:
            url = f"{BASE}/{src}"
            print(f"fetching {url}")
            with urllib.request.urlopen(url) as fh:
                raw = fh.read().decode()
        text = strip_comments(preprocess(raw, DEFS))
        name, emit = EMITTERS[src]
        dst = args.out / name
        dst.write_text(emit(text, DEFS))
        print(f"wrote {dst} ({len(dst.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
