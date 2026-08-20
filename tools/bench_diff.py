#!/usr/bin/env python3
"""Compare two `examples/bench.rs --csv` outputs and flag regressions.

    RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --csv=before.csv
    # ... make a change ...
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --csv=after.csv
    python3 tools/bench_diff.py before.csv after.csv

Each row is `function,metric,speedup[,ns_per_elem]`.
Rejects comparison if critical metadata (corpus, size, suite, widest_f64_lanes, widest_f32_lanes, fma)
does not match or is missing unless `--ignore-metadata` is provided.
Reports relative and absolute movement, gating both speedup drops and direct ns/elem increases.
"""

from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple


class Record(NamedTuple):
    speedup: float
    ns_per_elem: float | None


def load(path: Path) -> tuple[dict[str, str], dict[tuple[str, str], Record]]:
    metadata: dict[str, str] = {}
    rows: dict[tuple[str, str], Record] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if ":" in comment:
                k, v = comment.split(":", 1)
                metadata[k.strip()] = v.strip()
        elif stripped:
            data_lines.append(stripped)

    reader = csv.DictReader(data_lines)
    for row in reader:
        fn = row["function"]
        metric = row["metric"]
        speedup = float(row["speedup"])
        ns = float(row["ns_per_elem"]) if "ns_per_elem" in row and row["ns_per_elem"] else None
        rows[(fn, metric)] = Record(speedup=speedup, ns_per_elem=ns)
    return metadata, rows


def check_metadata(m_before: dict[str, str], m_after: dict[str, str], ignore: bool) -> bool:
    critical_keys = ["corpus", "size", "suite", "widest_f64_lanes", "widest_f32_lanes", "fma"]
    mismatches = []
    for k in critical_keys:
        v_before = m_before.get(k)
        v_after = m_after.get(k)
        if v_before is None or v_after is None:
            mismatches.append((k, str(v_before), str(v_after)))
        elif v_before != v_after:
            mismatches.append((k, v_before, v_after))

    if mismatches:
        print("ERROR: Incomparable or missing benchmark metadata:")
        for k, vb, va in mismatches:
            print(f"  {k}: before='{vb}' vs after='{va}'")
        if not ignore:
            print("Pass --ignore-metadata to override.")
            return False
        else:
            print("Proceeding anyway due to --ignore-metadata.")
    return True


def main() -> int:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="fractional change to report (default 0.03 = 3%%)",
    )
    ap.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="allow comparing benchmarks run with different metadata/corpora",
    )
    args = ap.parse_args()

    meta_b, before = load(args.before)
    meta_a, after = load(args.after)

    if not check_metadata(meta_b, meta_a, args.ignore_metadata):
        return 2

    keys = sorted(set(before) & set(after))
    missing_before = sorted(set(after) - set(before))
    missing_after = sorted(set(before) - set(after))

    moved = []
    regressions = []
    for key in keys:
        b, a = before[key], after[key]
        if b.speedup == 0:
            continue
        delta_speedup = (a.speedup - b.speedup) / b.speedup
        delta_ns = (
            (a.ns_per_elem - b.ns_per_elem) / b.ns_per_elem
            if (b.ns_per_elem and a.ns_per_elem and b.ns_per_elem > 0)
            else 0.0
        )
        # Movement is significant if speedup changed by threshold OR direct ns changed by threshold
        if abs(delta_speedup) >= args.threshold or abs(delta_ns) >= args.threshold:
            moved.append((delta_speedup, delta_ns, key, b, a))
        # Gating: regression if speedup dropped by > threshold OR direct ns increased by > threshold
        if delta_speedup < -args.threshold or delta_ns > args.threshold:
            regressions.append((key, delta_speedup, delta_ns))

    # Sort worst speedup regression first
    moved.sort(key=lambda x: (x[0], -x[1]))

    if not moved and not missing_before and not missing_after:
        print(f"no change past {args.threshold:.0%} across {len(keys)} rows")
        return 0

    if moved:
        has_ns = any(b.ns_per_elem is not None and a.ns_per_elem is not None for _, _, _, b, a in moved)
        if has_ns:
            print(
                f"{'function':<28} {'metric':<12} {'before':>8} {'after':>8} {'speedup_d':>10} {'ns_before':>10} {'ns_after':>10} {'ns_d':>8}"
            )
            for delta_sp, delta_ns, (fn, metric), b, a in moved:
                nb = f"{b.ns_per_elem:.2f}" if b.ns_per_elem is not None else "-"
                na = f"{a.ns_per_elem:.2f}" if a.ns_per_elem is not None else "-"
                ns_d = f"{delta_ns:>+7.1%}" if (b.ns_per_elem and a.ns_per_elem) else "-"
                print(
                    f"{fn:<28} {metric:<12} {b.speedup:>7.2f}x {a.speedup:>7.2f}x {delta_sp:>+9.1%} {nb:>10} {na:>10} {ns_d:>8}"
                )
        else:
            print(f"{'function':<28} {'metric':<12} {'before':>8} {'after':>8} {'delta':>8}")
            for delta_sp, _, (fn, metric), b, a in moved:
                print(f"{fn:<28} {metric:<12} {b.speedup:>7.2f}x {a.speedup:>7.2f}x {delta_sp:>+7.1%}")

    for fn, metric in missing_before:
        print(f"new row (no baseline): {fn} {metric}")
    for fn, metric in missing_after:
        print(f"row dropped: {fn} {metric}")

    if regressions:
        print(f"\nWARNING: {len(regressions)} row(s) regressed past {args.threshold:.0%} threshold:")
        for (fn, metric), d_sp, d_ns in regressions:
            print(f"  {fn} ({metric}): speedup {d_sp:>+7.1%}, ns/elem {d_ns:>+7.1%}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
