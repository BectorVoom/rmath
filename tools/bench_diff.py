#!/usr/bin/env python3
"""Compare two `examples/bench.rs --csv` outputs and flag regressions.

    RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --csv=before.csv
    # ... make a change ...
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --csv=after.csv
    python3 tools/bench_diff.py before.csv after.csv

Each row is `function,metric,speedup` (speedup over the scalar libm call, so
higher is better and the two files must come from the same corpus — a `pow`
row from a stale before.csv is not comparable to a rerun with a different
corpus). Prints every row that moved past `--threshold` (default 3%) either
way, sorted worst regression first, and exits 1 if any regression exceeds it
-- for wiring into a "did I just make something slower" check without eyeballing
a 60-row table.
"""

from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["function"], row["metric"])] = float(row["speedup"])
    return rows


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
    args = ap.parse_args()

    before, after = load(args.before), load(args.after)
    keys = sorted(set(before) & set(after))
    missing_before = sorted(set(after) - set(before))
    missing_after = sorted(set(before) - set(after))

    moved = []
    for key in keys:
        b, a = before[key], after[key]
        if b == 0:
            continue
        delta = (a - b) / b
        if abs(delta) >= args.threshold:
            moved.append((delta, key, b, a))
    moved.sort()  # worst regression (most negative delta) first

    if not moved and not missing_before and not missing_after:
        print(f"no change past {args.threshold:.0%} across {len(keys)} rows")
        return 0

    if moved:
        print(f"{'function':<12} {'metric':<10} {'before':>8} {'after':>8} {'delta':>8}")
        for delta, (fn, metric), b, a in moved:
            print(f"{fn:<12} {metric:<10} {b:>7.2f}x {a:>7.2f}x {delta:>+7.1%}")

    for fn, metric in missing_before:
        print(f"new row (no baseline): {fn} {metric}")
    for fn, metric in missing_after:
        print(f"row dropped: {fn} {metric}")

    return 1 if any(delta < -args.threshold for delta, *_ in moved) else 0


if __name__ == "__main__":
    sys.exit(main())
