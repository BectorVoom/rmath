#!/usr/bin/env python3
"""Run benchmark matrix over canonical sizes, corpora, and suites.

Usage:
    python3 tools/bench_matrix.py [--output-dir=DIR] [--quick]

Generates CSV files for each matrix entry and writes a manifest.json linking:
- git commit hash
- timestamp
- target CPU / target arch / features
- rustc version
- benchmark command and parameters
- CSV file paths
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench_results"),
        help="Directory to save CSVs and manifest (default: bench_results)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller subset of matrix configurations for quick validation",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = get_git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    corpora = ["in-domain", "boundary", "random-bit", "coherent", "mixed-special"]
    sizes = [64, 4096, 1 << 20] if not args.quick else [4096]
    suites = ["default", "traversal", "repair"]

    runs = []

    # Build bench example first once
    env = dict(os.environ)
    if "RUSTFLAGS" not in env:
        env["RUSTFLAGS"] = "-C target-cpu=native"

    print("Building benchmark binary...")
    subprocess.run(
        ["cargo", "build", "--release", "--example", "bench"],
        env=env,
        check=True,
    )

    for suite in suites:
        if suite in ("traversal", "repair"):
            csv_name = f"{suite}.csv"
            csv_path = out_dir / csv_name
            cmd = [
                "cargo",
                "run",
                "--release",
                "--example",
                "bench",
                "--",
                f"--suite={suite}",
                f"--csv={csv_path}",
            ]
            print(f"Running {' '.join(cmd)}...")
            subprocess.run(cmd, env=env, check=True)
            runs.append({
                "suite": suite,
                "corpus": "default",
                "size": 1 << 20,
                "csv": str(csv_path),
                "cmd": cmd,
            })
        else:
            for corpus in corpora:
                for size in sizes:
                    csv_name = f"{suite}_{corpus}_{size}.csv"
                    csv_path = out_dir / csv_name
                    cmd = [
                        "cargo",
                        "run",
                        "--release",
                        "--example",
                        "bench",
                        "--",
                        f"--suite={suite}",
                        f"--corpus={corpus}",
                        f"--size={size}",
                        f"--csv={csv_path}",
                    ]
                    print(f"Running {' '.join(cmd)}...")
                    subprocess.run(cmd, env=env, check=True)
                    runs.append({
                        "suite": suite,
                        "corpus": corpus,
                        "size": size,
                        "csv": str(csv_path),
                        "cmd": cmd,
                    })

    # Check for exact > fast inversions and Finite < FullRange inversions
    print("\nChecking for inversions across matrix runs...")
    inversions = 0
    for r in runs:
        if r["suite"] != "default":
            continue
        p = Path(r["csv"])
        if not p.exists():
            continue
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if not l.startswith("#") and l.strip()]
        import csv
        reader = csv.DictReader(lines)
        metrics: dict[str, dict[str, float]] = {}
        for row in reader:
            fn = row["function"]
            m = row["metric"]
            spd = float(row["speedup"])
            metrics.setdefault(fn, {})[m] = spd

        for fn, m_map in metrics.items():
            if "exact" in m_map and "fast" in m_map:
                exact_spd = m_map["exact"]
                fast_spd = m_map["fast"]
                # Exclude hardware instructions (sqrt) or within 5% noise margin
                if fn not in ("sqrt", "sqrtf", "floor", "trunc", "round", "abs", "ceil") and exact_spd > fast_spd * 1.05:
                    print(
                        f"  [INVERSION exact > fast] {fn} on corpus='{r['corpus']}', size={r['size']}: "
                        f"exact {exact_spd:.2f}x > fast {fast_spd:.2f}x (artifact or coherence bypass)"
                    )
                    inversions += 1

            # Check if Finite is unexpectedly slower than FullRange by > 10%
            if fn not in ("sqrt", "sqrtf", "floor", "trunc", "round", "abs", "ceil"):
                if "exact" in m_map and "exact/F" in m_map:
                    exact_spd = m_map["exact"]
                    exact_f_spd = m_map["exact/F"]
                    if exact_f_spd < exact_spd * 0.90:
                        print(
                            f"  [INVERSION Finite < FullRange] {fn} (exact) on corpus='{r['corpus']}', size={r['size']}: "
                            f"exact/F {exact_f_spd:.2f}x < exact {exact_spd:.2f}x (wiring bug or adversarial noise)"
                        )
                        inversions += 1
                if "fast" in m_map and "fast/F" in m_map:
                    fast_spd = m_map["fast"]
                    fast_f_spd = m_map["fast/F"]
                    if fast_f_spd < fast_spd * 0.90:
                        print(
                            f"  [INVERSION Finite < FullRange] {fn} (fast) on corpus='{r['corpus']}', size={r['size']}: "
                            f"fast/F {fast_f_spd:.2f}x < fast {fast_spd:.2f}x (wiring bug or adversarial noise)"
                        )
                        inversions += 1
    if inversions == 0:
        print("  No unexpected inversions found.")

    manifest = {
        "commit": commit,
        "timestamp": timestamp,
        "rustflags": env.get("RUSTFLAGS", ""),
        "runs": runs,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote benchmark manifest to {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
