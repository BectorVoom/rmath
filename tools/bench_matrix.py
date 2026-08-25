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
    parser.add_argument(
        "--check-inversions",
        action="store_true",
        help="Only run inversion checks on existing CSVs in output-dir without running benchmarks",
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

    if not args.check_inversions:
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
    else:
        for p in sorted(out_dir.glob("default_*.csv")):
            # default_{corpus}_{size}.csv
            stem = p.stem.removeprefix("default_")
            if "_" in stem:
                corpus, size_str = stem.rsplit("_", 1)
                try:
                    size = int(size_str)
                except ValueError:
                    size = 1 << 20
            else:
                corpus = "default"
                size = 1 << 20
            runs.append({
                "suite": "default",
                "corpus": corpus,
                "size": size,
                "csv": str(p),
                "cmd": [],
            })

    # Load dispersion.json if available to distinguish signal from noise
    dispersion_data = {}
    disp_json_path = out_dir / "dispersion.json"
    if not disp_json_path.exists():
        disp_json_path = Path("bench_results/dispersion.json")
    if disp_json_path.exists():
        try:
            dispersion_data = json.loads(disp_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Check for exact > fast inversions and Finite < FullRange inversions
    print("\nChecking for inversions across matrix runs...")
    inversions = 0
    dismissed = 0
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

        size_key = str(r["size"])
        corpus_key = r["corpus"]

        for fn, m_map in metrics.items():
            # Lookup recorded dispersion band if available
            fn_disp = None
            if corpus_key in dispersion_data and size_key in dispersion_data[corpus_key]:
                fn_disp = dispersion_data[corpus_key][size_key].get(fn)
            if fn_disp is None and size_key in dispersion_data:
                fn_disp = dispersion_data[size_key].get(fn)

            if "exact" in m_map and "fast" in m_map:
                exact_spd = m_map["exact"]
                fast_spd = m_map["fast"]
                if exact_spd > fast_spd * 1.05:
                    if fn_disp and "exact_speedup" in fn_disp and "fast_speedup" in fn_disp:
                        exact_band = fn_disp["exact_speedup"]
                        fast_band = fn_disp["fast_speedup"]
                        if exact_band["min"] <= fast_band["max"] * 1.05:
                            print(
                                f"  [DISMISSED exact > fast inversion] {fn} on corpus='{r['corpus']}', size={r['size']}: "
                                f"exact {exact_spd:.2f}x > fast {fast_spd:.2f}x (dispersion bands {exact_band['min']:.2f}..{exact_band['max']:.2f} vs {fast_band['min']:.2f}..{fast_band['max']:.2f})"
                            )
                            dismissed += 1
                        else:
                            print(
                                f"  [INVERSION exact > fast] {fn} on corpus='{r['corpus']}', size={r['size']}: "
                                f"exact {exact_spd:.2f}x > fast {fast_spd:.2f}x (artifact or coherence bypass)"
                            )
                            inversions += 1
                    else:
                        print(
                            f"  [INVERSION exact > fast] {fn} on corpus='{r['corpus']}', size={r['size']}: "
                            f"exact {exact_spd:.2f}x > fast {fast_spd:.2f}x (artifact or coherence bypass)"
                        )
                        inversions += 1

            # Check if Finite is unexpectedly slower than FullRange by > 10%
            if "exact" in m_map and "exact/F" in m_map:
                exact_spd = m_map["exact"]
                exact_f_spd = m_map["exact/F"]
                if exact_f_spd < exact_spd * 0.90:
                    disp_exact = fn_disp.get("exact_speedup") if fn_disp else None
                    disp_f = fn_disp.get("exact_f_speedup") or fn_disp.get("exact/F_speedup") if fn_disp else None
                    if disp_exact and disp_f and disp_f["max"] >= disp_exact["min"] * 0.90:
                        print(
                            f"  [DISMISSED Finite < FullRange inversion] {fn} (exact) on corpus='{r['corpus']}', size={r['size']}: "
                            f"exact/F {exact_f_spd:.2f}x < exact {exact_spd:.2f}x (dispersion bands overlap)"
                        )
                        dismissed += 1
                    else:
                        print(
                            f"  [INVERSION Finite < FullRange] {fn} (exact) on corpus='{r['corpus']}', size={r['size']}: "
                            f"exact/F {exact_f_spd:.2f}x < exact {exact_spd:.2f}x (wiring bug or adversarial noise)"
                        )
                        inversions += 1

            if "fast" in m_map and "fast/F" in m_map:
                fast_spd = m_map["fast"]
                fast_f_spd = m_map["fast/F"]
                if fast_f_spd < fast_spd * 0.90:
                    disp_fast = fn_disp.get("fast_speedup") if fn_disp else None
                    disp_f = fn_disp.get("fast_f_speedup") or fn_disp.get("fast/F_speedup") if fn_disp else None
                    if disp_fast and disp_f and disp_f["max"] >= disp_fast["min"] * 0.90:
                        print(
                            f"  [DISMISSED Finite < FullRange inversion] {fn} (fast) on corpus='{r['corpus']}', size={r['size']}: "
                            f"fast/F {fast_f_spd:.2f}x < fast {fast_spd:.2f}x (dispersion bands overlap)"
                        )
                        dismissed += 1
                    else:
                        print(
                            f"  [INVERSION Finite < FullRange] {fn} (fast) on corpus='{r['corpus']}', size={r['size']}: "
                            f"fast/F {fast_f_spd:.2f}x < fast {fast_spd:.2f}x (wiring bug or adversarial noise)"
                        )
                        inversions += 1

    if inversions == 0:
        print(f"  No unexpected inversions found ({dismissed} benign inversions dismissed by dispersion bands).")
    else:
        print(f"  Total unexpected inversions: {inversions} ({dismissed} dismissed by dispersion bands).")

    if not args.check_inversions:
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
