#!/usr/bin/env python3
"""Quantify benchmark dispersion for cheap rows (floor, round, trunc, sqrt)
across 10 alternating runs on identical binary at sizes 4096 and 1048576.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path


def run_bench(size: int, corpus: str = "in-domain") -> dict[str, dict[str, float]]:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        env = dict(os.environ)
        env["RUSTFLAGS"] = "-C target-cpu=native"
        cmd = [
            "cargo",
            "run",
            "--release",
            "--example",
            "bench",
            "--",
            f"--size={size}",
            f"--corpus={corpus}",
            f"--csv={tmp_path}",
        ]
        subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        lines = [l for l in tmp_path.read_text(encoding="utf-8").splitlines() if not l.startswith("#") and l.strip()]
        reader = csv.DictReader(lines)
        data: dict[str, dict[str, float]] = {}
        for row in reader:
            fn = row["function"]
            metric = row["metric"]
            speedup = float(row["speedup"])
            ns = float(row["ns_per_elem"])
            data.setdefault(fn, {})[f"{metric}_speedup"] = speedup
            data.setdefault(fn, {})[f"{metric}_ns"] = ns
        return data
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> int:
    target_fns = ["floor", "round", "trunc", "sqrt"]
    sizes = [4096, 1048576]
    reps = 10

    print(f"Building benchmark binary...")
    env = dict(os.environ)
    env["RUSTFLAGS"] = "-C target-cpu=native"
    subprocess.run(["cargo", "build", "--release", "--example", "bench"], env=env, check=True)

    results: dict[str, list[dict[str, dict[str, float]]]] = {str(s): [] for s in sizes}

    print(f"Running {reps} alternating runs across sizes {sizes}...")
    for rep in range(reps):
        for s in sizes:
            print(f"  Rep {rep+1}/{reps}, size {s}...", flush=True)
            data = run_bench(s)
            results[str(s)].append(data)

    out_csv = Path("bench_results/dispersion.csv")
    out_json = Path("bench_results/dispersion.json")

    rows = []
    summary_data = {}

    for s in sizes:
        s_key = str(s)
        summary_data[s_key] = {}
        for fn in target_fns:
            exact_speedups = [results[s_key][r][fn]["exact_speedup"] for r in range(reps)]
            exact_ns = [results[s_key][r][fn]["exact_ns"] for r in range(reps)]
            exact_f_speedups = [results[s_key][r][fn].get("exact/F_speedup", exact_speedups[r]) for r in range(reps)]
            exact_f_ns = [results[s_key][r][fn].get("exact/F_ns", exact_ns[r]) for r in range(reps)]

            def stats(vals: list[float]) -> dict[str, float]:
                s_vals = sorted(vals)
                med = 0.5 * (s_vals[4] + s_vals[5])
                mn = s_vals[0]
                mx = s_vals[-1]
                rng = mx - mn
                rel_disp = rng / med if med > 0 else 0.0
                return {
                    "min": mn,
                    "max": mx,
                    "median": med,
                    "range": rng,
                    "rel_dispersion": rel_disp,
                }

            fn_stats = {
                "exact_speedup": stats(exact_speedups),
                "exact_ns": stats(exact_ns),
                "exact_f_speedup": stats(exact_f_speedups),
                "exact_f_ns": stats(exact_f_ns),
            }
            summary_data[s_key][fn] = fn_stats

            rows.append({
                "size": s,
                "function": fn,
                "exact_spd_min": f"{fn_stats['exact_speedup']['min']:.4f}",
                "exact_spd_median": f"{fn_stats['exact_speedup']['median']:.4f}",
                "exact_spd_max": f"{fn_stats['exact_speedup']['max']:.4f}",
                "exact_spd_disp_pct": f"{fn_stats['exact_speedup']['rel_dispersion']*100:.2f}%",
                "exact_ns_min": f"{fn_stats['exact_ns']['min']:.4f}",
                "exact_ns_median": f"{fn_stats['exact_ns']['median']:.4f}",
                "exact_ns_max": f"{fn_stats['exact_ns']['max']:.4f}",
                "exact_ns_disp_pct": f"{fn_stats['exact_ns']['rel_dispersion']*100:.2f}%",
            })

    # Write CSV
    fieldnames = [
        "size",
        "function",
        "exact_spd_min",
        "exact_spd_median",
        "exact_spd_max",
        "exact_spd_disp_pct",
        "exact_ns_min",
        "exact_ns_median",
        "exact_ns_max",
        "exact_ns_disp_pct",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary_data, indent=2))
    print(f"\nWrote dispersion data to {out_csv} and {out_json}")

    # Print summary table
    print("\nDispersion summary across 10 identical-binary runs:")
    print(f"{'size':<8} {'function':<8} {'spd min..max (med)':<22} {'spd disp':<10} {'ns min..max (med)':<22} {'ns disp':<10}")
    print("-" * 82)
    for r in rows:
        print(
            f"{r['size']:<8} {r['function']:<8} "
            f"{r['exact_spd_min']}..{r['exact_spd_max']} ({r['exact_spd_median']})  "
            f"{r['exact_spd_disp_pct']:<10} "
            f"{r['exact_ns_min']}..{r['exact_ns_max']} ({r['exact_ns_median']})  "
            f"{r['exact_ns_disp_pct']:<10}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
