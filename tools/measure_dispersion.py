#!/usr/bin/env python3
"""Quantify benchmark dispersion across 10 alternating runs on identical binary.

Covers cheap rows (floor, round, trunc, sqrt, sqrtf, pow) on in-domain,
plus targeted corpus-adversarial rows (cosh, expm1 on boundary, erfc, y0 on random-bit,
acosh on coherent) at sizes 4096 and 1048576.
"""

from __future__ import annotations

import csv
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


def stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "range": 0.0, "rel_dispersion": 0.0}
    s_vals = sorted(vals)
    n = len(s_vals)
    if n % 2 == 0:
        med = 0.5 * (s_vals[n // 2 - 1] + s_vals[n // 2])
    else:
        med = s_vals[n // 2]
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


def main() -> int:
    reps = 10
    configs = [
        # (corpus, size, target_fns)
        ("in-domain", 4096, ["floor", "round", "trunc", "sqrt", "sqrtf", "pow"]),
        ("in-domain", 1048576, ["floor", "round", "trunc", "sqrt", "sqrtf", "pow"]),
        ("boundary", 1048576, ["cosh", "expm1", "pow"]),
        ("random-bit", 1048576, ["erfc", "y0"]),
        ("coherent", 1048576, ["acosh"]),
    ]

    print(f"Building benchmark binary...")
    env = dict(os.environ)
    env["RUSTFLAGS"] = "-C target-cpu=native"
    subprocess.run(["cargo", "build", "--release", "--example", "bench"], env=env, check=True)

    # Key: (corpus, size) -> list of run dicts
    collected: dict[tuple[str, int], list[dict[str, dict[str, float]]]] = {}

    print(f"Running {reps} alternating runs across configured targets...")
    for rep in range(reps):
        for corpus, size, _ in configs:
            key = (corpus, size)
            if key not in collected:
                collected[key] = []
            print(f"  Rep {rep+1}/{reps}, corpus={corpus}, size={size}...", flush=True)
            data = run_bench(size, corpus)
            collected[key].append(data)

    out_csv = Path("bench_results/dispersion.csv")
    out_json = Path("bench_results/dispersion.json")

    rows = []
    summary_data: dict[str, dict] = {}

    for corpus, size, target_fns in configs:
        key = (corpus, size)
        s_key = str(size)
        runs = collected[key]

        if corpus not in summary_data:
            summary_data[corpus] = {}
        if s_key not in summary_data[corpus]:
            summary_data[corpus][s_key] = {}

        # Also support legacy top-level size key for in-domain
        if corpus == "in-domain":
            if s_key not in summary_data:
                summary_data[s_key] = {}

        for fn in target_fns:
            fn_summary = {}
            metric_keys = set()
            for r in runs:
                if fn in r:
                    metric_keys.update(r[fn].keys())

            metrics = set(k.rsplit("_", 1)[0] for k in metric_keys)
            for m in sorted(metrics):
                spds = [runs[i][fn][f"{m}_speedup"] for i in range(reps) if fn in runs[i] and f"{m}_speedup" in runs[i][fn]]
                nss = [runs[i][fn][f"{m}_ns"] for i in range(reps) if fn in runs[i] and f"{m}_ns" in runs[i][fn]]
                spd_stat = stats(spds)
                ns_stat = stats(nss)
                fn_summary[f"{m}_speedup"] = spd_stat
                fn_summary[f"{m}_ns"] = ns_stat

                rows.append({
                    "corpus": corpus,
                    "size": size,
                    "function": fn,
                    "metric": m,
                    "spd_min": f"{spd_stat['min']:.4f}",
                    "spd_median": f"{spd_stat['median']:.4f}",
                    "spd_max": f"{spd_stat['max']:.4f}",
                    "spd_disp_pct": f"{spd_stat['rel_dispersion']*100:.2f}%",
                    "ns_min": f"{ns_stat['min']:.4f}",
                    "ns_median": f"{ns_stat['median']:.4f}",
                    "ns_max": f"{ns_stat['max']:.4f}",
                    "ns_disp_pct": f"{ns_stat['rel_dispersion']*100:.2f}%",
                })

            summary_data[corpus][s_key][fn] = fn_summary
            if corpus == "in-domain":
                summary_data[s_key][fn] = fn_summary

    fieldnames = [
        "corpus",
        "size",
        "function",
        "metric",
        "spd_min",
        "spd_median",
        "spd_max",
        "spd_disp_pct",
        "ns_min",
        "ns_median",
        "ns_max",
        "ns_disp_pct",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary_data, indent=2))
    print(f"\nWrote dispersion data to {out_csv} and {out_json}")

    print("\nDispersion summary across 10 identical-binary runs:")
    print(f"{'corpus':<12} {'size':<8} {'function':<8} {'metric':<8} {'spd min..max (med)':<22} {'spd disp':<10} {'ns min..max (med)':<22} {'ns disp':<10}")
    print("-" * 105)
    for r in rows:
        print(
            f"{r['corpus']:<12} {r['size']:<8} {r['function']:<8} {r['metric']:<8} "
            f"{r['spd_min']}..{r['spd_max']} ({r['spd_median']})  "
            f"{r['spd_disp_pct']:<10} "
            f"{r['ns_min']}..{r['ns_max']} ({r['ns_median']})  "
            f"{r['ns_disp_pct']:<10}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
