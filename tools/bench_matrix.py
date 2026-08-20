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
