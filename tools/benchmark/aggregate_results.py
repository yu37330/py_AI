#!/usr/bin/env python3
"""Aggregate small benchmark JSON records into a sortable CSV.

Expected per-run JSON keys are intentionally simple and model-agnostic.
Unknown keys are preserved only in the source JSON, not the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "run_id",
    "model",
    "protocol",
    "track_like",
    "success_rate",
    "steps_to_success",
    "episode_time_s",
    "policy_latency_ms",
    "peak_vram_mib",
    "train_wall_time_s",
    "git_sha",
    "dataset_manifest_sha256",
    "eval_split_sha256",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            rows.extend(data)
        else:
            rows.append(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
