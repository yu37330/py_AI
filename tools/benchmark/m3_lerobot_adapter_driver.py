#!/usr/bin/env python3
"""LeRobot-side M3 adapter driver for π0.5 and SmolVLA.

Preflight validates 72d/D10/schedule provenance without loading a model.
Smoke/benchmark equal-data execution delegates to ``m3_lerobot_train_entry.py``,
which injects the exact fixed sampler into the pinned upstream trainer.
Equal-wall still fails closed until its optimizer-boundary checkpoint stop hook
is implemented; it never falls back to a step approximation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.benchmark.m3_runner_core import D10_HASH, require_execution_guard, validate_batch_probe_summary
from tools.benchmark.m3_scheduled_data import load_schedule
from tools.data.openvla_lerobot_streaming import load_manifest

SUPPORTED = {"pi05", "smolvla"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(SUPPORTED), required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--batch-summary", type=Path, required=True)
    p.add_argument("--track", choices=("equal_data", "equal_wall"), required=True)
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--mode", choices=("preflight", "smoke", "benchmark"), required=True)
    p.add_argument("--micro-batch", type=int, required=True)
    p.add_argument("--grad-accum", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    require_execution_guard(args.mode)
    if args.micro_batch * args.grad_accum != 32:
        raise ValueError("M3 effective batch must equal 32")
    batch_summary = json.loads(args.batch_summary.read_text(encoding="utf-8"))
    batches = validate_batch_probe_summary(batch_summary)
    chosen = batches[args.model]
    if int(chosen["micro_batch"]) != args.micro_batch or int(chosen["gradient_accumulation"]) != args.grad_accum:
        raise ValueError("driver batch config differs from 72d summary")
    manifest = load_manifest(args.manifest)
    if manifest.get("episode_ids_sha256") != D10_HASH:
        raise ValueError("manifest D10 mismatch")
    schedule = load_schedule(args.schedule, require_equal_data=(args.track == "equal_data"))

    if args.mode == "preflight":
        payload = {
            "schema_version": 1,
            "stage": "M3_lerobot_adapter_driver",
            "status": "READY_FOR_TRAIN_LOOP",
            "model": args.model,
            "track": args.track,
            "order": args.order,
            "mode": args.mode,
            "selected_episode_ids_sha256": D10_HASH,
            "schedule_sha256": schedule.get("schedule_sha256"),
            "reference_count": len(schedule["references"]),
            "micro_batch": args.micro_batch,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": 32,
            "sampler_requirement": "resolve absolute dataset index through live LeRobotDataset absolute-to-relative map, then fixed schedule order",
            "native_random_sampler_allowed": False,
            "source_root": str(args.source_root),
            "benchmark_training_started": False,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    if args.track == "equal_wall":
        payload = {
            "schema_version": 1,
            "stage": "M3_lerobot_adapter_driver",
            "status": "BLOCKED_SAFE_WALL_STOP_NOT_IMPLEMENTED",
            "model": args.model,
            "track": args.track,
            "order": args.order,
            "selected_episode_ids_sha256": D10_HASH,
            "reason": "Do not approximate 1800 sec with a step budget. A checkpoint-preserving safe optimizer-boundary stop hook is required.",
            "benchmark_training_started": False,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 3

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.out.parent / "checkpoint_run"
    entry = repo_root / "tools/benchmark/m3_lerobot_train_entry.py"
    cmd = [
        sys.executable,
        "-u",
        str(entry),
        "--repo-root",
        str(repo_root),
        "--model",
        args.model,
        "--source-root",
        str(args.source_root),
        "--dataset-root",
        str(args.dataset_root),
        "--manifest",
        str(args.manifest),
        "--schedule",
        str(args.schedule),
        "--micro-batch",
        str(args.micro_batch),
        "--grad-accum",
        str(args.grad_accum),
        "--mode",
        args.mode,
        "--order",
        args.order,
        "--output-dir",
        str(output_dir),
        "--result-out",
        str(args.out),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
