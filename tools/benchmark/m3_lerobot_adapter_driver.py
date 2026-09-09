#!/usr/bin/env python3
"""LeRobot-side M3 adapter driver.

The driver validates the exact 72d batch choice and canonical schedule, resolves
all schedule references into the row order consumed by a D10-filtered
LeRobotDataset, and emits an execution contract for the model-specific train
loop.  It intentionally fails closed rather than falling back to LeRobot's
native random sampler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.benchmark.m3_runner_core import D10_HASH, require_execution_guard, validate_batch_probe_summary
from tools.benchmark.m3_scheduled_data import load_schedule, references_to_relative_indices
from tools.data.openvla_lerobot_streaming import load_episode_metadata, load_manifest

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
    ids = [int(x) for x in manifest["episode_ids"]]
    metadata = load_episode_metadata(args.dataset_root, ids)
    lengths = [int(metadata.loc[episode_id]["length"]) for episode_id in ids]
    schedule = load_schedule(args.schedule, require_equal_data=(args.track == "equal_data"))
    relative_indices = references_to_relative_indices(
        schedule["references"], episode_ids=ids, episode_lengths=lengths
    )
    if len(relative_indices) != len(schedule["references"]):
        raise RuntimeError("schedule row resolution lost samples")
    # The actual train-loop patch is deliberately represented explicitly.  The
    # adapter must consume these indices in order; falling back to shuffle or an
    # EpisodeAwareSampler would violate same_sampling_policy.
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
        "resolved_reference_count": len(relative_indices),
        "resolved_index_sha256": __import__("hashlib").sha256(
            json.dumps(relative_indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": 32,
        "sampler_requirement": "fixed_relative_indices_in_exact_schedule_order",
        "native_random_sampler_allowed": False,
        "source_root": str(args.source_root),
        "benchmark_training_started": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
