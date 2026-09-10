#!/usr/bin/env python3
"""OpenVLA-OFT-side M3 adapter driver preflight.

Validates the exact selected-pool streaming contract, 72d batch choice and
canonical M3 sample schedule.  The output is the immutable handoff consumed by
the real OFT train loop; it never falls back to modified_libero_rlds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.benchmark.m3_runner_core import D10_HASH, require_execution_guard, validate_batch_probe_summary
from tools.benchmark.m3_scheduled_data import load_schedule
from tools.data.openvla_lerobot_streaming import load_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("openvla_oft",), required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--batch-summary", type=Path, required=True)
    p.add_argument("--streaming-contract", type=Path, required=True)
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
    summary = json.loads(args.batch_summary.read_text(encoding="utf-8"))
    batch = validate_batch_probe_summary(summary)["openvla_oft"]
    if int(batch["micro_batch"]) != args.micro_batch or int(batch["gradient_accumulation"]) != args.grad_accum:
        raise ValueError("driver batch config differs from 72d summary")
    manifest = load_manifest(args.manifest)
    if manifest.get("episode_ids_sha256") != D10_HASH:
        raise ValueError("manifest D10 mismatch")
    schedule = load_schedule(args.schedule, require_equal_data=(args.track == "equal_data"))
    contract = json.loads(args.streaming_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise ValueError("69c streaming contract is not PASS lerobot_streaming")
    if contract.get("source_episode_ids_sha256") != D10_HASH:
        raise ValueError("69c streaming D10 mismatch")
    stats = contract.get("dataset_statistics")
    if not isinstance(stats, dict) or int(stats.get("num_transitions", 0)) <= 0:
        raise ValueError("69c selected-pool statistics missing")
    payload = {
        "schema_version": 1,
        "stage": "M3_openvla_adapter_driver",
        "status": "READY_FOR_TRAIN_LOOP",
        "model": "openvla_oft",
        "track": args.track,
        "order": args.order,
        "mode": args.mode,
        "selected_episode_ids_sha256": D10_HASH,
        "schedule_sha256": schedule.get("schedule_sha256"),
        "reference_count": len(schedule["references"]),
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": 32,
        "dataset_bridge": "lerobot_streaming",
        "full_rlds_materialization": False,
        "modified_libero_rlds_allowed": False,
        "image_aug_required_for_real_m3": True,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "source_root": str(args.source_root),
        "benchmark_training_started": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
