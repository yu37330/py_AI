#!/usr/bin/env python3
"""Execute exactly one promoted Track3 final-candidate run from its frozen config.

Requires PR #66 training adapters to be integrated. Planning is fail-closed and
live execution additionally requires `PARC_M3_EXECUTE=1` plus `--execute`.
No simulator evaluation or upload is started here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.submission.freeze_track3_final_candidate import D10_HASH, validate_manifest
from tools.submission.link_track3_final_training_result import (
    link_result,
    validate_final_run_config,
)

TRAINING_SCHEDULE_SEED = 20260906
MIN_A100_VRAM_MIB = 38_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--final-run-config", type=Path, required=True)
    p.add_argument("--batch-summary", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--plan-out", type=Path, required=True)
    p.add_argument("--linked-result-out", type=Path, required=True)
    p.add_argument("--streaming-contract", type=Path)
    p.add_argument("--parc-root", type=Path, default=Path("/content/parc2026"))
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def _load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def main() -> int:
    args = parse_args()
    final_cfg = validate_final_run_config(args.final_run_config)
    candidate = str(final_cfg["candidate"])
    track = str(final_cfg["budget"]["track"])
    validate_manifest(args.manifest)

    # PR #66 is intentionally imported late so this branch can exist before
    # integration. Final execution must fail closed until those adapters are on
    # the same checked-out source tree.
    try:
        from tools.benchmark.m3_batch_probe_common import gpu_info
        from tools.benchmark.m3_model_adapters import (
            build_post_training_command,
            build_training_command,
            default_runtimes,
            load_batch_summary,
            runtime_preflight,
        )
        from tools.benchmark.m3_scheduled_data import load_schedule
    except ImportError as exc:
        raise RuntimeError("final-candidate runner requires integrated PR #66 M3 adapters") from exc

    batch_summary = load_batch_summary(args.batch_summary)
    batch_cfg = batch_summary["models"][candidate]
    if batch_cfg.get("source_ref") != final_cfg["source_ref"]:
        raise ValueError("72d batch summary source_ref differs from final-run source_ref")
    if int(batch_cfg.get("micro_batch", 0)) * int(batch_cfg.get("gradient_accumulation", 0)) != 32:
        raise ValueError("72d batch summary does not preserve effective batch 32")

    schedule = load_schedule(args.schedule, require_equal_data=True)
    if schedule.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("final-run schedule D10 mismatch")
    if int(schedule.get("seed", -1)) != TRAINING_SCHEDULE_SEED:
        raise ValueError("final-run schedule must use frozen training schedule seed 20260906")

    runtimes = default_runtimes(args.parc_root, args.repo_root)
    runtime = runtimes[candidate]
    if runtime.source_ref != final_cfg["source_ref"]:
        raise ValueError("runtime source_ref differs from immutable final-run config")

    blockers = runtime_preflight(runtime, repo=args.repo_root, strict_files=True)
    if candidate == "openvla_oft":
        if args.streaming_contract is None or not args.streaming_contract.is_file():
            blockers.append("openvla_oft:validated_69c_streaming_contract_missing")
    if not args.dataset_root.exists():
        blockers.append(f"dataset_root_missing:{args.dataset_root}")
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        blockers.append("execution_guard_missing:PARC_M3_EXECUTE=1")

    try:
        gpu_name, gpu_vram_mib = gpu_info()
        if "A100" not in gpu_name or int(gpu_vram_mib) < MIN_A100_VRAM_MIB:
            blockers.append(f"hardware_gate_failed:{gpu_name}:{gpu_vram_mib}MiB")
    except Exception as exc:
        gpu_name, gpu_vram_mib = "UNKNOWN", 0
        blockers.append(f"hardware_probe_failed:{type(exc).__name__}:{exc}")

    run_root = args.output_root / candidate
    checkpoint_root = run_root / "checkpoint"
    raw_result = run_root / "raw_final_training_result.json"
    if run_root.exists() and any(run_root.iterdir()):
        blockers.append(f"final_output_not_empty:{run_root}; choose a fresh output root")

    command = None
    post_command = None
    if not blockers:
        # `order=forward` is only a compatibility label required by the shared
        # M3 training entry. The final candidate is a single-model run; order is
        # no longer a comparison variable, and the same frozen schedule is used.
        command = build_training_command(
            runtime=runtime,
            repo=args.repo_root,
            batch_cfg=batch_cfg,
            schedule_path=args.schedule,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            track=track,
            order="forward",
            mode="benchmark",
            output_dir=checkpoint_root,
            result_out=raw_result,
            streaming_contract=args.streaming_contract,
        )
        post_command = build_post_training_command(
            runtime=runtime,
            repo=args.repo_root,
            output_dir=checkpoint_root,
            result_out=raw_result,
        )

    plan = {
        "schema_version": 1,
        "stage": "Track3_final_candidate_execution_plan",
        "status": "READY_FOR_FINAL_RUN_EXECUTION" if not blockers else "BLOCKED",
        "candidate": candidate,
        "source_ref": final_cfg["source_ref"],
        "selected_episode_ids_sha256": D10_HASH,
        "final_run_config_sha256": final_cfg["final_run_config_sha256"],
        "track": track,
        "training_schedule_seed": TRAINING_SCHEDULE_SEED,
        "compatibility_order_label": "forward",
        "order_is_not_a_final_comparison_variable": True,
        "micro_batch": int(batch_cfg.get("micro_batch", 0)),
        "gradient_accumulation": int(batch_cfg.get("gradient_accumulation", 0)),
        "effective_batch_size": 32,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "hardware_gate": "single NVIDIA A100 with >=38000 MiB",
        "blockers": blockers,
        "command": command,
        "post_training_command": post_command,
        "raw_training_result": str(raw_result),
        "linked_training_result": str(args.linked_result_out),
        "automatic_training": False,
        "automatic_evaluation": False,
        "automatic_upload": False,
    }
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": plan["status"],
        "candidate": candidate,
        "track": track,
        "gpu": gpu_name,
        "blockers": blockers,
        "plan": str(args.plan_out),
    }, indent=2), flush=True)

    if not args.execute:
        return 0 if not blockers else 2
    if blockers:
        raise RuntimeError(f"final candidate execution is blocked: {blockers}")
    if command is None:
        raise RuntimeError("final training command missing")

    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")
    run_root.mkdir(parents=True, exist_ok=False)
    subprocess.run(command, cwd=str(args.repo_root), env=env, check=True)
    if post_command:
        # For a final OpenVLA candidate, retain this one evaluator-ready merged
        # artifact through final evaluation/freeze; #67 must be invoked with
        # PARC_M3_KEEP_MERGED_OPENVLA=1 for the final evaluation.
        subprocess.run(post_command, cwd=str(args.repo_root), env=env, check=True)
    if not raw_result.is_file():
        raise RuntimeError("final training did not produce raw training result")

    linked = link_result(config_path=args.final_run_config, raw_result_path=raw_result)
    args.linked_result_out.parent.mkdir(parents=True, exist_ok=True)
    args.linked_result_out.write_text(json.dumps(linked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "stage": linked["stage"],
        "candidate": candidate,
        "checkpoint_ref": linked["checkpoint_ref"],
        "samples_consumed": linked["samples_consumed"],
        "train_wall_time": linked["metrics"]["train_wall_time"],
        "linked_training_result": str(args.linked_result_out),
        "next": "run full #67 evaluation; if candidate=openvla_oft set PARC_M3_KEEP_MERGED_OPENVLA=1",
        "automatic_evaluation": False,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
