#!/usr/bin/env python3
"""Run one guarded M3 benchmark order (forward or reverse) on A100.

Each order executes equal-data and equal-wall training for all three models using
the exact same frozen training schedule seed/hash. Completed PASS results are
reused; partial outputs are never deleted automatically. OpenVLA remains in its
persistent LoRA/components form and is materialized only by the evaluation
lifecycle in PR #67.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"
TRAINING_SCHEDULE_SEED = 20260906
TRACKS = ("equal_data", "equal_wall")
MODELS_BY_ORDER = {
    "forward": ("pi05", "smolvla", "openvla_oft"),
    "reverse": ("openvla_oft", "smolvla", "pi05"),
}
EQUAL_DATA_SAMPLES = 4800
EQUAL_DATA_UPDATES = 150
EQUAL_WALL_SEC = 1800.0
EFFECTIVE_BATCH = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--attempt", default=os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "1"))
    return p.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_training_result(
    path: Path,
    *,
    model: str,
    order: str,
    track: str,
    expected_source_ref: str,
    equal_data_schedule_sha256: str,
) -> dict[str, Any]:
    result = _load(path)
    expected = {
        "status": "PASS",
        "model": model,
        "order": order,
        "track": track,
        "mode": "benchmark",
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": TRAINING_SCHEDULE_SEED,
        "source_ref": expected_source_ref,
        "effective_batch_size": EFFECTIVE_BATCH,
        "benchmark_training_started": True,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"benchmark result mismatch {path}: {key}={result.get(key)!r} != {value!r}")
    micro = int(result.get("micro_batch", 0))
    ga = int(result.get("gradient_accumulation", 0))
    if micro <= 0 or ga <= 0 or micro * ga != EFFECTIVE_BATCH:
        raise RuntimeError(f"invalid benchmark batch accounting: {path}")
    samples = int(result.get("samples_consumed", -1))
    updates = int(result.get("optimizer_updates", -1))
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"benchmark metrics missing: {path}")
    wall = float(metrics.get("train_wall_time", -1))
    peak = float(metrics.get("peak_train_vram", -1))
    if wall < 0 or peak < 0:
        raise RuntimeError(f"invalid benchmark train metrics: {path}")
    if track == "equal_data":
        if samples != EQUAL_DATA_SAMPLES:
            raise RuntimeError(f"equal-data samples mismatch: {samples} != {EQUAL_DATA_SAMPLES}")
        if updates != EQUAL_DATA_UPDATES:
            raise RuntimeError(f"equal-data optimizer updates mismatch: {updates} != {EQUAL_DATA_UPDATES}")
        if result.get("sampling_schedule_sha256") != equal_data_schedule_sha256:
            raise RuntimeError("equal-data benchmark schedule hash mismatch")
    else:
        if wall + 1e-9 < EQUAL_WALL_SEC:
            raise RuntimeError(f"equal-wall benchmark stopped early: {wall}")
        if samples <= 0 or samples % EFFECTIVE_BATCH != 0:
            raise RuntimeError(f"equal-wall benchmark sample boundary invalid: {samples}")
        if updates <= 0 or updates * EFFECTIVE_BATCH != samples:
            raise RuntimeError("equal-wall optimizer/sample accounting mismatch")
        if not str(result.get("sampling_schedule_sha256") or "").strip():
            raise RuntimeError("equal-wall stream hash missing")
    checkpoint = Path(str(result.get("checkpoint_ref") or ""))
    if not checkpoint.exists():
        raise FileNotFoundError(f"benchmark checkpoint missing: {checkpoint}")
    if model == "openvla_oft":
        if not (checkpoint / "lora_adapter").is_dir():
            raise FileNotFoundError("OpenVLA persistent LoRA adapter missing")
        if not any(checkpoint.glob("action_head--*checkpoint.pt")):
            raise FileNotFoundError("OpenVLA persistent action-head checkpoint missing")
        if not any(checkpoint.glob("proprio_projector--*checkpoint.pt")):
            raise FileNotFoundError("OpenVLA persistent proprio-projector checkpoint missing")
        if not (checkpoint / "dataset_statistics.json").is_file():
            raise FileNotFoundError("OpenVLA persistent dataset statistics missing")
        if result.get("checkpoint_eval_ready") is True:
            raise RuntimeError("intermediate M3 training unexpectedly retained merged OpenVLA weights")
    return result


def _validate_smoke_gate(path: Path) -> dict[str, Any]:
    smoke = _load(path)
    if smoke.get("status") != "PASS" or smoke.get("stage") != "M3_A100_training_smoke":
        raise RuntimeError("Notebook 73 training smoke is not PASS")
    if smoke.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("training smoke D10 mismatch")
    if int(smoke.get("training_schedule_seed", -1)) != TRAINING_SCHEDULE_SEED:
        raise RuntimeError("training smoke seed mismatch")
    if smoke.get("forward_reverse_reuse_same_schedule") is not True:
        raise RuntimeError("training smoke did not prove forward/reverse same schedule")
    if smoke.get("benchmark_training_started") is not False:
        raise RuntimeError("training smoke unexpectedly claims benchmark started")
    if smoke.get("full_1800_second_run_started") is not False:
        raise RuntimeError("training smoke unexpectedly claims full run started")
    return smoke


def _run_one(command: list[str], *, log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")
    with log_path.open("w", encoding="utf-8") as fh:
        subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 benchmark requires explicit PARC_M3_EXECUTE=1")
    if not args.attempt or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in args.attempt):
        raise ValueError("--attempt may contain only letters, digits, '-' and '_'")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).resolve()
    parc_root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    dataset_root = Path(os.environ.get("PARC_DRIVE_DATASET", drive / "datasets/lerobot_libero_plus_v3_train"))
    manifest = Path(
        os.environ.get(
            "PARC_D10_MANIFEST",
            drive / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json",
        )
    )
    batch_summary_path = drive / "model-benchmark-v1/m3_batch_probe_summary.json"
    streaming_contract = drive / "openvla-streaming-selected-v1/streaming_bridge_contract.json"
    smoke_summary_path = drive / "model-benchmark-v1/m3-training-smoke-v1/m3_training_smoke_summary.json"
    schedule_path = drive / f"model-benchmark-v1/m3-schedules-v1/m3_equal_data_seed-{TRAINING_SCHEDULE_SEED}.json"
    output_root = drive / f"model-benchmark-v1/m3-benchmark-v1/attempt-{args.attempt}"
    plan_root = output_root / "plans"
    order_summary_path = output_root / f"m3_training_{args.order}_summary.json"

    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_model_adapters import (  # noqa: PLC0415
        build_run_specs,
        default_runtimes,
        load_batch_summary,
        runtime_preflight,
    )
    from tools.benchmark.m3_scheduled_data import load_schedule  # noqa: PLC0415
    from tools.data.openvla_lerobot_streaming import load_manifest  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    _validate_smoke_gate(_require_file(smoke_summary_path))
    _require_file(batch_summary_path)
    _require_file(streaming_contract)
    _require_file(schedule_path)
    _require_file(manifest)
    _require_file(dataset_root / "meta/info.json")
    if _sha256_file(manifest) != D10_MANIFEST_SHA256:
        raise RuntimeError("D10 manifest file SHA mismatch")
    manifest_data = load_manifest(manifest)
    if manifest_data.get("episode_ids_sha256") != D10_HASH:
        raise RuntimeError("D10 manifest episode hash mismatch")
    schedule = load_schedule(schedule_path, require_equal_data=True)
    if schedule.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("M3 schedule D10 mismatch")
    if int(schedule.get("seed", -1)) != TRAINING_SCHEDULE_SEED:
        raise RuntimeError("M3 schedule seed mismatch")
    equal_data_schedule_sha256 = str(schedule.get("schedule_sha256") or "")
    if not equal_data_schedule_sha256:
        raise RuntimeError("M3 equal-data schedule hash missing")
    batch_summary = load_batch_summary(batch_summary_path)
    contract = _load(streaming_contract)
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming contract is not PASS lerobot_streaming")
    if contract.get("source_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("69c streaming D10 mismatch")

    runtimes = default_runtimes(parc_root, repo)
    blockers: list[str] = []
    for runtime in runtimes.values():
        blockers.extend(runtime_preflight(runtime, repo=repo, strict_files=True))
    if blockers:
        raise RuntimeError(f"M3 benchmark runtime blocked: {blockers}")

    results: list[dict[str, Any]] = []
    for track in TRACKS:
        plan_path = plan_root / f"{args.order}_{track}.json"
        specs = build_run_specs(
            batch_summary=batch_summary,
            schedule_path=schedule_path,
            batch_summary_path=batch_summary_path,
            dataset_root=dataset_root,
            manifest_path=manifest,
            output_root=output_root,
            root=parc_root,
            repo=repo,
            track=track,
            order=args.order,
            mode="benchmark",
            streaming_contract=streaming_contract,
        )
        expected_models = MODELS_BY_ORDER[args.order]
        if tuple(spec["model"] for spec in specs) != expected_models:
            raise RuntimeError(f"benchmark model order drift: {tuple(spec['model'] for spec in specs)} != {expected_models}")
        plan = {
            "schema_version": 1,
            "stage": "M3_benchmark_order_plan",
            "status": "READY_FOR_EXECUTION",
            "order": args.order,
            "track": track,
            "attempt": args.attempt,
            "training_schedule_seed": TRAINING_SCHEDULE_SEED,
            "equal_data_schedule_sha256": equal_data_schedule_sha256,
            "selected_episode_ids_sha256": D10_HASH,
            "runs": specs,
            "openvla_post_training_merge_executed": False,
            "openvla_merge_deferred_to_evaluation": True,
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

        for spec in specs:
            model = str(spec["model"])
            result_path = Path(spec["out"])
            if result_path.is_file():
                result = _validate_training_result(
                    result_path,
                    model=model,
                    order=args.order,
                    track=track,
                    expected_source_ref=str(spec["source_ref"]),
                    equal_data_schedule_sha256=equal_data_schedule_sha256,
                )
                results.append(result)
                print(f"skip completed benchmark: {args.order}/{track}/{model}", flush=True)
                continue

            checkpoint_dir = Path(spec["output_dir"])
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise RuntimeError(
                    f"partial benchmark checkpoint exists without PASS result: {checkpoint_dir}; "
                    "preserved for diagnosis. Use a new --attempt instead of deleting evidence."
                )
            command = spec.get("command")
            if not isinstance(command, list) or not command:
                raise RuntimeError(f"benchmark command missing for {model}/{track}")
            log_path = result_path.parent / "train.log"
            print(f"=== M3 benchmark start: order={args.order} track={track} model={model} ===", flush=True)
            _run_one(command, log_path=log_path, cwd=repo)
            if not result_path.is_file():
                raise RuntimeError(f"benchmark run finished without result: {result_path}")
            result = _validate_training_result(
                result_path,
                model=model,
                order=args.order,
                track=track,
                expected_source_ref=str(spec["source_ref"]),
                equal_data_schedule_sha256=equal_data_schedule_sha256,
            )
            # Deliberately ignore spec['post_training_command'] here. For
            # OpenVLA, the evaluator owns JIT merge/dematerialization.
            results.append(result)
            print(f"=== M3 benchmark PASS: order={args.order} track={track} model={model} ===", flush=True)

    if len(results) != len(TRACKS) * len(MODELS_BY_ORDER[args.order]):
        raise RuntimeError("M3 order benchmark result count mismatch")
    identities = {(r["track"], r["model"]) for r in results}
    if len(identities) != len(results):
        raise RuntimeError("duplicate/missing benchmark order result identities")

    summary = {
        "schema_version": 1,
        "stage": "M3_training_order_benchmark",
        "status": "PASS",
        "order": args.order,
        "attempt": args.attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "training_schedule_seed": TRAINING_SCHEDULE_SEED,
        "equal_data_schedule_sha256": equal_data_schedule_sha256,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "tracks": list(TRACKS),
        "model_sequence": list(MODELS_BY_ORDER[args.order]),
        "result_count": len(results),
        "openvla_merged_checkpoint_materialized": False,
        "openvla_merge_deferred_to_evaluation": True,
        "ready_for_screening_evaluation": True,
        "results": [
            {
                "model": r["model"],
                "track": r["track"],
                "result_path": str(output_root / args.order / r["track"] / f"seed-{TRAINING_SCHEDULE_SEED}" / r["model"] / "train_result.json"),
                "checkpoint_ref": r["checkpoint_ref"],
                "samples_consumed": r["samples_consumed"],
                "optimizer_updates": r["optimizer_updates"],
                "train_wall_time": r["metrics"]["train_wall_time"],
                "peak_train_vram": r["metrics"]["peak_train_vram"],
                "sampling_schedule_sha256": r["sampling_schedule_sha256"],
            }
            for r in results
        ],
    }
    order_summary_path.parent.mkdir(parents=True, exist_ok=True)
    order_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"=== M3 {args.order.upper()} TRAINING BENCHMARK: PASS ===", flush=True)
    print(json.dumps({
        "status": "PASS",
        "order": args.order,
        "result_count": len(results),
        "summary": str(order_summary_path),
        "next": "run M3 screening simulator evaluation for these six checkpoints",
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
