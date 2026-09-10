#!/usr/bin/env python3
"""Run the guarded M3 A100 training smoke matrix.

Smoke only. The exact same canonical training schedule is reused for forward
and reverse execution. OpenVLA smoke stops at the LoRA/component checkpoint;
7B merge/finalization is deliberately deferred to simulator evaluation so
multiple smoke runs cannot duplicate large merged checkpoints on Drive.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"
TRAINING_SCHEDULE_SEED = 20260906
ORDERS = ("forward", "reverse")
TRACKS = ("equal_data", "equal_wall")
MODELS = ("pi05", "smolvla", "openvla_oft")
SMOKE_EQUAL_DATA_SAMPLES = 64
SMOKE_WALL_SEC = 5.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _validate_result(path: Path, *, order: str, track: str, model: str) -> dict[str, Any]:
    result = _load_json(path)
    if result.get("status") != "PASS":
        raise RuntimeError(f"smoke result not PASS: {path}")
    expected = {
        "model": model,
        "mode": "smoke",
        "order": order,
        "track": track,
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": TRAINING_SCHEDULE_SEED,
        "effective_batch_size": 32,
        "benchmark_training_started": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"smoke result mismatch {path}: {key}={result.get(key)!r} != {value!r}")
    micro = int(result.get("micro_batch", 0))
    ga = int(result.get("gradient_accumulation", 0))
    if micro <= 0 or ga <= 0 or micro * ga != 32:
        raise RuntimeError(f"invalid smoke batch accounting: {path}")
    samples = int(result.get("samples_consumed", -1))
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"smoke metrics missing: {path}")
    wall = float(metrics.get("train_wall_time", -1))
    peak = float(metrics.get("peak_train_vram", -1))
    if wall < 0 or peak < 0:
        raise RuntimeError(f"invalid smoke train metrics: {path}")
    if track == "equal_data":
        if samples != SMOKE_EQUAL_DATA_SAMPLES:
            raise RuntimeError(f"equal-data smoke sample mismatch: {samples} != {SMOKE_EQUAL_DATA_SAMPLES}")
        if int(result.get("optimizer_updates", -1)) != SMOKE_EQUAL_DATA_SAMPLES // 32:
            raise RuntimeError("equal-data smoke optimizer-update mismatch")
    else:
        if wall + 1e-9 < SMOKE_WALL_SEC:
            raise RuntimeError(f"equal-wall smoke stopped early: {wall}")
        if samples <= 0 or samples % 32 != 0:
            raise RuntimeError(f"equal-wall smoke sample accounting invalid: {samples}")
    checkpoint = Path(str(result.get("checkpoint_ref") or ""))
    if not checkpoint.exists():
        raise FileNotFoundError(f"smoke checkpoint missing: {checkpoint}")
    if model == "openvla_oft":
        if not (checkpoint / "lora_adapter").is_dir():
            raise FileNotFoundError(f"OpenVLA smoke LoRA adapter missing: {checkpoint / 'lora_adapter'}")
        if not any(checkpoint.glob("action_head--*checkpoint.pt")):
            raise FileNotFoundError("OpenVLA smoke action-head checkpoint missing")
        if not any(checkpoint.glob("proprio_projector--*checkpoint.pt")):
            raise FileNotFoundError("OpenVLA smoke proprio-projector checkpoint missing")
        if not (checkpoint / "dataset_statistics.json").is_file():
            raise FileNotFoundError("OpenVLA smoke dataset statistics missing")
        if result.get("checkpoint_eval_ready") is True:
            raise RuntimeError("training smoke unexpectedly materialized a merged OpenVLA checkpoint")
    return result


def _execute_training_plan(plan_path: Path) -> None:
    plan = _load_json(plan_path)
    if plan.get("status") != "READY_FOR_EXECUTION" or plan.get("mode") != "smoke":
        raise RuntimeError(f"invalid M3 smoke plan: {plan_path}")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise RuntimeError(f"M3 smoke plan must contain three model runs: {plan_path}")
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")
    for run in runs:
        command = run.get("command")
        if not isinstance(command, list) or not command:
            raise RuntimeError(f"missing training command in plan: {plan_path}")
        # Intentionally do NOT execute post_training_command here. For OpenVLA
        # that command merges the 7B base model and is an evaluation concern.
        subprocess.run(command, check=True, env=env)


def main() -> int:
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 smoke requires explicit PARC_M3_EXECUTE=1")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).resolve()
    parc_root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    dataset_root = Path(os.environ.get("PARC_DRIVE_DATASET", drive / "datasets/lerobot_libero_plus_v3_train"))
    manifest = Path(
        os.environ.get(
            "PARC_D10_MANIFEST",
            drive
            / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json",
        )
    )
    batch_summary = drive / "model-benchmark-v1/m3_batch_probe_summary.json"
    streaming_contract = drive / "openvla-streaming-selected-v1/streaming_bridge_contract.json"
    schedule_root = drive / "model-benchmark-v1/m3-schedules-v1"
    output_root = drive / "model-benchmark-v1/m3-training-smoke-v1"
    plan_root = output_root / "plans"
    summary_out = output_root / "m3_training_smoke_summary.json"

    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_runner_core import validate_batch_probe_summary  # noqa: PLC0415
    from tools.benchmark.m3_sampling_schedule import (  # noqa: PLC0415
        build_equal_data_schedule,
        load_d10_episode_lengths,
        write_schedule,
    )
    from tools.benchmark.m3_scheduled_data import load_schedule  # noqa: PLC0415
    from tools.data.openvla_lerobot_streaming import load_manifest  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    _require_file(batch_summary)
    _require_file(streaming_contract)
    _require_file(manifest)
    _require_file(dataset_root / "meta/info.json")
    if _sha256_file(manifest) != D10_MANIFEST_SHA256:
        raise RuntimeError("D10 manifest file SHA mismatch")
    manifest_data = load_manifest(manifest)
    if manifest_data.get("episode_ids_sha256") != D10_HASH:
        raise RuntimeError("D10 manifest episode hash mismatch")
    validate_batch_probe_summary(_load_json(batch_summary))
    contract = _load_json(streaming_contract)
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming contract is not PASS lerobot_streaming")
    if contract.get("source_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("69c streaming D10 mismatch")

    episode_ids, episode_lengths = load_d10_episode_lengths(
        manifest_path=manifest,
        dataset_root=dataset_root,
    )
    payload = build_equal_data_schedule(
        episode_ids=episode_ids,
        episode_lengths=episode_lengths,
        seed=TRAINING_SCHEDULE_SEED,
    )
    schedule_root.mkdir(parents=True, exist_ok=True)
    schedule = schedule_root / f"m3_equal_data_seed-{TRAINING_SCHEDULE_SEED}.json"
    if schedule.exists():
        existing = load_schedule(schedule, require_equal_data=True)
        if existing.get("schedule_sha256") != payload.get("schedule_sha256"):
            raise RuntimeError(f"existing M3 schedule differs from deterministic schedule: {schedule}")
    else:
        write_schedule(schedule, payload)

    results: list[dict[str, Any]] = []
    runner = repo / "tools/benchmark/m3_model_adapters.py"
    _require_file(runner)
    reset = os.environ.get("PARC_M3_SMOKE_RESET") == "1"
    for order in ORDERS:
        for track in TRACKS:
            plan = plan_root / f"seed-{TRAINING_SCHEDULE_SEED}_{order}_{track}.json"
            expected_paths = {
                model: output_root / order / track / f"seed-{TRAINING_SCHEDULE_SEED}" / model / "train_result.json"
                for model in MODELS
            }
            if all(path.is_file() for path in expected_paths.values()):
                for model, path in expected_paths.items():
                    results.append(_validate_result(path, order=order, track=track, model=model))
                print(f"skip completed smoke: order={order} track={track}", flush=True)
                continue
            run_root = output_root / order / track / f"seed-{TRAINING_SCHEDULE_SEED}"
            if run_root.exists():
                if not reset:
                    raise RuntimeError(
                        f"partial smoke output exists: {run_root}; set PARC_M3_SMOKE_RESET=1 to reset only this smoke run"
                    )
                shutil.rmtree(run_root)

            cmd = [
                sys.executable,
                "-u",
                str(runner),
                "--batch-summary",
                str(batch_summary),
                "--schedule",
                str(schedule),
                "--dataset-root",
                str(dataset_root),
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--track",
                track,
                "--order",
                order,
                "--mode",
                "smoke",
                "--repo-root",
                str(repo),
                "--parc-root",
                str(parc_root),
                "--streaming-contract",
                str(streaming_contract),
                "--plan-out",
                str(plan),
            ]
            subprocess.run(cmd, cwd=str(repo), check=True, env=os.environ.copy())
            _execute_training_plan(plan)
            for model, path in expected_paths.items():
                results.append(_validate_result(path, order=order, track=track, model=model))

    expected_count = len(ORDERS) * len(TRACKS) * len(MODELS)
    if len(results) != expected_count:
        raise RuntimeError(f"smoke result count mismatch: {len(results)} != {expected_count}")
    identities = {(r["order"], r["track"], r["model"]) for r in results}
    if len(identities) != expected_count:
        raise RuntimeError("duplicate/missing M3 smoke result identities")
    schedule_hashes = {r["sampling_schedule_sha256"] for r in results if r["track"] == "equal_data"}
    if schedule_hashes != {payload["schedule_sha256"]}:
        raise RuntimeError("forward/reverse equal-data smoke did not reuse the exact same schedule hash")
    for model in MODELS:
        wall_hashes = {
            r["sampling_schedule_sha256"]
            for r in results
            if r["track"] == "equal_wall" and r["model"] == model
        }
        if len(wall_hashes) != 1:
            raise RuntimeError(f"forward/reverse equal-wall stream hash mismatch for {model}")

    summary = {
        "schema_version": 2,
        "stage": "M3_A100_training_smoke",
        "status": "PASS",
        "selected_episode_ids_sha256": D10_HASH,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "training_schedule_seed": TRAINING_SCHEDULE_SEED,
        "orders": list(ORDERS),
        "tracks": list(TRACKS),
        "models": list(MODELS),
        "equal_data_schedule_sha256": payload["schedule_sha256"],
        "forward_reverse_reuse_same_schedule": True,
        "openvla_smoke_merged_checkpoint_materialized": False,
        "openvla_merge_deferred_to_evaluation": True,
        "result_count": expected_count,
        "benchmark_training_started": False,
        "full_1800_second_run_started": False,
        "results": [
            {
                "order": r["order"],
                "track": r["track"],
                "model": r["model"],
                "samples_consumed": r["samples_consumed"],
                "optimizer_updates": r["optimizer_updates"],
                "train_wall_time": r["metrics"]["train_wall_time"],
                "peak_train_vram": r["metrics"]["peak_train_vram"],
                "checkpoint_ref": r["checkpoint_ref"],
                "sampling_schedule_sha256": r["sampling_schedule_sha256"],
            }
            for r in results
        ],
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== 73 M3 A100 TRAINING SMOKE: PASS ===", flush=True)
    print(json.dumps({"status": "PASS", "result_count": expected_count, "summary": str(summary_out)}, indent=2), flush=True)
    print("Benchmark training has NOT started. Full 1800-second runs remain disabled.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
