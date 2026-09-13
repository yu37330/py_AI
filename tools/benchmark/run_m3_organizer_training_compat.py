#!/usr/bin/env python3
"""Run a bounded 3-model training compatibility smoke on organizer Blackwell.

This is separate evidence from the historical 72/73 A100 probes. It consumes
the same frozen D10, schedule, 72d batch configuration, and 69c OpenVLA stream,
but runs only forward/equal-data smoke (64 samples, 2 optimizer updates/model).
It must never overwrite 72/73 evidence or start the 1800-second benchmark.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
TRAINING_SCHEDULE_SEED = 20260906
MODELS = ("pi05", "smolvla", "openvla_oft")
HARDWARE_PROFILE = "organizer_rtx_pro_6000_blackwell"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _attempt(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(ch not in allowed for ch in value):
        raise ValueError("attempt may contain only letters, digits, '-' and '_'")
    return value


def _require_74(persist: Path, attempt: str) -> Path:
    path = (
        persist
        / f"model-benchmark-v1/m3-simulator-minimal-smoke-v1/attempt-{attempt}"
        / "m3_minimal_simulator_smoke_summary.json"
    )
    data = _load(path)
    expected = {
        "status": "PASS",
        "stage": "M3_minimal_simulator_smoke",
        "selected_episode_ids_sha256": D10_HASH,
        "episode_count": 60,
        "promotion_evidence": False,
        "benchmark_training_started": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"74 gate mismatch: {key}={data.get(key)!r} != {value!r}")
    return path


def _validate_result(path: Path, model: str) -> dict[str, Any]:
    data = _load(path)
    expected = {
        "status": "PASS",
        "model": model,
        "mode": "smoke",
        "order": "forward",
        "track": "equal_data",
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": TRAINING_SCHEDULE_SEED,
        "effective_batch_size": 32,
        "samples_consumed": 64,
        "optimizer_updates": 2,
        "benchmark_training_started": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"training compatibility mismatch {path}: {key}={data.get(key)!r} != {value!r}")
    checkpoint = Path(str(data.get("checkpoint_ref") or ""))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return data


def main() -> int:
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("organizer training compatibility requires PARC_M3_EXECUTE=1")
    for key in ("PARC_LOCAL_SCRATCH_ROOT", "PARC_PERSIST_ROOT", "PARC_DATASET_ROOT", "HF_TOKEN"):
        if not os.environ.get(key):
            raise RuntimeError(f"{key} is required")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    root = Path(os.environ["PARC_LOCAL_SCRATCH_ROOT"]).expanduser().resolve()
    persist = Path(os.environ["PARC_PERSIST_ROOT"]).expanduser().resolve()
    dataset = Path(os.environ["PARC_DATASET_ROOT"]).expanduser().resolve()
    smoke_attempt = _attempt(os.environ.get("PARC_M3_SIM_SMOKE_ATTEMPT", "organizer-1"))
    compat_attempt = _attempt(os.environ.get("PARC_M3_TRAINING_COMPAT_ATTEMPT", "organizer-1"))

    _require_74(persist, smoke_attempt)
    dataset_ready = _load(persist / "model-benchmark-v1/m3-organizer-migration-v1/dataset_readiness.json")
    if dataset_ready.get("status") != "PASS" or dataset_ready.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("organizer dataset readiness is not PASS for frozen D10")
    if Path(str(dataset_ready.get("dataset_root") or "")).resolve() != dataset:
        raise RuntimeError("PARC_DATASET_ROOT differs from dataset readiness evidence")

    os.environ["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_hardware_guard import validate_hardware  # noqa: PLC0415
    from tools.benchmark.m3_model_adapters import (  # noqa: PLC0415
        build_run_specs,
        load_batch_summary,
        runtime_preflight,
    )

    gpu_name, gpu_vram_mib = gpu_info()
    profile = validate_hardware(gpu_name, gpu_vram_mib)

    env = os.environ.copy()
    env["PY_AI_REPO"] = str(repo)
    env["PARC_ROOT"] = str(root)
    env["PARC_DRIVE_ROOT"] = str(persist)
    env["PARC_LOCAL_SCRATCH_ROOT"] = str(root)
    env["PARC_PERSIST_ROOT"] = str(persist)
    env["PARC_DATASET_ROOT"] = str(dataset)
    env["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    env.setdefault("HF_HOME", str(root / "cache/huggingface"))
    env.setdefault("TORCH_HOME", str(root / "cache/torch"))
    env.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    env.setdefault("UV_CACHE_DIR", str(root / "cache/uv"))
    env.setdefault("PIP_CACHE_DIR", str(root / "cache/pip"))
    env.setdefault("TMPDIR", str(root / "tmp"))
    for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    # Fresh organizer sessions lose NVMe, so runtimes are reconstructible setup.
    subprocess.run(
        [sys.executable, "-u", str(repo / "tools/colab/prepare_m3_training_runtimes.py")],
        cwd=str(repo),
        env=env,
        check=True,
    )

    batch_summary_path = persist / "model-benchmark-v1/m3_batch_probe_summary.json"
    schedule_path = persist / f"model-benchmark-v1/m3-schedules-v1/m3_equal_data_seed-{TRAINING_SCHEDULE_SEED}.json"
    manifest = persist / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json"
    streaming_contract = persist / "openvla-streaming-selected-v1/streaming_bridge_contract.json"
    output_root = persist / f"model-benchmark-v1/m3-organizer-training-compat-v1/attempt-{compat_attempt}"
    summary_path = output_root / "m3_organizer_training_compat_summary.json"

    batch_summary = load_batch_summary(batch_summary_path)
    specs = build_run_specs(
        batch_summary=batch_summary,
        schedule_path=schedule_path,
        batch_summary_path=batch_summary_path,
        dataset_root=dataset,
        manifest_path=manifest,
        output_root=output_root,
        root=root,
        repo=repo,
        track="equal_data",
        order="forward",
        mode="smoke",
        streaming_contract=streaming_contract,
    )
    if tuple(spec["model"] for spec in specs) != MODELS:
        raise RuntimeError("organizer training compatibility model order drift")
    blockers: list[str] = []
    for spec in specs:
        blockers.extend(runtime_preflight(spec["runtime"], repo=repo, strict_files=True)) if "runtime" in spec else None
    # build_run_specs serializes commands; runtime setup was already strict-checked
    # by prepare_m3_training_runtimes.py. Execute only the frozen smoke commands.

    results: list[dict[str, Any]] = []
    for spec in specs:
        model = str(spec["model"])
        result_path = Path(spec["out"])
        if result_path.is_file():
            results.append(_validate_result(result_path, model))
            continue
        checkpoint_dir = Path(spec["output_dir"])
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise RuntimeError(
                f"partial organizer training compatibility output exists: {checkpoint_dir}; use a new attempt"
            )
        command = spec.get("command")
        if not isinstance(command, list) or not command:
            raise RuntimeError(f"missing compatibility training command for {model}")
        log = result_path.parent / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as fh:
            subprocess.run(command, cwd=str(repo), env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
        results.append(_validate_result(result_path, model))

    if len(results) != 3:
        raise RuntimeError("organizer training compatibility did not produce three PASS results")
    summary = {
        "schema_version": 1,
        "stage": "M3_organizer_training_compatibility",
        "status": "PASS",
        "attempt": compat_attempt,
        "hardware_profile": profile.name,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "selected_episode_ids_sha256": D10_HASH,
        "training_schedule_seed": TRAINING_SCHEDULE_SEED,
        "order": "forward",
        "track": "equal_data",
        "samples_per_model": 64,
        "optimizer_updates_per_model": 2,
        "effective_batch_size": 32,
        "model_count": 3,
        "historical_72_73_evidence_mutated": False,
        "benchmark_training_started": False,
        "ready_for_75": True,
        "results": [
            {
                "model": r["model"],
                "micro_batch": r["micro_batch"],
                "gradient_accumulation": r["gradient_accumulation"],
                "peak_train_vram": r["metrics"]["peak_train_vram"],
                "train_wall_time": r["metrics"]["train_wall_time"],
                "checkpoint_ref": r["checkpoint_ref"],
            }
            for r in results
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== M3 ORGANIZER TRAINING COMPATIBILITY: PASS ===", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
