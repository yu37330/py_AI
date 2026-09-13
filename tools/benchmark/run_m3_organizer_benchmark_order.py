#!/usr/bin/env python3
"""Run one frozen M3 benchmark order on the organizer Blackwell environment.

This wrapper adds organizer-specific gates around the canonical benchmark runner:
74 simulator PASS, checksum handoff PASS, exact dataset readiness, and a bounded
Blackwell training-compatibility PASS. Run forward and reverse as separate
restartable sessions when desired; persistent checkpoints/results remain in
~/data while runtimes, caches, and dataset live on NVMe scratch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
HARDWARE_PROFILE = "organizer_rtx_pro_6000_blackwell"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--attempt", default=os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "organizer-1"))
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _safe_attempt(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(ch not in allowed for ch in value):
        raise ValueError("attempt may contain only letters, digits, '-' and '_'")
    return value


def _require_74(persist: Path, attempt: str) -> None:
    path = persist / f"model-benchmark-v1/m3-simulator-minimal-smoke-v1/attempt-{attempt}/m3_minimal_simulator_smoke_summary.json"
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


def _require_training_compat(persist: Path, attempt: str) -> None:
    path = persist / f"model-benchmark-v1/m3-organizer-training-compat-v1/attempt-{attempt}/m3_organizer_training_compat_summary.json"
    data = _load(path)
    expected = {
        "status": "PASS",
        "stage": "M3_organizer_training_compatibility",
        "hardware_profile": HARDWARE_PROFILE,
        "selected_episode_ids_sha256": D10_HASH,
        "effective_batch_size": 32,
        "model_count": 3,
        "historical_72_73_evidence_mutated": False,
        "benchmark_training_started": False,
        "ready_for_75": True,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"training compatibility gate mismatch: {key}={data.get(key)!r} != {value!r}")


def main() -> int:
    args = parse_args()
    attempt = _safe_attempt(args.attempt)
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("organizer benchmark requires explicit PARC_M3_EXECUTE=1")
    for key in ("PARC_LOCAL_SCRATCH_ROOT", "PARC_PERSIST_ROOT", "PARC_DATASET_ROOT", "HF_TOKEN"):
        if not os.environ.get(key):
            raise RuntimeError(f"{key} is required")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    root = Path(os.environ["PARC_LOCAL_SCRATCH_ROOT"]).expanduser().resolve()
    persist = Path(os.environ["PARC_PERSIST_ROOT"]).expanduser().resolve()
    dataset = Path(os.environ["PARC_DATASET_ROOT"]).expanduser().resolve()
    smoke_attempt = _safe_attempt(os.environ.get("PARC_M3_SIM_SMOKE_ATTEMPT", "organizer-1"))
    compat_attempt = _safe_attempt(os.environ.get("PARC_M3_TRAINING_COMPAT_ATTEMPT", "organizer-1"))

    if not dataset.is_dir() or not (dataset / "meta/info.json").is_file():
        raise FileNotFoundError(f"organizer NVMe dataset is not materialized: {dataset}")
    dataset_ready_path = persist / "model-benchmark-v1/m3-organizer-migration-v1/dataset_readiness.json"
    dataset_ready = _load(dataset_ready_path)
    if dataset_ready.get("status") != "PASS" or dataset_ready.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("dataset readiness evidence is not PASS")
    if Path(str(dataset_ready.get("dataset_root") or "")).resolve() != dataset:
        raise RuntimeError("dataset readiness path differs from PARC_DATASET_ROOT")

    _require_74(persist, smoke_attempt)
    _require_training_compat(persist, compat_attempt)

    os.environ["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_hardware_guard import validate_hardware  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    profile = validate_hardware(gpu_name, gpu_vram_mib)

    env = os.environ.copy()
    env["PY_AI_REPO"] = str(repo)
    env["PARC_ROOT"] = str(root)
    env["PARC_DRIVE_ROOT"] = str(persist)
    env["PARC_LOCAL_SCRATCH_ROOT"] = str(root)
    env["PARC_PERSIST_ROOT"] = str(persist)
    env["PARC_DATASET_ROOT"] = str(dataset)
    env["PARC_DRIVE_DATASET"] = str(dataset)  # canonical runner compatibility
    env["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    env["PARC_M3_BENCHMARK_ATTEMPT"] = attempt
    env.setdefault("HF_HOME", str(root / "cache/huggingface"))
    env.setdefault("TORCH_HOME", str(root / "cache/torch"))
    env.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    env.setdefault("UV_CACHE_DIR", str(root / "cache/uv"))
    env.setdefault("PIP_CACHE_DIR", str(root / "cache/pip"))
    env.setdefault("TMPDIR", str(root / "tmp"))
    for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    # Re-verify transferred immutable evidence every session; additions under
    # ~/data do not affect the original handoff manifest inventory.
    subprocess.run(
        [sys.executable, "-u", str(repo / "tools/benchmark/verify_m3_organizer_handoff.py"), "--root", str(persist)],
        cwd=str(repo),
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-u", str(repo / "tools/colab/prepare_m3_training_runtimes.py")],
        cwd=str(repo),
        env=env,
        check=True,
    )

    runner = repo / "tools/colab/run_m3_benchmark_order.py"
    subprocess.run(
        [sys.executable, "-u", str(runner), "--order", args.order, "--attempt", attempt],
        cwd=str(repo),
        env=env,
        check=True,
    )

    summary_path = persist / f"model-benchmark-v1/m3-benchmark-v1/attempt-{attempt}/m3_training_{args.order}_summary.json"
    summary = _load(summary_path)
    expected = {
        "status": "PASS",
        "stage": "M3_training_order_benchmark",
        "order": args.order,
        "attempt": attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "result_count": 6,
        "openvla_merged_checkpoint_materialized": False,
        "openvla_merge_deferred_to_evaluation": True,
        "ready_for_screening_evaluation": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"benchmark summary mismatch: {key}={summary.get(key)!r} != {value!r}")

    handoff = {
        "schema_version": 1,
        "stage": "M3_organizer_benchmark_order_handoff",
        "status": "PASS",
        "hardware_profile": profile.name,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "order": args.order,
        "attempt": attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "summary": str(summary_path),
        "result_count": 6,
        "ready_for_screening_evaluation": True,
        "next_action": "parc-home-sync data-push before stopping the server",
    }
    out = persist / f"model-benchmark-v1/m3-organizer-migration-v1/benchmark-{args.order}-attempt-{attempt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"=== M3 ORGANIZER {args.order.upper()} BENCHMARK: PASS ===", flush=True)
    print(json.dumps(handoff, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
