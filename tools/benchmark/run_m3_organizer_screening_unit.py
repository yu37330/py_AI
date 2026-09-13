#!/usr/bin/env python3
"""Run one 80-episode M3 screening checkpoint on organizer Blackwell.

A unit is exactly one (order, track, model) checkpoint. Evaluation first runs in
an attempt-scoped staging directory. Only a fully validated PASS is promoted to
the canonical screening path used by the promotion controller. A forced 12-hour
session stop therefore never poisons previously completed checkpoint evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
TRAINING_SCHEDULE_SEED = 20260906
HARDWARE_PROFILE = "organizer_rtx_pro_6000_blackwell"
TRACKS = ("equal_data", "equal_wall")
MODELS_BY_ORDER = {
    "forward": ("pi05", "smolvla", "openvla_oft"),
    "reverse": ("openvla_oft", "smolvla", "pi05"),
}
SCREENING_EPISODES = 80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--model", choices=("pi05", "smolvla", "openvla_oft"), required=True)
    p.add_argument("--attempt", default=os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "organizer-1"))
    p.add_argument("--unit-attempt", default=os.environ.get("PARC_M3_SCREENING_UNIT_ATTEMPT", "unit-1"))
    return p.parse_args()


def _safe(value: str, label: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(ch not in allowed for ch in value):
        raise ValueError(f"{label} may contain only letters, digits, '-' and '_'")
    return value


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_training(path: Path, *, order: str, track: str, model: str) -> dict[str, Any]:
    data = _load(path)
    expected = {
        "status": "PASS",
        "model": model,
        "order": order,
        "track": track,
        "mode": "benchmark",
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": TRAINING_SCHEDULE_SEED,
        "effective_batch_size": 32,
        "benchmark_training_started": True,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"training mismatch {path}: {key}={data.get(key)!r} != {value!r}")
    checkpoint = Path(str(data.get("checkpoint_ref") or ""))
    if not checkpoint.exists():
        raise FileNotFoundError(f"training checkpoint missing: {checkpoint}")
    return data


def _validate_eval(path: Path, *, training: dict[str, Any], training_path: Path) -> dict[str, Any]:
    data = _load(path)
    expected = {
        "status": "PASS",
        "stage": "M3_simulator_evaluation",
        "model": training["model"],
        "order": training["order"],
        "track": training["track"],
        "source_ref": training["source_ref"],
        "checkpoint_ref": training["checkpoint_ref"],
        "selected_episode_ids_sha256": D10_HASH,
        "episode_count": SCREENING_EPISODES,
        "max_steps_per_episode": 300,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"evaluation mismatch {path}: {key}={data.get(key)!r} != {value!r}")
    if tuple(int(x) for x in data.get("seed_set", [])) != (20260906, 20260907):
        raise RuntimeError("screening evaluation seed set mismatch")
    ref = str(data.get("training_result_ref") or "").strip()
    if not ref or Path(ref).resolve() != training_path.resolve():
        raise RuntimeError("screening evaluation does not link exact training result")
    return data


def _prepare_runtime(repo: Path, env: dict[str, str], persist: Path) -> None:
    chain = [
        ("tools/benchmark/verify_m3_organizer_handoff.py", "--root", str(persist)),
        ("tools/colab/prepare_m3_disk_headroom.py", "--phase", "pre-setup"),
        ("tools/colab/prepare_m3_libero_noninteractive_config.py",),
        ("tools/colab/prepare_m3_simulator_runtimes.py",),
        ("tools/colab/prepare_m3_mujoco_compat.py",),
        ("tools/colab/prepare_m3_disk_headroom.py", "--phase", "pre-eval"),
    ]
    for parts in chain:
        subprocess.run(
            [sys.executable, "-u", str(repo / parts[0]), *parts[1:]],
            cwd=str(repo), env=env, check=True,
        )


def main() -> int:
    args = parse_args()
    attempt = _safe(args.attempt, "attempt")
    unit_attempt = _safe(args.unit_attempt, "unit-attempt")
    if args.model not in MODELS_BY_ORDER[args.order]:
        raise RuntimeError(f"model {args.model} is not in {args.order} order")
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("screening unit requires PARC_M3_EXECUTE=1")
    for key in ("PARC_LOCAL_SCRATCH_ROOT", "PARC_PERSIST_ROOT", "HF_TOKEN"):
        if not os.environ.get(key):
            raise RuntimeError(f"{key} is required")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    root = Path(os.environ["PARC_LOCAL_SCRATCH_ROOT"]).expanduser().resolve()
    persist = Path(os.environ["PARC_PERSIST_ROOT"]).expanduser().resolve()
    training_root = persist / f"model-benchmark-v1/m3-benchmark-v1/attempt-{attempt}"
    training_summary_path = training_root / f"m3_training_{args.order}_summary.json"
    training_summary = _load(_require(training_summary_path))
    if (
        training_summary.get("status") != "PASS"
        or training_summary.get("order") != args.order
        or training_summary.get("selected_episode_ids_sha256") != D10_HASH
        or training_summary.get("ready_for_screening_evaluation") is not True
    ):
        raise RuntimeError(f"benchmark order is not screening-ready: {training_summary_path}")

    training_path = _require(
        training_root / args.order / args.track / f"seed-{TRAINING_SCHEDULE_SEED}" / args.model / "train_result.json"
    )
    training = _validate_training(training_path, order=args.order, track=args.track, model=args.model)

    canonical_root = persist / f"model-benchmark-v1/m3-screening-eval-v1/attempt-{attempt}/{args.order}/{args.track}/{args.model}"
    canonical_summary = canonical_root / "m3_screening_evaluation_summary.json"
    promotion_path = persist / f"model-benchmark-v1/m3-promotion-records-v1/attempt-{attempt}/{args.order}/{args.track}/{args.model}/promotion_record.json"

    sys.path.insert(0, str(repo))
    os.environ["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_hardware_guard import validate_hardware  # noqa: PLC0415
    from tools.benchmark.m3_promotion_record import build_promotion_record  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    profile = validate_hardware(gpu_name, gpu_vram_mib)

    if canonical_summary.is_file():
        _validate_eval(canonical_summary, training=training, training_path=training_path)
        record = build_promotion_record(
            training_path=training_path,
            evaluation_path=canonical_summary,
            require_screening_episodes=True,
        )
        promotion_path.parent.mkdir(parents=True, exist_ok=True)
        promotion_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("=== M3 ORGANIZER SCREENING UNIT: REUSE PASS ===", flush=True)
        return 0
    if canonical_root.exists() and any(canonical_root.iterdir()):
        raise RuntimeError(
            f"canonical screening root contains partial evidence: {canonical_root}; "
            "do not delete it automatically; inspect/move it before retrying"
        )

    env = os.environ.copy()
    env["PY_AI_REPO"] = str(repo)
    env["PARC_ROOT"] = str(root)
    env["PARC_DRIVE_ROOT"] = str(persist)
    env["PARC_LOCAL_SCRATCH_ROOT"] = str(root)
    env["PARC_PERSIST_ROOT"] = str(persist)
    env["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    env["LIBERO_CONFIG_PATH"] = str(root / "m3-libero-config/shared")
    env.setdefault("HF_HOME", str(root / "cache/huggingface"))
    env.setdefault("TORCH_HOME", str(root / "cache/torch"))
    env.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    env.setdefault("UV_CACHE_DIR", str(root / "cache/uv"))
    env.setdefault("PIP_CACHE_DIR", str(root / "cache/pip"))
    env.setdefault("TMPDIR", str(root / "tmp"))
    for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    env["MPLBACKEND"] = "Agg"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    _prepare_runtime(repo, env, persist)

    staging_root = (
        persist
        / f"model-benchmark-v1/m3-screening-unit-staging-v1/benchmark-attempt-{attempt}"
        / args.order / args.track / args.model / f"unit-attempt-{unit_attempt}"
    )
    if staging_root.exists() and any(staging_root.iterdir()):
        raise RuntimeError(
            f"screening unit staging output already exists: {staging_root}; use a new --unit-attempt"
        )
    staging_summary = staging_root / "m3_screening_evaluation_summary.json"
    staging_plan = staging_root / "m3_screening_evaluation_plan.json"
    wrapper = repo / "tools/benchmark/run_m3_libero_evaluation.py"
    cmd = [
        sys.executable, "-u", str(wrapper),
        "--training-result", str(training_path),
        "--output-root", str(staging_root),
        "--plan-out", str(staging_plan),
        "--summary-out", str(staging_summary),
        "--parc-root", str(root),
        "--repo-root", str(repo),
        "--mode", "smoke",
        "--execute",
    ]
    subprocess.run(cmd, cwd=str(repo), env=env, check=True)
    _validate_eval(staging_summary, training=training, training_path=training_path)

    canonical_root.parent.mkdir(parents=True, exist_ok=True)
    if canonical_root.exists():
        raise RuntimeError(f"canonical root appeared during unit execution: {canonical_root}")
    shutil.move(str(staging_root), str(canonical_root))
    _validate_eval(canonical_summary, training=training, training_path=training_path)

    record = build_promotion_record(
        training_path=training_path,
        evaluation_path=canonical_summary,
        require_screening_episodes=True,
    )
    promotion_path.parent.mkdir(parents=True, exist_ok=True)
    promotion_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "stage": "M3_organizer_screening_unit",
        "status": "PASS",
        "benchmark_attempt": attempt,
        "unit_attempt": unit_attempt,
        "hardware_profile": profile.name,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "order": args.order,
        "track": args.track,
        "model": args.model,
        "selected_episode_ids_sha256": D10_HASH,
        "episode_count": SCREENING_EPISODES,
        "evaluation_summary": str(canonical_summary),
        "promotion_record": str(promotion_path),
        "promotion_record_sha256": record["promotion_record_sha256"],
        "final_800_episode_evaluation_started": False,
    }
    evidence_path = persist / f"model-benchmark-v1/m3-organizer-migration-v1/screening-unit-{args.order}-{args.track}-{args.model}-attempt-{attempt}.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== M3 ORGANIZER SCREENING UNIT: PASS ===", flush=True)
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
