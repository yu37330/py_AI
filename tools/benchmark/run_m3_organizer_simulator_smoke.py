#!/usr/bin/env python3
"""Run the Notebook 74-equivalent smoke on the organizer GPU/JupyterHub.

This launcher is intentionally separate from the Colab notebook. It preserves
the same 60-episode non-promotion contract while mapping organizer scratch and
persistent storage into the existing helper interfaces.

Required environment:
- PARC_M3_EXECUTE=1
- PARC_LOCAL_SCRATCH_ROOT=/opt/dlami/nvme/... などの高速一時領域
- PARC_PERSIST_ROOT=~/data/... などの永続領域
- HF_TOKEN

The launcher never reruns 72 probes, never starts the 1800-second benchmark,
and never mutates the original Notebook73 train_result.json files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
TRAINING_SCHEDULE_SEED = 20260906
MODELS = ("pi05", "smolvla", "openvla_oft")
ARTIFACT_MARKER = "model-benchmark-v1"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _validate_attempt(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(ch not in allowed for ch in value):
        raise ValueError("PARC_M3_SIM_SMOKE_ATTEMPT may contain only letters, digits, '-' and '_'")
    return value


def _disk(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    gib = 1024**3
    return {
        "path": str(path),
        "total_gib": round(usage.total / gib, 2),
        "used_gib": round(usage.used / gib, 2),
        "free_gib": round(usage.free / gib, 2),
    }


def _command_text(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "output": (proc.stdout or "")[-12000:],
    }


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _training_result_path(persist_root: Path, model: str) -> Path:
    return (
        persist_root
        / "model-benchmark-v1/m3-training-smoke-v1/forward/equal_data"
        / f"seed-{TRAINING_SCHEDULE_SEED}"
        / model
        / "train_result.json"
    )


def _rebased_checkpoint(original: Path, persist_root: Path) -> Path:
    if original.exists():
        return original.resolve()
    parts = original.parts
    try:
        marker_index = parts.index(ARTIFACT_MARKER)
    except ValueError as exc:
        raise RuntimeError(
            f"checkpoint_ref has no safe {ARTIFACT_MARKER!r} relocation marker: {original}"
        ) from exc
    candidate = persist_root.joinpath(*parts[marker_index:]).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"relocated checkpoint is missing: original={original} candidate={candidate}"
        )
    return candidate


def _validate_handoff(persist_root: Path) -> dict[str, Any]:
    summary_path = persist_root / "model-benchmark-v1/m3-training-smoke-v1/m3_training_smoke_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _load(summary_path)
    if summary.get("status") != "PASS" or summary.get("stage") != "M3_A100_training_smoke":
        raise RuntimeError("Notebook73 training smoke summary is not PASS")
    if summary.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("Notebook73 training smoke D10 mismatch")
    if summary.get("benchmark_training_started") is not False:
        raise RuntimeError("Notebook73 summary unexpectedly claims benchmark training started")

    models: list[dict[str, Any]] = []
    for model in MODELS:
        path = _training_result_path(persist_root, model)
        if not path.is_file():
            raise FileNotFoundError(path)
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
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise RuntimeError(f"handoff mismatch {path}: {key}={data.get(key)!r} != {value!r}")
        original = Path(str(data.get("checkpoint_ref") or ""))
        if not str(data.get("checkpoint_ref") or ""):
            raise RuntimeError(f"checkpoint_ref missing: {path}")
        resolved = _rebased_checkpoint(original, persist_root)
        models.append(
            {
                "model": model,
                "train_result": str(path),
                "original_checkpoint_ref": str(original),
                "resolved_checkpoint_ref": str(resolved),
                "checkpoint_rebased": not original.exists(),
            }
        )

    return {
        "schema_version": 1,
        "stage": "M3_organizer_handoff_validation",
        "status": "PASS",
        "selected_episode_ids_sha256": D10_HASH,
        "training_summary": str(summary_path),
        "models": models,
        "source_training_results_mutated": False,
        "probes_rerun": False,
        "benchmark_training_started": False,
    }


def _run_helper(repo: Path, env: dict[str, str], *args: str) -> None:
    command = [sys.executable, "-u", str(repo / args[0]), *args[1:]]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(repo), env=env, check=True)


def main() -> int:
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("organizer simulator smoke requires explicit PARC_M3_EXECUTE=1")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    if not os.environ.get("PARC_LOCAL_SCRATCH_ROOT"):
        raise RuntimeError("PARC_LOCAL_SCRATCH_ROOT is required on organizer GPU")
    if not os.environ.get("PARC_PERSIST_ROOT"):
        raise RuntimeError("PARC_PERSIST_ROOT is required on organizer GPU")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    local_root = Path(os.environ["PARC_LOCAL_SCRATCH_ROOT"]).expanduser().resolve()
    persist_root = Path(os.environ["PARC_PERSIST_ROOT"]).expanduser().resolve()
    attempt = _validate_attempt(os.environ.get("PARC_M3_SIM_SMOKE_ATTEMPT", "organizer-1"))

    if not (repo / ".git").is_dir():
        raise FileNotFoundError(f"PY_AI_REPO is not a Git checkout: {repo}")
    local_root.mkdir(parents=True, exist_ok=True)
    if not persist_root.is_dir():
        raise FileNotFoundError(f"persistent root must already contain restored artifacts: {persist_root}")

    source_sha = _git_head(repo)
    expected_sha = os.environ.get("PARC_EXPECTED_SOURCE_SHA")
    if expected_sha and source_sha != expected_sha:
        raise RuntimeError(f"source SHA mismatch: {source_sha} != {expected_sha}")

    migration_root = (
        persist_root
        / "model-benchmark-v1/m3-organizer-migration-v1"
        / f"attempt-{attempt}"
    )
    migration_root.mkdir(parents=True, exist_ok=True)
    status_path = migration_root / "launcher_status.json"
    inventory_path = migration_root / "environment_inventory.json"
    handoff_path = migration_root / "handoff_validation.json"

    inventory = {
        "schema_version": 1,
        "stage": "M3_organizer_environment_inventory",
        "status": "CAPTURED",
        "source_sha": source_sha,
        "local_scratch": _disk(local_root),
        "persistent_storage": _disk(persist_root),
        "nvidia_smi": _command_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvcc": _command_text(["bash", "-lc", "command -v nvcc >/dev/null && nvcc --version || true"]),
        "python": sys.version,
        "probes_rerun": False,
        "benchmark_training_started": False,
    }
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    handoff = _validate_handoff(persist_root)
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PY_AI_REPO"] = str(repo)
    env["PARC_ROOT"] = str(local_root)
    env["PARC_DRIVE_ROOT"] = str(persist_root)  # legacy helper alias
    env["PARC_LOCAL_SCRATCH_ROOT"] = str(local_root)
    env["PARC_PERSIST_ROOT"] = str(persist_root)
    env["PARC_M3_ALLOW_ARTIFACT_REBASE"] = "1"
    env["PARC_M3_SIM_SMOKE_ATTEMPT"] = attempt
    env["LIBERO_CONFIG_PATH"] = str(local_root / "m3-libero-config/shared")
    env.setdefault("HF_HOME", str(local_root / "cache/huggingface"))
    env.setdefault("TORCH_HOME", str(local_root / "cache/torch"))
    env.setdefault("XDG_CACHE_HOME", str(local_root / "cache"))
    env["MPLBACKEND"] = "Agg"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    status = {
        "schema_version": 1,
        "stage": "M3_organizer_simulator_smoke_launcher",
        "status": "RUNNING",
        "attempt": attempt,
        "source_sha": source_sha,
        "selected_episode_ids_sha256": D10_HASH,
        "probes_rerun": False,
        "benchmark_training_started": False,
        "final_800_episode_evaluation_started": False,
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        _run_helper(repo, env, "tools/colab/prepare_m3_disk_headroom.py", "--phase", "pre-setup")
        _run_helper(repo, env, "tools/colab/prepare_m3_libero_noninteractive_config.py")
        _run_helper(repo, env, "tools/colab/prepare_m3_simulator_runtimes.py")
        _run_helper(repo, env, "tools/colab/prepare_m3_mujoco_compat.py")
        _run_helper(repo, env, "tools/colab/prepare_m3_disk_headroom.py", "--phase", "pre-eval")
        _run_helper(repo, env, "tools/colab/run_m3_minimal_simulator_smoke.py")

        summary_path = (
            persist_root
            / f"model-benchmark-v1/m3-simulator-minimal-smoke-v1/attempt-{attempt}"
            / "m3_minimal_simulator_smoke_summary.json"
        )
        summary = _load(summary_path)
        if summary.get("status") != "PASS" or summary.get("episode_count") != 60:
            raise RuntimeError(f"organizer smoke summary is not PASS/60: {summary_path}")
        if summary.get("selected_episode_ids_sha256") != D10_HASH:
            raise RuntimeError("organizer smoke D10 mismatch")
        if summary.get("promotion_evidence") is not False:
            raise RuntimeError("organizer smoke must remain non-promotion evidence")
        if summary.get("benchmark_training_started") is not False:
            raise RuntimeError("organizer smoke unexpectedly started benchmark training")

        status.update(
            status="PASS",
            smoke_summary=str(summary_path),
            episode_count=60,
            artifact_rebased_models=summary.get("artifact_rebased_models", []),
        )
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("=== ORGANIZER GPU 74 EQUIVALENT: PASS ===", flush=True)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("=== ORGANIZER GPU FAILURE DIAGNOSTICS ===", flush=True)
        subprocess.run(["df", "-h", str(local_root), str(persist_root)], check=False)
        subprocess.run(["df", "-i", str(local_root), str(persist_root)], check=False)
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
