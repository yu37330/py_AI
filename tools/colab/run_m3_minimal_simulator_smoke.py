#!/usr/bin/env python3
"""Run a minimal simulator smoke before expensive M3 benchmark training.

Uses only Notebook73 forward/equal-data smoke checkpoints. For each model it
runs `libero_spatial`, all 10 tasks, one trial per task, for both frozen eval
seeds: 20 episodes/model, 60 total. This validates checkpoint loading,
simulator stepping, inference instrumentation, and OpenVLA JIT merge/cleanup.
It is not promotion evidence and never starts training.
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
SEEDS = (20260906, 20260907)
SUITE = "libero_spatial"
EXPECTED_PER_MODEL = 20
EXPECTED_TOTAL = 60


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _headless_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    return env


def _training_path(drive: Path, model: str) -> Path:
    return (
        drive
        / "model-benchmark-v1/m3-training-smoke-v1/forward/equal_data"
        / f"seed-{TRAINING_SCHEDULE_SEED}"
        / model
        / "train_result.json"
    )


def _validate_training(path: Path, model: str) -> dict[str, Any]:
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
            raise RuntimeError(f"training smoke mismatch {path}: {key}={data.get(key)!r} != {value!r}")
    if not Path(str(data.get("checkpoint_ref") or "")).exists():
        raise FileNotFoundError(f"training smoke checkpoint missing: {data.get('checkpoint_ref')}")
    return data


def _run_job(command: list[str], *, cwd: Path, log_path: Path) -> list[dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        subprocess.run(
            command,
            cwd=str(cwd),
            env=_headless_env(),
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=True,
        )
    records_path = Path(command[command.index("--records-out") + 1])
    payload = _load(records_path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise RuntimeError(f"minimal simulator smoke expected 10 task records: {records_path}")
    return episodes


def _validate_attempt(value: str) -> str:
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in value):
        raise ValueError("PARC_M3_SIM_SMOKE_ATTEMPT may contain only letters, digits, '-' and '_'")
    return value


def main() -> int:
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("minimal simulator smoke requires explicit PARC_M3_EXECUTE=1")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).resolve()
    parc_root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    attempt = _validate_attempt(os.environ.get("PARC_M3_SIM_SMOKE_ATTEMPT", "1"))
    smoke_summary_path = drive / "model-benchmark-v1/m3-training-smoke-v1/m3_training_smoke_summary.json"
    output_root = drive / f"model-benchmark-v1/m3-simulator-minimal-smoke-v1/attempt-{attempt}"
    summary_path = output_root / "m3_minimal_simulator_smoke_summary.json"

    if not smoke_summary_path.is_file():
        raise FileNotFoundError(smoke_summary_path)
    smoke_summary = _load(smoke_summary_path)
    if smoke_summary.get("status") != "PASS" or smoke_summary.get("stage") != "M3_A100_training_smoke":
        raise RuntimeError("Notebook73 training smoke summary is not PASS")
    if smoke_summary.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("Notebook73 training smoke D10 mismatch")

    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_evaluation_metrics import aggregate_episode_records  # noqa: PLC0415
    from tools.benchmark.m3_libero_executor import SOURCE_REFS, default_eval_runtimes  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    runtimes = default_eval_runtimes(parc_root)
    model_summaries: list[dict[str, Any]] = []

    for model in MODELS:
        training_path = _training_path(drive, model)
        training = _validate_training(training_path, model)
        if training.get("source_ref") != SOURCE_REFS[model]:
            raise RuntimeError(f"training smoke source mismatch for {model}")
        runtime = runtimes[model]
        model_root = output_root / model
        if model_root.exists() and any(model_root.iterdir()) and not summary_path.is_file():
            raise RuntimeError(
                f"partial minimal simulator smoke output exists: {model_root}; preserved for diagnosis. "
                "Use a new PARC_M3_SIM_SMOKE_ATTEMPT instead of deleting evidence."
            )

        finalizer = repo / "tools/benchmark/m3_openvla_finalize_checkpoint.py"
        dematerializer = repo / "tools/benchmark/m3_openvla_dematerialize_checkpoint.py"
        merged_present_before = bool(model == "openvla_oft" and training.get("checkpoint_eval_ready") is True)
        finalization_attempted = False
        materialized_openvla = False
        episodes: list[dict[str, Any]] = []
        try:
            if model == "openvla_oft":
                if not finalizer.is_file() or not dematerializer.is_file():
                    raise RuntimeError("minimal OpenVLA simulator smoke requires integrated PR #66 lifecycle tools")
                if not merged_present_before:
                    finalization_attempted = True
                    subprocess.run([
                        runtime.python,
                        "-u",
                        str(finalizer),
                        "--source-root",
                        runtime.command_root,
                        "--checkpoint-root",
                        str(training["checkpoint_ref"]),
                        "--result",
                        str(training_path),
                    ], cwd=str(repo), env=_headless_env(), check=True)
                    training = _load(training_path)
                    if training.get("checkpoint_eval_ready") is not True:
                        raise RuntimeError("OpenVLA minimal smoke finalizer did not produce eval-ready checkpoint")
                    materialized_openvla = True

            for seed in SEEDS:
                out = model_root / f"seed-{seed}" / SUITE
                records_out = out / "episode_records.json"
                log = out / "eval.log"
                if records_out.is_file():
                    payload = _load(records_out)
                    current = payload.get("episodes")
                    if not isinstance(current, list) or len(current) != 10:
                        raise RuntimeError(f"invalid existing minimal smoke records: {records_out}")
                    episodes.extend(current)
                    continue
                if out.exists() and any(out.iterdir()):
                    raise RuntimeError(
                        f"partial minimal simulator smoke output exists: {out}; preserved for diagnosis. "
                        "Use a new PARC_M3_SIM_SMOKE_ATTEMPT instead of deleting evidence."
                    )
                if model in {"pi05", "smolvla"}:
                    entry = repo / "tools/benchmark/m3_lerobot_eval_entry.py"
                    command = [
                        runtime.python, "-u", str(entry),
                        "--repo-root", str(repo),
                        "--source-root", runtime.command_root,
                        "--training-result", str(training_path),
                        "--model", model,
                        "--suite", SUITE,
                        "--seed", str(seed),
                        "--episodes", "1",
                        "--output-dir", str(out),
                        "--records-out", str(records_out),
                    ]
                else:
                    entry = repo / "tools/benchmark/m3_openvla_eval_entry.py"
                    command = [
                        runtime.python, "-u", str(entry),
                        "--repo-root", str(repo),
                        "--source-root", runtime.command_root,
                        "--training-result", str(training_path),
                        "--suite", SUITE,
                        "--seed", str(seed),
                        "--trials-per-task", "1",
                        "--output-dir", str(out),
                        "--records-out", str(records_out),
                    ]
                episodes.extend(_run_job(command, cwd=Path(runtime.command_root), log_path=log))
        finally:
            # If this smoke created (or even only started creating) merged 7B
            # weights, remove them on both success and failure. Never remove a
            # merged checkpoint that already existed before this attempt.
            cleanup_needed = (
                model == "openvla_oft"
                and not merged_present_before
                and (materialized_openvla or finalization_attempted)
            )
            if cleanup_needed:
                active_exception = sys.exc_info()[0] is not None
                cleanup = subprocess.run([
                    runtime.python,
                    "-u",
                    str(dematerializer),
                    "--checkpoint-root",
                    str(training["checkpoint_ref"]),
                    "--result",
                    str(training_path),
                ], cwd=str(repo), env=_headless_env(), check=False)
                if cleanup.returncode != 0:
                    message = f"OpenVLA merged-weight cleanup failed rc={cleanup.returncode}"
                    if active_exception:
                        print("WARNING: " + message, file=sys.stderr, flush=True)
                    else:
                        raise RuntimeError(message)

        if len(episodes) != EXPECTED_PER_MODEL:
            raise RuntimeError(f"minimal simulator smoke episode count mismatch for {model}: {len(episodes)}")
        aggregate = aggregate_episode_records(
            episodes,
            training_result=_load(training_path),
            training_result_ref=str(training_path),
        )
        model_summary_path = model_root / "minimal_smoke_evaluation_summary.json"
        aggregate["smoke_scope"] = {
            "suite": SUITE,
            "task_count": 10,
            "seed_set": list(SEEDS),
            "trials_per_task": 1,
            "episode_count": EXPECTED_PER_MODEL,
            "promotion_evidence": False,
        }
        model_summary_path.parent.mkdir(parents=True, exist_ok=True)
        model_summary_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        model_summaries.append({
            "model": model,
            "episode_count": EXPECTED_PER_MODEL,
            "simulator_success_rate": aggregate["metrics"]["simulator_success_rate"],
            "inference_latency": aggregate["metrics"]["inference_latency"],
            "peak_inference_vram": aggregate["metrics"]["peak_inference_vram"],
            "summary": str(model_summary_path),
        })

    if sum(item["episode_count"] for item in model_summaries) != EXPECTED_TOTAL:
        raise RuntimeError("minimal simulator smoke total episode count mismatch")
    summary = {
        "schema_version": 2,
        "stage": "M3_minimal_simulator_smoke",
        "status": "PASS",
        "attempt": attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "training_checkpoint_scope": "notebook73_forward_equal_data_smoke_only",
        "suite": SUITE,
        "evaluation_seed_set": list(SEEDS),
        "episodes_per_model": EXPECTED_PER_MODEL,
        "episode_count": EXPECTED_TOTAL,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "models": model_summaries,
        "promotion_evidence": False,
        "benchmark_training_started": False,
        "final_800_episode_evaluation_started": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== 74 M3 MINIMAL SIMULATOR SMOKE: PASS ===", flush=True)
    print(json.dumps({
        "status": "PASS",
        "attempt": attempt,
        "episode_count": EXPECTED_TOTAL,
        "models": MODELS,
        "summary": str(summary_path),
        "promotion_evidence": False,
        "next": "run 75a/75b M3 benchmark training",
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
