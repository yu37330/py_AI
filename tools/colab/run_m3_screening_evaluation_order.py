#!/usr/bin/env python3
"""Evaluate one completed M3 training order for screening/promotion.

This runner consumes the six benchmark training results from Notebook 75a or
75b (3 models x equal-data/equal-wall), evaluates every checkpoint on all 40
LIBERO tasks with both frozen evaluation seeds and one trial per task (80
episodes/checkpoint), and writes promotion-ready records. It never launches
training or the final 800-episode evaluation.
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
TRAINING_SCHEDULE_SEED = 20260906
TRACKS = ("equal_data", "equal_wall")
MODELS_BY_ORDER = {
    "forward": ("pi05", "smolvla", "openvla_oft"),
    "reverse": ("openvla_oft", "smolvla", "pi05"),
}
SCREENING_EPISODES = 80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--attempt", default=os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "1"))
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _training_result_path(root: Path, order: str, track: str, model: str) -> Path:
    return root / order / track / f"seed-{TRAINING_SCHEDULE_SEED}" / model / "train_result.json"


def _validate_training_result(path: Path, *, order: str, track: str, model: str) -> dict[str, Any]:
    result = _load(path)
    expected = {
        "status": "PASS",
        "model": model,
        "order": order,
        "track": track,
        "mode": "benchmark",
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": TRAINING_SCHEDULE_SEED,
        "effective_batch_size": 32,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"training result mismatch {path}: {key}={result.get(key)!r} != {value!r}")
    if not str(result.get("sampling_schedule_sha256") or "").strip():
        raise RuntimeError(f"training schedule hash missing: {path}")
    if not Path(str(result.get("checkpoint_ref") or "")).exists():
        raise FileNotFoundError(f"training checkpoint missing: {result.get('checkpoint_ref')}")
    return result


def _validate_eval_summary(path: Path, *, training: dict[str, Any], training_path: Path) -> dict[str, Any]:
    summary = _load(path)
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
        if summary.get(key) != value:
            raise RuntimeError(f"screening evaluation mismatch {path}: {key}={summary.get(key)!r} != {value!r}")
    if tuple(int(x) for x in summary.get("seed_set", [])) != (20260906, 20260907):
        raise RuntimeError("screening evaluation seed set mismatch")
    ref = str(summary.get("training_result_ref") or "").strip()
    if not ref or Path(ref).resolve() != training_path.resolve():
        raise RuntimeError("screening evaluation does not link exact training result")
    return summary


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 screening evaluation requires explicit PARC_M3_EXECUTE=1")
    if not args.attempt or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in args.attempt):
        raise ValueError("--attempt may contain only letters, digits, '-' and '_'")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).resolve()
    parc_root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    training_root = drive / f"model-benchmark-v1/m3-benchmark-v1/attempt-{args.attempt}"
    screening_root = drive / f"model-benchmark-v1/m3-screening-eval-v1/attempt-{args.attempt}" / args.order
    promotion_root = drive / f"model-benchmark-v1/m3-promotion-records-v1/attempt-{args.attempt}" / args.order
    order_training_summary = training_root / f"m3_training_{args.order}_summary.json"
    order_eval_summary = screening_root / f"m3_screening_{args.order}_summary.json"

    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_promotion_record import build_promotion_record  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    training_summary = _load(_require_file(order_training_summary))
    if training_summary.get("status") != "PASS" or training_summary.get("order") != args.order:
        raise RuntimeError(f"M3 {args.order} training summary is not PASS")
    if training_summary.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("M3 training order summary D10 mismatch")
    if training_summary.get("ready_for_screening_evaluation") is not True:
        raise RuntimeError("M3 training order summary is not ready for screening evaluation")

    wrapper = repo / "tools/benchmark/run_m3_libero_evaluation.py"
    if not wrapper.is_file():
        raise FileNotFoundError(wrapper)
    promotion_records: list[dict[str, Any]] = []
    result_refs: list[dict[str, Any]] = []

    for track in TRACKS:
        for model in MODELS_BY_ORDER[args.order]:
            training_path = _require_file(_training_result_path(training_root, args.order, track, model))
            training = _validate_training_result(training_path, order=args.order, track=track, model=model)
            eval_root = screening_root / track / model
            summary_path = eval_root / "m3_screening_evaluation_summary.json"
            plan_path = eval_root / "m3_screening_evaluation_plan.json"
            promotion_path = promotion_root / track / model / "promotion_record.json"

            if summary_path.is_file():
                _validate_eval_summary(summary_path, training=training, training_path=training_path)
                print(f"skip completed screening eval: {args.order}/{track}/{model}", flush=True)
            else:
                # Any pre-existing output without a valid summary is preserved; use a
                # new attempt rather than deleting simulator evidence.
                if eval_root.exists() and any(eval_root.iterdir()):
                    raise RuntimeError(
                        f"partial screening evaluation output exists: {eval_root}; "
                        "preserved for diagnosis. Use a new --attempt instead of deleting evidence."
                    )
                eval_root.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-u",
                    str(wrapper),
                    "--training-result",
                    str(training_path),
                    "--output-root",
                    str(eval_root),
                    "--plan-out",
                    str(plan_path),
                    "--summary-out",
                    str(summary_path),
                    "--parc-root",
                    str(parc_root),
                    "--repo-root",
                    str(repo),
                    "--mode",
                    "smoke",
                    "--execute",
                ]
                print(f"=== M3 screening eval start: {args.order}/{track}/{model} ===", flush=True)
                subprocess.run(cmd, cwd=str(repo), env=os.environ.copy(), check=True)
                _validate_eval_summary(summary_path, training=training, training_path=training_path)
                print(f"=== M3 screening eval PASS: {args.order}/{track}/{model} ===", flush=True)

            record = build_promotion_record(
                training_path=training_path,
                evaluation_path=summary_path,
                require_screening_episodes=True,
            )
            promotion_path.parent.mkdir(parents=True, exist_ok=True)
            promotion_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            promotion_records.append(record)
            result_refs.append({
                "model": model,
                "track": track,
                "training_result": str(training_path),
                "evaluation_summary": str(summary_path),
                "promotion_record": str(promotion_path),
                "promotion_record_sha256": record["promotion_record_sha256"],
            })

    expected = len(TRACKS) * len(MODELS_BY_ORDER[args.order])
    if len(promotion_records) != expected:
        raise RuntimeError(f"promotion record count mismatch: {len(promotion_records)} != {expected}")
    identities = {(r["model"], r["track"], r["order"]) for r in promotion_records}
    if len(identities) != expected:
        raise RuntimeError("duplicate/missing screening promotion record identities")
    for model in MODELS_BY_ORDER[args.order]:
        if {(r["track"]) for r in promotion_records if r["model"] == model} != set(TRACKS):
            raise RuntimeError(f"screening evaluation missing track for {model}")

    output = {
        "schema_version": 1,
        "stage": "M3_screening_evaluation_order",
        "status": "PASS",
        "order": args.order,
        "attempt": args.attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "evaluation_seed_set": [20260906, 20260907],
        "trials_per_task": 1,
        "episodes_per_checkpoint": SCREENING_EPISODES,
        "checkpoint_count": expected,
        "total_episode_records": expected * SCREENING_EPISODES,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "promotion_record_count": len(promotion_records),
        "ready_for_forward_reverse_promotion": True,
        "final_800_episode_evaluation_started": False,
        "records": result_refs,
    }
    order_eval_summary.parent.mkdir(parents=True, exist_ok=True)
    order_eval_summary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"=== M3 {args.order.upper()} SCREENING EVALUATION: PASS ===", flush=True)
    print(json.dumps({
        "status": "PASS",
        "order": args.order,
        "checkpoints": expected,
        "episodes_per_checkpoint": SCREENING_EPISODES,
        "total_episode_records": output["total_episode_records"],
        "promotion_records": len(promotion_records),
        "summary": str(order_eval_summary),
        "final_800_episode_evaluation_started": False,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
