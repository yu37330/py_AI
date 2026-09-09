#!/usr/bin/env python3
"""Validate and aggregate model-independent M3 simulator episode records.

The actual simulator/model adapters can be wired later.  This module freezes the
per-episode evidence and required metric semantics now so checkpoint evaluation
can begin immediately once M3 training outputs exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
MODELS = ("pi05", "smolvla", "openvla_oft")
ORDERS = ("forward", "reverse")
TRACKS = ("equal_data", "equal_wall")
SEEDS = (20260906, 20260907)
MAX_STEPS = 300
REQUIRED_METRICS = (
    "simulator_success_rate",
    "steps_to_success",
    "episode_duration",
    "inference_latency",
    "peak_inference_vram",
    "train_wall_time",
    "peak_train_vram",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0")
    return number


def validate_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    model = record.get("model")
    order = record.get("order")
    track = record.get("track")
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    if order not in ORDERS:
        raise ValueError(f"unknown order: {order}")
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track}")
    checkpoint_ref = str(record.get("checkpoint_ref") or "").strip()
    source_ref = str(record.get("source_ref") or "").strip()
    task_id = str(record.get("task_id") or "").strip()
    if not checkpoint_ref:
        raise ValueError("checkpoint_ref is required")
    if not source_ref:
        raise ValueError("source_ref is required")
    if not task_id:
        raise ValueError("task_id is required")
    seed = int(record.get("seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"unexpected evaluation seed: {seed}")
    success = record.get("success")
    if not isinstance(success, bool):
        raise ValueError("success must be boolean")
    steps = int(record.get("steps", -1))
    if not 0 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be within [0,{MAX_STEPS}]")
    if success and steps <= 0:
        raise ValueError("successful episode must report at least one step")

    normalized = dict(record)
    normalized.update(
        {
            "checkpoint_ref": checkpoint_ref,
            "source_ref": source_ref,
            "task_id": task_id,
            "seed": seed,
            "success": success,
            "steps": steps,
            "episode_duration_sec": _nonnegative_number(
                record.get("episode_duration_sec"), "episode_duration_sec"
            ),
            "mean_inference_latency_ms": _nonnegative_number(
                record.get("mean_inference_latency_ms"), "mean_inference_latency_ms"
            ),
            "peak_inference_vram_mib": _nonnegative_number(
                record.get("peak_inference_vram_mib"), "peak_inference_vram_mib"
            ),
        }
    )
    return normalized


def _training_metric(training_result: dict[str, Any], name: str) -> float:
    metrics = training_result.get("metrics")
    if isinstance(metrics, dict) and name in metrics:
        return _nonnegative_number(metrics[name], f"training.metrics.{name}")
    if name in training_result:
        return _nonnegative_number(training_result[name], f"training.{name}")
    raise ValueError(f"training result missing required metric: {name}")


def validate_training_link(
    training_result: dict[str, Any], episode_identity: dict[str, str]
) -> tuple[float, float]:
    if training_result.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("training result D10 hash mismatch")
    for field in ("model", "checkpoint_ref", "source_ref", "order", "track"):
        expected = episode_identity[field]
        actual = str(training_result.get(field) or "")
        if actual != expected:
            raise ValueError(
                f"training/evaluation identity mismatch for {field}: {actual!r} != {expected!r}"
            )
    train_wall = _training_metric(training_result, "train_wall_time")
    peak_train_vram = _training_metric(training_result, "peak_train_vram")
    return train_wall, peak_train_vram


def aggregate_episode_records(
    records: Iterable[dict[str, Any]],
    *,
    training_result: dict[str, Any],
    training_result_ref: str,
) -> dict[str, Any]:
    normalized = [validate_episode_record(record) for record in records]
    if not normalized:
        raise ValueError("at least one simulator episode record is required")

    identity_fields = ("model", "checkpoint_ref", "source_ref", "order", "track")
    identity = {field: str(normalized[0][field]) for field in identity_fields}
    for record in normalized[1:]:
        for field in identity_fields:
            if str(record[field]) != identity[field]:
                raise ValueError(f"mixed evaluation identity for {field}")

    observed_seeds = {int(record["seed"]) for record in normalized}
    if observed_seeds != set(SEEDS):
        raise ValueError(
            f"evaluation must contain exact seed set {list(SEEDS)}, got {sorted(observed_seeds)}"
        )

    train_wall, peak_train_vram = validate_training_link(training_result, identity)
    successes = [record for record in normalized if record["success"]]
    success_rate = len(successes) / len(normalized)
    steps_censored = not successes
    steps_to_success = (
        sum(record["steps"] for record in successes) / len(successes)
        if successes
        else float(MAX_STEPS)
    )
    episode_duration = sum(
        record["episode_duration_sec"] for record in normalized
    ) / len(normalized)
    inference_latency = sum(
        record["mean_inference_latency_ms"] for record in normalized
    ) / len(normalized)
    peak_inference_vram = max(
        record["peak_inference_vram_mib"] for record in normalized
    )

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        subset = [record for record in normalized if record["seed"] == seed]
        per_seed[str(seed)] = {
            "episode_count": len(subset),
            "success_count": sum(1 for record in subset if record["success"]),
            "simulator_success_rate": (
                sum(1 for record in subset if record["success"]) / len(subset)
            ),
        }

    metrics = {
        "simulator_success_rate": success_rate,
        "steps_to_success": steps_to_success,
        "episode_duration": episode_duration,
        "inference_latency": inference_latency,
        "peak_inference_vram": peak_inference_vram,
        "train_wall_time": train_wall,
        "peak_train_vram": peak_train_vram,
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M3_simulator_evaluation",
        "status": "PASS",
        **identity,
        "selected_episode_ids_sha256": D10_HASH,
        "seed_set": list(SEEDS),
        "max_steps_per_episode": MAX_STEPS,
        "episode_count": len(normalized),
        "success_count": len(successes),
        "steps_to_success_censored": steps_censored,
        "training_result_ref": str(training_result_ref),
        "metrics": metrics,
        "per_seed": per_seed,
        "episode_records_sha256": _sha256_json(normalized),
    }
    output["evaluation_summary_sha256"] = _sha256_json(output)
    return output


def _load_episode_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        return data["episodes"]
    raise ValueError("episode record file must be a JSON list or {'episodes': [...]} object")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    training_result = json.loads(args.training_result.read_text(encoding="utf-8"))
    if not isinstance(training_result, dict):
        raise ValueError("training result must be a JSON object")
    summary = aggregate_episode_records(
        _load_episode_records(args.episodes),
        training_result=training_result,
        training_result_ref=str(args.training_result),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "model": summary["model"],
        "order": summary["order"],
        "track": summary["track"],
        "episode_count": summary["episode_count"],
        "metrics": summary["metrics"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
