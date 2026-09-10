#!/usr/bin/env python3
"""Build one promotion-ready M3 record from training + simulator evidence.

The merged promotion controller requires sampling/sample accounting from the
training result and simulator metrics from evaluation. This helper binds those
two evidence objects without weakening either provenance chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
SEEDS = (20260906, 20260907)
SCREENING_EPISODES = 80
EFFECTIVE_BATCH = 32
EQUAL_DATA_SAMPLES = 4800
EQUAL_WALL_SEC = 1800.0
REQUIRED_METRICS = (
    "simulator_success_rate",
    "steps_to_success",
    "episode_duration",
    "inference_latency",
    "peak_inference_vram",
    "train_wall_time",
    "peak_train_vram",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def build_promotion_record(
    *, training_path: Path, evaluation_path: Path, require_screening_episodes: bool = True
) -> dict[str, Any]:
    training = load_object(training_path)
    evaluation = load_object(evaluation_path)
    if training.get("status") != "PASS" or training.get("mode") != "benchmark":
        raise ValueError("promotion training evidence must be PASS benchmark result")
    if evaluation.get("status") != "PASS" or evaluation.get("stage") != "M3_simulator_evaluation":
        raise ValueError("promotion evaluation evidence must be PASS M3 simulator summary")

    for field in ("model", "order", "track", "source_ref", "checkpoint_ref"):
        if training.get(field) != evaluation.get(field):
            raise ValueError(f"training/evaluation identity mismatch for {field}")
    if training.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("training D10 mismatch")
    if evaluation.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("evaluation D10 mismatch")
    if tuple(int(x) for x in evaluation.get("seed_set", [])) != SEEDS:
        raise ValueError("evaluation seed set mismatch")
    if require_screening_episodes and int(evaluation.get("episode_count", -1)) != SCREENING_EPISODES:
        raise ValueError(f"M3 screening evaluation must contain exactly {SCREENING_EPISODES} episodes")

    schedule_hash = str(training.get("sampling_schedule_sha256") or "").strip()
    if not schedule_hash:
        raise ValueError("training sampling_schedule_sha256 missing")
    samples = training.get("samples_consumed")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("training samples_consumed missing/invalid")
    if int(training.get("effective_batch_size", -1)) != EFFECTIVE_BATCH:
        raise ValueError("training effective batch mismatch")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("evaluation metrics missing")
    normalized_metrics: dict[str, float] = {}
    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            raise ValueError(f"evaluation metric invalid: {name}")
        normalized_metrics[name] = float(value)
    if not 0 <= normalized_metrics["simulator_success_rate"] <= 1:
        raise ValueError("simulator success rate outside [0,1]")

    track = str(training["track"])
    if track == "equal_data":
        if samples != EQUAL_DATA_SAMPLES:
            raise ValueError(f"equal-data promotion record requires {EQUAL_DATA_SAMPLES} samples")
    elif track == "equal_wall":
        if normalized_metrics["train_wall_time"] + 1e-9 < EQUAL_WALL_SEC:
            raise ValueError(f"equal-wall promotion record requires >= {EQUAL_WALL_SEC} sec")
        if samples <= 0 or samples % EFFECTIVE_BATCH != 0:
            raise ValueError("equal-wall samples must end on effective-batch boundary")
    else:
        raise ValueError(f"unexpected training track: {track}")

    ref = str(evaluation.get("training_result_ref") or "").strip()
    if not ref or Path(ref).resolve() != training_path.resolve():
        raise ValueError("evaluation does not point to exact training result")

    record: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M3_screening_promotion_record",
        "status": "PASS",
        "model": training["model"],
        "order": training["order"],
        "track": track,
        "source_ref": training["source_ref"],
        "checkpoint_ref": training["checkpoint_ref"],
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "seed_set": list(SEEDS),
        "sampling_schedule_sha256": schedule_hash,
        "samples_consumed": samples,
        "metrics": normalized_metrics,
        "evaluation_episode_count": int(evaluation.get("episode_count", -1)),
        "training_result_ref": str(training_path.resolve()),
        "training_result_sha256": sha256_file(training_path),
        "evaluation_summary_ref": str(evaluation_path.resolve()),
        "evaluation_summary_sha256": sha256_file(evaluation_path),
        "training_loss_used_for_promotion": False,
    }
    record["promotion_record_sha256"] = hashlib.sha256(canonical_json(record)).hexdigest()
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--training-result", type=Path, required=True)
    p.add_argument("--evaluation-summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    record = build_promotion_record(
        training_path=args.training_result,
        evaluation_path=args.evaluation_summary,
        require_screening_episodes=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "model": record["model"],
        "order": record["order"],
        "track": record["track"],
        "episodes": record["evaluation_episode_count"],
        "promotion_record_sha256": record["promotion_record_sha256"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
