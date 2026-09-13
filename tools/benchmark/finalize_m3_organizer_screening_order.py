#!/usr/bin/env python3
"""Finalize one organizer screening order after six checkpoint units PASS.

This is CPU-only. It never starts simulator evaluation. It validates the six
canonical 80-episode summaries plus exact training links, rebuilds promotion
records, and writes the same order-level summary contract consumed by M3 77.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    p.add_argument("--attempt", default=os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "organizer-1"))
    return p.parse_args()


def _safe(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(ch not in allowed for ch in value):
        raise ValueError("attempt may contain only letters, digits, '-' and '_'")
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
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"training mismatch {path}: {key}={data.get(key)!r} != {value!r}")
    if not Path(str(data.get("checkpoint_ref") or "")).exists():
        raise FileNotFoundError(f"training checkpoint missing: {data.get('checkpoint_ref')}")
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
        raise RuntimeError("screening seed set mismatch")
    ref = str(data.get("training_result_ref") or "").strip()
    if not ref or Path(ref).resolve() != training_path.resolve():
        raise RuntimeError("screening evaluation does not link exact training result")
    return data


def main() -> int:
    args = parse_args()
    attempt = _safe(args.attempt)
    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    persist = Path(
        os.environ.get(
            "PARC_PERSIST_ROOT",
            os.environ.get("PARC_DRIVE_ROOT", str(Path.home() / "data/parc2026-cache")),
        )
    ).expanduser().resolve()
    training_root = persist / f"model-benchmark-v1/m3-benchmark-v1/attempt-{attempt}"
    screening_root = persist / f"model-benchmark-v1/m3-screening-eval-v1/attempt-{attempt}/{args.order}"
    promotion_root = persist / f"model-benchmark-v1/m3-promotion-records-v1/attempt-{attempt}/{args.order}"
    output_path = screening_root / f"m3_screening_{args.order}_summary.json"

    training_summary = _load(_require(training_root / f"m3_training_{args.order}_summary.json"))
    if (
        training_summary.get("status") != "PASS"
        or training_summary.get("order") != args.order
        or training_summary.get("selected_episode_ids_sha256") != D10_HASH
        or training_summary.get("ready_for_screening_evaluation") is not True
    ):
        raise RuntimeError("benchmark order is not screening-ready")

    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_promotion_record import build_promotion_record  # noqa: PLC0415

    records: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for track in TRACKS:
        for model in MODELS_BY_ORDER[args.order]:
            training_path = _require(
                training_root / args.order / track / f"seed-{TRAINING_SCHEDULE_SEED}" / model / "train_result.json"
            )
            training = _validate_training(training_path, order=args.order, track=track, model=model)
            eval_path = _require(screening_root / track / model / "m3_screening_evaluation_summary.json")
            _validate_eval(eval_path, training=training, training_path=training_path)
            record = build_promotion_record(
                training_path=training_path,
                evaluation_path=eval_path,
                require_screening_episodes=True,
            )
            promotion_path = promotion_root / track / model / "promotion_record.json"
            promotion_path.parent.mkdir(parents=True, exist_ok=True)
            promotion_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            records.append(record)
            refs.append(
                {
                    "model": model,
                    "track": track,
                    "training_result": str(training_path),
                    "evaluation_summary": str(eval_path),
                    "promotion_record": str(promotion_path),
                    "promotion_record_sha256": record["promotion_record_sha256"],
                }
            )

    expected_count = len(TRACKS) * len(MODELS_BY_ORDER[args.order])
    if len(records) != expected_count:
        raise RuntimeError(f"promotion record count mismatch: {len(records)} != {expected_count}")
    identities = {(r["model"], r["track"], r["order"]) for r in records}
    if len(identities) != expected_count:
        raise RuntimeError("duplicate/missing screening promotion identities")

    output = {
        "schema_version": 1,
        "stage": "M3_screening_evaluation_order",
        "status": "PASS",
        "order": args.order,
        "attempt": attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "evaluation_seed_set": [20260906, 20260907],
        "trials_per_task": 1,
        "episodes_per_checkpoint": SCREENING_EPISODES,
        "checkpoint_count": expected_count,
        "total_episode_records": expected_count * SCREENING_EPISODES,
        "promotion_record_count": len(records),
        "ready_for_forward_reverse_promotion": True,
        "final_800_episode_evaluation_started": False,
        "organizer_checkpoint_scoped_execution": True,
        "records": refs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"=== M3 ORGANIZER {args.order.upper()} SCREENING FINALIZED: PASS ===", flush=True)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
