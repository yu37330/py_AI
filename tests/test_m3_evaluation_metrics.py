import json
from pathlib import Path

import pytest

from tools.benchmark.m3_evaluation_metrics import (
    D10_HASH,
    MAX_STEPS,
    REQUIRED_METRICS,
    aggregate_episode_records,
)

ROOT = Path(__file__).resolve().parents[1]


def _episode(seed, success=True, steps=100, *, task="task-a"):
    return {
        "model": "pi05",
        "checkpoint_ref": "checkpoint-forward-equal-data",
        "source_ref": "v0.4.4",
        "order": "forward",
        "track": "equal_data",
        "seed": seed,
        "task_id": task,
        "success": success,
        "steps": steps,
        "episode_duration_sec": 5.0,
        "mean_inference_latency_ms": 40.0,
        "peak_inference_vram_mib": 12000.0,
    }


def _training_result():
    return {
        "model": "pi05",
        "checkpoint_ref": "checkpoint-forward-equal-data",
        "source_ref": "v0.4.4",
        "order": "forward",
        "track": "equal_data",
        "selected_episode_ids_sha256": D10_HASH,
        "metrics": {
            "train_wall_time": 130.0,
            "peak_train_vram": 30000.0,
        },
    }


def test_evaluation_contract_contains_all_required_metrics_and_seeds():
    plan = json.loads(
        (ROOT / "experiments/plans/m3_evaluation_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(plan["required_metrics"]) == REQUIRED_METRICS
    assert plan["seed_set"] == [20260906, 20260907]
    assert plan["max_steps_per_episode"] == 300
    assert plan["safety"]["simulator_execution_by_default"] is False


def test_aggregate_produces_required_metrics_and_seed_breakdown():
    records = [
        _episode(20260906, True, 80, task="a"),
        _episode(20260906, False, 300, task="b"),
        _episode(20260907, True, 120, task="a"),
        _episode(20260907, True, 100, task="b"),
    ]
    summary = aggregate_episode_records(
        records,
        training_result=_training_result(),
        training_result_ref="training.json",
    )
    assert summary["status"] == "PASS"
    assert summary["episode_count"] == 4
    assert summary["success_count"] == 3
    assert summary["metrics"]["simulator_success_rate"] == 0.75
    assert summary["metrics"]["steps_to_success"] == 100.0
    assert summary["metrics"]["train_wall_time"] == 130.0
    assert summary["metrics"]["peak_train_vram"] == 30000.0
    assert set(summary["metrics"]) == set(REQUIRED_METRICS)
    assert set(summary["per_seed"]) == {"20260906", "20260907"}
    assert summary["steps_to_success_censored"] is False


def test_zero_success_uses_documented_censored_max_steps_value():
    records = [
        _episode(20260906, False, 300),
        _episode(20260907, False, 300),
    ]
    summary = aggregate_episode_records(
        records,
        training_result=_training_result(),
        training_result_ref="training.json",
    )
    assert summary["metrics"]["simulator_success_rate"] == 0.0
    assert summary["metrics"]["steps_to_success"] == float(MAX_STEPS)
    assert summary["steps_to_success_censored"] is True


def test_exact_seed_set_is_required():
    records = [_episode(20260906, True, 100)]
    with pytest.raises(ValueError, match="exact seed set"):
        aggregate_episode_records(
            records,
            training_result=_training_result(),
            training_result_ref="training.json",
        )


def test_mixed_checkpoint_identity_is_rejected():
    records = [_episode(20260906), _episode(20260907)]
    records[1]["checkpoint_ref"] = "other-checkpoint"
    with pytest.raises(ValueError, match="mixed evaluation identity"):
        aggregate_episode_records(
            records,
            training_result=_training_result(),
            training_result_ref="training.json",
        )


def test_training_result_must_link_to_same_checkpoint_and_d10():
    records = [_episode(20260906), _episode(20260907)]
    training = _training_result()
    training["selected_episode_ids_sha256"] = "wrong"
    with pytest.raises(ValueError, match="D10 hash mismatch"):
        aggregate_episode_records(
            records,
            training_result=training,
            training_result_ref="training.json",
        )


def test_episode_steps_never_exceed_frozen_max():
    records = [_episode(20260906), _episode(20260907)]
    records[0]["steps"] = 301
    with pytest.raises(ValueError, match="within"):
        aggregate_episode_records(
            records,
            training_result=_training_result(),
            training_result_ref="training.json",
        )
