import json
from pathlib import Path

import pytest

from tools.benchmark.aggregate_m3_results import (
    D10_HASH,
    MODELS,
    REQUIRED_METRICS,
    aggregate,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
    "openvla_oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}


def _record(model, order, track, success, *, schedule_hash=None, loss=1.0):
    schedule_hash = schedule_hash or f"schedule-{track}"
    train_wall = 1810.0 if track == "equal_wall" else 120.0
    samples = 6400 if track == "equal_wall" else 4800
    return {
        "schema_version": 1,
        "status": "PASS",
        "model": model,
        "order": order,
        "track": track,
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "source_ref": SOURCE_REFS[model],
        "sampling_schedule_sha256": schedule_hash,
        "seed_set": [20260906, 20260907],
        "samples_consumed": samples,
        "training_loss": loss,
        "metrics": {
            "simulator_success_rate": success,
            "steps_to_success": 100.0 + (0.01 if order == "reverse" else 0.0),
            "episode_duration": 5.0,
            "inference_latency": 0.05,
            "peak_inference_vram": 12000.0,
            "train_wall_time": train_wall,
            "peak_train_vram": 30000.0,
        },
    }


def _complete_model(model, success, *, loss=1.0, schedule_suffix=""):
    records = []
    for order in ("forward", "reverse"):
        for track in ("equal_data", "equal_wall"):
            records.append(
                _record(
                    model,
                    order,
                    track,
                    success,
                    schedule_hash=f"schedule-{track}{schedule_suffix}",
                    loss=loss,
                )
            )
    return records


def test_contract_keeps_forward_reverse_and_loss_out_of_promotion():
    plan = json.loads(
        (ROOT / "experiments/plans/m3_promotion_aggregation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["orders"] == ["forward", "reverse"]
    assert plan["tracks"] == ["equal_data", "equal_wall"]
    assert plan["promotion"]["max_models"] == 2
    assert plan["promotion"]["primary"] == "simulator_success_rate"
    assert plan["promotion"]["training_loss_alone_is_not_sufficient"] is True


def test_complete_matrix_promotes_top_two_by_success_not_loss():
    records = []
    # Best model deliberately has the worst training loss.
    records += _complete_model("pi05", 0.80, loss=99.0)
    records += _complete_model("smolvla", 0.70, loss=0.01)
    records += _complete_model("openvla_oft", 0.60, loss=0.001)
    summary = aggregate(records)
    assert summary["status"] == "READY_FOR_PROMOTION"
    assert summary["ranking"] == ["pi05", "smolvla", "openvla_oft"]
    assert summary["promoted_models"] == ["pi05", "smolvla"]
    assert summary["training_loss_used_for_promotion"] is False
    assert len(summary["promoted_models"]) <= 2


def test_missing_reverse_record_makes_model_incomplete_and_not_promotable():
    records = _complete_model("pi05", 0.8)
    records = [
        r
        for r in records
        if not (r["order"] == "reverse" and r["track"] == "equal_wall")
    ]
    records += _complete_model("smolvla", 0.7)
    summary = aggregate(records)
    assert summary["model_summaries"]["pi05"]["status"] == "INCOMPLETE"
    assert "pi05" not in summary["promoted_models"]
    assert summary["promoted_models"] == ["smolvla"]


def test_schedule_must_match_between_forward_and_reverse():
    records = _complete_model("pi05", 0.8)
    for record in records:
        if record["order"] == "reverse" and record["track"] == "equal_data":
            record["sampling_schedule_sha256"] = "different-order-schedule"
    summary = aggregate(records)
    assert summary["model_summaries"]["pi05"]["status"] == "INCOMPLETE"
    assert summary["status"] == "NOT_READY"


def test_schedule_must_match_across_promotable_models():
    records = _complete_model("pi05", 0.8)
    records += _complete_model("smolvla", 0.7, schedule_suffix="-drift")
    with pytest.raises(ValueError, match="sampling schedule differs across promotable models"):
        aggregate(records)


def test_explicit_exclusion_with_evidence_allows_other_models_to_continue():
    records = _complete_model("pi05", 0.8)
    records += _complete_model("smolvla", 0.7)
    records.append(
        {
            "status": "EXCLUDED_WITH_EVIDENCE",
            "model": "openvla_oft",
            "selected_episode_ids_sha256": D10_HASH,
            "evidence": "Sep 10 cut rule: reproducible dependency failure after focused repair cycle",
        }
    )
    summary = aggregate(records)
    assert summary["model_summaries"]["openvla_oft"]["status"] == "EXCLUDED_WITH_EVIDENCE"
    assert summary["promoted_models"] == ["pi05", "smolvla"]


def test_equal_data_refuses_sample_overshoot():
    record = _record("pi05", "forward", "equal_data", 0.5)
    record["samples_consumed"] = 4801
    with pytest.raises(ValueError, match="exactly 4800"):
        aggregate([record])


def test_equal_wall_refuses_early_stop():
    record = _record("pi05", "forward", "equal_wall", 0.5)
    record["metrics"]["train_wall_time"] = 1799.99
    with pytest.raises(ValueError, match=">= 1800.0"):
        aggregate([record])


def test_all_required_metrics_are_present_in_contract():
    plan = json.loads(
        (ROOT / "experiments/plans/m3_promotion_aggregation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(plan["required_metrics"]) == REQUIRED_METRICS
    assert set(MODELS) == {"pi05", "smolvla", "openvla_oft"}
