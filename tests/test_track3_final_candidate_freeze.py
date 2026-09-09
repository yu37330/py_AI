import json
from pathlib import Path

import pytest

from tools.submission.freeze_track3_final_candidate import (
    D10_HASH,
    build_selection_record,
    build_submission_spec,
    validate_evaluation,
    validate_final_training,
    validate_promotion,
    validate_source_sha,
)


def _write(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _promotion():
    return {
        "status": "READY_FOR_PROMOTION",
        "selected_episode_ids_sha256": D10_HASH,
        "primary_metric": "simulator_success_rate",
        "training_loss_used_for_promotion": False,
        "promoted_models": ["pi05", "smolvla"],
        "ranking": ["pi05", "smolvla"],
    }


def _training():
    return {
        "status": "PASS",
        "model": "pi05",
        "selected_episode_ids_sha256": D10_HASH,
        "checkpoint_ref": "/tmp/final/pi05",
        "source_ref": "v0.4.4",
    }


def _evaluation():
    return {
        "status": "PASS",
        "stage": "M3_simulator_evaluation",
        "model": "pi05",
        "checkpoint_ref": "/tmp/final/pi05",
        "source_ref": "v0.4.4",
        "selected_episode_ids_sha256": D10_HASH,
        "metrics": {
            "simulator_success_rate": 0.8,
            "steps_to_success": 100,
            "episode_duration": 10,
            "inference_latency": 20,
            "peak_inference_vram": 12000,
            "train_wall_time": 2000,
            "peak_train_vram": 30000,
        },
    }


def test_source_sha_must_be_full_commit():
    assert validate_source_sha("a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="40-hex"):
        validate_source_sha("abc")


def test_candidate_must_come_from_promotion(tmp_path):
    path = _write(tmp_path / "promotion.json", _promotion())
    assert validate_promotion(path, "pi05")["promoted_models"][0] == "pi05"
    with pytest.raises(ValueError, match="was not promoted"):
        validate_promotion(path, "openvla_oft")


def test_training_and_eval_must_link_same_checkpoint(tmp_path):
    training_path = _write(tmp_path / "train.json", _training())
    training = validate_final_training(training_path, "pi05")
    eval_path = _write(tmp_path / "eval.json", _evaluation())
    assert validate_evaluation(
        eval_path, "pi05", training["checkpoint_ref"], training["source_ref"]
    )["status"] == "PASS"
    broken = _evaluation()
    broken["checkpoint_ref"] = "other"
    _write(eval_path, broken)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_evaluation(eval_path, "pi05", training["checkpoint_ref"], training["source_ref"])


def test_selection_never_uses_training_loss():
    record = build_selection_record(candidate="pi05", promotion=_promotion(), rationale="best aggregated simulator success")
    assert record["candidate"] == "pi05"
    assert record["training_loss_used_for_selection"] is False
    with pytest.raises(ValueError, match="rationale"):
        build_selection_record(candidate="pi05", promotion=_promotion(), rationale="")


def test_submission_spec_matches_integrity_builder_shape(tmp_path):
    spec = build_submission_spec(
        source_sha="b" * 40,
        manifest_path=tmp_path / "manifest.json",
        candidate="pi05",
        artifact_path=tmp_path / "artifact",
        artifact_sha256="c" * 64,
        config_path=tmp_path / "config.yaml",
        promotion_path=tmp_path / "promotion.json",
        evaluation_path=tmp_path / "eval.json",
        reproduction_command="python reproduce.py",
        checkpoint_ref="immutable-checkpoint-ref",
    )
    assert spec["dataset"]["episode_ids_sha256"] == D10_HASH
    assert spec["model"]["name"] == "pi05"
    assert spec["model"]["artifact_sha256"] == "c" * 64
    assert spec["benchmark"]["promotion_summary_path"].endswith("promotion.json")
    assert spec["evaluation"]["metrics_path"].endswith("eval.json")
    assert spec["reproduction"]["command"] == "python reproduce.py"


def test_freeze_contract_forbids_automatic_upload():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "experiments/plans/track3_final_candidate_freeze_v1.json").read_text())
    assert contract["final_status"] == "SUBMISSION_ARTIFACT_FROZEN"
    assert contract["candidate_selection"]["training_loss_may_not_select"] is True
    assert contract["automatic_long_training"] is False
    assert contract["automatic_omnicampus_upload"] is False
