import json
from pathlib import Path

import pytest

from tools.submission.freeze_track3_final_candidate import (
    D10_HASH,
    EXPECTED_FINAL_EVAL_EPISODES,
    build_readiness_report,
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


def _pass_record(model: str, order: str, track: str):
    return {
        "status": "PASS",
        "model": model,
        "order": order,
        "track": track,
        "selected_episode_ids_sha256": D10_HASH,
        "source_ref": {"pi05": "v0.4.4", "smolvla": "smol-ref", "openvla_oft": "oft-ref"}[model],
        "seed_set": [20260906, 20260907],
        "sampling_schedule_sha256": f"{track}-schedule",
    }


def _model_summary(model: str):
    return {
        "status": "PASS",
        "records": [
            _pass_record(model, order, track)
            for order in ("forward", "reverse")
            for track in ("equal_data", "equal_wall")
        ],
    }


def _promotion():
    return {
        "status": "READY_FOR_PROMOTION",
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "required_orders": ["forward", "reverse"],
        "required_tracks": ["equal_data", "equal_wall"],
        "seed_set": [20260906, 20260907],
        "primary_metric": "simulator_success_rate",
        "training_loss_used_for_promotion": False,
        "promoted_models": ["pi05", "smolvla"],
        "ranking": ["pi05", "smolvla"],
        "model_summaries": {
            "pi05": _model_summary("pi05"),
            "smolvla": _model_summary("smolvla"),
            "openvla_oft": {
                "status": "EXCLUDED_WITH_EVIDENCE",
                "evidence": "documented A100 runtime exclusion",
            },
        },
    }


def _training(model="pi05"):
    data = {
        "status": "PASS",
        "model": model,
        "selected_episode_ids_sha256": D10_HASH,
        "checkpoint_ref": f"/tmp/final/{model}",
        "source_ref": {"pi05": "v0.4.4", "smolvla": "smol-ref", "openvla_oft": "oft-ref"}[model],
        "metrics": {"train_wall_time": 2000.0, "peak_train_vram": 30000.0},
    }
    if model == "openvla_oft":
        data.update(
            {
                "checkpoint_eval_ready": True,
                "lora_merged_for_evaluation": True,
                "merge_excluded_from_train_wall_time": True,
            }
        )
    return data


def _evaluation(training_path: Path, model="pi05"):
    source_ref = _training(model)["source_ref"]
    return {
        "status": "PASS",
        "stage": "M3_simulator_evaluation",
        "model": model,
        "checkpoint_ref": f"/tmp/final/{model}",
        "source_ref": source_ref,
        "selected_episode_ids_sha256": D10_HASH,
        "seed_set": [20260906, 20260907],
        "max_steps_per_episode": 300,
        "episode_count": EXPECTED_FINAL_EVAL_EPISODES,
        "per_seed": {
            "20260906": {"episode_count": 400},
            "20260907": {"episode_count": 400},
        },
        "training_result_ref": str(training_path),
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


def test_candidate_requires_complete_forward_reverse_matrix(tmp_path):
    promotion = _promotion()
    path = _write(tmp_path / "promotion.json", promotion)
    assert validate_promotion(path, "pi05")["promoted_models"][0] == "pi05"

    promotion["model_summaries"]["pi05"]["records"].pop()
    _write(path, promotion)
    with pytest.raises(ValueError, match="matrix incomplete"):
        validate_promotion(path, "pi05")


def test_missing_competitor_evidence_requires_documented_exclusion(tmp_path):
    promotion = _promotion()
    promotion["model_summaries"]["openvla_oft"] = {"status": "MISSING"}
    path = _write(tmp_path / "promotion.json", promotion)
    with pytest.raises(ValueError, match="documented exclusion"):
        validate_promotion(path, "pi05")


def test_candidate_must_come_from_promotion(tmp_path):
    path = _write(tmp_path / "promotion.json", _promotion())
    with pytest.raises(ValueError, match="was not promoted"):
        validate_promotion(path, "openvla_oft")


def test_openvla_training_requires_eval_ready_merge(tmp_path):
    training = _training("openvla_oft")
    training["checkpoint_eval_ready"] = False
    path = _write(tmp_path / "train.json", training)
    with pytest.raises(ValueError, match="not evaluator-ready"):
        validate_final_training(path, "openvla_oft")


def test_final_eval_must_be_full_800_and_link_training_result(tmp_path):
    training_path = _write(tmp_path / "train.json", _training())
    training = validate_final_training(training_path, "pi05")
    eval_path = _write(tmp_path / "eval.json", _evaluation(training_path))
    assert validate_evaluation(
        eval_path,
        "pi05",
        training["checkpoint_ref"],
        training["source_ref"],
        final_training_path=training_path,
    )["episode_count"] == 800

    smoke = _evaluation(training_path)
    smoke["episode_count"] = 80
    smoke["per_seed"]["20260906"]["episode_count"] = 40
    smoke["per_seed"]["20260907"]["episode_count"] = 40
    _write(eval_path, smoke)
    with pytest.raises(ValueError, match="800 episodes"):
        validate_evaluation(
            eval_path,
            "pi05",
            training["checkpoint_ref"],
            training["source_ref"],
            final_training_path=training_path,
        )


def test_dry_run_before_promotion_reports_not_ready(tmp_path):
    report = build_readiness_report(
        candidate=None,
        promotion_path=tmp_path / "not-created-yet.json",
    )
    assert report["status"] == "NOT_READY"
    assert report["automatic_training"] is False
    assert report["automatic_evaluation"] is False
    assert any(blocker.startswith("promotion:missing") for blocker in report["blockers"])


def test_selection_never_uses_training_loss():
    record = build_selection_record(
        candidate="pi05",
        promotion=_promotion(),
        rationale="best aggregated simulator success",
    )
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


def test_freeze_contract_records_acceptance_and_forbids_automatic_actions():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "experiments/plans/track3_final_candidate_freeze_v1.json").read_text())
    assert contract["final_status"] == "SUBMISSION_ARTIFACT_FROZEN"
    assert contract["candidate_selection"]["training_loss_may_not_select"] is True
    assert contract["candidate_selection"]["forward_reverse_matrix_required"] is True
    assert contract["final_evaluation"]["episode_count"] == 800
    assert contract["dry_run"]["not_ready_is_nonfatal"] is True
    assert contract["automatic_long_training"] is False
    assert contract["automatic_simulator_evaluation"] is False
    assert contract["automatic_omnicampus_upload"] is False
