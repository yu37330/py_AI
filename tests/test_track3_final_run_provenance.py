import ast
import json
from pathlib import Path

import pytest

from tools.submission.finalize_track3_submission import validate_chain
from tools.submission.link_track3_final_training_result import link_result
from tools.submission.prepare_track3_final_run import build_final_run_config

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"


def _pass_record(model: str, order: str, track: str, source_ref: str):
    return {
        "status": "PASS",
        "model": model,
        "order": order,
        "track": track,
        "selected_episode_ids_sha256": D10_HASH,
        "source_ref": source_ref,
        "seed_set": [20260906, 20260907],
        "sampling_schedule_sha256": f"{track}-schedule",
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
        "promoted_models": ["pi05"],
        "ranking": ["pi05"],
        "model_summaries": {
            "pi05": {
                "status": "PASS",
                "records": [
                    _pass_record("pi05", order, track, "v0.4.4")
                    for order in ("forward", "reverse")
                    for track in ("equal_data", "equal_wall")
                ],
            },
            "smolvla": {
                "status": "EXCLUDED_WITH_EVIDENCE",
                "evidence": "documented deadline fallback",
            },
            "openvla_oft": {
                "status": "EXCLUDED_WITH_EVIDENCE",
                "evidence": "documented deadline fallback",
            },
        },
    }


def _raw_training(source_ref="v0.4.4", mode="benchmark", samples=4800, updates=150):
    return {
        "schema_version": 2,
        "stage": "M3_model_training",
        "status": "PASS",
        "model": "pi05",
        "source_ref": source_ref,
        "checkpoint_ref": "/tmp/final/pi05",
        "order": "forward",
        "track": "equal_data",
        "mode": mode,
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": 20260906,
        "sampling_policy": "uniform_selected_frames_with_replacement_v1",
        "sampling_schedule_sha256": "equal_data-schedule",
        "samples_consumed": samples,
        "micro_batch": 8,
        "gradient_accumulation": 4,
        "effective_batch_size": 32,
        "micro_steps": 600,
        "optimizer_updates": updates,
        "metrics": {"train_wall_time": 123.0, "peak_train_vram": 12345.0},
    }


def _evaluation(linked_training_path: Path):
    return {
        "status": "PASS",
        "stage": "M3_simulator_evaluation",
        "model": "pi05",
        "checkpoint_ref": "/tmp/final/pi05",
        "source_ref": "v0.4.4",
        "selected_episode_ids_sha256": D10_HASH,
        "seed_set": [20260906, 20260907],
        "max_steps_per_episode": 300,
        "episode_count": 800,
        "per_seed": {
            "20260906": {"episode_count": 400},
            "20260907": {"episode_count": 400},
        },
        "training_result_ref": str(linked_training_path.resolve()),
        "metrics": {
            "simulator_success_rate": 0.5,
            "steps_to_success": 100.0,
            "episode_duration": 10.0,
            "inference_latency": 20.0,
            "peak_inference_vram": 12000.0,
            "train_wall_time": 123.0,
            "peak_train_vram": 12345.0,
        },
    }


def test_new_submission_tools_are_syntax_valid():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "tools/submission/prepare_track3_final_run.py",
        "tools/submission/link_track3_final_training_result.py",
        "tools/submission/finalize_track3_submission.py",
    ):
        ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)


def test_prepare_binds_promotion_registry_and_recipe_source(tmp_path):
    root = Path(__file__).resolve().parents[1]
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(json.dumps(_promotion()), encoding="utf-8")
    cfg = build_final_run_config(
        repo_root=root,
        promotion_path=promotion_path,
        candidate="pi05",
        track="equal_data",
    )
    assert cfg["status"] == "READY_FOR_FINAL_RUN"
    assert cfg["source_ref"] == "v0.4.4"
    assert cfg["budget"]["sample_budget"] == 4800
    assert cfg["budget"]["optimizer_updates"] == 150
    assert cfg["sampling"]["training_schedule_seed"] == 20260906
    assert cfg["automatic_training"] is False


def test_link_rejects_smoke_source_drift_and_budget_drift(tmp_path):
    root = Path(__file__).resolve().parents[1]
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(json.dumps(_promotion()), encoding="utf-8")
    cfg = build_final_run_config(
        repo_root=root,
        promotion_path=promotion_path,
        candidate="pi05",
        track="equal_data",
    )
    cfg_path = tmp_path / "final-run.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(_raw_training(mode="smoke")), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark/final"):
        link_result(config_path=cfg_path, raw_result_path=raw_path)

    raw_path.write_text(json.dumps(_raw_training(source_ref="drifted")), encoding="utf-8")
    with pytest.raises(ValueError, match="source_ref"):
        link_result(config_path=cfg_path, raw_result_path=raw_path)

    raw_path.write_text(json.dumps(_raw_training(samples=4799)), encoding="utf-8")
    with pytest.raises(ValueError, match="4800"):
        link_result(config_path=cfg_path, raw_result_path=raw_path)

    raw_path.write_text(json.dumps(_raw_training(updates=149)), encoding="utf-8")
    with pytest.raises(ValueError, match="150"):
        link_result(config_path=cfg_path, raw_result_path=raw_path)


def test_canonical_chain_rejects_recipe_or_promotion_mutation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(json.dumps(_promotion()), encoding="utf-8")
    cfg = build_final_run_config(
        repo_root=root,
        promotion_path=promotion_path,
        candidate="pi05",
        track="equal_data",
    )
    cfg_path = tmp_path / "final-run.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(_raw_training()), encoding="utf-8")
    linked = link_result(config_path=cfg_path, raw_result_path=raw_path)
    linked_path = tmp_path / "linked.json"
    linked_path.write_text(json.dumps(linked), encoding="utf-8")
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(_evaluation(linked_path)), encoding="utf-8")
    model_config = root / "experiments/configs/pi05.yaml"

    _, checked_cfg, training = validate_chain(
        candidate="pi05",
        promotion_path=promotion_path,
        final_run_config_path=cfg_path,
        linked_training_path=linked_path,
        evaluation_path=eval_path,
        model_config_path=model_config,
    )
    assert checked_cfg["source_ref"] == training["source_ref"] == "v0.4.4"

    changed = _promotion()
    changed["ranking"] = ["pi05", "changed-after-config"]
    promotion_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after final-run config"):
        validate_chain(
            candidate="pi05",
            promotion_path=promotion_path,
            final_run_config_path=cfg_path,
            linked_training_path=linked_path,
            evaluation_path=eval_path,
            model_config_path=model_config,
        )
