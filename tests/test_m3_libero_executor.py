import json
from pathlib import Path

import pytest

from tools.benchmark.m3_eval_instrumentation import build_episode_record
from tools.benchmark.m3_libero_executor import (
    SEEDS,
    SUITES,
    SOURCE_REFS,
    build_jobs,
)


def _training(model: str):
    return {
        "status": "PASS",
        "model": model,
        "checkpoint_ref": f"/tmp/{model}/checkpoint",
        "source_ref": SOURCE_REFS[model],
        "selected_episode_ids_sha256": "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239",
        "order": "forward",
        "track": "equal_data",
        "metrics": {"train_wall_time": 100.0, "peak_train_vram": 1000.0},
    }


def _jobs(tmp_path: Path, model: str, smoke: bool):
    training_path = tmp_path / f"{model}-train.json"
    training_path.write_text(json.dumps(_training(model)))
    return build_jobs(
        training_result=_training(model),
        training_result_path=training_path,
        root=tmp_path,
        repo_root=tmp_path / "repo",
        output_root=tmp_path / "out",
        smoke=smoke,
    )


def test_lerobot_models_build_suite_x_seed_instrumented_jobs(tmp_path):
    for model in ("pi05", "smolvla"):
        jobs = _jobs(tmp_path, model, smoke=True)
        assert len(jobs) == len(SEEDS) * len(SUITES)
        assert {job["seed"] for job in jobs} == set(SEEDS)
        assert {job["suite"] for job in jobs} == set(SUITES)
        for job in jobs:
            text = " ".join(job["command"])
            assert "m3_lerobot_eval_entry.py" in text
            assert "--episodes 1" in text
            assert job["records_out"].endswith("episode_records.json")


def test_openvla_builds_suite_x_seed_instrumented_jobs(tmp_path):
    jobs = _jobs(tmp_path, "openvla_oft", smoke=False)
    assert len(jobs) == len(SEEDS) * len(SUITES)
    assert {job["suite"] for job in jobs} == set(SUITES)
    for job in jobs:
        text = " ".join(job["command"])
        assert "m3_openvla_eval_entry.py" in text
        assert "--trials-per-task 10" in text
        assert job["records_out"].endswith("episode_records.json")


def test_episode_record_enforces_step_boundary():
    rec = build_episode_record(
        model="pi05",
        checkpoint_ref="ckpt",
        source_ref=SOURCE_REFS["pi05"],
        order="forward",
        track="equal_data",
        seed=SEEDS[0],
        task_id="libero_spatial/0",
        success=True,
        steps=42,
        episode_duration_sec=4.2,
        mean_inference_latency_ms=25.0,
        peak_inference_vram_mib=12000,
    )
    assert rec["steps"] == 42
    with pytest.raises(ValueError, match="within"):
        build_episode_record(
            model="pi05", checkpoint_ref="c", source_ref="r", order="forward", track="equal_data",
            seed=SEEDS[0], task_id="t", success=False, steps=301,
            episode_duration_sec=1, mean_inference_latency_ms=1, peak_inference_vram_mib=1,
        )


def test_entries_enforce_300_and_instrument_upstream_rollout():
    root = Path(__file__).resolve().parents[1]
    lerobot_entry = (root / "tools/benchmark/m3_lerobot_eval_entry.py").read_text()
    openvla_entry = (root / "tools/benchmark/m3_openvla_eval_entry.py").read_text()
    assert "--env.episode_length=300" in lerobot_entry
    assert "instrument_select_action" in lerobot_entry
    assert "done_indices" in lerobot_entry
    assert "module.TASK_MAX_STEPS[key] = 300" in openvla_entry
    assert "timed_get_action" in openvla_entry
    assert "save_rollout_video = lambda" in openvla_entry


def test_executor_contract_has_required_instrumentation():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "experiments/plans/m3_libero_executor_v1.json").read_text())
    assert contract["seed_set"] == list(SEEDS)
    assert contract["max_steps_per_episode"] == 300
    assert contract["instrumentation"]["synchronized_select_action_latency"] is True
    assert contract["automatic_full_evaluation"] is False
