import json
from pathlib import Path

import pytest

from tools.benchmark.m3_eval_instrumentation import build_episode_record
from tools.benchmark.m3_libero_executor import (
    SEEDS,
    SUITES,
    SOURCE_REFS,
    build_jobs,
    default_eval_runtimes,
)


def _training(model: str):
    return {
        "model": model,
        "checkpoint_ref": f"/tmp/{model}/checkpoint",
        "source_ref": SOURCE_REFS[model],
        "selected_episode_ids_sha256": "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239",
        "order": "forward",
        "track": "equal_data",
    }


def test_lerobot_models_build_one_multisuite_job_per_seed(tmp_path):
    for model in ("pi05", "smolvla"):
        jobs = build_jobs(training_result=_training(model), root=tmp_path, output_root=tmp_path / "out", smoke=True)
        assert len(jobs) == len(SEEDS)
        assert {job["seed"] for job in jobs} == set(SEEDS)
        for job in jobs:
            text = " ".join(job["command"])
            assert "--env.type=libero" in text
            assert ",".join(SUITES) in text
            assert "--eval.n_episodes=1" in text
        if model == "pi05":
            assert "--policy.n_action_steps=10" in " ".join(jobs[0]["command"])


def test_openvla_builds_suite_x_seed_jobs_and_center_crop(tmp_path):
    jobs = build_jobs(training_result=_training("openvla_oft"), root=tmp_path, output_root=tmp_path / "out", smoke=False)
    assert len(jobs) == len(SEEDS) * len(SUITES)
    assert {job["suite"] for job in jobs} == set(SUITES)
    for job in jobs:
        text = " ".join(job["command"])
        assert "run_libero_eval.py" in text
        assert "--center_crop True" in text
        assert "--num_trials_per_task 10" in text


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


def test_executor_contract_has_required_instrumentation():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "experiments/plans/m3_libero_executor_v1.json").read_text())
    assert contract["seed_set"] == list(SEEDS)
    assert contract["max_steps_per_episode"] == 300
    assert contract["instrumentation"]["synchronized_select_action_latency"] is True
    assert contract["automatic_full_evaluation"] is False
