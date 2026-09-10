import json
from pathlib import Path

import pytest

from tools.benchmark.m3_eval_instrumentation import build_episode_record
from tools.benchmark.m3_libero_executor import (
    SEEDS,
    SUITES,
    SOURCE_REFS,
    TASKS_PER_SUITE,
    build_jobs,
)


def _training(model: str):
    payload = {
        "status": "PASS",
        "model": model,
        "checkpoint_ref": f"/tmp/{model}/checkpoint",
        "source_ref": SOURCE_REFS[model],
        "selected_episode_ids_sha256": "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239",
        "order": "forward",
        "track": "equal_data",
        "metrics": {"train_wall_time": 100.0, "peak_train_vram": 1000.0},
    }
    if model == "openvla_oft":
        payload["checkpoint_eval_ready"] = True
        payload["lora_merged_for_evaluation"] = True
    return payload


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
            assert "--source-root" in text
            assert "--episodes 1" in text
            assert job["expected_episode_records"] == TASKS_PER_SUITE
            assert job["records_out"].endswith("episode_records.json")


def test_openvla_builds_suite_x_seed_instrumented_jobs(tmp_path):
    jobs = _jobs(tmp_path, "openvla_oft", smoke=False)
    assert len(jobs) == len(SEEDS) * len(SUITES)
    assert {job["suite"] for job in jobs} == set(SUITES)
    for job in jobs:
        text = " ".join(job["command"])
        assert "m3_openvla_eval_entry.py" in text
        assert "--source-root" in text
        assert "--trials-per-task 10" in text
        assert job["expected_episode_records"] == TASKS_PER_SUITE * 10
        assert job["records_out"].endswith("episode_records.json")


def test_smoke_and_benchmark_record_totals_are_frozen(tmp_path):
    smoke_jobs = _jobs(tmp_path, "pi05", smoke=True)
    benchmark_jobs = _jobs(tmp_path, "pi05", smoke=False)
    assert sum(j["expected_episode_records"] for j in smoke_jobs) == 80
    assert sum(j["expected_episode_records"] for j in benchmark_jobs) == 800


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
            model="pi05",
            checkpoint_ref="c",
            source_ref="r",
            order="forward",
            track="equal_data",
            seed=SEEDS[0],
            task_id="t",
            success=False,
            steps=301,
            episode_duration_sec=1,
            mean_inference_latency_ms=1,
            peak_inference_vram_mib=1,
        )


def test_runtime_guard_requires_opt_in_a100_and_pinned_source():
    root = Path(__file__).resolve().parents[1]
    guard = (root / "tools/benchmark/m3_eval_runtime_guard.py").read_text()
    for token in (
        "PARC_M3_EXECUTE",
        '"A100" not in name',
        "MIN_A100_VRAM_MIB = 38_000",
        "rev-parse",
        "v0.4.4",
        "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
        "e4287e94541f459edc4feabc4e181f537cd569a8",
    ):
        assert token in guard


def test_entries_enforce_300_hard_reset_and_instrument_upstream_rollout():
    root = Path(__file__).resolve().parents[1]
    lerobot_entry = (root / "tools/benchmark/m3_lerobot_eval_entry.py").read_text()
    openvla_entry = (root / "tools/benchmark/m3_openvla_eval_entry.py").read_text()
    assert "--env.episode_length=300" in lerobot_entry
    assert "--env.hard_reset=true" in lerobot_entry
    assert "instrument_select_action" in lerobot_entry
    assert "done_indices" in lerobot_entry
    assert "validate_runtime(args.model" in lerobot_entry
    assert "module.TASK_MAX_STEPS[key] = 300" in openvla_entry
    assert "timed_get_action" in openvla_entry
    assert "save_rollout_video = lambda" in openvla_entry
    assert 'text.startswith("Episode error:")' in openvla_entry
    assert "_validate_checkpoint" in openvla_entry
    assert "checkpoint_eval_ready" in openvla_entry
    assert "lora_merged_for_evaluation" in openvla_entry
    assert "model*.safetensors" in openvla_entry
    assert 'validate_runtime("openvla_oft"' in openvla_entry


def test_openvla_lifecycle_wrapper_materializes_then_cleans_by_default():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "tools/benchmark/run_m3_libero_evaluation.py").read_text()
    for token in (
        "m3_openvla_finalize_checkpoint.py",
        "m3_openvla_dematerialize_checkpoint.py",
        "PARC_M3_KEEP_MERGED_OPENVLA",
        "materialized_here",
        "finally:",
        "openvla_dematerialized_after_evaluation",
        "automatic_upload",
    ):
        assert token in wrapper
    assert "subprocess.run(finalize_command" in wrapper
    assert "subprocess.run(\n                cleanup_command" in wrapper


def test_lifecycle_wrapper_requires_pr66_tools_in_integrated_tree():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "tools/benchmark/run_m3_libero_evaluation.py").read_text()
    assert "requires #54 finalizer to be present" in wrapper
    assert "requires #54 dematerializer to be present" in wrapper
    contract = json.loads(
        (root / "experiments/plans/m3_libero_executor_v1.json").read_text()
    )
    assert contract["openvla_storage_lifecycle"]["requires_training_adapter_tools_from_pr_66"] is True
    assert contract["integration_order"][0].startswith("merge PR #66")


def test_instrumentation_scope_is_policy_inference_not_preprocessing():
    root = Path(__file__).resolve().parents[1]
    helper = (root / "tools/benchmark/m3_eval_instrumentation.py").read_text()
    assert "torch.cuda.synchronize" in helper
    assert "policy.select_action" in helper
    contract = json.loads(
        (root / "experiments/plans/m3_libero_executor_v1.json").read_text()
    )
    assert "policy inference call only" in contract["instrumentation"]["inference_latency_scope"]


def test_executor_contract_records_live_guard_storage_lifecycle_and_metric_set():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "experiments/plans/m3_libero_executor_v1.json").read_text()
    )
    assert contract["status"] == "IMPLEMENTED_PENDING_A100_SMOKE"
    assert contract["canonical_live_entry"] == "tools/benchmark/run_m3_libero_evaluation.py"
    assert contract["seed_set"] == list(SEEDS)
    assert contract["max_steps_per_episode"] == 300
    assert contract["evaluation_protocol"]["hard_reset"] is True
    assert contract["evaluation_protocol"]["smoke_episode_records_per_model"] == 80
    assert contract["evaluation_protocol"]["benchmark_episode_records_per_model"] == 800
    assert contract["runtime_gate"]["minimum_gpu_vram_mib"] == 38000
    openvla = contract["models"]["openvla_oft"]
    assert openvla["silent_upstream_episode_errors_are_fatal"] is True
    assert openvla["jit_merge_before_evaluation"] is True
    assert openvla["default_dematerialize_merged_weights_after_evaluation"] is True
    lifecycle = contract["openvla_storage_lifecycle"]
    assert lifecycle["remove_only_merged_model_weights_after_evaluation"] is True
    assert lifecycle["preserve_lora_adapter"] is True
    assert lifecycle["preserve_action_head"] is True
    assert lifecycle["preserve_proprio_projector"] is True
    assert lifecycle["preserve_d10_statistics"] is True
    assert contract["evaluation_protocol"]["automatic_full_evaluation"] is False
    assert contract["aggregation"]["required_metrics"] == [
        "simulator_success_rate",
        "steps_to_success",
        "episode_duration",
        "inference_latency",
        "peak_inference_vram",
        "train_wall_time",
        "peak_train_vram",
    ]
