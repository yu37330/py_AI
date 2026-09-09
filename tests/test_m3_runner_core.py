import json
from pathlib import Path

import pytest

from tools.benchmark.m3_runner_core import (
    D10_HASH,
    EFFECTIVE_BATCH,
    EQUAL_DATA_BUDGET,
    EQUAL_DATA_OPTIMIZER_UPDATES,
    EQUAL_WALL_SEC,
    EqualDataCounter,
    EqualWallTimer,
    build_execution_plan,
    build_run_matrix,
    require_execution_guard,
    result_path,
    validate_batch_probe_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def _batch_summary():
    return {
        "schema_version": 1,
        "stage": "M3_A100_batch_probe_controller",
        "status": "READY_FOR_M3_RUNNER_IMPLEMENTATION",
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "effective_batch_size": 32,
        "equal_data_protocol": {
            "sample_budget": 4800,
            "optimizer_updates": 150,
            "samples_per_optimizer_update": 32,
        },
        "models": {
            "pi05": {
                "source_ref": "v0.4.4",
                "micro_batch": 8,
                "gradient_accumulation": 4,
                "effective_batch_size": 32,
            },
            "smolvla": {
                "source_ref": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
                "micro_batch": 16,
                "gradient_accumulation": 2,
                "effective_batch_size": 32,
            },
            "openvla_oft": {
                "source_ref": "e4287e94541f459edc4feabc4e181f537cd569a8",
                "micro_batch": 4,
                "gradient_accumulation": 8,
                "effective_batch_size": 32,
            },
        },
        "probe_only": True,
        "benchmark_training_started": False,
        "checkpoint_promoted": False,
    }


def test_runner_contract_freezes_forward_reverse_and_guard():
    plan = json.loads(
        (ROOT / "experiments/plans/m3_execution_runner_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["orders"]["forward"] == ["pi05", "smolvla", "openvla_oft"]
    assert plan["orders"]["reverse"] == ["openvla_oft", "smolvla", "pi05"]
    assert plan["tracks"]["equal_data"]["sample_budget"] == 4800
    assert plan["tracks"]["equal_wall"]["train_loop_sec"] == 1800
    assert plan["execution_guard"]["environment_variable"] == "PARC_M3_EXECUTE"
    assert plan["safety"]["automatic_benchmark_start"] is False


def test_equal_data_counter_completes_exactly_without_overshoot():
    counter = EqualDataCounter()
    for _ in range(EQUAL_DATA_OPTIMIZER_UPDATES):
        # Simulate four micro-batches of 8 examples per optimizer update.
        for _ in range(4):
            counter.after_batch(8)
        counter.on_optimizer_step()
    assert counter.consumed_samples == EQUAL_DATA_BUDGET
    assert counter.optimizer_updates == EQUAL_DATA_OPTIMIZER_UPDATES
    assert counter.complete is True
    counter.assert_complete()
    with pytest.raises(RuntimeError, match="overshoot blocked"):
        counter.after_batch(1)


def test_equal_wall_timer_starts_only_at_first_batch_fetch_and_stops_on_boundary():
    now = [100.0]
    timer = EqualWallTimer(clock=lambda: now[0])
    # Arbitrary setup time before the hook must not count.
    now[0] = 500.0
    assert timer.elapsed() == 0.0
    timer.before_first_batch_fetch()
    now[0] = 500.0 + EQUAL_WALL_SEC - 0.1
    assert timer.optimizer_boundary_should_stop() is False
    now[0] = 500.0 + EQUAL_WALL_SEC + 0.25
    assert timer.optimizer_boundary_should_stop() is True
    timer.assert_stopped_at_or_after_target()
    assert timer.elapsed() >= EQUAL_WALL_SEC


def test_execution_guard_blocks_smoke_and_benchmark_without_explicit_opt_in():
    require_execution_guard("preflight", env={})
    for mode in ("smoke", "benchmark"):
        with pytest.raises(RuntimeError, match="training not started"):
            require_execution_guard(mode, env={})
        require_execution_guard(mode, env={"PARC_M3_EXECUTE": "1"})


def test_batch_probe_summary_is_late_bound_and_validated():
    normalized = validate_batch_probe_summary(_batch_summary())
    assert normalized["pi05"]["micro_batch"] == 8
    assert normalized["smolvla"]["gradient_accumulation"] == 2
    assert normalized["openvla_oft"]["effective_batch_size"] == EFFECTIVE_BATCH


def test_batch_probe_summary_rejects_effective_batch_drift():
    summary = _batch_summary()
    summary["models"]["pi05"]["gradient_accumulation"] = 3
    with pytest.raises(ValueError, match="invalid batch config"):
        validate_batch_probe_summary(summary)


def test_run_matrix_contains_both_orders_tracks_and_exact_reverse_sequence():
    rows = build_run_matrix()
    assert len(rows) == 12
    for track in ("equal_data", "equal_wall"):
        forward = [
            row["model"]
            for row in rows
            if row["order"] == "forward" and row["track"] == track
        ]
        reverse = [
            row["model"]
            for row in rows
            if row["order"] == "reverse" and row["track"] == track
        ]
        assert forward == ["pi05", "smolvla", "openvla_oft"]
        assert reverse == list(reversed(forward))


def test_preflight_plan_never_claims_training_started(monkeypatch):
    monkeypatch.delenv("PARC_M3_EXECUTE", raising=False)
    plan = build_execution_plan(
        batch_summary=_batch_summary(),
        equal_data_schedule_sha256="equal-data-schedule",
        equal_wall_schedule_sha256="equal-wall-stream-contract",
        mode="preflight",
    )
    assert plan["status"] == "READY_FOR_SMOKE"
    assert plan["benchmark_training_started"] is False
    assert plan["automatic_full_run"] is False
    assert len(plan["run_matrix"]) == 12


def test_result_paths_separate_order_track_and_model(tmp_path):
    assert result_path(
        tmp_path, order="reverse", track="equal_wall", model="openvla_oft"
    ) == tmp_path / "runs/reverse/equal_wall/openvla_oft/result.json"
