import json
from pathlib import Path

import pytest

from tools.benchmark.run_m3_guarded import _completed_matches, _sequence
from tools.benchmark.workers.m3_worker_common import (
    SMOKE_OPTIMIZER_TARGET,
    SMOKE_SAMPLE_TARGET,
    build_runtime_spec,
)

ROOT = Path(__file__).resolve().parents[1]
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"


def _plan():
    return {
        "status": "READY_FOR_GUARDED_SMOKE_IMPLEMENTATION",
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "training_started": False,
        "automatic_benchmark_start": False,
        "equal_data_schedule": "/tmp/schedule.json",
        "orders": {
            "forward": ["pi05", "smolvla", "openvla_oft"],
            "reverse": ["openvla_oft", "smolvla", "pi05"],
        },
    }


def _run(model="pi05", order="forward", track="equal_data"):
    return {
        "model": model,
        "order": order,
        "track": track,
        "source_ref": "v0.4.4" if model == "pi05" else "source",
        "micro_batch": 8,
        "gradient_accumulation": 4,
        "effective_batch_size": 32,
        "schedule_sha256": "schedule-sha",
        "seed": 20260906,
        "result_path": f"/tmp/runs/{order}/{track}/{model}/result.json",
    }


def test_sequence_is_exact_forward_and_reverse():
    plan = _plan()
    assert _sequence(plan, "forward") == ["pi05", "smolvla", "openvla_oft"]
    assert _sequence(plan, "reverse") == ["openvla_oft", "smolvla", "pi05"]
    broken = _plan()
    broken["orders"]["reverse"] = ["pi05", "smolvla", "openvla_oft"]
    with pytest.raises(ValueError, match="sequence drift"):
        _sequence(broken, "reverse")


def test_smoke_runtime_is_only_64_samples_and_two_optimizer_updates(monkeypatch, tmp_path):
    monkeypatch.setenv("PARC_M3_EXECUTE", "1")
    plan = _plan()
    run = _run()
    runtime = build_runtime_spec(
        plan=plan,
        run=run,
        mode="smoke",
        manifest_path=tmp_path / "manifest.json",
        dataset_root=tmp_path / "dataset",
        evidence_path=tmp_path / "evidence.json",
        result_path=tmp_path / "result.json",
    )
    assert runtime["sample_target"] == SMOKE_SAMPLE_TARGET == 64
    assert runtime["optimizer_target"] == SMOKE_OPTIMIZER_TARGET == 2
    assert runtime["train_loop_sec"] is None
    assert runtime["effective_batch_size"] == 32


def test_benchmark_equal_data_and_wall_targets_are_immutable(monkeypatch, tmp_path):
    monkeypatch.setenv("PARC_M3_EXECUTE", "1")
    plan = _plan()
    equal_data = build_runtime_spec(
        plan=plan,
        run=_run(track="equal_data"),
        mode="benchmark",
        manifest_path=tmp_path / "manifest.json",
        dataset_root=tmp_path / "dataset",
        evidence_path=tmp_path / "data-evidence.json",
        result_path=tmp_path / "data-result.json",
    )
    assert equal_data["sample_target"] == 4800
    assert equal_data["optimizer_target"] == 150
    assert equal_data["train_loop_sec"] is None

    equal_wall = build_runtime_spec(
        plan=plan,
        run=_run(track="equal_wall"),
        mode="benchmark",
        manifest_path=tmp_path / "manifest.json",
        dataset_root=tmp_path / "dataset",
        evidence_path=tmp_path / "wall-evidence.json",
        result_path=tmp_path / "wall-result.json",
    )
    assert equal_wall["train_loop_sec"] == 1800


def test_worker_runtime_is_blocked_without_explicit_execution_guard(monkeypatch, tmp_path):
    monkeypatch.delenv("PARC_M3_EXECUTE", raising=False)
    with pytest.raises(RuntimeError, match="training not started"):
        build_runtime_spec(
            plan=_plan(),
            run=_run(),
            mode="smoke",
            manifest_path=tmp_path / "manifest.json",
            dataset_root=tmp_path / "dataset",
            evidence_path=tmp_path / "evidence.json",
            result_path=tmp_path / "result.json",
        )


def test_resume_skip_requires_immutable_pass_identity(tmp_path):
    run = _run()
    result_path = tmp_path / "training_result.json"
    result = {
        "status": "PASS",
        "mode": "smoke",
        "model": "pi05",
        "order": "forward",
        "track": "equal_data",
        "source_ref": "v0.4.4",
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_schedule_sha256": "schedule-sha",
        "micro_batch": 8,
        "gradient_accumulation": 4,
        "effective_batch_size": 32,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert _completed_matches(
        result_path,
        run=run,
        mode="smoke",
        order="forward",
        track="equal_data",
        model="pi05",
    )
    result["sampling_schedule_sha256"] = "wrong"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not _completed_matches(
        result_path,
        run=run,
        mode="smoke",
        order="forward",
        track="equal_data",
        model="pi05",
    )


def test_orchestrator_never_auto_starts_other_order_or_track():
    source = (ROOT / "tools/benchmark/run_m3_guarded.py").read_text(encoding="utf-8")
    assert '"automatic_reverse_start": False' in source
    assert '"automatic_other_track_start": False' in source
    assert 'p.add_argument("--order", choices=("forward", "reverse"), required=True)' in source
    assert 'p.add_argument("--mode", choices=("smoke", "benchmark"), required=True)' in source
    assert "PARC_M3_EXECUTE" in source
