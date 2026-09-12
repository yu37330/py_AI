import json
from pathlib import Path

import pytest

from tools.benchmark.m3_runner_core import D10_HASH
from tools.benchmark.m3_sampling_schedule import POLICY, sha256_json
from tools.benchmark.m3_training_adapters import (
    ADAPTERS,
    build_adapter_plan,
    build_equal_wall_stream_descriptor,
    load_equal_data_schedule,
)


def _batch_summary() -> dict:
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
        "benchmark_training_started": False,
    }


def _schedule() -> dict:
    refs = [
        {"sample_index": i, "episode_id": i % 10758, "timestep": i % 11}
        for i in range(4800)
    ]
    payload = {
        "schema_version": 1,
        "stage": "M3_equal_data_sampling_schedule",
        "status": "FROZEN",
        "policy": POLICY,
        "seed": 20260906,
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "selected_frame_count": 1620614,
        "sample_budget": 4800,
        "references": refs,
        "references_sha256": sha256_json(refs),
    }
    payload["schedule_sha256"] = sha256_json(payload)
    return payload


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_adapter_definitions_pin_all_three_frameworks_and_workers():
    assert set(ADAPTERS) == {"pi05", "smolvla", "openvla_oft"}
    assert ADAPTERS["pi05"].source_ref == "v0.4.4"
    assert ADAPTERS["smolvla"].source_ref.startswith("3f2c29ef")
    assert ADAPTERS["openvla_oft"].source_ref.startswith("e4287e94")
    assert ADAPTERS["pi05"].worker.endswith("run_m3_pi05.py")
    assert ADAPTERS["smolvla"].worker.endswith("run_m3_smolvla.py")
    assert ADAPTERS["openvla_oft"].worker.endswith("run_m3_openvla_oft.py")
    assert "never materialize full RLDS" in ADAPTERS["openvla_oft"].notes


def test_equal_data_schedule_revalidates_hash_and_exact_budget(tmp_path):
    path = _write_json(tmp_path / "schedule.json", _schedule())
    loaded = load_equal_data_schedule(path)
    assert loaded["sample_budget"] == 4800
    assert len(loaded["references"]) == 4800

    bad = _schedule()
    bad["references"][0]["timestep"] += 1
    bad_path = _write_json(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError, match="reference hash mismatch"):
        load_equal_data_schedule(bad_path)


def test_equal_wall_descriptor_is_same_counter_stream_and_1800_seconds():
    a = build_equal_wall_stream_descriptor(seed=20260906)
    b = build_equal_wall_stream_descriptor(seed=20260906)
    assert a == b
    assert a["policy"] == POLICY
    assert a["start_sample_index"] == 0
    assert a["train_loop_sec"] == 1800.0
    assert a["materialized"] is False


def test_adapter_plan_binds_72d_batches_to_forward_and_reverse_without_training(tmp_path):
    batch = _write_json(tmp_path / "summary.json", _batch_summary())
    schedule = _write_json(tmp_path / "schedule.json", _schedule())
    results = tmp_path / "results"
    plan = build_adapter_plan(
        batch_summary_path=batch,
        equal_data_schedule_path=schedule,
        results_root=results,
        seed=20260906,
    )

    assert plan["status"] == "READY_FOR_GUARDED_SMOKE_IMPLEMENTATION"
    assert plan["training_started"] is False
    assert plan["automatic_benchmark_start"] is False
    assert plan["run_count"] == 12
    assert plan["orders"]["forward"] == ["pi05", "smolvla", "openvla_oft"]
    assert plan["orders"]["reverse"] == ["openvla_oft", "smolvla", "pi05"]

    for track in ("equal_data", "equal_wall"):
        forward = [
            row for row in plan["runs"] if row["order"] == "forward" and row["track"] == track
        ]
        reverse = [
            row for row in plan["runs"] if row["order"] == "reverse" and row["track"] == track
        ]
        assert [row["model"] for row in reverse] == list(
            reversed([row["model"] for row in forward])
        )
        assert len({row["schedule_sha256"] for row in forward + reverse}) == 1

    pi05 = next(
        row for row in plan["runs"]
        if row["order"] == "forward" and row["track"] == "equal_data" and row["model"] == "pi05"
    )
    assert pi05["micro_batch"] == 8
    assert pi05["gradient_accumulation"] == 4
    assert pi05["effective_batch_size"] == 32
    assert pi05["sample_budget"] == 4800
    assert pi05["execution_guard"] == "PARC_M3_EXECUTE=1"
    assert pi05["result_path"].endswith("runs/forward/equal_data/pi05/result.json")


def test_adapter_plan_rejects_seed_mismatch(tmp_path):
    batch = _write_json(tmp_path / "summary.json", _batch_summary())
    schedule = _write_json(tmp_path / "schedule.json", _schedule())
    with pytest.raises(ValueError, match="schedule seed"):
        build_adapter_plan(
            batch_summary_path=batch,
            equal_data_schedule_path=schedule,
            results_root=tmp_path / "results",
            seed=20260907,
        )
