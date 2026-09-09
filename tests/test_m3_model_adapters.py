import json
from pathlib import Path

import pytest

from tools.benchmark.m3_model_adapters import (
    SOURCE_REFS,
    build_adapter_command,
    build_run_specs,
    default_runtimes,
)
from tools.benchmark.m3_scheduled_data import (
    D10_HASH,
    build_episode_offsets,
    references_to_relative_indices,
    sha256_json,
)


def _batch_summary():
    return {
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
            "pi05": {"source_ref": SOURCE_REFS["pi05"], "micro_batch": 8, "gradient_accumulation": 4, "effective_batch_size": 32},
            "smolvla": {"source_ref": SOURCE_REFS["smolvla"], "micro_batch": 16, "gradient_accumulation": 2, "effective_batch_size": 32},
            "openvla_oft": {"source_ref": SOURCE_REFS["openvla_oft"], "micro_batch": 4, "gradient_accumulation": 8, "effective_batch_size": 32},
        },
        "benchmark_training_started": False,
    }


def _schedule_payload():
    refs = [
        {"sample_index": 0, "episode_id": 10, "timestep": 1},
        {"sample_index": 1, "episode_id": 11, "timestep": 0},
        {"sample_index": 2, "episode_id": 10, "timestep": 2},
    ]
    return {
        "status": "FROZEN",
        "policy": "uniform_selected_frames_with_replacement_v1",
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "references": refs,
        "references_sha256": sha256_json(refs),
        "schedule_sha256": "sched",
        "sample_budget": len(refs),
    }


def test_reference_mapping_preserves_schedule_order():
    offsets = build_episode_offsets([10, 11], [3, 2])
    assert offsets == {10: (0, 3), 11: (3, 2)}
    refs = _schedule_payload()["references"]
    assert references_to_relative_indices(refs, episode_ids=[10, 11], episode_lengths=[3, 2]) == [1, 3, 2]


def test_reference_mapping_rejects_episode_outside_pool():
    with pytest.raises(ValueError, match="outside D10 pool"):
        references_to_relative_indices(
            [{"episode_id": 99, "timestep": 0}], episode_ids=[10], episode_lengths=[1]
        )


def test_runtime_source_refs_are_frozen(tmp_path):
    runtimes = default_runtimes(tmp_path, tmp_path / "repo")
    assert runtimes["pi05"].source_ref == "v0.4.4"
    assert runtimes["smolvla"].source_ref == "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
    assert runtimes["openvla_oft"].source_ref == "e4287e94541f459edc4feabc4e181f537cd569a8"


def test_openvla_command_requires_streaming_contract(tmp_path):
    rt = default_runtimes(tmp_path, tmp_path / "repo")["openvla_oft"]
    with pytest.raises(ValueError, match="streaming contract"):
        build_adapter_command(
            runtime=rt,
            batch_cfg={"micro_batch": 4, "gradient_accumulation": 8},
            schedule_path=tmp_path / "s.json",
            batch_summary_path=tmp_path / "b.json",
            dataset_root=tmp_path / "data",
            manifest_path=tmp_path / "m.json",
            track="equal_data",
            order="forward",
            mode="preflight",
            out=tmp_path / "o.json",
            streaming_contract=None,
        )


def test_plan_contract_disallows_native_random_sampler():
    contract = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments/plans/m3_model_adapter_contract_v1.json").read_text()
    )
    assert contract["sampling"]["native_random_sampler_allowed"] is False
    assert contract["execution"]["forward"] == ["pi05", "smolvla", "openvla_oft"]
    assert contract["execution"]["reverse"] == ["openvla_oft", "smolvla", "pi05"]
    assert contract["openvla"]["full_rlds_materialization"] is False
