import json
from pathlib import Path

import pytest

from tools.benchmark.m3_equal_wall_schedule import stream_descriptor
from tools.benchmark.m3_model_adapters import (
    SOURCE_REFS,
    build_adapter_command,
    build_training_command,
    default_runtimes,
)
from tools.benchmark.m3_scheduled_data import (
    D10_HASH,
    build_episode_offsets,
    references_to_lerobot_indices,
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
            "pi05": {
                "source_ref": SOURCE_REFS["pi05"],
                "micro_batch": 8,
                "gradient_accumulation": 4,
                "effective_batch_size": 32,
            },
            "smolvla": {
                "source_ref": SOURCE_REFS["smolvla"],
                "micro_batch": 16,
                "gradient_accumulation": 2,
                "effective_batch_size": 32,
            },
            "openvla_oft": {
                "source_ref": SOURCE_REFS["openvla_oft"],
                "micro_batch": 4,
                "gradient_accumulation": 8,
                "effective_batch_size": 32,
            },
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
        "seed": 20260906,
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": D10_HASH,
        "references": refs,
        "references_sha256": sha256_json(refs),
        "schedule_sha256": "sched",
        "sample_budget": len(refs),
    }


def test_reference_mapping_preserves_explicit_concatenation_order():
    offsets = build_episode_offsets([10, 11], [3, 2])
    assert offsets == {10: (0, 3), 11: (3, 2)}
    refs = _schedule_payload()["references"]
    assert references_to_relative_indices(
        refs, episode_ids=[10, 11], episode_lengths=[3, 2]
    ) == [1, 3, 2]


def test_live_lerobot_mapping_uses_absolute_to_relative_map_not_manifest_order():
    metadata = {
        10: {"dataset_from_index": 100, "length": 3},
        11: {"dataset_from_index": 50, "length": 2},
    }
    absolute_to_relative = {50: 0, 51: 1, 100: 2, 101: 3, 102: 4}
    refs = _schedule_payload()["references"]
    assert references_to_lerobot_indices(
        refs,
        episode_metadata=metadata,
        absolute_to_relative_idx=absolute_to_relative,
    ) == [3, 0, 4]


def test_live_lerobot_mapping_rejects_missing_absolute_frame():
    with pytest.raises(ValueError, match="absent from filtered"):
        references_to_lerobot_indices(
            [{"episode_id": 10, "timestep": 1}],
            episode_metadata={10: {"dataset_from_index": 100, "length": 2}},
            absolute_to_relative_idx={100: 0},
        )


def test_reference_mapping_rejects_episode_outside_pool():
    with pytest.raises(ValueError, match="outside D10 pool"):
        references_to_relative_indices(
            [{"episode_id": 99, "timestep": 0}],
            episode_ids=[10],
            episode_lengths=[1],
        )


def test_runtime_source_refs_are_frozen(tmp_path):
    runtimes = default_runtimes(tmp_path, tmp_path / "repo")
    assert runtimes["pi05"].source_ref == "v0.4.4"
    assert runtimes["smolvla"].source_ref == "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
    assert runtimes["openvla_oft"].source_ref == "e4287e94541f459edc4feabc4e181f537cd569a8"


def test_openvla_preflight_requires_streaming_contract(tmp_path):
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


def test_live_training_commands_use_real_entries(monkeypatch, tmp_path):
    monkeypatch.setenv("PARC_M3_EXECUTE", "1")
    runtimes = default_runtimes(tmp_path, tmp_path / "repo")
    pi_cmd = build_training_command(
        runtime=runtimes["pi05"],
        repo=tmp_path / "repo",
        batch_cfg={"micro_batch": 8, "gradient_accumulation": 4},
        schedule_path=tmp_path / "schedule.json",
        dataset_root=tmp_path / "data",
        manifest_path=tmp_path / "manifest.json",
        track="equal_wall",
        order="forward",
        mode="smoke",
        output_dir=tmp_path / "out/pi05",
        result_out=tmp_path / "result-pi05.json",
    )
    assert "m3_lerobot_train_entry_v2.py" in " ".join(pi_cmd)
    assert "--model pi05" in " ".join(pi_cmd)
    assert "--track equal_wall" in " ".join(pi_cmd)

    ov_cmd = build_training_command(
        runtime=runtimes["openvla_oft"],
        repo=tmp_path / "repo",
        batch_cfg={"micro_batch": 4, "gradient_accumulation": 8},
        schedule_path=tmp_path / "schedule.json",
        dataset_root=tmp_path / "data",
        manifest_path=tmp_path / "manifest.json",
        track="equal_data",
        order="reverse",
        mode="smoke",
        output_dir=tmp_path / "out/openvla",
        result_out=tmp_path / "result-openvla.json",
        streaming_contract=tmp_path / "69c.json",
    )
    text = " ".join(ov_cmd)
    assert "torch.distributed.run" in text
    assert "m3_openvla_train_entry.py" in text
    assert "--streaming-contract" in text


def test_equal_wall_stream_descriptor_is_deterministic_and_seed_scoped():
    a = stream_descriptor(seed=20260906)
    b = stream_descriptor(seed=20260906)
    c = stream_descriptor(seed=20260907)
    assert a == b
    assert a["materialized"] is False
    assert a["stream_sha256"] != c["stream_sha256"]


def test_train_entries_enforce_frozen_boundaries_and_no_rlds_fallback():
    root = Path(__file__).resolve().parents[1]
    lerobot_entry = (root / "tools/benchmark/m3_lerobot_train_entry_v2.py").read_text()
    openvla_entry = (root / "tools/benchmark/m3_openvla_train_entry.py").read_text()
    openvla_stream = (root / "tools/data/openvla_scheduled_stream.py").read_text()
    for token in (
        "PARC_M3_EXECUTE",
        "EQUAL_WALL_SEC = 1800.0",
        "EQUAL_DATA_BUDGET = 4800",
        "make_counter_stream_sampler",
        "checkpoint_save_excluded_from_train_wall_time",
    ):
        assert token in lerobot_entry
    for token in (
        "PARC_M3_EXECUTE",
        "EQUAL_WALL_SEC = 1800.0",
        "EQUAL_DATA_BUDGET = 4800",
        "make_counter_stream_iterable_dataset",
        '"full_rlds_materialization": False',
        '"modified_libero_rlds_used": False',
        '"image_aug_applied": True',
    ):
        assert token in openvla_entry
    assert "iter_references" in openvla_stream


def test_plan_contract_records_live_execution_and_no_random_sampler():
    contract = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments/plans/m3_model_adapter_contract_v1.json").read_text()
    )
    assert contract["status"] == "IMPLEMENTED_PENDING_A100_SMOKE"
    assert contract["seed_set"] == [20260906, 20260907]
    assert contract["sampling"]["native_random_sampler_allowed"] is False
    assert contract["sampling"]["equal_wall_does_not_cycle_4800_prefix"] is True
    assert contract["execution"]["forward"] == ["pi05", "smolvla", "openvla_oft"]
    assert contract["execution"]["reverse"] == ["openvla_oft", "smolvla", "pi05"]
    assert contract["openvla"]["full_rlds_materialization"] is False
