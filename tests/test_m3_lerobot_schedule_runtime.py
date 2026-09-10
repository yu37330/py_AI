from types import SimpleNamespace

import pytest

from tools.benchmark.m3_lerobot_schedule_runtime import (
    CanonicalReferenceSampler,
    M3ScheduleEvidence,
    episode_start_lookup,
    extract_raw_batch_references,
    reference_to_dataset_index,
)


class FakeEpisodes(dict):
    pass


class FakeDataset:
    def __init__(self):
        self.meta = SimpleNamespace(
            episodes=FakeEpisodes(
                episode_index=[10, 20],
                dataset_from_index=[100, 200],
                dataset_to_index=[105, 204],
            )
        )
        self._absolute_to_relative_idx = {
            100: 0,
            101: 1,
            102: 2,
            103: 3,
            104: 4,
            200: 5,
            201: 6,
            202: 7,
            203: 8,
        }


def test_episode_ranges_and_reference_mapping_support_filtered_lerobot_subset():
    ds = FakeDataset()
    assert episode_start_lookup(ds) == {10: (100, 5), 20: (200, 4)}
    assert reference_to_dataset_index(ds, {"episode_id": 10, "timestep": 2}) == 2
    assert reference_to_dataset_index(ds, {"episode_id": 20, "timestep": 3}) == 8
    with pytest.raises(IndexError, match="timestep out of range"):
        reference_to_dataset_index(ds, {"episode_id": 20, "timestep": 4})


def test_canonical_sampler_yields_reference_order_and_exact_finite_length():
    refs = [
        {"episode_id": 20, "timestep": 1},
        {"episode_id": 10, "timestep": 4},
        {"episode_id": 20, "timestep": 0},
    ]
    sampler = CanonicalReferenceSampler(FakeDataset(), refs, finite_length=3)
    assert list(sampler) == [6, 4, 5]
    assert len(sampler) == 3


def test_raw_batch_reference_extraction_handles_tensorlike_lists():
    assert extract_raw_batch_references(
        {"episode_index": [10, 20], "frame_index": [3, 1]}
    ) == [(10, 3), (20, 1)]


def test_equal_data_evidence_checks_actual_order_and_exact_4800_samples():
    refs = iter(
        {"episode_id": i % 2, "timestep": i % 7}
        for i in range(4800)
    )
    now = [10.0]
    evidence = M3ScheduleEvidence(
        track="equal_data",
        micro_batch=8,
        gradient_accumulation=4,
        expected_references=refs,
        clock=lambda: now[0],
    )
    evidence.before_first_batch_fetch()
    sample_index = 0
    for update in range(150):
        for _ in range(4):
            episodes = []
            frames = []
            for _ in range(8):
                episodes.append(sample_index % 2)
                frames.append(sample_index % 7)
                sample_index += 1
            evidence.observe_raw_batch(
                {"episode_index": episodes, "frame_index": frames}
            )
            assert evidence.optimizer_boundary(sync_gradients=False) is False
        should_stop = evidence.optimizer_boundary(sync_gradients=True)
        assert should_stop is (update == 149)
    result = evidence.finalize()
    assert result["consumed_samples"] == 4800
    assert result["optimizer_updates"] == 150
    assert result["canonical_sample_order_verified"] is True


def test_equal_data_evidence_fails_immediately_on_actual_sample_mismatch():
    evidence = M3ScheduleEvidence(
        track="equal_data",
        micro_batch=1,
        gradient_accumulation=32,
        expected_references=iter([{"episode_id": 1, "timestep": 2}]),
    )
    evidence.before_first_batch_fetch()
    with pytest.raises(RuntimeError, match="canonical sample mismatch"):
        evidence.observe_raw_batch({"episode_index": [1], "frame_index": [3]})


def test_equal_wall_stops_only_on_optimizer_boundary_at_or_after_target():
    now = [100.0]
    refs = iter({"episode_id": 1, "timestep": i} for i in range(1000))
    evidence = M3ScheduleEvidence(
        track="equal_wall",
        micro_batch=8,
        gradient_accumulation=4,
        expected_references=refs,
        wall_target_sec=1800.0,
        clock=lambda: now[0],
    )
    evidence.before_first_batch_fetch()
    now[0] = 1899.9
    assert evidence.optimizer_boundary(sync_gradients=True) is False
    now[0] = 1900.1
    assert evidence.optimizer_boundary(sync_gradients=False) is False
    assert evidence.optimizer_boundary(sync_gradients=True) is True
    result = evidence.finalize()
    assert result["train_wall_time_sec"] >= 1800.0
    assert result["optimizer_updates"] == 2
