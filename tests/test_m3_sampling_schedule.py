import json
from pathlib import Path

from tools.benchmark.m3_sampling_schedule import (
    D10_EPISODE_IDS_SHA256,
    D10_FRAME_COUNT,
    EQUAL_DATA_SAMPLE_BUDGET,
    POLICY,
    build_equal_data_schedule,
    iter_references,
    resolve_global_position,
    sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixture_pool():
    # Preserve the exact D10 cardinality/frame count while keeping the fixture simple.
    ids = list(range(10758))
    lengths = [D10_FRAME_COUNT - 10757] + [1] * 10757
    assert sum(lengths) == D10_FRAME_COUNT
    return ids, lengths


def test_schedule_contract_is_frozen_and_model_independent():
    plan = json.loads(
        (ROOT / "experiments/plans/m3_sampling_schedule_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["status"] == "FROZEN"
    assert plan["policy"]["name"] == POLICY
    assert plan["equal_data"]["sample_budget"] == 4800
    assert plan["selected_episode_ids_sha256"] == D10_EPISODE_IDS_SHA256
    assert plan["forward_reverse_rule"]


def test_same_seed_builds_byte_stable_equal_data_schedule():
    ids, lengths = _fixture_pool()
    a = build_equal_data_schedule(
        episode_ids=ids, episode_lengths=lengths, seed=20260906
    )
    b = build_equal_data_schedule(
        episode_ids=ids, episode_lengths=lengths, seed=20260906
    )
    assert a == b
    assert sha256_json(a) == sha256_json(b)
    assert len(a["references"]) == EQUAL_DATA_SAMPLE_BUDGET
    assert a["sample_budget"] == EQUAL_DATA_SAMPLE_BUDGET


def test_different_seed_changes_schedule_but_not_contract():
    ids, lengths = _fixture_pool()
    a = build_equal_data_schedule(
        episode_ids=ids, episode_lengths=lengths, seed=20260906
    )
    b = build_equal_data_schedule(
        episode_ids=ids, episode_lengths=lengths, seed=20260907
    )
    assert a["references_sha256"] != b["references_sha256"]
    assert a["selected_episode_ids_sha256"] == b["selected_episode_ids_sha256"]
    assert a["sample_budget"] == b["sample_budget"] == 4800


def test_every_reference_stays_inside_selected_pool_and_episode_length():
    ids, lengths = _fixture_pool()
    schedule = build_equal_data_schedule(
        episode_ids=ids, episode_lengths=lengths, seed=20260906
    )
    length_by_id = dict(zip(ids, lengths, strict=True))
    for index, ref in enumerate(schedule["references"]):
        assert ref["sample_index"] == index
        assert ref["episode_id"] in length_by_id
        assert 0 <= ref["timestep"] < length_by_id[ref["episode_id"]]


def test_global_position_mapping_handles_episode_boundary():
    ids, lengths = _fixture_pool()
    first_len = lengths[0]
    assert resolve_global_position(ids, lengths, first_len - 1) == {
        "episode_id": 0,
        "timestep": first_len - 1,
    }
    assert resolve_global_position(ids, lengths, first_len) == {
        "episode_id": 1,
        "timestep": 0,
    }


def test_equal_wall_stream_is_a_repeatable_prefix_stream():
    ids, lengths = _fixture_pool()
    stream_a = iter_references(
        episode_ids=ids, episode_lengths=lengths, seed=20260906
    )
    stream_b = iter_references(
        episode_ids=ids, episode_lengths=lengths, seed=20260906
    )
    prefix_a = [next(stream_a) for _ in range(64)]
    prefix_b = [next(stream_b) for _ in range(64)]
    assert prefix_a == prefix_b
