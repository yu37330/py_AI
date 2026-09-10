#!/usr/bin/env python3
"""Infinite deterministic M3 schedule helpers for the equal-wall track.

The frozen equal-data JSON contains the first 4,800 counter-generated samples.
Equal-wall must *continue the same counter stream* instead of cycling that file.
This module verifies the materialized prefix against the canonical sampler and
then exposes an effectively-unbounded LeRobot sampler with no reference-list
materialization.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator, Mapping, Sequence

from tools.benchmark.m3_sampling_schedule import iter_references
from tools.benchmark.m3_scheduled_data import D10_HASH, SCHEDULE_POLICY, sha256_json

SEEDS = (20260906, 20260907)
DEFAULT_MAX_STREAM_SAMPLES = 2_147_483_648


def verify_materialized_prefix(
    schedule: dict[str, Any], *, episode_ids: Sequence[int], episode_lengths: Sequence[int]
) -> int:
    """Prove that the schedule JSON is the canonical prefix for its declared seed."""
    seed = int(schedule.get("seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"unexpected M3 sampling seed: {seed}")
    if schedule.get("policy") != SCHEDULE_POLICY:
        raise ValueError("M3 sampling policy mismatch")
    if schedule.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("M3 schedule D10 hash mismatch")
    references = schedule.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("materialized schedule prefix is required")
    canonical = iter_references(
        episode_ids=episode_ids,
        episode_lengths=episode_lengths,
        seed=seed,
        start_index=0,
    )
    for expected, actual in zip(canonical, references, strict=False):
        normalized = {
            "sample_index": int(actual["sample_index"]),
            "episode_id": int(actual["episode_id"]),
            "timestep": int(actual["timestep"]),
        }
        if expected != normalized:
            raise ValueError(
                "materialized M3 schedule is not the canonical counter-stream prefix: "
                f"sample_index={normalized['sample_index']} expected={expected} actual={normalized}"
            )
    return seed


def stream_descriptor(
    *, seed: int, episode_ids_sha256: str = D10_HASH, start_index: int = 0
) -> dict[str, Any]:
    if int(seed) not in SEEDS:
        raise ValueError(f"unexpected M3 sampling seed: {seed}")
    if episode_ids_sha256 != D10_HASH:
        raise ValueError("equal-wall stream D10 hash mismatch")
    payload: dict[str, Any] = {
        "policy": SCHEDULE_POLICY,
        "seed": int(seed),
        "selected_episode_ids_sha256": D10_HASH,
        "start_index": int(start_index),
        "materialized": False,
        "stream_rule": "counter_based_sha256_until_safe_optimizer_boundary",
    }
    payload["stream_sha256"] = sha256_json(payload)
    return payload


def iter_lerobot_relative_indices(
    *,
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    dataset_from_indices: Sequence[int],
    absolute_to_relative_idx: Mapping[int, int],
    seed: int,
    start_index: int = 0,
) -> Iterator[int]:
    ids = [int(x) for x in episode_ids]
    lengths = [int(x) for x in episode_lengths]
    starts = [int(x) for x in dataset_from_indices]
    if not ids or len(ids) != len(lengths) or len(ids) != len(starts):
        raise ValueError("episode IDs/lengths/dataset starts must have equal non-zero length")
    if not absolute_to_relative_idx:
        raise ValueError("LeRobot absolute-to-relative map is required")
    episode_to_meta = {
        episode_id: (length, start)
        for episode_id, length, start in zip(ids, lengths, starts, strict=True)
    }
    refs = iter_references(
        episode_ids=ids,
        episode_lengths=lengths,
        seed=int(seed),
        start_index=int(start_index),
    )
    for ref in refs:
        episode_id = int(ref["episode_id"])
        timestep = int(ref["timestep"])
        length, start = episode_to_meta[episode_id]
        if not 0 <= timestep < length:
            raise RuntimeError("canonical sampler emitted timestep outside episode")
        absolute = start + timestep
        if absolute not in absolute_to_relative_idx:
            raise ValueError(
                "canonical equal-wall frame missing from filtered dataset: "
                f"episode={episode_id} timestep={timestep} absolute={absolute}"
            )
        yield int(absolute_to_relative_idx[absolute])


def make_counter_stream_sampler(
    *,
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    dataset_from_indices: Sequence[int],
    absolute_to_relative_idx: Mapping[int, int],
    seed: int,
    max_samples: int = DEFAULT_MAX_STREAM_SAMPLES,
):
    """Create an effectively-unbounded torch sampler without materializing refs."""
    import torch

    if int(max_samples) <= 0:
        raise ValueError("max_samples must be positive")

    class CounterStreamSampler(torch.utils.data.Sampler[int]):
        def __iter__(self):
            stream = iter_lerobot_relative_indices(
                episode_ids=episode_ids,
                episode_lengths=episode_lengths,
                dataset_from_indices=dataset_from_indices,
                absolute_to_relative_idx=absolute_to_relative_idx,
                seed=int(seed),
            )
            for _, index in zip(range(int(max_samples)), stream, strict=False):
                yield index

        def __len__(self) -> int:
            return int(max_samples)

    return CounterStreamSampler()
