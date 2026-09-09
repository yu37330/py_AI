#!/usr/bin/env python3
"""Resolve frozen M3 sample schedules into framework-consumable indices.

This module is intentionally training-framework neutral.  It validates the
canonical schedule produced by ``m3_sampling_schedule.py`` and maps every
(episode_id, timestep) reference into the relative row index of a LeRobot
Dataset instantiated with the frozen D10 episode list in manifest order.

The same validated reference stream is also exposed unchanged for the
OpenVLA selected-pool streaming adapter.  No dataset copy is created.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_VARIANT = "V2_SQRT_BALANCED_RAW"
EQUAL_DATA_BUDGET = 4800
SCHEDULE_POLICY = "uniform_selected_frames_with_replacement_v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class SampleRef:
    sample_index: int
    episode_id: int
    timestep: int


def load_schedule(path: Path, *, require_equal_data: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN":
        raise ValueError("M3 sampling schedule must be FROZEN")
    if payload.get("policy") != SCHEDULE_POLICY:
        raise ValueError("M3 sampling policy mismatch")
    if payload.get("selected_dataset_variant") != D10_VARIANT:
        raise ValueError("M3 schedule dataset variant mismatch")
    if payload.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("M3 schedule D10 hash mismatch")
    refs = payload.get("references")
    if not isinstance(refs, list) or not refs:
        raise ValueError("M3 schedule references must be a non-empty list")
    normalized: list[dict[str, int]] = []
    for expected_i, raw in enumerate(refs):
        if not isinstance(raw, dict):
            raise ValueError("schedule reference must be an object")
        sample_index = int(raw.get("sample_index", -1))
        episode_id = int(raw.get("episode_id", -1))
        timestep = int(raw.get("timestep", -1))
        if sample_index != expected_i:
            raise ValueError(f"non-contiguous sample_index at {expected_i}: {sample_index}")
        if episode_id < 0 or timestep < 0:
            raise ValueError("negative episode/timestep in schedule")
        normalized.append({"sample_index": sample_index, "episode_id": episode_id, "timestep": timestep})
    if payload.get("references_sha256") != sha256_json(normalized):
        raise ValueError("schedule references SHA mismatch")
    if require_equal_data:
        if int(payload.get("sample_budget", -1)) != EQUAL_DATA_BUDGET or len(normalized) != EQUAL_DATA_BUDGET:
            raise ValueError("equal-data schedule must contain exactly 4800 references")
    out = dict(payload)
    out["references"] = normalized
    return out


def build_episode_offsets(episode_ids: Sequence[int], episode_lengths: Sequence[int]) -> dict[int, tuple[int, int]]:
    ids = [int(x) for x in episode_ids]
    lengths = [int(x) for x in episode_lengths]
    if len(ids) != len(lengths) or not ids:
        raise ValueError("episode IDs/lengths mismatch")
    if len(ids) != len(set(ids)):
        raise ValueError("episode IDs must be unique")
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for episode_id, length in zip(ids, lengths, strict=True):
        if length <= 0:
            raise ValueError(f"invalid episode length: {episode_id}={length}")
        offsets[episode_id] = (cursor, length)
        cursor += length
    return offsets


def references_to_relative_indices(
    references: Iterable[dict[str, int]],
    *,
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
) -> list[int]:
    offsets = build_episode_offsets(episode_ids, episode_lengths)
    indices: list[int] = []
    for ref in references:
        episode_id = int(ref["episode_id"])
        timestep = int(ref["timestep"])
        if episode_id not in offsets:
            raise ValueError(f"schedule episode outside D10 pool: {episode_id}")
        start, length = offsets[episode_id]
        if not 0 <= timestep < length:
            raise ValueError(f"schedule timestep outside episode: episode={episode_id} timestep={timestep} length={length}")
        indices.append(start + timestep)
    return indices


def iter_sample_refs(schedule: dict[str, Any]) -> Iterator[SampleRef]:
    for ref in schedule["references"]:
        yield SampleRef(
            sample_index=int(ref["sample_index"]),
            episode_id=int(ref["episode_id"]),
            timestep=int(ref["timestep"]),
        )


def make_fixed_index_sampler(indices: Sequence[int]):
    """Create a torch Sampler lazily so static tests do not require torch."""
    import torch

    frozen = tuple(int(x) for x in indices)
    if not frozen:
        raise ValueError("fixed sampler requires at least one index")

    class FixedIndexSampler(torch.utils.data.Sampler[int]):
        def __iter__(self):
            yield from frozen

        def __len__(self) -> int:
            return len(frozen)

    return FixedIndexSampler()
