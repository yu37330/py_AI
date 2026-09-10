#!/usr/bin/env python3
"""Canonical LeRobot sample-order runtime for PARC2026 M3.

The M3 fairness contract is stronger than selecting the same episode pool: each
framework must consume the same ordered `(episode_id, timestep)` references for
a given seed. This module maps those references into a LeRobot map-style
dataset and validates the raw batches that actually leave the DataLoader.

It contains no model code and starts no training.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from tools.benchmark.m3_runner_core import (
    EFFECTIVE_BATCH,
    EQUAL_DATA_BUDGET,
    EQUAL_DATA_OPTIMIZER_UPDATES,
    EQUAL_WALL_SEC,
)


def _column_values(episodes: Any, key: str) -> list[int]:
    try:
        values = episodes[key]
    except Exception as exc:  # noqa: BLE001
        raise KeyError(f"episode metadata missing column {key}") from exc
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x.item() if hasattr(x, "item") else x) for x in values]


def episode_start_lookup(dataset: Any) -> dict[int, tuple[int, int]]:
    """Return episode_id -> (absolute_dataset_start, length)."""
    episodes = dataset.meta.episodes
    ids = _column_values(episodes, "episode_index")
    starts = _column_values(episodes, "dataset_from_index")
    stops = _column_values(episodes, "dataset_to_index")
    if not (len(ids) == len(starts) == len(stops)):
        raise ValueError("episode metadata column lengths differ")
    out: dict[int, tuple[int, int]] = {}
    for episode_id, start, stop in zip(ids, starts, stops, strict=True):
        if stop <= start:
            raise ValueError(f"invalid episode range {episode_id}: {start}:{stop}")
        out[episode_id] = (start, stop - start)
    return out


def _absolute_to_relative_map(dataset: Any) -> Any:
    mapping = getattr(dataset, "absolute_to_relative_idx", None)
    if mapping is None:
        mapping = getattr(dataset, "_absolute_to_relative_idx", None)
    return mapping


def reference_to_dataset_index(
    dataset: Any,
    ref: dict[str, Any],
    *,
    starts: dict[int, tuple[int, int]] | None = None,
) -> int:
    episode_id = int(ref["episode_id"])
    timestep = int(ref["timestep"])
    starts = episode_start_lookup(dataset) if starts is None else starts
    if episode_id not in starts:
        raise KeyError(f"scheduled episode {episode_id} is not present in dataset metadata")
    absolute_start, length = starts[episode_id]
    if not 0 <= timestep < length:
        raise IndexError(
            f"scheduled timestep out of range: episode={episode_id} timestep={timestep} length={length}"
        )
    absolute_index = absolute_start + timestep
    mapping = _absolute_to_relative_map(dataset)
    if mapping is None:
        return absolute_index
    try:
        return int(mapping[absolute_index])
    except (KeyError, IndexError) as exc:
        raise KeyError(
            f"scheduled absolute frame {absolute_index} is not loaded in the LeRobot subset"
        ) from exc


class CanonicalReferenceSampler:
    """PyTorch-compatible sampler yielding dataset indices in canonical order."""

    def __init__(
        self,
        dataset: Any,
        references: Iterable[dict[str, Any]],
        *,
        finite_length: int | None,
    ) -> None:
        self.dataset = dataset
        self.references = references
        self.finite_length = finite_length
        self._starts = episode_start_lookup(dataset)

    def __iter__(self) -> Iterator[int]:
        count = 0
        for ref in self.references:
            if self.finite_length is not None and count >= self.finite_length:
                return
            yield reference_to_dataset_index(self.dataset, ref, starts=self._starts)
            count += 1
        if self.finite_length is not None and count != self.finite_length:
            raise RuntimeError(
                f"canonical sampler ended early: {count}/{self.finite_length} references"
            )

    def __len__(self) -> int:
        if self.finite_length is None:
            raise TypeError("infinite equal-wall canonical sampler has no length")
        return int(self.finite_length)


def _as_int_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    out: list[int] = []
    for x in value:
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        if hasattr(x, "item"):
            x = x.item()
        out.append(int(x))
    return out


def extract_raw_batch_references(batch: dict[str, Any]) -> list[tuple[int, int]]:
    """Extract actual episode/frame identities before policy preprocessing."""
    if "episode_index" not in batch or "frame_index" not in batch:
        raise KeyError("raw LeRobot batch must retain episode_index and frame_index for M3 evidence")
    episodes = _as_int_list(batch["episode_index"])
    frames = _as_int_list(batch["frame_index"])
    if len(episodes) != len(frames):
        raise ValueError("raw batch episode/frame lengths differ")
    if not episodes:
        raise ValueError("empty raw training batch")
    return list(zip(episodes, frames, strict=True))


@dataclass
class M3ScheduleEvidence:
    track: str
    micro_batch: int
    gradient_accumulation: int
    expected_references: Iterator[dict[str, Any]]
    wall_target_sec: float = EQUAL_WALL_SEC
    sample_target: int = EQUAL_DATA_BUDGET
    optimizer_target: int = EQUAL_DATA_OPTIMIZER_UPDATES
    clock: Any = time.perf_counter
    consumed_samples: int = 0
    optimizer_updates: int = 0
    started_at: float | None = None
    stopped_at: float | None = None
    first_mismatches: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.track not in {"equal_data", "equal_wall"}:
            raise ValueError(f"invalid M3 track: {self.track}")
        if self.micro_batch <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("micro-batch and gradient accumulation must be positive")
        if self.micro_batch * self.gradient_accumulation != EFFECTIVE_BATCH:
            raise ValueError("M3 effective batch must remain 32")
        if self.sample_target <= 0 or self.optimizer_target <= 0:
            raise ValueError("M3 evidence targets must be positive")
        if self.track == "equal_data":
            if self.sample_target % EFFECTIVE_BATCH != 0:
                raise ValueError("equal-data sample target must be divisible by effective batch 32")
            if self.optimizer_target * EFFECTIVE_BATCH != self.sample_target:
                raise ValueError("equal-data optimizer target does not match sample target")

    def before_first_batch_fetch(self) -> None:
        if self.started_at is None:
            self.started_at = float(self.clock())

    def observe_raw_batch(self, batch: dict[str, Any]) -> None:
        if self.started_at is None:
            raise RuntimeError("M3 timer/evidence must start before first batch fetch")
        actual = extract_raw_batch_references(batch)
        if len(actual) > self.micro_batch:
            raise RuntimeError(
                f"actual raw batch exceeds selected micro-batch: {len(actual)} > {self.micro_batch}"
            )
        if self.track == "equal_data" and self.consumed_samples + len(actual) > self.sample_target:
            raise RuntimeError("equal-data actual sample overshoot blocked")
        for episode_id, timestep in actual:
            try:
                expected = next(self.expected_references)
            except StopIteration as exc:
                raise RuntimeError("canonical reference stream ended before actual training samples") from exc
            expected_pair = (int(expected["episode_id"]), int(expected["timestep"]))
            actual_pair = (episode_id, timestep)
            if actual_pair != expected_pair:
                mismatch = {
                    "sample_index": self.consumed_samples,
                    "expected": list(expected_pair),
                    "actual": list(actual_pair),
                }
                self.first_mismatches.append(mismatch)
                raise RuntimeError(f"canonical sample mismatch: {mismatch}")
            self.consumed_samples += 1

    def optimizer_boundary(self, *, sync_gradients: bool) -> bool:
        """Record one safe optimizer boundary and return whether the run should stop."""
        if not sync_gradients:
            return False
        self.optimizer_updates += 1
        if self.track == "equal_data":
            if self.consumed_samples == self.sample_target:
                self.stopped_at = float(self.clock())
                return True
            return False
        if self.started_at is None:
            raise RuntimeError("equal-wall timer never started")
        if float(self.clock()) - self.started_at >= self.wall_target_sec:
            self.stopped_at = float(self.clock())
            return True
        return False

    @property
    def elapsed_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None else float(self.clock())
        return max(0.0, end - self.started_at)

    def finalize(self) -> dict[str, Any]:
        if self.first_mismatches:
            raise RuntimeError(f"canonical sampling mismatches observed: {self.first_mismatches[:3]}")
        if self.track == "equal_data":
            if self.consumed_samples != self.sample_target:
                raise RuntimeError(
                    f"equal-data incomplete: {self.consumed_samples}/{self.sample_target} samples"
                )
            if self.optimizer_updates != self.optimizer_target:
                raise RuntimeError(
                    f"equal-data optimizer updates mismatch: {self.optimizer_updates}/{self.optimizer_target}"
                )
        else:
            if self.stopped_at is None or self.elapsed_sec < self.wall_target_sec:
                raise RuntimeError(
                    f"equal-wall stopped before target: {self.elapsed_sec:.6f}/{self.wall_target_sec} sec"
                )
        return {
            "track": self.track,
            "micro_batch": self.micro_batch,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch_size": self.micro_batch * self.gradient_accumulation,
            "sample_target": self.sample_target,
            "optimizer_target": self.optimizer_target,
            "consumed_samples": self.consumed_samples,
            "optimizer_updates": self.optimizer_updates,
            "train_wall_time_sec": self.elapsed_sec,
            "canonical_sample_order_verified": True,
            "sample_mismatch_count": 0,
        }
