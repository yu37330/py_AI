#!/usr/bin/env python3
"""Bounded-memory iterable datasets for scheduled OpenVLA-OFT M3 training."""
from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from openvla_scheduled_training import load_scheduled_windows
from openvla_streaming_training_adapter import to_upstream_window

DEFAULT_CHUNK_SIZE = 256


def _iter_chunks(references: Iterable[dict[str, int]], chunk_size: int) -> Iterator[list[dict[str, int]]]:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    iterator = iter(references)
    while True:
        chunk = list(islice(iterator, int(chunk_size)))
        if not chunk:
            return
        yield chunk


def iter_transformed_chunks(
    *,
    root: Path,
    references: Iterable[dict[str, int]],
    statistics: dict[str, Any],
    batch_transform,
    image_aug: bool,
    schedule_seed: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[dict[str, Any]]:
    """Decode at most one bounded reference chunk and yield in canonical order."""
    for chunk in _iter_chunks(references, int(chunk_size)):
        windows = load_scheduled_windows(
            Path(root),
            chunk,
            statistics,
            image_aug=bool(image_aug),
            schedule_seed=int(schedule_seed),
        )
        if len(windows) != len(chunk):
            raise RuntimeError("OpenVLA scheduled chunk lost references")
        for window in windows:
            yield batch_transform(to_upstream_window(window))


def make_finite_scheduled_iterable_dataset(
    *,
    root: Path,
    references: Sequence[dict[str, int]],
    statistics: dict[str, Any],
    batch_transform,
    image_aug: bool,
    schedule_seed: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    import torch

    frozen = tuple(dict(ref) for ref in references)
    if not frozen:
        raise ValueError("finite OpenVLA schedule cannot be empty")

    class FiniteScheduledDataset(torch.utils.data.IterableDataset):
        dataset_statistics = statistics

        def __iter__(self):
            yield from iter_transformed_chunks(
                root=Path(root),
                references=frozen,
                statistics=statistics,
                batch_transform=batch_transform,
                image_aug=image_aug,
                schedule_seed=schedule_seed,
                chunk_size=chunk_size,
            )

        def __len__(self) -> int:
            return len(frozen)

    return FiniteScheduledDataset()


def make_counter_stream_iterable_dataset(
    *,
    root: Path,
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    statistics: dict[str, Any],
    batch_transform,
    image_aug: bool,
    schedule_seed: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """Create an effectively-unbounded OpenVLA stream without materializing refs."""
    import torch
    from tools.benchmark.m3_sampling_schedule import iter_references

    ids = tuple(int(x) for x in episode_ids)
    lengths = tuple(int(x) for x in episode_lengths)
    if not ids or len(ids) != len(lengths):
        raise ValueError("OpenVLA counter stream requires matching episode IDs/lengths")

    class CounterStreamDataset(torch.utils.data.IterableDataset):
        dataset_statistics = statistics

        def __iter__(self):
            references = iter_references(
                episode_ids=ids,
                episode_lengths=lengths,
                seed=int(schedule_seed),
                start_index=0,
            )
            yield from iter_transformed_chunks(
                root=Path(root),
                references=references,
                statistics=statistics,
                batch_transform=batch_transform,
                image_aug=image_aug,
                schedule_seed=schedule_seed,
                chunk_size=chunk_size,
            )

        # DataLoader only needs a finite length for progress/accounting. The
        # runner always terminates via the equal-wall optimizer-boundary gate.
        def __len__(self) -> int:
            return 2_147_483_648

    return CounterStreamDataset()
