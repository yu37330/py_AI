#!/usr/bin/env python3
"""Training-facing wrapper for the validated 69c LeRobot -> OpenVLA stream.

69c intentionally validates a model-neutral pre-tokenization window. The pinned
OpenVLA-OFT RLDSBatchTransform expects a leading observation window dimension
(`image_primary[0]`, `image_wrist[0]`) while actions already carry the future
8-action chunk. This module adds only that structural dimension and delegates
all token/image transforms to the pinned upstream batch transform.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from openvla_lerobot_streaming import iter_openvla_windows, iter_selected_trajectories


def to_openvla_rlds_batch(window: dict[str, Any]) -> dict[str, Any]:
    """Convert one validated 69c window to the shape expected by RLDSBatchTransform."""
    obs = window["observation"]
    primary = np.asarray(obs["image_primary"], dtype=np.uint8)
    wrist = np.asarray(obs["image_wrist"], dtype=np.uint8)
    proprio = np.asarray(obs["proprio"], dtype=np.float32)
    action = np.asarray(window["action"], dtype=np.float32)

    if primary.shape != (256, 256, 3) or wrist.shape != (256, 256, 3):
        raise ValueError(f"expected 256x256 RGB images, got primary={primary.shape} wrist={wrist.shape}")
    if proprio.shape != (8,):
        raise ValueError(f"expected proprio[8], got {proprio.shape}")
    if action.shape != (8, 7):
        raise ValueError(f"expected action chunk[8,7], got {action.shape}")
    language = window["task"]["language_instruction"]
    if not isinstance(language, (bytes, bytearray)) or not language:
        raise ValueError("language_instruction must be non-empty bytes")

    return {
        "dataset_name": window["dataset_name"],
        "observation": {
            # Pinned OpenVLA-OFT RLDSBatchTransform indexes observation images at [0].
            "image_primary": primary[None, ...],
            "image_wrist": wrist[None, ...],
            "proprio": proprio[None, ...],
        },
        "task": {"language_instruction": bytes(language)},
        "action": action,
        "episode_id": int(window["episode_id"]),
        "timestep": int(window["timestep"]),
    }


def iter_transformed_openvla_samples(
    source_root: Path,
    manifest: dict[str, Any],
    statistics: dict[str, Any],
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_episodes: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield samples in the same post-RLDSBatchTransform shape as upstream RLDSDataset."""
    for trajectory in iter_selected_trajectories(source_root, manifest, max_episodes=max_episodes):
        for window in iter_openvla_windows(trajectory, statistics):
            yield batch_transform(to_openvla_rlds_batch(window))


def make_torch_iterable_dataset(
    source_root: Path,
    manifest: dict[str, Any],
    statistics: dict[str, Any],
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_episodes: int | None = None,
):
    """Build a torch IterableDataset lazily, keeping torch optional for contract tests."""
    import torch

    total = int(statistics.get("num_transitions", 0))
    if total <= 0:
        raise ValueError("dataset statistics must include positive num_transitions")

    class OpenVLASelectedStreamingDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield from iter_transformed_openvla_samples(
                Path(source_root),
                manifest,
                statistics,
                batch_transform,
                max_episodes=max_episodes,
            )

        def __len__(self) -> int:
            return total

    return OpenVLASelectedStreamingDataset()
