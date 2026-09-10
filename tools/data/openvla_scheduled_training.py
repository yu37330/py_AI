#!/usr/bin/env python3
"""Canonical-schedule loader for OpenVLA-OFT on selected LeRobot data.

Only schedule-referenced frames are decoded.  References are grouped by episode
for I/O, then restored to the exact canonical sample order before training.
This avoids materializing the projected 56.6 GiB RLDS copy while preserving the
same frame-level M3 schedule used by the LeRobot models.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from openvla_lerobot_streaming import (
    ACTION_NORMALIZATION_MASK,
    FRONT_KEY,
    WRIST_KEY,
    STATE_KEY,
    ACTION_KEY,
    _data_path,
    _video_path,
    bounds_q99_normalize,
    decode_video_targets,
    load_episode_metadata,
    load_source_info,
    load_task_lookup,
    make_action_chunk,
    standardize_action,
    standardize_proprio,
)
from openvla_streaming_training_adapter import to_upstream_window

AUGMENT_KWARGS = {
    "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
    "random_brightness": [0.2],
    "random_contrast": [0.8, 1.2],
    "random_saturation": [0.8, 1.2],
    "random_hue": [0.05],
    "augment_order": [
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
}


def _group_references(references: Iterable[dict[str, int]]) -> dict[int, list[dict[str, int]]]:
    groups: dict[int, list[dict[str, int]]] = {}
    for ref in references:
        groups.setdefault(int(ref["episode_id"]), []).append(ref)
    return groups


def _augmentation_seed(schedule_seed: int, sample_index: int) -> tuple[int, int]:
    payload = f"parc2026-openvla-aug-v1|{int(schedule_seed)}|{int(sample_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Keep signed-int32-safe positive values.
    return (
        int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
        int.from_bytes(digest[4:8], "big") & 0x7FFFFFFF,
    )


def augment_window_images(window: dict[str, Any], *, schedule_seed: int, sample_index: int) -> dict[str, Any]:
    """Apply the exact pinned OpenVLA RLDS image augmentation recipe."""
    import tensorflow as tf
    from prismatic.vla.datasets.rlds import obs_transforms

    obs = dict(window["observation"])
    aug_obs = {
        "image_primary": tf.convert_to_tensor(obs["image_primary"], dtype=tf.uint8),
        "image_wrist": tf.convert_to_tensor(obs["image_wrist"], dtype=tf.uint8),
        "pad_mask_dict": {
            "image_primary": tf.constant(True),
            "image_wrist": tf.constant(True),
        },
    }
    seed = tf.convert_to_tensor(_augmentation_seed(schedule_seed, sample_index), dtype=tf.int32)
    aug_obs = obs_transforms.augment(aug_obs, seed=seed, augment_kwargs=AUGMENT_KWARGS)
    obs["image_primary"] = np.asarray(aug_obs["image_primary"].numpy(), dtype=np.uint8)
    obs["image_wrist"] = np.asarray(aug_obs["image_wrist"].numpy(), dtype=np.uint8)
    out = dict(window)
    out["observation"] = obs
    return out


def load_scheduled_windows(
    root: Path,
    references: list[dict[str, int]],
    statistics: dict[str, Any],
    *,
    image_aug: bool,
    schedule_seed: int,
) -> list[dict[str, Any]]:
    """Load only referenced frames while restoring exact schedule order."""
    import pyarrow.parquet as pq

    if not references:
        raise ValueError("OpenVLA scheduled loader requires references")
    info = load_source_info(root)
    groups = _group_references(references)
    episode_ids = sorted(groups)
    metadata = load_episode_metadata(root, episode_ids)
    task_lookup = load_task_lookup(root)
    action_stats = statistics["action"]
    proprio_stats = statistics["proprio"]
    windows: dict[int, dict[str, Any]] = {}
    fps = float(info["fps"])

    for episode_id in episode_ids:
        row = metadata.loc[episode_id]
        table = pq.read_table(
            _data_path(root, info, row),
            columns=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "task_index"],
            filters=[("episode_index", "=", int(episode_id))],
        )
        df = table.to_pandas().sort_values("frame_index").reset_index(drop=True)
        expected_len = int(row["length"])
        if len(df) != expected_len:
            raise RuntimeError(f"episode {episode_id}: rows={len(df)} metadata={expected_len}")
        raw_state = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df[STATE_KEY].tolist()])
        raw_action = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df[ACTION_KEY].tolist()])
        proprio = standardize_proprio(raw_state)
        actions = standardize_action(raw_action)
        action_norm = bounds_q99_normalize(
            actions,
            np.asarray(action_stats["q01"]),
            np.asarray(action_stats["q99"]),
            np.asarray(action_stats["min"]),
            np.asarray(action_stats["max"]),
            mask=np.asarray(ACTION_NORMALIZATION_MASK),
        )
        proprio_norm = bounds_q99_normalize(
            proprio,
            np.asarray(proprio_stats["q01"]),
            np.asarray(proprio_stats["q99"]),
            np.asarray(proprio_stats["min"]),
            np.asarray(proprio_stats["max"]),
        )
        task_ids = np.asarray(df["task_index"], dtype=np.int64)
        if len(set(task_ids.tolist())) != 1:
            raise RuntimeError(f"episode {episode_id}: multiple task IDs")
        instruction = task_lookup[int(task_ids[0])]
        refs = groups[episode_id]
        timesteps = sorted({int(ref["timestep"]) for ref in refs})
        if any(t < 0 or t >= expected_len for t in timesteps):
            raise ValueError(f"episode {episode_id}: scheduled timestep out of range")
        local_ts = np.asarray(df["timestamp"], dtype=np.float64)
        front_start = float(row[f"videos/{FRONT_KEY}/from_timestamp"])
        wrist_start = float(row[f"videos/{WRIST_KEY}/from_timestamp"])
        front = decode_video_targets(
            _video_path(root, info, row, FRONT_KEY),
            front_start + local_ts[timesteps],
            fps,
        )
        wrist = decode_video_targets(
            _video_path(root, info, row, WRIST_KEY),
            wrist_start + local_ts[timesteps],
            fps,
        )
        decoded = {t: (front[i], wrist[i]) for i, t in enumerate(timesteps)}
        for ref in refs:
            sample_index = int(ref["sample_index"])
            t = int(ref["timestep"])
            primary, wrist_img = decoded[t]
            window = {
                "dataset_name": "parc_libero_selected_streaming",
                "episode_id": episode_id,
                "timestep": t,
                "observation": {
                    "image_primary": primary,
                    "image_wrist": wrist_img,
                    "proprio": proprio_norm[t],
                },
                "task": {"language_instruction": instruction.encode("utf-8")},
                "action": make_action_chunk(action_norm, t),
            }
            if image_aug:
                window = augment_window_images(
                    window, schedule_seed=schedule_seed, sample_index=sample_index
                )
            windows[sample_index] = window

    if len(windows) != len(references):
        raise RuntimeError(f"scheduled window count mismatch: {len(windows)} != {len(references)}")
    # Keep the caller-provided canonical order even when this is a later chunk
    # whose absolute sample_index does not start at zero.
    return [windows[int(ref["sample_index"])] for ref in references]


def make_scheduled_torch_dataset(
    root: Path,
    references: list[dict[str, int]],
    statistics: dict[str, Any],
    batch_transform,
    *,
    image_aug: bool,
    schedule_seed: int,
):
    import torch

    class ScheduledDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.windows = load_scheduled_windows(
                root,
                references,
                statistics,
                image_aug=image_aug,
                schedule_seed=schedule_seed,
            )
            self.dataset_statistics = statistics

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index: int):
            return batch_transform(to_upstream_window(self.windows[int(index)]))

    return ScheduledDataset()
