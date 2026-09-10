#!/usr/bin/env python3
"""Canonical-schedule sample source for guarded OpenVLA-OFT M3 training.

This module keeps the validated 69c LeRobot -> OpenVLA semantics while allowing
M3 to request arbitrary `(episode_id, timestep)` references in the exact order
produced by the framework-independent sampling controller.  It never builds a
full RLDS copy.

For efficiency, vector data are cached per episode and video frames are decoded
once per camera for the requested timesteps in each micro-batch.  The returned
sample identities are verified by the worker before every forward pass.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from openvla_lerobot_streaming import (
    ACTION_KEY,
    ACTION_NORMALIZATION_MASK,
    FRONT_KEY,
    STATE_KEY,
    WRIST_KEY,
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
from openvla_streaming_training_adapter import to_openvla_rlds_batch


IMAGE_AUGMENT_KWARGS = {
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


@dataclass
class EpisodeVectors:
    episode_id: int
    meta_row: Any
    timestamps: np.ndarray
    action: np.ndarray
    proprio: np.ndarray
    language_instruction: str


class ScheduledOpenVLASource:
    """Load canonical OpenVLA training examples from the existing LeRobot dataset."""

    def __init__(
        self,
        source_root: Path,
        manifest: dict[str, Any],
        statistics: dict[str, Any],
        *,
        vector_cache_episodes: int = 64,
    ) -> None:
        self.source_root = Path(source_root)
        self.manifest = manifest
        self.statistics = statistics
        self.info = load_source_info(self.source_root)
        self.task_lookup = load_task_lookup(self.source_root)
        self.episode_ids = [int(x) for x in manifest["episode_ids"]]
        self.episode_id_set = set(self.episode_ids)
        self.metadata = load_episode_metadata(self.source_root, self.episode_ids)
        self.vector_cache_episodes = max(1, int(vector_cache_episodes))
        self._vectors: OrderedDict[int, EpisodeVectors] = OrderedDict()

    def _load_vectors(self, episode_id: int) -> EpisodeVectors:
        import pyarrow.parquet as pq

        episode_id = int(episode_id)
        if episode_id not in self.episode_id_set:
            raise KeyError(f"scheduled episode {episode_id} is outside frozen D10")
        cached = self._vectors.get(episode_id)
        if cached is not None:
            self._vectors.move_to_end(episode_id)
            return cached

        row = self.metadata.loc[episode_id]
        table = pq.read_table(
            _data_path(self.source_root, self.info, row),
            columns=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "task_index"],
            filters=[("episode_index", "=", episode_id)],
        )
        df = table.to_pandas().sort_values("frame_index").reset_index(drop=True)
        expected_len = int(row["length"])
        if len(df) != expected_len:
            raise RuntimeError(
                f"episode {episode_id}: rows={len(df)} metadata length={expected_len}"
            )
        frames = np.asarray(df["frame_index"], dtype=np.int64)
        if not np.array_equal(frames, np.arange(expected_len, dtype=np.int64)):
            raise RuntimeError(f"episode {episode_id}: frame_index is not contiguous from zero")

        raw_state = np.stack(
            [np.asarray(x, dtype=np.float32).reshape(-1) for x in df[STATE_KEY].tolist()]
        )
        raw_action = np.stack(
            [np.asarray(x, dtype=np.float32).reshape(-1) for x in df[ACTION_KEY].tolist()]
        )
        action_std = standardize_action(raw_action)
        proprio_std = standardize_proprio(raw_state)
        action_stats = self.statistics["action"]
        proprio_stats = self.statistics["proprio"]
        action = bounds_q99_normalize(
            action_std,
            np.asarray(action_stats["q01"]),
            np.asarray(action_stats["q99"]),
            np.asarray(action_stats["min"]),
            np.asarray(action_stats["max"]),
            mask=np.asarray(ACTION_NORMALIZATION_MASK),
        )
        proprio = bounds_q99_normalize(
            proprio_std,
            np.asarray(proprio_stats["q01"]),
            np.asarray(proprio_stats["q99"]),
            np.asarray(proprio_stats["min"]),
            np.asarray(proprio_stats["max"]),
        )
        task_ids = np.asarray(df["task_index"], dtype=np.int64)
        if len(set(task_ids.tolist())) != 1:
            raise RuntimeError(f"episode {episode_id}: multiple task_index values")
        task_id = int(task_ids[0])
        if task_id not in self.task_lookup:
            raise KeyError(f"task_index {task_id} missing from tasks.parquet")
        language = self.task_lookup[task_id]
        if not language:
            raise ValueError(f"episode {episode_id}: empty language instruction")

        result = EpisodeVectors(
            episode_id=episode_id,
            meta_row=row,
            timestamps=np.asarray(df["timestamp"], dtype=np.float64),
            action=action,
            proprio=proprio,
            language_instruction=language,
        )
        self._vectors[episode_id] = result
        self._vectors.move_to_end(episode_id)
        while len(self._vectors) > self.vector_cache_episodes:
            self._vectors.popitem(last=False)
        return result

    def _decode_batch_images(
        self, refs: list[dict[str, Any]]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        primary: list[np.ndarray | None] = [None] * len(refs)
        wrist: list[np.ndarray | None] = [None] * len(refs)
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for slot, ref in enumerate(refs):
            episode_id = int(ref["episode_id"])
            timestep = int(ref["timestep"])
            grouped[episode_id].append((slot, timestep))

        fps = float(self.info["fps"])
        for episode_id, slot_steps in grouped.items():
            vectors = self._load_vectors(episode_id)
            for _, timestep in slot_steps:
                if not 0 <= timestep < len(vectors.timestamps):
                    raise IndexError(
                        f"scheduled timestep out of range: episode={episode_id} timestep={timestep}"
                    )
            ordered = sorted(slot_steps, key=lambda item: item[1])
            steps = np.asarray([step for _, step in ordered], dtype=np.int64)
            local_ts = vectors.timestamps[steps]
            row = vectors.meta_row
            front_start = float(row[f"videos/{FRONT_KEY}/from_timestamp"])
            wrist_start = float(row[f"videos/{WRIST_KEY}/from_timestamp"])
            front_frames = decode_video_targets(
                _video_path(self.source_root, self.info, row, FRONT_KEY),
                front_start + local_ts,
                fps,
            )
            wrist_frames = decode_video_targets(
                _video_path(self.source_root, self.info, row, WRIST_KEY),
                wrist_start + local_ts,
                fps,
            )
            for i, (slot, _) in enumerate(ordered):
                primary[slot] = front_frames[i]
                wrist[slot] = wrist_frames[i]

        if any(frame is None for frame in primary) or any(frame is None for frame in wrist):
            raise RuntimeError("scheduled image decode left an empty batch slot")
        return [np.asarray(x, dtype=np.uint8) for x in primary], [np.asarray(x, dtype=np.uint8) for x in wrist]

    def load_rlds_batch(self, refs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return pre-tokenization RLDS-shaped examples in the requested order."""
        refs = [dict(ref) for ref in refs]
        if not refs:
            raise ValueError("scheduled OpenVLA batch cannot be empty")
        primary, wrist = self._decode_batch_images(refs)
        out: list[dict[str, Any]] = []
        for slot, ref in enumerate(refs):
            episode_id = int(ref["episode_id"])
            timestep = int(ref["timestep"])
            vectors = self._load_vectors(episode_id)
            window = {
                "dataset_name": "parc_libero_selected_streaming",
                "episode_id": episode_id,
                "timestep": timestep,
                "observation": {
                    "image_primary": primary[slot],
                    "image_wrist": wrist[slot],
                    "proprio": vectors.proprio[timestep],
                },
                "task": {
                    "language_instruction": vectors.language_instruction.encode("utf-8")
                },
                "action": make_action_chunk(vectors.action, timestep),
            }
            out.append(to_openvla_rlds_batch(window))
        return out


def augmentation_seed_pair(run_seed: int, sample_index: int) -> tuple[int, int]:
    payload = f"parc2026-openvla-image-aug-v1|{int(run_seed)}|{int(sample_index)}".encode(
        "ascii"
    )
    digest = hashlib.sha256(payload).digest()
    limit = 2**31 - 1
    return (
        int.from_bytes(digest[:4], "big") % limit,
        int.from_bytes(digest[4:8], "big") % limit,
    )


def augment_openvla_rlds_sample(
    sample: dict[str, Any], *, run_seed: int, sample_index: int
) -> dict[str, Any]:
    """Apply the pinned OpenVLA-OFT image augmentation recipe before tokenization.

    The augmentation parameters match `RLDSDataset(..., image_aug=True)`.  Seeds
    are deterministic from the M3 seed and canonical sample index, so forward and
    reverse order receive identical augmented observations for a given schedule.
    """
    import dlimp as dl
    import tensorflow as tf

    first, second = augmentation_seed_pair(run_seed, sample_index)
    obs = dict(sample["observation"])
    for offset, key in enumerate(("image_primary", "image_wrist")):
        image = np.asarray(obs[key][0], dtype=np.uint8)
        seed = tf.constant([first, (second + offset) % (2**31 - 1)], dtype=tf.int32)
        augmented = dl.transforms.augment_image(
            tf.convert_to_tensor(image, dtype=tf.uint8),
            **IMAGE_AUGMENT_KWARGS,
            seed=seed,
        )
        image_out = np.asarray(augmented.numpy(), dtype=np.uint8)
        if image_out.shape != image.shape:
            raise RuntimeError(f"OpenVLA augmentation changed image shape: {image.shape} -> {image_out.shape}")
        obs[key] = image_out[None, ...]
    result = dict(sample)
    result["observation"] = obs
    return result


def transform_scheduled_batch(
    source: ScheduledOpenVLASource,
    refs: list[dict[str, Any]],
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    run_seed: int,
    image_aug: bool,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Load, optionally augment, and tokenize one canonical micro-batch."""
    raw = source.load_rlds_batch(refs)
    transformed: list[dict[str, Any]] = []
    identities: list[tuple[int, int]] = []
    for ref, sample in zip(refs, raw, strict=True):
        episode_id = int(sample["episode_id"])
        timestep = int(sample["timestep"])
        expected = (int(ref["episode_id"]), int(ref["timestep"]))
        actual = (episode_id, timestep)
        if actual != expected:
            raise RuntimeError(f"OpenVLA scheduled sample mismatch: expected={expected} actual={actual}")
        if image_aug:
            sample = augment_openvla_rlds_sample(
                sample,
                run_seed=int(run_seed),
                sample_index=int(ref["sample_index"]),
            )
        transformed.append(batch_transform(sample))
        identities.append(actual)
    return transformed, identities
