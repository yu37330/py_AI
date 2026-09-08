#!/usr/bin/env python3
"""Direct LeRobot -> OpenVLA streaming bridge for the D10-selected LIBERO pool.

This module deliberately does not materialize a second RLDS/TFDS copy. It reads
only manifest-selected episodes from the prefetched LeRobot v3 dataset, applies
the same LIBERO standardization contract used by the pinned OpenVLA-OFT stack,
and can expose trajectory/window data for a future training runner.

The bridge contract is provenance-first:
- exact schema-v2 group-aware manifest only;
- source dataset/revision must match the D10 decision;
- raw movement actions are preserved;
- LIBERO gripper standardization is 1 - clip(raw_gripper, 0, 1);
- proprio is state[:6] + state[-2:];
- q01/q99 normalization is applied only after selected-pool statistics exist;
- action chunks contain 8 actions; beyond episode end, relative dims are zero
  and the absolute gripper repeats the last valid value.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DATASET_ID = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
VARIANT = "V2_SQRT_BALANCED_RAW"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_DIM = 7
PROPRIO_DIM = 8
ACTION_CHUNK = 8
NORMALIZATION_TYPE = "bounds_q99"
ABSOLUTE_ACTION_MASK = [False] * 6 + [True]
ACTION_NORMALIZATION_MASK = [True] * 6 + [False]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data.get("episode_ids")
    if (
        data.get("schema_version") != 2
        or data.get("group_aware") is not True
        or data.get("kind") != "training_episode_manifest"
        or data.get("variant") != VARIANT
    ):
        raise ValueError(f"not the selected group-aware manifest: {path}")
    if data.get("dataset_id") != DATASET_ID or data.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("manifest dataset provenance mismatch")
    if not isinstance(ids, list) or len(ids) != 10758 or len(ids) != len(set(ids)):
        raise ValueError("manifest episode_ids must contain the exact 10,758 selected episodes")
    ids = [int(x) for x in ids]
    digest = _sha256_json(ids)
    if data.get("episode_ids_sha256") != digest:
        raise ValueError("manifest episode_ids hash mismatch")
    summary = data.get("summary", {})
    if int(summary.get("episode_count", -1)) != len(ids):
        raise ValueError("manifest episode count mismatch")
    if int(summary.get("frame_count", 0)) <= 0:
        raise ValueError("manifest frame count missing")
    data["episode_ids"] = ids
    return data


def load_source_info(root: Path) -> dict[str, Any]:
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0" or int(info.get("fps", -1)) != 20:
        raise ValueError("expected LeRobot v3.0 at 20 Hz")
    features = info.get("features", {})
    for key in (FRONT_KEY, WRIST_KEY, STATE_KEY, ACTION_KEY):
        if key not in features:
            raise KeyError(f"source feature missing: {key}")
    if list(features[STATE_KEY]["shape"]) != [8]:
        raise ValueError(f"expected state8, got {features[STATE_KEY]['shape']}")
    if list(features[ACTION_KEY]["shape"]) != [7]:
        raise ValueError(f"expected action7, got {features[ACTION_KEY]['shape']}")
    for key in (FRONT_KEY, WRIST_KEY):
        if list(features[key]["shape"]) != [256, 256, 3]:
            raise ValueError(f"expected 256x256 RGB for {key}")
    return info


def standardize_action(raw_action: np.ndarray) -> np.ndarray:
    """Apply pinned OpenVLA-OFT LIBERO action standardization."""
    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected action[...,7], got {raw.shape}")
    out = raw.copy()
    out[..., 6] = 1.0 - np.clip(raw[..., 6], 0.0, 1.0)
    return out


def standardize_proprio(raw_state: np.ndarray) -> np.ndarray:
    """Construct LIBERO proprio: state[:6] + state[-2:]."""
    state = np.asarray(raw_state, dtype=np.float32)
    if state.shape[-1] != 8:
        raise ValueError(f"expected state[...,8], got {state.shape}")
    return np.concatenate([state[..., :6], state[..., -2:]], axis=-1).astype(
        np.float32, copy=False
    )


def bounds_q99_normalize(
    values: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """OpenVLA BOUNDS_Q99 normalization, with optional per-dimension mask."""
    x = np.asarray(values, dtype=np.float32)
    low = np.asarray(q01, dtype=np.float32)
    high = np.asarray(q99, dtype=np.float32)
    scaled = np.clip(2.0 * (x - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    scaled = np.where(low == high, 0.0, scaled)
    if mask is None:
        return scaled.astype(np.float32, copy=False)
    mask = np.asarray(mask, dtype=bool)
    return np.where(mask, scaled, x).astype(np.float32, copy=False)


def make_action_chunk(
    standardized_actions: np.ndarray,
    start: int,
    chunk_size: int = ACTION_CHUNK,
) -> np.ndarray:
    """Create OpenVLA future-action chunk with relative/absolute end padding."""
    actions = np.asarray(standardized_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected [T,7] actions, got {actions.shape}")
    if not (0 <= start < len(actions)):
        raise IndexError(start)
    out = np.zeros((chunk_size, ACTION_DIM), dtype=np.float32)
    stop = min(len(actions), start + chunk_size)
    valid = stop - start
    out[:valid] = actions[start:stop]
    if valid < chunk_size:
        # Movement dims are relative -> neutral zero. Gripper is absolute ->
        # repeat the last valid absolute gripper state.
        out[valid:, 6] = actions[stop - 1, 6]
    return out


def load_task_lookup(root: Path) -> dict[int, str]:
    import pandas as pd

    tasks = pd.read_parquet(Path(root) / "meta/tasks.parquet")
    if "task_index" not in tasks.columns:
        raise KeyError("meta/tasks.parquet missing task_index")
    return {int(row.task_index): str(idx) for idx, row in tasks.iterrows()}


def load_episode_metadata(root: Path, episode_ids: list[int]):
    import pyarrow.dataset as pads

    files = sorted((Path(root) / "meta/episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(Path(root) / "meta/episodes")
    ds = pads.dataset([str(p) for p in files], format="parquet")
    table = ds.to_table(filter=pads.field("episode_index").isin(episode_ids))
    df = table.to_pandas().sort_values("episode_index").reset_index(drop=True)
    found = set(int(x) for x in df["episode_index"].tolist())
    missing = sorted(set(episode_ids) - found)
    if missing:
        raise KeyError(f"episode metadata missing IDs: {missing[:20]}")
    return df.set_index("episode_index", drop=False)


def _data_path(root: Path, info: dict[str, Any], row) -> Path:
    return Path(root) / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )


def _video_path(root: Path, info: dict[str, Any], row, key: str) -> Path:
    return Path(root) / info["video_path"].format(
        video_key=key,
        chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]),
    )


def decode_video_targets(path: Path, targets_sec: np.ndarray, fps: float) -> np.ndarray:
    import av

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    targets = np.asarray(targets_sec, dtype=np.float64)
    if len(targets) == 0:
        return np.empty((0, 256, 256, 3), dtype=np.uint8)
    if np.any(np.diff(targets) < -1e-6):
        raise ValueError("video target timestamps must be monotonic")

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        seek_sec = max(0.0, float(targets[0]) - max(1.0, 8.0 / fps))
        container.seek(int(seek_sec / tb), stream=stream, any_frame=False, backward=True)
        decoded_t: list[float] = []
        decoded_img: list[np.ndarray] = []
        stop_after = float(targets[-1]) + max(0.15, 3.0 / fps)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t + 1.0 / fps < float(targets[0]):
                continue
            decoded_t.append(t)
            decoded_img.append(frame.to_ndarray(format="rgb24"))
            if t >= stop_after:
                break

    if not decoded_t:
        raise RuntimeError(f"no frames decoded from {path}")
    times = np.asarray(decoded_t)
    nearest = np.abs(targets[:, None] - times[None, :]).argmin(axis=1)
    error = np.abs(targets - times[nearest])
    tolerance = max(0.03, 0.80 / fps)
    if float(error.max(initial=0.0)) > tolerance:
        raise RuntimeError(
            f"video timestamp mismatch {path}: max={float(error.max()):.4f}s > {tolerance:.4f}s"
        )
    frames = np.stack([decoded_img[int(i)] for i in nearest], axis=0)
    if frames.shape[1:] != (256, 256, 3):
        raise ValueError(f"unexpected decoded frame shape: {frames.shape}")
    return frames.astype(np.uint8, copy=False)


@dataclass
class StreamingTrajectory:
    episode_id: int
    image_primary: np.ndarray
    image_wrist: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    language_instruction: str
    raw_state: np.ndarray
    raw_action: np.ndarray


def load_streaming_trajectory(
    root: Path,
    info: dict[str, Any],
    meta_row,
    task_lookup: dict[int, str],
) -> StreamingTrajectory:
    import pyarrow.parquet as pq

    episode_id = int(meta_row["episode_index"])
    table = pq.read_table(
        _data_path(root, info, meta_row),
        columns=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "task_index"],
        filters=[("episode_index", "=", episode_id)],
    )
    df = table.to_pandas().sort_values("frame_index").reset_index(drop=True)
    expected_len = int(meta_row["length"])
    if len(df) != expected_len:
        raise RuntimeError(
            f"episode {episode_id}: rows={len(df)} metadata length={expected_len}"
        )

    raw_state = np.stack(
        [np.asarray(x, dtype=np.float32).reshape(-1) for x in df[STATE_KEY].tolist()]
    )
    raw_action = np.stack(
        [np.asarray(x, dtype=np.float32).reshape(-1) for x in df[ACTION_KEY].tolist()]
    )
    local_ts = np.asarray(df["timestamp"], dtype=np.float64)
    task_ids = np.asarray(df["task_index"], dtype=np.int64)
    if len(set(task_ids.tolist())) != 1:
        raise RuntimeError(f"episode {episode_id}: multiple task_index values")
    task_id = int(task_ids[0])
    if task_id not in task_lookup:
        raise KeyError(f"task_index {task_id} missing from tasks.parquet")
    instruction = task_lookup[task_id]
    if not instruction:
        raise ValueError(f"episode {episode_id}: empty language instruction")

    fps = float(info["fps"])
    front_start = float(meta_row[f"videos/{FRONT_KEY}/from_timestamp"])
    wrist_start = float(meta_row[f"videos/{WRIST_KEY}/from_timestamp"])
    front = decode_video_targets(
        _video_path(root, info, meta_row, FRONT_KEY), front_start + local_ts, fps
    )
    wrist = decode_video_targets(
        _video_path(root, info, meta_row, WRIST_KEY), wrist_start + local_ts, fps
    )
    if len(front) != len(df) or len(wrist) != len(df):
        raise RuntimeError(f"episode {episode_id}: decoded video length mismatch")

    return StreamingTrajectory(
        episode_id=episode_id,
        image_primary=front,
        image_wrist=wrist,
        proprio=standardize_proprio(raw_state),
        action=standardize_action(raw_action),
        language_instruction=instruction,
        raw_state=raw_state,
        raw_action=raw_action,
    )


def iter_selected_trajectories(
    root: Path,
    manifest: dict[str, Any],
    *,
    max_episodes: int | None = None,
) -> Iterator[StreamingTrajectory]:
    info = load_source_info(root)
    ids = manifest["episode_ids"][:max_episodes] if max_episodes else manifest["episode_ids"]
    tasks = load_task_lookup(root)
    metadata = load_episode_metadata(root, ids)
    for episode_id in ids:
        yield load_streaming_trajectory(root, info, metadata.loc[episode_id], tasks)


def iter_openvla_windows(
    trajectory: StreamingTrajectory,
    statistics: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield pre-tokenization OpenVLA windows from one standardized trajectory."""
    action_stats = statistics["action"]
    proprio_stats = statistics["proprio"]
    action_norm = bounds_q99_normalize(
        trajectory.action,
        np.asarray(action_stats["q01"]),
        np.asarray(action_stats["q99"]),
        mask=np.asarray(ACTION_NORMALIZATION_MASK),
    )
    proprio_norm = bounds_q99_normalize(
        trajectory.proprio,
        np.asarray(proprio_stats["q01"]),
        np.asarray(proprio_stats["q99"]),
    )
    for i in range(len(trajectory.action)):
        yield {
            "dataset_name": "parc_libero_selected_streaming",
            "episode_id": trajectory.episode_id,
            "timestep": i,
            "observation": {
                "image_primary": trajectory.image_primary[i],
                "image_wrist": trajectory.image_wrist[i],
                "proprio": proprio_norm[i],
            },
            "task": {"language_instruction": trajectory.language_instruction.encode("utf-8")},
            "action": make_action_chunk(action_norm, i),
        }
