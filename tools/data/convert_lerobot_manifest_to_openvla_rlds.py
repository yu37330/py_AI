#!/usr/bin/env python3
"""Convert an exact LeRobot episode manifest into an OpenVLA-compatible RLDS/TFDS dataset.

The bridge is intentionally provenance-first:
- input episode IDs come only from the supplied schema-v2 manifest;
- source state/action values are preserved exactly;
- source front/wrist video frames are decoded at the original LeRobot timestamps;
- no no-op filtering or action rewriting is performed during conversion;
- a full ``conversion_contract.json`` is written only when every manifest episode was converted.

A small ``--max-episodes`` run is the recommended first step.  It writes a
``bridge_smoke_report.json`` with bytes/frame and a projected full-dataset size,
so we can decide whether materialising the whole RLDS copy is sensible before
spending tens of GiB of Drive storage.

OpenVLA's LIBERO standardisation later performs its usual gripper transform and
constructs proprio from state[:6] + state[-2:].  ``joint_state`` is present only
to match the upstream LIBERO RLDS feature schema; this LeRobot source does not
contain joint angles, so it is zero padding and is not consumed by the selected
OpenVLA-OFT LIBERO configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import tensorflow_datasets as tfds

DATASET_NAME = "parc_libero_selected"
VERSION = tfds.core.Version("1.0.0")
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 2 or data.get("group_aware") is not True:
        raise ValueError(f"manifest must be group-aware schema-v2: {path}")
    ids = data.get("episode_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("manifest episode_ids is empty")
    ids = [int(x) for x in ids]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest episode_ids contains duplicates")
    expected_hash = data.get("episode_ids_sha256")
    got_hash = _sha256_json(ids)
    if expected_hash and expected_hash != got_hash:
        raise ValueError(f"manifest episode_ids hash mismatch: {expected_hash} != {got_hash}")
    data["episode_ids"] = ids
    data["episode_ids_sha256"] = got_hash
    return data


def _load_info(root: Path) -> dict[str, Any]:
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected LeRobot v3.0 dataset, got {info.get('codebase_version')}")
    if int(info.get("fps", -1)) != 20:
        raise ValueError(f"expected 20 Hz source, got {info.get('fps')}")
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
            raise ValueError(f"expected 256x256 RGB for {key}, got {features[key]['shape']}")
    return info


def _load_tasks(root: Path) -> dict[int, str]:
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if "task_index" not in tasks.columns:
        raise KeyError("meta/tasks.parquet missing task_index")
    # LeRobot v3 stores the natural-language task as the dataframe index.
    return {int(row.task_index): str(idx) for idx, row in tasks.iterrows()}


def _load_episode_metadata(root: Path, episode_ids: list[int]) -> pd.DataFrame:
    files = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(root / "meta/episodes")
    ds = pads.dataset([str(p) for p in files], format="parquet")
    table = ds.to_table(filter=pads.field("episode_index").isin(episode_ids))
    df = table.to_pandas().sort_values("episode_index").reset_index(drop=True)
    found = set(int(x) for x in df["episode_index"].tolist())
    missing = sorted(set(episode_ids) - found)
    if missing:
        raise KeyError(f"episode metadata missing IDs: {missing[:20]}")
    return df.set_index("episode_index", drop=False)


def _data_path(root: Path, info: dict[str, Any], row: pd.Series) -> Path:
    rel = info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )
    return root / rel


def _video_path(root: Path, info: dict[str, Any], row: pd.Series, key: str) -> Path:
    rel = info["video_path"].format(
        video_key=key,
        chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]),
    )
    return root / rel


def _decode_video_targets(path: Path, targets_sec: np.ndarray, fps: float) -> np.ndarray:
    """Decode nearest RGB frames for monotonically increasing absolute video timestamps."""
    if not path.exists():
        raise FileNotFoundError(path)
    if len(targets_sec) == 0:
        return np.empty((0, 256, 256, 3), dtype=np.uint8)
    targets = np.asarray(targets_sec, dtype=np.float64)
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
        raise RuntimeError(f"no frames decoded from {path} for {targets[0]:.3f}..{targets[-1]:.3f}s")
    times = np.asarray(decoded_t)
    # Episode length is normally ~100-200 frames, so this small dense distance matrix is cheap.
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


def _stack_vector_column(series: pd.Series, width: int, name: str) -> np.ndarray:
    rows = [np.asarray(x, dtype=np.float32).reshape(-1) for x in series.tolist()]
    out = np.stack(rows)
    if out.ndim != 2 or out.shape[1] != width:
        raise ValueError(f"{name}: expected [T,{width}], got {out.shape}")
    return out


def _load_episode(
    root: Path,
    info: dict[str, Any],
    meta_row: pd.Series,
    task_lookup: dict[int, str],
) -> tuple[list[dict[str, Any]], int]:
    episode_index = int(meta_row["episode_index"])
    path = _data_path(root, info, meta_row)
    table = pq.read_table(
        path,
        columns=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "task_index"],
        filters=[("episode_index", "=", episode_index)],
    )
    df = table.to_pandas().sort_values("frame_index").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"episode {episode_index}: no rows in {path}")
    expected_len = int(meta_row["length"])
    if len(df) != expected_len:
        raise RuntimeError(f"episode {episode_index}: rows={len(df)} metadata length={expected_len}")

    state = _stack_vector_column(df[STATE_KEY], 8, STATE_KEY)
    action = _stack_vector_column(df[ACTION_KEY], 7, ACTION_KEY)
    local_ts = np.asarray(df["timestamp"], dtype=np.float64).reshape(-1)
    task_indices = np.asarray(df["task_index"], dtype=np.int64).reshape(-1)
    if len(set(task_indices.tolist())) != 1:
        raise RuntimeError(f"episode {episode_index}: multiple task_index values")
    task_index = int(task_indices[0])
    if task_index not in task_lookup:
        raise KeyError(f"task_index {task_index} missing from tasks.parquet")
    instruction = task_lookup[task_index]

    front_start = float(meta_row[f"videos/{FRONT_KEY}/from_timestamp"])
    wrist_start = float(meta_row[f"videos/{WRIST_KEY}/from_timestamp"])
    front = _decode_video_targets(
        _video_path(root, info, meta_row, FRONT_KEY), front_start + local_ts, float(info["fps"])
    )
    wrist = _decode_video_targets(
        _video_path(root, info, meta_row, WRIST_KEY), wrist_start + local_ts, float(info["fps"])
    )
    if len(front) != len(df) or len(wrist) != len(df):
        raise RuntimeError(f"episode {episode_index}: decoded video length mismatch")

    steps: list[dict[str, Any]] = []
    last = len(df) - 1
    for i in range(len(df)):
        steps.append(
            {
                "observation": {
                    "image": front[i],
                    "wrist_image": wrist[i],
                    "state": state[i],
                    # Upstream LIBERO RLDS exposes this key, but OpenVLA's selected
                    # LIBERO config ignores it.  Keep a deterministic padding value.
                    "joint_state": np.zeros((7,), dtype=np.float32),
                },
                "action": action[i],
                "language_instruction": instruction,
                "is_first": i == 0,
                "is_last": i == last,
                "is_terminal": i == last,
                "reward": np.float32(1.0 if i == last else 0.0),
                "discount": np.float32(1.0),
            }
        )
    return steps, len(df)


class ParcLiberoSelected(tfds.core.GeneratorBasedBuilder):
    VERSION = VERSION
    RELEASE_NOTES = {"1.0.0": "PARC2026 selected LeRobot episode pool bridge."}

    def __init__(
        self,
        *,
        source_root: Path,
        episode_ids: list[int],
        source_repo_id: str,
        source_revision: str,
        **kwargs: Any,
    ) -> None:
        self._source_root = Path(source_root)
        self._episode_ids = list(episode_ids)
        self._source_repo_id = source_repo_id
        self._source_revision = source_revision
        self._source_info = _load_info(self._source_root)
        self._task_lookup = _load_tasks(self._source_root)
        self._episode_meta = _load_episode_metadata(self._source_root, self._episode_ids)
        self.converted_frames = 0
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                                "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                                "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            },
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "language_instruction": tfds.features.Text(),
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "reward": np.float32,
                            "discount": np.float32,
                        }
                    ),
                    "episode_metadata": {"file_path": tfds.features.Text()},
                }
            ),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/lerobot/libero_plus",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _generate_examples(self) -> Iterable[tuple[str, dict[str, Any]]]:
        for pos, episode_id in enumerate(self._episode_ids, start=1):
            print(f"[bridge] episode {pos}/{len(self._episode_ids)} id={episode_id}", flush=True)
            row = self._episode_meta.loc[episode_id]
            steps, frames = _load_episode(self._source_root, self._source_info, row, self._task_lookup)
            self.converted_frames += frames
            yield str(episode_id), {
                "steps": steps,
                "episode_metadata": {
                    "file_path": f"lerobot://{self._source_repo_id}@{self._source_revision}/episode/{episode_id}"
                },
            }


def _directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--dataset-repo-id", default="lerobot/libero_plus")
    p.add_argument("--dataset-revision", default="f3f49f426d75030177b18778374005bc12ccd588")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = _load_manifest(args.manifest)
    all_ids = manifest["episode_ids"]
    selected_ids = all_ids[: args.max_episodes] if args.max_episodes else all_ids
    if not selected_ids:
        raise RuntimeError("no selected episodes")
    is_full = len(selected_ids) == len(all_ids)

    info = _load_info(args.lerobot_root)
    if manifest.get("dataset_id") and manifest["dataset_id"] != args.dataset_repo_id:
        raise ValueError(f"dataset id mismatch: {manifest['dataset_id']} != {args.dataset_repo_id}")
    if manifest.get("dataset_revision") and manifest["dataset_revision"] != args.dataset_revision:
        raise ValueError(
            f"dataset revision mismatch: {manifest['dataset_revision']} != {args.dataset_revision}"
        )

    # Keep smoke outputs separate from the persistent full-conversion root.
    args.out_root.mkdir(parents=True, exist_ok=True)
    generated_root = args.out_root / DATASET_NAME
    if generated_root.exists() and args.overwrite:
        shutil.rmtree(generated_root)

    builder = ParcLiberoSelected(
        source_root=args.lerobot_root,
        episode_ids=selected_ids,
        source_repo_id=args.dataset_repo_id,
        source_revision=args.dataset_revision,
        data_dir=str(args.out_root),
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(try_download_gcs=False),
    )
    prepared_dir = Path(builder.data_dir)
    prepared_bytes = _directory_size(prepared_dir)
    converted_frames = int(builder.converted_frames)
    if converted_frames <= 0:
        raise RuntimeError("converted_frames == 0")

    # Validate that the generated TFDS can be opened without the converter class.
    readonly = tfds.builder_from_directory(str(prepared_dir))
    first = next(iter(tfds.as_numpy(readonly.as_dataset(split="train").take(1))))
    if "steps" not in first or "episode_metadata" not in first:
        raise RuntimeError("builder_from_directory schema gate failed")
    step0 = next(iter(tfds.as_numpy(first["steps"].take(1)))) if hasattr(first["steps"], "take") else None
    # Nested DatasetFeature materialises as a Dataset in eager mode; if TFDS returns another
    # representation in a future version, the feature schema validation below still protects us.
    feature_text = json.dumps(readonly.info.features.to_json(), sort_keys=True)
    for token in ("wrist_image", "language_instruction", "joint_state"):
        if token not in feature_text:
            raise RuntimeError(f"prepared feature schema missing {token}")

    bpf = prepared_bytes / converted_frames
    projected_frames = int(manifest.get("summary", {}).get("frame_count", 0))
    projected_bytes = int(math.ceil(bpf * projected_frames)) if projected_frames else None
    report = {
        "schema_version": 1,
        "stage": "openvla_selected_rlds_bridge",
        "status": "FULL_CONVERSION_PASS" if is_full else "SMOKE_PASS",
        "selected_dataset_variant": manifest.get("variant"),
        "source_dataset_id": args.dataset_repo_id,
        "source_dataset_revision": args.dataset_revision,
        "source_manifest": str(args.manifest),
        "source_episode_ids_sha256": manifest["episode_ids_sha256"],
        "source_episode_count": len(all_ids),
        "source_frame_count": projected_frames,
        "converted_episode_count": len(selected_ids),
        "converted_episode_ids": selected_ids,
        "converted_frames": converted_frames,
        "prepared_dir": str(prepared_dir),
        "prepared_bytes": prepared_bytes,
        "bytes_per_frame": bpf,
        "projected_full_bytes": projected_bytes,
        "projected_full_gib": projected_bytes / 1024**3 if projected_bytes is not None else None,
        "tfds_builder_from_directory": "PASS",
        "raw_action_preserved": True,
        "raw_state_preserved": True,
        "no_noop_filter_applied": True,
        "joint_state": "zero_padding_unused_by_openvla_libero_config",
        "source_fps": info["fps"],
        "camera_mapping": {"image": FRONT_KEY, "wrist_image": WRIST_KEY},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)

    if is_full:
        contract_path = args.report.parent / "conversion_contract.json"
        contract = dict(report)
        contract.update(
            {
                "status": "PASS",
                "openvla_dataset_name": DATASET_NAME,
                "prepared_dir": str(prepared_dir),
                "converted_episode_ids_sha256": _sha256_json(selected_ids),
            }
        )
        if contract["converted_episode_ids_sha256"] != manifest["episode_ids_sha256"]:
            raise RuntimeError("full conversion episode hash mismatch")
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
        print(f"[bridge] full conversion contract: {contract_path}", flush=True)
    else:
        print("[bridge] smoke only: conversion_contract.json intentionally NOT written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
