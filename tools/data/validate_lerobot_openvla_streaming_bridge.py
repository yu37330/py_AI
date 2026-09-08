#!/usr/bin/env python3
"""Validate the 69c selected-pool LeRobot -> OpenVLA streaming bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openvla_lerobot_streaming import (
    ABSOLUTE_ACTION_MASK,
    ACTION_CHUNK,
    ACTION_DIM,
    ACTION_KEY,
    ACTION_NORMALIZATION_MASK,
    DATASET_ID,
    DATASET_REVISION,
    NORMALIZATION_TYPE,
    PROPRIO_DIM,
    STATE_KEY,
    VARIANT,
    iter_selected_trajectories,
    load_manifest,
    load_source_info,
    standardize_action,
    standardize_proprio,
)

OPENVLA_OFT_REPO = "small-zeng/openvla-oft"
OPENVLA_OFT_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
EXPECTED_MANIFEST_SHA256 = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"invalid statistics input: shape={x.shape}")
    return {
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "min": x.min(axis=0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def _validate_69b(
    report: dict[str, Any],
    capacity: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    digest = manifest["episode_ids_sha256"]
    if report.get("status") != "SMOKE_PASS":
        raise ValueError("69b smoke report is not PASS")
    if report.get("selected_dataset_variant") != VARIANT:
        raise ValueError("69b variant mismatch")
    if report.get("source_dataset_id") != DATASET_ID or report.get("source_dataset_revision") != DATASET_REVISION:
        raise ValueError("69b source provenance mismatch")
    if report.get("source_episode_ids_sha256") != digest:
        raise ValueError("69b manifest hash mismatch")
    if report.get("converted_episode_count") != 8 or report.get("converted_frames", 0) <= 0:
        raise ValueError("69b reference sample is not the required 8-episode smoke")
    if report.get("tfds_builder_from_directory") != "PASS":
        raise ValueError("69b TFDS readback gate failed")
    for key in ("raw_action_preserved", "raw_state_preserved", "no_noop_filter_applied"):
        if report.get(key) is not True:
            raise ValueError(f"69b {key} gate failed")

    if capacity.get("status") != "PASS" or capacity.get("source_episode_ids_sha256") != digest:
        raise ValueError("69b capacity decision provenance mismatch")
    if capacity.get("decision") != "STREAMING_BRIDGE_RECOMMENDED":
        raise ValueError("69b did not select the streaming bridge path")
    if float(capacity.get("projected_full_gib", 0.0)) <= float(
        capacity.get("materialize_threshold_gib", 35.0)
    ):
        raise ValueError("streaming decision is inconsistent with capacity projection")


def _scan_selected_pool(
    source_root: Path,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int, set[int], set[int]]:
    import pyarrow.dataset as pads

    files = sorted((Path(source_root) / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(Path(source_root) / "data")
    selected = np.asarray(manifest["episode_ids"], dtype=np.int64)
    ds = pads.dataset([str(p) for p in files], format="parquet")
    table = ds.to_table(
        columns=[ACTION_KEY, STATE_KEY, "episode_index", "task_index"],
        filter=pads.field("episode_index").isin(selected.tolist()),
    )
    df = table.to_pandas()
    if df.empty:
        raise RuntimeError("selected-pool parquet scan returned no rows")

    raw_actions = np.stack(
        [np.asarray(x, dtype=np.float32).reshape(-1) for x in df[ACTION_KEY].tolist()]
    )
    raw_states = np.stack(
        [np.asarray(x, dtype=np.float32).reshape(-1) for x in df[STATE_KEY].tolist()]
    )
    if raw_actions.shape[1] != ACTION_DIM or raw_states.shape[1] != PROPRIO_DIM:
        raise ValueError(
            f"source shape mismatch actions={raw_actions.shape} states={raw_states.shape}"
        )
    actions = standardize_action(raw_actions)
    proprios = standardize_proprio(raw_states)
    episode_ids = set(int(x) for x in df["episode_index"].tolist())
    task_ids = set(int(x) for x in df["task_index"].tolist())
    return actions, proprios, len(df), episode_ids, task_ids


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--69b-report", dest="report69b", type=Path, required=True)
    p.add_argument("--69b-capacity", dest="capacity69b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    if manifest["episode_ids_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("selected manifest hash is not the fixed D10 recovery identity")
    load_source_info(args.lerobot_root)

    report69b = _read_json(args.report69b)
    capacity69b = _read_json(args.capacity69b)
    _validate_69b(report69b, capacity69b, manifest)
    print("[1/3] 69b provenance + streaming capacity decision PASS", flush=True)

    print("[2/3] scanning exact 10,758-episode selected pool for OpenVLA statistics...", flush=True)
    actions, proprios, row_count, episode_ids, task_ids = _scan_selected_pool(
        args.lerobot_root, manifest
    )
    expected_frames = int(manifest["summary"]["frame_count"])
    if row_count != expected_frames:
        raise RuntimeError(f"selected-pool rows={row_count} expected={expected_frames}")
    if episode_ids != set(manifest["episode_ids"]):
        raise RuntimeError("selected-pool episode identity mismatch")
    print(f"[2/3] selected pool PASS episodes={len(episode_ids)} frames={row_count}", flush=True)

    # Validate the exact 69b reference sample through the direct streaming path.
    sample_ids = list(report69b["converted_episode_ids"])
    if sample_ids != manifest["episode_ids"][:8]:
        raise RuntimeError("69b sample IDs do not match first 8 selected episodes")
    sample_frames = 0
    sample_languages: set[str] = set()
    sample_window_count = 0
    sample_image_shapes: set[tuple[int, ...]] = set()
    sample_wrist_shapes: set[tuple[int, ...]] = set()
    action_stats = _stats(actions)
    proprio_stats = _stats(proprios)
    statistics = {
        "action": action_stats,
        "proprio": proprio_stats,
        "num_transitions": row_count,
        "num_trajectories": len(episode_ids),
    }

    from openvla_lerobot_streaming import iter_openvla_windows

    print("[3/3] validating 69b reference episodes through direct streaming...", flush=True)
    for traj in iter_selected_trajectories(
        args.lerobot_root, manifest, max_episodes=8
    ):
        print(f"[stream] episode={traj.episode_id} frames={len(traj.action)}", flush=True)
        if traj.episode_id not in sample_ids:
            raise RuntimeError(f"unexpected sample episode {traj.episode_id}")
        sample_frames += len(traj.action)
        sample_languages.add(traj.language_instruction)
        sample_image_shapes.add(tuple(traj.image_primary.shape[1:]))
        sample_wrist_shapes.add(tuple(traj.image_wrist.shape[1:]))
        first_window = next(iter_openvla_windows(traj, statistics))
        if first_window["action"].shape != (ACTION_CHUNK, ACTION_DIM):
            raise RuntimeError("OpenVLA action chunk shape mismatch")
        if first_window["observation"]["proprio"].shape != (PROPRIO_DIM,):
            raise RuntimeError("OpenVLA proprio shape mismatch")
        sample_window_count += len(traj.action)

    if sample_frames != int(report69b["converted_frames"]):
        raise RuntimeError(
            f"streaming sample frames={sample_frames} != 69b={report69b['converted_frames']}"
        )
    if sample_window_count != sample_frames:
        raise RuntimeError("streaming window count mismatch")
    if sample_image_shapes != {(256, 256, 3)} or sample_wrist_shapes != {(256, 256, 3)}:
        raise RuntimeError("streaming image shape gate failed")
    if not sample_languages:
        raise RuntimeError("streaming language gate failed")

    contract = {
        "schema_version": 1,
        "stage": "openvla_selected_streaming_bridge",
        "status": "PASS",
        "bridge_type": "lerobot_streaming",
        "selected_dataset_variant": VARIANT,
        "source_dataset_id": DATASET_ID,
        "source_dataset_revision": DATASET_REVISION,
        "source_manifest": str(args.manifest),
        "source_episode_ids_sha256": manifest["episode_ids_sha256"],
        "source_episode_count": len(manifest["episode_ids"]),
        "source_frame_count": row_count,
        "source_task_count_observed": len(task_ids),
        "openvla_oft_repo": OPENVLA_OFT_REPO,
        "openvla_oft_revision": OPENVLA_OFT_REVISION,
        "openvla_contract": {
            "action_dim": ACTION_DIM,
            "proprio_dim": PROPRIO_DIM,
            "action_chunk": ACTION_CHUNK,
            "normalization_type": NORMALIZATION_TYPE,
            "absolute_action_mask": ABSOLUTE_ACTION_MASK,
            "action_normalization_mask": ACTION_NORMALIZATION_MASK,
            "camera_mapping": {
                "image_primary": "observation.images.front",
                "image_wrist": "observation.images.wrist",
            },
            "proprio_transform": "concat(state[:6], state[-2:])",
            "gripper_transform": "1 - clip(raw_action[6], 0, 1)",
            "future_padding": "relative_dims_zero_absolute_gripper_repeat_last",
        },
        "dataset_statistics": statistics,
        "equivalence_evidence": {
            "reference_stage": "69b",
            "reference_attempt_id": capacity69b.get("attempt_id"),
            "reference_status": report69b["status"],
            "reference_tfds_builder_from_directory": report69b[
                "tfds_builder_from_directory"
            ],
            "reference_raw_action_preserved": report69b["raw_action_preserved"],
            "reference_raw_state_preserved": report69b["raw_state_preserved"],
            "reference_no_noop_filter_applied": report69b[
                "no_noop_filter_applied"
            ],
            "sample_episode_ids": sample_ids,
            "sample_episode_count": len(sample_ids),
            "sample_frames": sample_frames,
            "streaming_image_shape": [256, 256, 3],
            "streaming_window_count": sample_window_count,
        },
        "storage_policy": {
            "full_rlds_materialized": False,
            "projected_full_gib": capacity69b["projected_full_gib"],
            "materialize_threshold_gib": capacity69b["materialize_threshold_gib"],
            "decision": capacity69b["decision"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
    print("=== 69c COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
