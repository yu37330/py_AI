from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools" / "data"))

from static_quality_analyzer import (  # noqa: E402
    AnalyzerConfig,
    add_task_relative_flags,
    analyze_dataset,
    compute_episode_metrics,
    gripper_switches,
    task_map_from_meta,
)


def test_compute_episode_metrics_for_constant_velocity_xyz():
    fps = 20.0
    n = 10
    t = np.arange(n) / fps
    state = np.zeros((n, 8), dtype=np.float64)
    state[:, 0] = 0.1 * t
    state[:, 6] = 0.02
    state[:, 7] = -0.02
    action = np.zeros((n, 7), dtype=np.float64)
    action[:, 0] = 0.1
    action[:, 6] = -1.0
    frame_index = np.arange(n)

    got = compute_episode_metrics(
        state,
        action,
        t,
        frame_index,
        AnalyzerConfig(fps=fps, idle_motion_threshold=0.02),
    )

    assert got["frames"] == n
    assert got["duration_sec"] == n / fps
    assert np.isclose(got["eef_path_m"], state[-1, 0] - state[0, 0])
    assert np.isclose(got["eef_displacement_m"], got["eef_path_m"])
    assert np.isclose(got["path_efficiency"], 1.0)
    assert got["rms_cart_jerk_raw"] < 1e-8
    assert got["idle_ratio"] == 0.0
    assert got["timestamp_nonmonotonic_count"] == 0
    assert got["frame_gap_count"] == 0
    assert got["invalid_value_count"] == 0


def test_integrity_and_gripper_switch_detection():
    assert gripper_switches(np.array([-1, -1, 0, 1, 1, -1], dtype=float), 0.1) == 2

    state = np.zeros((5, 8), dtype=np.float64)
    state[2, 0] = np.nan
    action = np.zeros((5, 7), dtype=np.float64)
    ts = np.array([0.0, 0.05, 0.05, 0.15, 0.20])
    fi = np.array([0, 1, 3, 4, 5])
    got = compute_episode_metrics(state, action, ts, fi, AnalyzerConfig(fps=20.0))
    assert got["invalid_value_count"] == 1
    assert got["timestamp_nonmonotonic_count"] == 1
    assert got["frame_gap_count"] == 1


def test_task_relative_flags_mark_extreme_episode_for_review():
    rows = []
    for i in range(20):
        rows.append(
            {
                "episode_index": i,
                "task_index": 0,
                "duration_sec": 5.0 + (i % 3) * 0.02,
                "eef_path_m": 0.5 + (i % 4) * 0.005,
                "rms_cart_jerk_smooth": 1.0 + (i % 5) * 0.01,
                "idle_ratio": 0.1 + (i % 2) * 0.005,
                "motion_action_rms": 0.2 + (i % 3) * 0.002,
                "invalid_value_count": 0,
                "timestamp_nonmonotonic_count": 0,
                "frame_gap_count": 0,
            }
        )
    rows.append(
        {
            "episode_index": 99,
            "task_index": 0,
            "duration_sec": 30.0,
            "eef_path_m": 5.0,
            "rms_cart_jerk_smooth": 50.0,
            "idle_ratio": 0.95,
            "motion_action_rms": 3.0,
            "invalid_value_count": 0,
            "timestamp_nonmonotonic_count": 0,
            "frame_gap_count": 0,
        }
    )
    flagged = add_task_relative_flags(pd.DataFrame(rows), z_threshold=5.0)
    extreme = flagged.loc[flagged["episode_index"] == 99].iloc[0]
    assert bool(extreme["quality_review_candidate"])
    assert extreme["quality_status_v1"] == "REVIEW"
    assert int(extreme["quality_flag_count"]) >= 3


def test_integrity_issue_always_creates_review_candidate():
    df = pd.DataFrame(
        [
            {
                "episode_index": 1,
                "task_index": 0,
                "duration_sec": 5.0,
                "eef_path_m": 0.5,
                "rms_cart_jerk_smooth": 1.0,
                "idle_ratio": 0.1,
                "motion_action_rms": 0.2,
                "invalid_value_count": 1,
                "timestamp_nonmonotonic_count": 0,
                "frame_gap_count": 0,
            }
        ]
    )
    flagged = add_task_relative_flags(df, z_threshold=5.0)
    assert bool(flagged.iloc[0]["flag_integrity"])
    assert bool(flagged.iloc[0]["quality_review_candidate"])


def test_task_map_supports_v3_tasks_parquet(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {"task_index": [0, 1], "task": ["task zero", "task one"]}
    ).to_parquet(meta / "tasks.parquet", index=False)
    assert task_map_from_meta(meta) == {0: "task zero", 1: "task one"}


def test_analyze_dataset_supports_multi_episode_v3_parquet(tmp_path: Path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)

    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 2,
                "total_frames": 10,
                "total_tasks": 2,
                "fps": 20,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        )
    )
    pd.DataFrame(
        {"task_index": [0, 1], "task": ["task zero", "task one"]}
    ).to_parquet(meta / "tasks.parquet", index=False)

    rows = []
    for episode_index, task_index in [(0, 0), (1, 1)]:
        for frame_index in range(5):
            state = np.zeros(8, dtype=np.float32)
            state[0] = episode_index + frame_index * 0.01
            action = np.zeros(7, dtype=np.float32)
            action[0] = 0.1
            rows.append(
                {
                    "observation.state": state.tolist(),
                    "action": action.tolist(),
                    "timestamp": frame_index / 20.0,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "task_index": task_index,
                }
            )
    pd.DataFrame(rows).to_parquet(data / "file-000.parquet", index=False)

    out = tmp_path / "out"
    summary = analyze_dataset(root, out, AnalyzerConfig(fps=20.0))
    metrics = pd.read_csv(out / "episode_quality_metrics.csv")

    assert summary["episodes_analyzed"] == 2
    assert summary["missing_episode_count"] == 0
    assert summary["data_parquet_file_count"] == 1
    assert metrics["episode_index"].tolist() == [0, 1]
    assert metrics["task_name"].tolist() == ["task zero", "task one"]
