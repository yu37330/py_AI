#!/usr/bin/env python3
"""Static episode-level quality analyzer for LeRobot LIBERO-style datasets.

This V1 intentionally stays simulator-free. It reads episode parquet files and
computes descriptive trajectory/action metrics plus task-relative review flags.
It does NOT infer task success, harmful collision, or replayability.

For the 8D LIBERO state layout we use the LeRobot convention:
  [eef_xyz(3), eef_axis_angle(3), gripper_qpos(2)]
Actions are expected to be 7D:
  [eef_delta_xyz(3), eef_delta_rot(3), gripper(1)]

The analyzer records this assumption in its summary and fails closed if the
required state/action shapes are not present.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7


@dataclass
class AnalyzerConfig:
    fps: float
    idle_motion_threshold: float = 0.02
    gripper_zero_threshold: float = 0.1
    smooth_window: int = 5
    robust_z_threshold: float = 5.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_dataset_root(root: Path) -> Path:
    if (root / "meta" / "info.json").exists():
        return root
    matches = list(root.glob("**/meta/info.json"))
    if len(matches) == 1:
        return matches[0].parent.parent
    if not matches:
        raise FileNotFoundError(f"meta/info.json not found under {root}")
    raise RuntimeError(
        "multiple dataset roots found: "
        + ", ".join(str(p.parent.parent) for p in matches[:10])
    )


def task_map_from_meta(meta: Path) -> dict[int, str]:
    path = meta / "tasks.jsonl"
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    for row in read_jsonl(path):
        idx = row.get("task_index")
        name = row.get("task") or row.get("task_name") or row.get("name")
        if idx is not None and name is not None:
            out[int(idx)] = str(name)
    return out


def episode_meta_map(meta: Path) -> dict[int, dict[str, Any]]:
    path = meta / "episodes.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {
        int(row.get("episode_index", i)): row
        for i, row in enumerate(rows)
    }


def moving_average_positions(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    cols = [np.convolve(padded[:, i], kernel, mode="valid") for i in range(x.shape[1])]
    return np.stack(cols, axis=1)


def rms(v: np.ndarray) -> float:
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(v))))


def vector_rms(rows: np.ndarray) -> float:
    if rows.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(np.square(rows), axis=1))))


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def jerk_norms(points: np.ndarray, dt: float) -> np.ndarray:
    if len(points) < 4 or dt <= 0:
        return np.empty(0, dtype=np.float64)
    jerk = np.diff(points, n=3, axis=0) / (dt ** 3)
    return np.linalg.norm(jerk, axis=1)


def gripper_switches(gripper_action: np.ndarray, zero_threshold: float) -> int:
    if len(gripper_action) < 2:
        return 0
    signs = np.zeros_like(gripper_action, dtype=np.int8)
    signs[gripper_action > zero_threshold] = 1
    signs[gripper_action < -zero_threshold] = -1
    active = signs[signs != 0]
    if len(active) < 2:
        return 0
    return int(np.count_nonzero(active[1:] != active[:-1]))


def safe_numeric_matrix(series: pd.Series, expected_dim: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(series.tolist(), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive for malformed parquet
        raise ValueError(f"{name}: cannot convert to numeric matrix") from exc
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"{name}: expected shape [T,{expected_dim}], got {arr.shape}")
    return arr


def compute_episode_metrics(
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray | None,
    frame_index: np.ndarray | None,
    cfg: AnalyzerConfig,
) -> dict[str, Any]:
    """Compute descriptive metrics for one episode.

    Exposed as a pure function so it can be regression-tested without parquet IO.
    """
    if state.ndim != 2 or state.shape[1] != LIBERO_STATE_DIM:
        raise ValueError(f"state must be [T,{LIBERO_STATE_DIM}], got {state.shape}")
    if action.ndim != 2 or action.shape[1] != LIBERO_ACTION_DIM:
        raise ValueError(f"action must be [T,{LIBERO_ACTION_DIM}], got {action.shape}")
    if len(state) != len(action):
        raise ValueError(f"state/action length mismatch: {len(state)} != {len(action)}")

    n = len(state)
    dt = 1.0 / cfg.fps
    eef = state[:, :3]
    axis_angle = state[:, 3:6]
    gripper_state = state[:, 6:8]
    motion_action = action[:, :6]
    xyz_action = action[:, :3]
    gripper_action = action[:, 6]

    finite_state = np.isfinite(state)
    finite_action = np.isfinite(action)
    invalid_value_count = int((~finite_state).sum() + (~finite_action).sum())

    # Replace non-finite values only for metric calculation; the episode remains flagged.
    state_calc = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    action_calc = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    eef = state_calc[:, :3]
    axis_angle = state_calc[:, 3:6]
    gripper_state = state_calc[:, 6:8]
    motion_action = action_calc[:, :6]
    xyz_action = action_calc[:, :3]
    gripper_action = action_calc[:, 6]

    eef_path = path_length(eef)
    displacement = float(np.linalg.norm(eef[-1] - eef[0])) if n else float("nan")
    efficiency = displacement / eef_path if eef_path > 1e-12 else float("nan")
    smooth_eef = moving_average_positions(eef, cfg.smooth_window)
    jerk_raw = jerk_norms(eef, dt)
    jerk_smooth = jerk_norms(smooth_eef, dt)

    motion_norm = np.linalg.norm(motion_action, axis=1) if n else np.empty(0)
    idle_ratio = (
        float(np.mean(motion_norm <= cfg.idle_motion_threshold)) if n else float("nan")
    )

    timestamp_nonmonotonic = 0
    timestamp_dt_mae = float("nan")
    if timestamp is not None and len(timestamp) == n and n >= 2:
        ts = np.asarray(timestamp, dtype=np.float64)
        dts = np.diff(ts)
        timestamp_nonmonotonic = int(np.count_nonzero(dts <= 0))
        timestamp_dt_mae = float(np.mean(np.abs(dts - dt)))

    frame_gap_count = 0
    if frame_index is not None and len(frame_index) == n and n >= 2:
        fi = np.asarray(frame_index, dtype=np.int64)
        frame_gap_count = int(np.count_nonzero(np.diff(fi) != 1))

    return {
        "frames": int(n),
        "duration_sec": float(n / cfg.fps),
        "eef_path_m": eef_path,
        "eef_displacement_m": displacement,
        "path_efficiency": efficiency,
        "eef_step_rms_m": vector_rms(np.diff(eef, axis=0)) if n >= 2 else 0.0,
        "axis_angle_path_l2": path_length(axis_angle),
        "gripper_state_path_l2": path_length(gripper_state),
        "rms_cart_jerk_raw": rms(jerk_raw),
        "max_cart_jerk_raw": float(np.max(jerk_raw)) if jerk_raw.size else float("nan"),
        "rms_cart_jerk_smooth": rms(jerk_smooth),
        "max_cart_jerk_smooth": float(np.max(jerk_smooth)) if jerk_smooth.size else float("nan"),
        "action_rms": vector_rms(action_calc),
        "motion_action_rms": vector_rms(motion_action),
        "xyz_action_rms": vector_rms(xyz_action),
        "action_max_abs": float(np.max(np.abs(action_calc))) if action_calc.size else float("nan"),
        "idle_ratio": idle_ratio,
        "gripper_switches": gripper_switches(gripper_action, cfg.gripper_zero_threshold),
        "timestamp_nonmonotonic_count": timestamp_nonmonotonic,
        "timestamp_dt_mae_sec": timestamp_dt_mae,
        "frame_gap_count": frame_gap_count,
        "invalid_value_count": invalid_value_count,
    }


def robust_z_abs(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        return pd.Series(np.zeros(len(x), dtype=np.float64), index=x.index)
    return 0.67448975 * (x - med).abs() / mad


def add_task_relative_flags(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    out = df.copy()
    metrics = [
        "duration_sec",
        "eef_path_m",
        "rms_cart_jerk_smooth",
        "idle_ratio",
        "motion_action_rms",
    ]
    flag_cols: list[str] = []
    for metric in metrics:
        z_col = f"rz_{metric}"
        flag_col = f"flag_{metric}_outlier"
        out[z_col] = out.groupby("task_index", dropna=False)[metric].transform(robust_z_abs)
        out[flag_col] = out[z_col] > z_threshold
        flag_cols.append(flag_col)

    out["flag_integrity"] = (
        (out["invalid_value_count"] > 0)
        | (out["timestamp_nonmonotonic_count"] > 0)
        | (out["frame_gap_count"] > 0)
    )
    flag_cols.append("flag_integrity")
    out["quality_flag_count"] = out[flag_cols].sum(axis=1).astype(int)
    out["quality_review_candidate"] = out["quality_flag_count"] > 0
    out["quality_status_v1"] = np.where(out["quality_review_candidate"], "REVIEW", "OK")
    return out


def iter_episode_parquet_paths(root: Path, info: dict[str, Any], episode_ids: Iterable[int]) -> Iterable[tuple[int, Path]]:
    pattern = info.get("data_path")
    chunk_size = int(info.get("chunks_size", 1000))
    if isinstance(pattern, str) and "{episode_index" in pattern:
        for idx in episode_ids:
            chunk = idx // chunk_size
            rel = pattern.format(episode_chunk=chunk, episode_index=idx)
            yield idx, root / rel
        return

    # Fallback for datasets with a different LeRobot layout.
    discovered: dict[int, Path] = {}
    for p in root.glob("data/**/*.parquet"):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            discovered[int(digits)] = p
    for idx in episode_ids:
        if idx in discovered:
            yield idx, discovered[idx]


def analyze_dataset(root: Path, out_dir: Path, cfg: AnalyzerConfig, max_episodes: int | None = None) -> dict[str, Any]:
    root = resolve_dataset_root(root)
    meta = root / "meta"
    info = read_json(meta / "info.json")
    tasks = task_map_from_meta(meta)
    episodes_meta = episode_meta_map(meta)

    episode_ids = sorted(episodes_meta) if episodes_meta else list(range(int(info.get("total_episodes", 0))))
    if max_episodes is not None:
        episode_ids = episode_ids[:max_episodes]

    rows: list[dict[str, Any]] = []
    missing_paths: list[str] = []

    for pos, (episode_index, parquet_path) in enumerate(iter_episode_parquet_paths(root, info, episode_ids), start=1):
        if not parquet_path.exists():
            missing_paths.append(str(parquet_path))
            continue
        frame = pd.read_parquet(parquet_path)
        required = {"observation.state", "action"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{parquet_path}: missing columns {sorted(missing)}")

        state = safe_numeric_matrix(frame["observation.state"], LIBERO_STATE_DIM, "observation.state")
        action = safe_numeric_matrix(frame["action"], LIBERO_ACTION_DIM, "action")
        timestamp = frame["timestamp"].to_numpy() if "timestamp" in frame else None
        frame_index = frame["frame_index"].to_numpy() if "frame_index" in frame else None
        metrics = compute_episode_metrics(state, action, timestamp, frame_index, cfg)

        ep_meta = episodes_meta.get(episode_index, {})
        if "task_index" in frame and len(frame):
            task_index = int(frame["task_index"].iloc[0])
        else:
            task_index = int(ep_meta.get("task_index", -1))
        row = {
            "episode_index": episode_index,
            "task_index": task_index,
            "task_name": tasks.get(task_index, f"task_index:{task_index}"),
            "parquet_path": str(parquet_path.relative_to(root)),
            **metrics,
        }
        rows.append(row)
        if pos % 500 == 0:
            print(f"processed {pos}/{len(episode_ids)} episodes")

    if not rows:
        sample = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            "no episode parquet files were analyzed. Download data/**/*.parquet first."
            + (f"\nmissing examples:\n{sample}" if sample else "")
        )

    df = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
    df = add_task_relative_flags(df, cfg.robust_z_threshold)
    review = df[df["quality_review_candidate"]].copy()

    task_summary = (
        df.groupby(["task_index", "task_name"], dropna=False)
        .agg(
            episodes=("episode_index", "count"),
            mean_duration_sec=("duration_sec", "mean"),
            median_eef_path_m=("eef_path_m", "median"),
            median_rms_cart_jerk_smooth=("rms_cart_jerk_smooth", "median"),
            median_idle_ratio=("idle_ratio", "median"),
            review_candidates=("quality_review_candidate", "sum"),
        )
        .reset_index()
    )
    task_summary["review_candidate_rate"] = task_summary["review_candidates"] / task_summary["episodes"]

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "episode_quality_metrics.csv"
    review_path = out_dir / "quality_review_candidates.csv"
    task_path = out_dir / "task_quality_summary.csv"
    summary_path = out_dir / "static_quality_summary.json"
    df.to_csv(metrics_path, index=False)
    review.to_csv(review_path, index=False)
    task_summary.to_csv(task_path, index=False)

    summary = {
        "dataset_root": str(root),
        "fps": cfg.fps,
        "episodes_requested": len(episode_ids),
        "episodes_analyzed": int(len(df)),
        "missing_episode_parquet_count": int(len(missing_paths)),
        "review_candidate_count": int(df["quality_review_candidate"].sum()),
        "review_candidate_rate": float(df["quality_review_candidate"].mean()),
        "config": asdict(cfg),
        "state_layout_assumption": {
            "observation.state": "LIBERO 8D = eef xyz(3) + eef axis-angle(3) + gripper qpos(2)",
            "action": "LIBERO 7D = eef delta pose(6) + gripper(1)",
        },
        "outputs": {
            "episode_quality_metrics": str(metrics_path),
            "quality_review_candidates": str(review_path),
            "task_quality_summary": str(task_path),
        },
        "limitations": [
            "V1 is static/descriptive only; it does not infer task success, harmful collision, or replayability",
            "jerk from 20 Hz position is noise-sensitive; raw and 5-frame-smoothed values are both retained",
            "task-relative robust-z flags create REVIEW candidates only; they are not automatic reject labels",
            "idle threshold is action-space descriptive and must not be treated as a universal quality threshold",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--idle-motion-threshold", type=float, default=0.02)
    ap.add_argument("--gripper-zero-threshold", type=float, default=0.1)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--robust-z-threshold", type=float, default=5.0)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    root = resolve_dataset_root(args.root)
    info = read_json(root / "meta" / "info.json")
    fps = float(args.fps if args.fps is not None else info.get("fps", 0))
    if fps <= 0:
        raise ValueError("fps must be provided or present in meta/info.json")
    cfg = AnalyzerConfig(
        fps=fps,
        idle_motion_threshold=args.idle_motion_threshold,
        gripper_zero_threshold=args.gripper_zero_threshold,
        smooth_window=args.smooth_window,
        robust_z_threshold=args.robust_z_threshold,
    )
    analyze_dataset(root, args.out, cfg, args.max_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
