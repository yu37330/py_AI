#!/usr/bin/env python3
"""Build episode/task inventory CSVs from a LeRobot-style dataset metadata tree.

The analyzer intentionally starts from metadata only. It does not assume that the
converted LeRobot dataset contains enough simulator state for true replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


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
    direct = root / "meta" / "info.json"
    if direct.exists():
        return root
    matches = list(root.glob("**/meta/info.json"))
    if len(matches) == 1:
        return matches[0].parent.parent
    if not matches:
        raise FileNotFoundError(f"meta/info.json not found under {root}")
    raise RuntimeError(
        f"multiple dataset roots found under {root}: "
        + ", ".join(str(p.parent.parent) for p in matches[:10])
    )


def get_fps(info: dict[str, Any]) -> float | None:
    for key in ("fps", "frequency_hz", "frequency"):
        value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def build_task_maps(meta: Path) -> tuple[dict[int, str], list[dict[str, Any]]]:
    path = meta / "tasks.jsonl"
    if not path.exists():
        return {}, []
    rows = read_jsonl(path)
    task_map: dict[int, str] = {}
    for row in rows:
        idx = row.get("task_index")
        name = row.get("task") or row.get("task_name") or row.get("name")
        if idx is not None and name is not None:
            task_map[int(idx)] = str(name)
    return task_map, rows


def normalize_tasks(ep: dict[str, Any], task_map: dict[int, str]) -> list[str]:
    raw_tasks = ep.get("tasks")
    if isinstance(raw_tasks, str):
        return [raw_tasks]
    if isinstance(raw_tasks, list):
        return [str(x) for x in raw_tasks]

    for key in ("task", "task_name"):
        if ep.get(key) is not None:
            return [str(ep[key])]

    idx = ep.get("task_index")
    if idx is not None:
        idx_i = int(idx)
        return [task_map.get(idx_i, f"task_index:{idx_i}")]

    indices = ep.get("task_indices")
    if isinstance(indices, list):
        return [task_map.get(int(i), f"task_index:{int(i)}") for i in indices]

    return ["unknown"]


def infer_source(ep: dict[str, Any]) -> str:
    for key in ("source", "dataset", "suite", "dataset_source", "origin"):
        if ep.get(key) is not None:
            return str(ep[key])
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    root = resolve_dataset_root(args.root)
    meta = root / "meta"
    info = read_json(meta / "info.json")
    fps = get_fps(info)

    task_map, task_meta_rows = build_task_maps(meta)
    episodes_path = meta / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"missing {episodes_path}")
    episodes = read_jsonl(episodes_path)

    episode_rows: list[dict[str, Any]] = []
    exploded: list[dict[str, Any]] = []

    for fallback_index, ep in enumerate(episodes):
        episode_index = int(ep.get("episode_index", fallback_index))
        length_raw = ep.get("length", ep.get("num_frames", ep.get("frames")))
        frames = int(length_raw) if length_raw is not None else None
        duration = (frames / fps) if (frames is not None and fps) else None
        tasks = normalize_tasks(ep, task_map)
        source = infer_source(ep)

        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": json.dumps(tasks, ensure_ascii=False),
                "task_count": len(tasks),
                "frames": frames,
                "duration_sec_approx": duration,
                "source": source,
            }
        )
        for task_name in tasks:
            exploded.append(
                {
                    "episode_index": episode_index,
                    "task_name": task_name,
                    "frames": frames,
                    "duration_sec_approx": duration,
                    "source": source,
                }
            )

    ep_df = pd.DataFrame(episode_rows).sort_values("episode_index")
    ex_df = pd.DataFrame(exploded)

    if ex_df.empty:
        task_df = pd.DataFrame(
            columns=[
                "task_name",
                "episodes",
                "frames",
                "avg_frames",
                "avg_duration_sec_approx",
                "sources",
            ]
        )
    else:
        grouped = ex_df.groupby("task_name", dropna=False)
        task_df = grouped.agg(
            episodes=("episode_index", "nunique"),
            frames=("frames", "sum"),
            avg_frames=("frames", "mean"),
            avg_duration_sec_approx=("duration_sec_approx", "mean"),
        ).reset_index()
        sources = (
            grouped["source"]
            .apply(lambda s: json.dumps(sorted(set(map(str, s))), ensure_ascii=False))
            .reset_index(name="sources")
        )
        task_df = task_df.merge(sources, on="task_name", how="left")
        task_df = task_df.sort_values(["episodes", "task_name"], ascending=[False, True])

    args.out.mkdir(parents=True, exist_ok=True)
    ep_path = args.out / "episode_inventory.csv"
    task_path = args.out / "task_inventory.csv"
    summary_path = args.out / "dataset_inventory_summary.json"
    task_meta_path = args.out / "task_metadata_raw.json"

    ep_df.to_csv(ep_path, index=False)
    task_df.to_csv(task_path, index=False)
    task_meta_path.write_text(
        json.dumps(task_meta_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    observed_frames = None
    for key in ("total_frames", "num_frames", "frames"):
        if info.get(key) is not None:
            observed_frames = info[key]
            break
    observed_episodes = None
    for key in ("total_episodes", "num_episodes", "episodes"):
        if info.get(key) is not None:
            observed_episodes = info[key]
            break

    summary = {
        "dataset_root": str(root),
        "fps": fps,
        "info_reported_frames": observed_frames,
        "info_reported_episodes": observed_episodes,
        "episodes_from_metadata": int(len(ep_df)),
        "tasks_from_metadata": int(task_df["task_name"].nunique()) if len(task_df) else 0,
        "frames_from_episode_lengths": int(ep_df["frames"].sum())
        if len(ep_df) and ep_df["frames"].notna().any()
        else None,
        "source_values": sorted(ep_df["source"].dropna().astype(str).unique().tolist())
        if len(ep_df)
        else [],
        "outputs": {
            "episode_inventory": str(ep_path),
            "task_inventory": str(task_path),
            "task_metadata_raw": str(task_meta_path),
        },
        "notes": [
            "duration_sec_approx is frames/fps, intended for inventory/relative comparison",
            "source remains unknown unless the converted dataset metadata carries origin labels",
            "success/collision/replayability require simulator-capable source data and are not inferred here",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop tasks by episode count:")
    if len(task_df):
        print(task_df.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
