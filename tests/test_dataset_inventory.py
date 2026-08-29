"""Dataset Inventory V1 metadata parser tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def test_build_dataset_inventory_from_lerobot_metadata(tmp_path: Path):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)

    (meta / "info.json").write_text(
        json.dumps({"fps": 20, "total_frames": 60, "total_episodes": 3}),
        encoding="utf-8",
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": 0, "task": "put bowl on plate"},
            {"task_index": 1, "task": "open drawer"},
        ],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["put bowl on plate"], "length": 20},
            {"episode_index": 1, "task_index": 0, "length": 10},
            {"episode_index": 2, "task_index": 1, "length": 30},
        ],
    )

    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(_ROOT / "tools/data/build_dataset_inventory.py"),
            "--root",
            str(dataset),
            "--out",
            str(out),
        ],
        check=True,
    )

    episodes = pd.read_csv(out / "episode_inventory.csv")
    tasks = pd.read_csv(out / "task_inventory.csv")
    summary = json.loads((out / "dataset_inventory_summary.json").read_text())

    assert len(episodes) == 3
    assert episodes["frames"].sum() == 60
    assert summary["fps"] == 20
    assert summary["tasks_from_metadata"] == 2

    by_name = tasks.set_index("task_name")
    assert by_name.loc["put bowl on plate", "episodes"] == 2
    assert by_name.loc["put bowl on plate", "frames"] == 30
    assert by_name.loc["open drawer", "episodes"] == 1
    assert by_name.loc["open drawer", "avg_duration_sec_approx"] == 1.5
