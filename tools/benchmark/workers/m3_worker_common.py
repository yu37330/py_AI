#!/usr/bin/env python3
"""Shared guarded-worker utilities for PARC2026 M3."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from tools.benchmark.m3_runner_core import (
    D10_HASH,
    D10_VARIANT,
    EQUAL_DATA_BUDGET,
    EQUAL_DATA_OPTIMIZER_UPDATES,
    EQUAL_WALL_SEC,
    require_execution_guard,
)

SMOKE_SAMPLE_TARGET = 64
SMOKE_OPTIMIZER_TARGET = 2


def require_a100() -> tuple[str, int]:
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    total = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    if "A100" not in name or total < 38000:
        raise RuntimeError(f"M3 training worker requires NVIDIA A100 with >=38 GiB; got {name} {total} MiB")
    return name, total


def load_adapter_plan(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "READY_FOR_GUARDED_SMOKE_IMPLEMENTATION":
        raise ValueError(f"adapter preflight not ready: {data.get('status')}")
    if data.get("selected_dataset_variant") != D10_VARIANT:
        raise ValueError("adapter plan dataset variant mismatch")
    if data.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("adapter plan D10 hash mismatch")
    if data.get("training_started") is not False:
        raise ValueError("adapter plan unexpectedly claims training started")
    if data.get("automatic_benchmark_start") is not False:
        raise ValueError("adapter plan permits automatic benchmark start")
    return data


def select_run_spec(
    plan: dict[str, Any], *, model: str, order: str, track: str
) -> dict[str, Any]:
    matches = [
        row
        for row in plan.get("runs", [])
        if row.get("model") == model and row.get("order") == order and row.get("track") == track
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one adapter run spec for {order}/{track}/{model}, got {len(matches)}")
    row = dict(matches[0])
    if int(row.get("effective_batch_size", -1)) != 32:
        raise ValueError("adapter run effective batch drift")
    if int(row["micro_batch"]) * int(row["gradient_accumulation"]) != 32:
        raise ValueError("adapter run micro/GA mismatch")
    return row


def build_runtime_spec(
    *,
    plan: dict[str, Any],
    run: dict[str, Any],
    mode: str,
    manifest_path: Path,
    dataset_root: Path,
    evidence_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    require_execution_guard(mode)
    if mode not in {"smoke", "benchmark"}:
        raise ValueError(mode)
    track = run["track"]
    if mode == "smoke" and track != "equal_data":
        raise ValueError("guarded smoke uses the equal-data canonical prefix only")

    if mode == "smoke":
        sample_target = SMOKE_SAMPLE_TARGET
        optimizer_target = SMOKE_OPTIMIZER_TARGET
        train_loop_sec = None
    elif track == "equal_data":
        sample_target = EQUAL_DATA_BUDGET
        optimizer_target = EQUAL_DATA_OPTIMIZER_UPDATES
        train_loop_sec = None
    else:
        # Ignored by equal-wall evidence, but kept positive for one common schema.
        sample_target = 32
        optimizer_target = 1
        train_loop_sec = EQUAL_WALL_SEC

    schedule_sha = run["schedule_sha256"]
    spec = {
        "schema_version": 1,
        "stage": "M3_guarded_worker_runtime",
        "mode": mode,
        "model": run["model"],
        "order": run["order"],
        "track": track,
        "seed": int(run["seed"]),
        "source_ref": run["source_ref"],
        "micro_batch": int(run["micro_batch"]),
        "gradient_accumulation": int(run["gradient_accumulation"]),
        "effective_batch_size": int(run["effective_batch_size"]),
        "schedule_sha256": schedule_sha,
        "equal_data_schedule": plan["equal_data_schedule"] if track == "equal_data" else None,
        "manifest_path": str(Path(manifest_path)),
        "dataset_root": str(Path(dataset_root)),
        "result_path": str(Path(result_path)),
        "evidence_path": str(Path(evidence_path)),
        "sample_target": sample_target,
        "optimizer_target": optimizer_target,
        "train_loop_sec": train_loop_sec,
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_HASH,
        "execution_guard_verified": os.environ.get("PARC_M3_EXECUTE") == "1",
    }
    return spec


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest_episode_ids(path: Path) -> list[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("variant") != D10_VARIANT or data.get("episode_ids_sha256") != D10_HASH:
        raise ValueError("worker manifest is not exact frozen D10")
    ids = data.get("episode_ids")
    if not isinstance(ids, list) or len(ids) != 10758:
        raise ValueError("worker manifest must contain exactly 10,758 D10 episodes")
    return [int(x) for x in ids]
