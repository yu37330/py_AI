#!/usr/bin/env python3
"""Build deterministic episode manifests for PARC2026 dataset ablations.

Input is the episode-level CSV emitted by Static Quality Analyzer V1. The
script keeps filtering, held-out evaluation episodes, and task balancing as
separate axes so a cheap ablation can attribute changes correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "episode_index",
    "task_index",
    "task_name",
    "frames",
    "flag_integrity",
    "quality_flag_count",
    "quality_review_candidate",
}


def _json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.lower().map(mapping)
    if converted.isna().any():
        bad = sorted(series[converted.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"cannot parse boolean values: {bad}")
    return converted.astype(bool)


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    df = df.copy()
    df["episode_index"] = pd.to_numeric(df["episode_index"], errors="raise").astype(int)
    df["task_index"] = pd.to_numeric(df["task_index"], errors="raise").astype(int)
    df["frames"] = pd.to_numeric(df["frames"], errors="raise").astype(int)
    df["quality_flag_count"] = pd.to_numeric(df["quality_flag_count"], errors="raise").astype(int)
    df["flag_integrity"] = _as_bool(df["flag_integrity"])
    df["quality_review_candidate"] = _as_bool(df["quality_review_candidate"])
    if df["episode_index"].duplicated().any():
        dup = df.loc[df["episode_index"].duplicated(), "episode_index"].head().tolist()
        raise ValueError(f"duplicate episode_index values: {dup}")
    return df.sort_values("episode_index").reset_index(drop=True)


def build_eval_holdout(df: pd.DataFrame, per_task: int, seed: int) -> list[int]:
    """Pick a fixed OK-only task-stratified eval set shared by every variant."""
    if per_task <= 0:
        return []
    rng = np.random.default_rng(seed)
    eligible = df.loc[~df["quality_review_candidate"]]
    selected: list[int] = []
    for task_index, group in eligible.groupby("task_index", sort=True):
        ids = np.array(sorted(group["episode_index"].astype(int).tolist()), dtype=np.int64)
        if len(ids) < per_task:
            raise ValueError(
                f"task {task_index} has only {len(ids)} non-REVIEW episodes; cannot hold out {per_task}"
            )
        chosen = rng.choice(ids, size=per_task, replace=False)
        selected.extend(int(x) for x in sorted(chosen.tolist()))
    return sorted(selected)


def sqrt_balance_by_frames(df: pd.DataFrame, seed: int) -> tuple[list[int], list[dict[str, Any]]]:
    """Return deterministic sqrt task balancing based on frame exposure.

    LeRobot samples frames from the selected dataset, so frame counts are a
    closer proxy for task exposure than episode counts. The rarest task keeps
    all of its exposure; larger tasks are reduced toward sqrt exposure.
    """
    task_frames = df.groupby("task_index", sort=True)["frames"].sum().astype(int)
    if task_frames.empty:
        return [], []
    min_frames = int(task_frames.min())
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    task_rows: list[dict[str, Any]] = []

    for task_index, group in df.groupby("task_index", sort=True):
        raw_frames = int(group["frames"].sum())
        target_frames = float(np.sqrt(min_frames * raw_frames))
        shuffled = group.iloc[rng.permutation(len(group))]
        kept_ids: list[int] = []
        kept_frames = 0
        for row in shuffled.itertuples(index=False):
            if kept_frames >= target_frames:
                break
            kept_ids.append(int(row.episode_index))
            kept_frames += int(row.frames)
        selected.extend(kept_ids)
        task_rows.append(
            {
                "task_index": int(task_index),
                "task_name": str(group["task_name"].iloc[0]),
                "raw_episodes": int(len(group)),
                "selected_episodes": int(len(kept_ids)),
                "raw_frames": raw_frames,
                "target_frames": target_frames,
                "selected_frames": int(kept_frames),
            }
        )
    return sorted(selected), task_rows


def summarize_variant(df: pd.DataFrame, episode_ids: list[int]) -> dict[str, Any]:
    chosen = df[df["episode_index"].isin(episode_ids)]
    task = (
        chosen.groupby(["task_index", "task_name"], sort=True)
        .agg(episodes=("episode_index", "count"), frames=("frames", "sum"))
        .reset_index()
    )
    return {
        "episode_count": int(len(chosen)),
        "frame_count": int(chosen["frames"].sum()),
        "task_count": int(chosen["task_index"].nunique()),
        "task_distribution": task.to_dict(orient="records"),
    }


def build_manifests(
    metrics_csv: Path,
    out_dir: Path,
    *,
    dataset_id: str,
    dataset_revision: str | None,
    seed: int = 20260830,
    eval_per_task: int = 2,
) -> dict[str, Any]:
    df = load_metrics(metrics_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_ids = build_eval_holdout(df, eval_per_task, seed)
    eval_set = set(eval_ids)
    train_pool = df.loc[~df["episode_index"].isin(eval_set)].copy()

    variant_ids: dict[str, list[int]] = {
        "V0_RAW": sorted(train_pool["episode_index"].astype(int).tolist()),
        "V1_INTEGRITY_ONLY": sorted(
            train_pool.loc[~train_pool["flag_integrity"], "episode_index"].astype(int).tolist()
        ),
        "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL": sorted(
            train_pool.loc[train_pool["quality_flag_count"] < 2, "episode_index"].astype(int).tolist()
        ),
        "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL": sorted(
            train_pool.loc[~train_pool["quality_review_candidate"], "episode_index"].astype(int).tolist()
        ),
    }
    sqrt_ids, sqrt_task_rows = sqrt_balance_by_frames(train_pool, seed)
    variant_ids["V2_SQRT_BALANCED_RAW"] = sqrt_ids

    source_summary = {
        "episode_count": int(len(df)),
        "frame_count": int(df["frames"].sum()),
        "task_count": int(df["task_index"].nunique()),
        "review_count": int(df["quality_review_candidate"].sum()),
        "integrity_failure_count": int(df["flag_integrity"].sum()),
    }

    eval_summary = summarize_variant(df, eval_ids)
    eval_manifest = {
        "schema_version": 1,
        "kind": "fixed_eval_holdout",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "eval_per_task": eval_per_task,
        "episode_ids": eval_ids,
        "episode_ids_sha256": _json_sha256(eval_ids),
        "summary": eval_summary,
        "selection": "OK-only task-stratified holdout shared by all training variants",
    }
    (out_dir / "FIXED_EVAL_HOLDOUT.json").write_text(
        json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    variants_summary: dict[str, Any] = {}
    for name, ids in variant_ids.items():
        rule = {
            "V0_RAW": "all post-holdout episodes",
            "V1_INTEGRITY_ONLY": "drop hard integrity failures only",
            "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL": "drop quality_flag_count >= 2",
            "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL": "drop every Static Quality REVIEW candidate",
            "V2_SQRT_BALANCED_RAW": "post-holdout raw pool; deterministic sqrt task balancing by frame exposure",
        }[name]
        summary = summarize_variant(df, ids)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "training_episode_manifest",
            "variant": name,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "seed": seed,
            "source_metrics_csv": str(metrics_csv),
            "source_summary": source_summary,
            "fixed_eval_manifest": "FIXED_EVAL_HOLDOUT.json",
            "selection_rule": rule,
            "episode_ids": ids,
            "episode_ids_sha256": _json_sha256(ids),
            "summary": summary,
        }
        if name == "V2_SQRT_BALANCED_RAW":
            manifest["balancing"] = {
                "basis": "task frame exposure",
                "formula": "target_frames(task) = sqrt(min_task_frames * raw_task_frames)",
                "task_rows": sqrt_task_rows,
            }
        (out_dir / f"{name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        variants_summary[name] = summary

    cheap_order = ["V0_RAW"]
    skip_reason: dict[str, str] = {}
    if variant_ids["V1_INTEGRITY_ONLY"] == variant_ids["V0_RAW"]:
        skip_reason["V1_INTEGRITY_ONLY"] = "identical to V0 because integrity_failure_count == 0"
    else:
        cheap_order.append("V1_INTEGRITY_ONLY")
    cheap_order.extend(
        [
            "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL",
            "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL",
            "V2_SQRT_BALANCED_RAW",
        ]
    )

    run_matrix = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "source_summary": source_summary,
        "fixed_eval": eval_summary,
        "variants": variants_summary,
        "cheap_ablation_order": cheap_order,
        "skip_reason": skip_reason,
        "promotion_rule": "screening only; do not select final data from training loss alone",
    }
    (out_dir / "run_matrix.json").write_text(
        json.dumps(run_matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-id", default="lerobot/libero_plus")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--eval-per-task", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    matrix = build_manifests(
        args.metrics_csv,
        args.out,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        seed=args.seed,
        eval_per_task=args.eval_per_task,
    )
    print(json.dumps(matrix, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
