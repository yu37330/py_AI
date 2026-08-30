#!/usr/bin/env python3
"""Build group-aware dataset ablation manifests.

This consumes:
- Static Quality Analyzer episode metrics
- trajectory_group_members.csv from check_trajectory_group_leakage.py

The fixed eval split selects distinct exact trajectory groups per task and uses
one representative OK episode from each group. Every episode belonging to the
selected eval groups is removed from every training variant, eliminating
non-visual trajectory sibling leakage while keeping eval size compact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_METRICS = {
    "episode_index",
    "task_index",
    "task_name",
    "frames",
    "flag_integrity",
    "quality_flag_count",
    "quality_review_candidate",
}
REQUIRED_GROUPS = {
    "episode_index",
    "task_index",
    "exact_group_hash",
    "exact_group_id",
}


def json_sha256(obj: Any) -> str:
    payload = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.lower().map(mapping)
    if converted.isna().any():
        bad = sorted(series[converted.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"cannot parse boolean values: {bad}")
    return converted.astype(bool)


def load_inputs(metrics_csv: Path, groups_csv: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_csv)
    missing = REQUIRED_METRICS.difference(metrics.columns)
    if missing:
        raise KeyError(f"{metrics_csv}: missing columns {sorted(missing)}")
    metrics = metrics.copy()
    metrics["episode_index"] = pd.to_numeric(
        metrics["episode_index"], errors="raise"
    ).astype(int)
    metrics["task_index"] = pd.to_numeric(metrics["task_index"], errors="raise").astype(int)
    metrics["frames"] = pd.to_numeric(metrics["frames"], errors="raise").astype(int)
    metrics["quality_flag_count"] = pd.to_numeric(
        metrics["quality_flag_count"], errors="raise"
    ).astype(int)
    metrics["flag_integrity"] = as_bool(metrics["flag_integrity"])
    metrics["quality_review_candidate"] = as_bool(metrics["quality_review_candidate"])
    if metrics["episode_index"].duplicated().any():
        raise ValueError("duplicate episode_index in metrics")

    groups = pd.read_csv(groups_csv)
    missing = REQUIRED_GROUPS.difference(groups.columns)
    if missing:
        raise KeyError(f"{groups_csv}: missing columns {sorted(missing)}")
    groups = groups[list(REQUIRED_GROUPS)].copy()
    groups["episode_index"] = pd.to_numeric(groups["episode_index"], errors="raise").astype(int)
    groups["task_index_group"] = pd.to_numeric(
        groups.pop("task_index"), errors="raise"
    ).astype(int)
    if groups["episode_index"].duplicated().any():
        raise ValueError("duplicate episode_index in trajectory groups")

    merged = metrics.merge(groups, on="episode_index", how="left", validate="one_to_one")
    if merged["exact_group_hash"].isna().any():
        missing_ids = merged.loc[
            merged["exact_group_hash"].isna(), "episode_index"
        ].head(10).tolist()
        raise KeyError(f"metrics episodes missing trajectory groups: {missing_ids}")
    mismatch = merged["task_index"] != merged["task_index_group"]
    if mismatch.any():
        bad = merged.loc[mismatch, ["episode_index", "task_index", "task_index_group"]].head(10)
        raise ValueError(f"task_index mismatch between metrics/groups:\n{bad}")
    return merged.drop(columns=["task_index_group"]).sort_values("episode_index").reset_index(drop=True)


def summarize(df: pd.DataFrame, episode_ids: list[int]) -> dict[str, Any]:
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


def select_group_aware_eval(
    df: pd.DataFrame, *, per_task: int, seed: int
) -> tuple[list[int], list[str], list[int], list[dict[str, Any]]]:
    if per_task <= 0:
        return [], [], [], []

    rng = np.random.default_rng(seed)
    eligible = df.loc[~df["quality_review_candidate"]].copy()
    eval_ids: list[int] = []
    selected_hashes: list[str] = []
    rows: list[dict[str, Any]] = []

    for task_index, task_df in eligible.groupby("task_index", sort=True):
        candidates = (
            task_df.groupby("exact_group_hash", sort=True)
            .agg(
                exact_group_id=("exact_group_id", "first"),
                ok_episode_count=("episode_index", "count"),
                representative_episode=("episode_index", "min"),
            )
            .reset_index()
        )
        if len(candidates) < per_task:
            raise ValueError(
                f"task {task_index} has only {len(candidates)} eligible exact groups; "
                f"cannot select {per_task}"
            )
        chosen_pos = rng.choice(len(candidates), size=per_task, replace=False)
        chosen = candidates.iloc[np.sort(chosen_pos)]
        task_name = str(task_df["task_name"].iloc[0])
        for row in chosen.itertuples(index=False):
            group_hash = str(row.exact_group_hash)
            representative = int(row.representative_episode)
            selected_hashes.append(group_hash)
            eval_ids.append(representative)
            rows.append(
                {
                    "task_index": int(task_index),
                    "task_name": task_name,
                    "exact_group_hash": group_hash,
                    "exact_group_id": str(row.exact_group_id),
                    "eval_episode_index": representative,
                    "ok_episode_count_in_group": int(row.ok_episode_count),
                }
            )

    selected_hash_set = set(selected_hashes)
    protected = sorted(
        df.loc[df["exact_group_hash"].isin(selected_hash_set), "episode_index"]
        .astype(int)
        .tolist()
    )
    return sorted(eval_ids), sorted(selected_hashes), protected, rows


def sqrt_balance_by_frames(
    df: pd.DataFrame, *, seed: int
) -> tuple[list[int], list[dict[str, Any]]]:
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


def build(
    *,
    metrics_csv: Path,
    groups_csv: Path,
    out_dir: Path,
    dataset_id: str,
    dataset_revision: str | None,
    seed: int,
    eval_per_task: int,
) -> dict[str, Any]:
    df = load_inputs(metrics_csv, groups_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_ids, eval_hashes, protected_ids, eval_group_rows = select_group_aware_eval(
        df, per_task=eval_per_task, seed=seed
    )
    protected_set = set(protected_ids)
    train_pool = df.loc[~df["episode_index"].isin(protected_set)].copy()

    if set(eval_ids) - protected_set:
        raise AssertionError("eval representatives must be inside protected groups")
    if set(train_pool["exact_group_hash"]).intersection(eval_hashes):
        raise AssertionError("protected eval trajectory group leaked into train_pool")
    expected_eval = int(df["task_index"].nunique()) * eval_per_task
    if len(eval_ids) != expected_eval or len(eval_hashes) != expected_eval:
        raise AssertionError(
            f"expected {expected_eval} eval episodes/groups, got "
            f"{len(eval_ids)} episodes / {len(eval_hashes)} groups"
        )

    variant_ids: dict[str, list[int]] = {
        "V0_RAW": sorted(train_pool["episode_index"].astype(int).tolist()),
        "V1_INTEGRITY_ONLY": sorted(
            train_pool.loc[~train_pool["flag_integrity"], "episode_index"].astype(int).tolist()
        ),
        "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL": sorted(
            train_pool.loc[
                train_pool["quality_flag_count"] < 2, "episode_index"
            ].astype(int).tolist()
        ),
        "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL": sorted(
            train_pool.loc[
                ~train_pool["quality_review_candidate"], "episode_index"
            ].astype(int).tolist()
        ),
    }
    sqrt_ids, sqrt_rows = sqrt_balance_by_frames(train_pool, seed=seed)
    variant_ids["V2_SQRT_BALANCED_RAW"] = sqrt_ids

    source_summary = {
        "episode_count": int(len(df)),
        "frame_count": int(df["frames"].sum()),
        "task_count": int(df["task_index"].nunique()),
        "exact_group_count": int(df["exact_group_hash"].nunique()),
        "review_count": int(df["quality_review_candidate"].sum()),
        "integrity_failure_count": int(df["flag_integrity"].sum()),
    }

    eval_manifest = {
        "schema_version": 2,
        "kind": "fixed_eval_holdout",
        "group_aware": True,
        "group_key": "exact_group_hash",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "eval_per_task": eval_per_task,
        "selection": (
            "OK-only, task-stratified, distinct exact trajectory groups; "
            "one representative episode per group; all group siblings excluded from training"
        ),
        "episode_ids": eval_ids,
        "episode_ids_sha256": json_sha256(eval_ids),
        "exact_group_hashes": eval_hashes,
        "exact_group_hashes_sha256": json_sha256(eval_hashes),
        "protected_episode_ids": protected_ids,
        "protected_episode_ids_sha256": json_sha256(protected_ids),
        "protected_episode_count": len(protected_ids),
        "summary": summarize(df, eval_ids),
        "group_rows": eval_group_rows,
    }
    (out_dir / "FIXED_EVAL_HOLDOUT.json").write_text(
        json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    variants_summary: dict[str, Any] = {}
    for name, ids in variant_ids.items():
        variant_groups = set(
            df.loc[df["episode_index"].isin(ids), "exact_group_hash"].astype(str)
        )
        if variant_groups.intersection(eval_hashes):
            raise AssertionError(f"{name}: exact trajectory group leakage")
        rule = {
            "V0_RAW": "all episodes outside protected eval trajectory groups",
            "V1_INTEGRITY_ONLY": "V0 group exclusion + drop hard integrity failures",
            "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL": "V0 group exclusion + drop quality_flag_count >= 2",
            "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL": "V0 group exclusion + drop every Static Quality REVIEW candidate",
            "V2_SQRT_BALANCED_RAW": "V0 group exclusion + deterministic sqrt task balancing by frame exposure",
        }[name]
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "kind": "training_episode_manifest",
            "group_aware": True,
            "group_key": "exact_group_hash",
            "variant": name,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "seed": seed,
            "source_metrics_csv": str(metrics_csv),
            "source_groups_csv": str(groups_csv),
            "source_summary": source_summary,
            "fixed_eval_manifest": "FIXED_EVAL_HOLDOUT.json",
            "selection_rule": rule,
            "episode_ids": ids,
            "episode_ids_sha256": json_sha256(ids),
            "protected_eval_group_count": len(eval_hashes),
            "protected_episode_count": len(protected_ids),
            "summary": summarize(df, ids),
        }
        if name == "V2_SQRT_BALANCED_RAW":
            manifest["balancing"] = {
                "basis": "task frame exposure after group-aware eval exclusion",
                "formula": "target_frames(task) = sqrt(min_task_frames * raw_task_frames)",
                "task_rows": sqrt_rows,
            }
        (out_dir / f"{name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        variants_summary[name] = manifest["summary"]

    cheap_order = ["V0_RAW"]
    skip_reason: dict[str, str] = {}
    if variant_ids["V1_INTEGRITY_ONLY"] == variant_ids["V0_RAW"]:
        skip_reason["V1_INTEGRITY_ONLY"] = (
            "identical to V0 because integrity_failure_count == 0"
        )
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
        "schema_version": 2,
        "group_aware": True,
        "group_key": "exact_group_hash",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "source_summary": source_summary,
        "fixed_eval": eval_manifest["summary"],
        "fixed_eval_group_count": len(eval_hashes),
        "protected_episode_count": len(protected_ids),
        "protected_fraction": len(protected_ids) / len(df),
        "variants": variants_summary,
        "cheap_ablation_order": cheap_order,
        "skip_reason": skip_reason,
        "promotion_rule": (
            "screening only; group-aware leakage gate must PASS and final selection "
            "requires fixed/simulator evaluation"
        ),
    }
    (out_dir / "run_matrix.json").write_text(
        json.dumps(run_matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-csv", type=Path, required=True)
    p.add_argument("--trajectory-groups-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-id", default="lerobot/libero_plus")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--eval-per-task", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    matrix = build(
        metrics_csv=args.metrics_csv,
        groups_csv=args.trajectory_groups_csv,
        out_dir=args.out,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        seed=args.seed,
        eval_per_task=args.eval_per_task,
    )
    print(json.dumps(matrix, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
