#!/usr/bin/env python3
"""Detect trajectory-group leakage between fixed eval and training manifests.

This is a simulator-free diagnostic for LeRobot-style datasets. It hashes the
non-visual trajectory of each episode so visual/domain perturbation copies of
an otherwise identical control trajectory are treated as one group.

Two fingerprints are emitted:

- exact_group_id: task + quantized observation.state + action sequence
- action_group_id: task + quantized action sequence only

An exact-group overlap between eval and train is a hard leakage signal.
Action-only overlap is reported separately as a softer "control-equivalent"
candidate because different states can legitimately share an action sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_matrix(series: pd.Series, name: str) -> np.ndarray:
    values = series.tolist()
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        arr = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in values], axis=0)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected matrix [T,D], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite values are not hashable")
    return arr


def quantize(arr: np.ndarray, decimals: int) -> np.ndarray:
    if decimals < 0 or decimals > 9:
        raise ValueError("round_decimals must be between 0 and 9")
    scale = 10**decimals
    q = np.rint(np.asarray(arr, dtype=np.float64) * scale)
    limit = np.iinfo(np.int64).max
    if np.abs(q).max(initial=0.0) > limit:
        raise OverflowError("quantized trajectory exceeds int64 range")
    return q.astype("<i8", copy=False)


def fingerprint_arrays(
    *,
    task_index: int,
    state: np.ndarray,
    action: np.ndarray,
    decimals: int,
) -> tuple[str, str, str]:
    """Return (exact_hash, action_hash, state_hash) for one episode."""
    if len(state) != len(action):
        raise ValueError(f"state/action length mismatch: {len(state)} != {len(action)}")
    q_state = quantize(state, decimals)
    q_action = quantize(action, decimals)

    def digest(kind: str, *arrays: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(
            f"trajectory-group-v1|{kind}|task={int(task_index)}|n={len(state)}|d={decimals}|".encode()
        )
        for arr in arrays:
            h.update(str(tuple(arr.shape)).encode())
            h.update(b"|")
            h.update(arr.tobytes(order="C"))
            h.update(b"|")
        return h.hexdigest()

    return (
        digest("state_action", q_state, q_action),
        digest("action", q_action),
        digest("state", q_state),
    )


def iter_episode_frames(root: Path) -> Iterator[tuple[int, int, Path, pd.DataFrame]]:
    parquet_paths = sorted(root.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no data/**/*.parquet under {root}")

    wanted = ["episode_index", "task_index", "frame_index", "observation.state", "action"]
    for pos, path in enumerate(parquet_paths, start=1):
        print(f"read parquet {pos}/{len(parquet_paths)}: {path.relative_to(root)}")
        table = pq.read_table(path)
        available = set(table.column_names)
        missing = {"episode_index", "task_index", "observation.state", "action"}.difference(available)
        if missing:
            raise KeyError(f"{path}: missing columns {sorted(missing)}")
        frame = table.select([c for c in wanted if c in available]).to_pandas()
        for episode_index, ep in frame.groupby("episode_index", sort=False):
            ep = ep.copy()
            if "frame_index" in ep.columns:
                ep = ep.sort_values("frame_index", kind="stable")
            task_values = pd.to_numeric(ep["task_index"], errors="raise").astype(int).unique()
            if len(task_values) != 1:
                raise ValueError(
                    f"episode {episode_index}: multiple task_index values {task_values.tolist()}"
                )
            yield int(episode_index), int(task_values[0]), path, ep.reset_index(drop=True)


def build_group_members(root: Path, decimals: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_index, task_index, path, ep in iter_episode_frames(root):
        state = safe_matrix(ep["observation.state"], "observation.state")
        action = safe_matrix(ep["action"], "action")
        exact_hash, action_hash, state_hash = fingerprint_arrays(
            task_index=task_index,
            state=state,
            action=action,
            decimals=decimals,
        )
        rows.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "frames": int(len(ep)),
                "parquet_path": str(path.relative_to(root)),
                "exact_group_hash": exact_hash,
                "exact_group_id": exact_hash[:16],
                "action_group_hash": action_hash,
                "action_group_id": action_hash[:16],
                "state_group_hash": state_hash,
                "state_group_id": state_hash[:16],
            }
        )
    members = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
    if members["episode_index"].duplicated().any():
        raise ValueError("duplicate episode_index detected while hashing dataset")
    members["exact_group_size"] = members.groupby("exact_group_hash")["episode_index"].transform("size")
    members["action_group_size"] = members.groupby("action_group_hash")["episode_index"].transform("size")
    return members


def load_optional_quality(metrics_csv: Path | None) -> pd.DataFrame | None:
    if metrics_csv is None or not metrics_csv.exists():
        return None
    q = pd.read_csv(metrics_csv)
    cols = [
        c
        for c in [
            "episode_index",
            "quality_review_candidate",
            "quality_flag_count",
            "quality_status_v1",
        ]
        if c in q.columns
    ]
    if "episode_index" not in cols:
        return None
    return q[cols].copy()


def group_summary(members: pd.DataFrame, group_col: str, id_col: str) -> pd.DataFrame:
    return (
        members.groupby([group_col, id_col, "task_index"], sort=True)
        .agg(
            group_size=("episode_index", "count"),
            frames_min=("frames", "min"),
            frames_max=("frames", "max"),
            episode_min=("episode_index", "min"),
            episode_max=("episode_index", "max"),
        )
        .reset_index()
        .sort_values(["group_size", "task_index", "episode_min"], ascending=[False, True, True])
    )


def read_manifest_files(manifests_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    paths = sorted(manifests_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no manifest JSON files under {manifests_dir}")
    eval_manifest: dict[str, Any] | None = None
    training: dict[str, dict[str, Any]] = {}
    for path in paths:
        obj = read_json(path)
        kind = obj.get("kind")
        if kind == "fixed_eval_holdout":
            if eval_manifest is not None:
                raise ValueError("multiple fixed_eval_holdout manifests found")
            eval_manifest = obj
        elif kind == "training_episode_manifest":
            name = str(obj.get("variant") or path.stem)
            training[name] = obj
    if eval_manifest is None:
        raise FileNotFoundError("fixed_eval_holdout manifest not found")
    if not training:
        raise FileNotFoundError("training_episode_manifest files not found")
    return eval_manifest, training


def _group_leakage(
    members: pd.DataFrame,
    eval_ids: set[int],
    train_ids: set[int],
    group_hash_col: str,
) -> tuple[set[str], pd.DataFrame]:
    indexed = members.set_index("episode_index")
    missing_eval = sorted(eval_ids.difference(indexed.index))
    missing_train = sorted(train_ids.difference(indexed.index))
    if missing_eval or missing_train:
        raise KeyError(
            f"manifest episode IDs absent from dataset: eval={missing_eval[:10]} train={missing_train[:10]}"
        )

    eval_rows = indexed.loc[sorted(eval_ids)]
    train_rows = indexed.loc[sorted(train_ids)]
    eval_groups = set(eval_rows[group_hash_col].astype(str))
    train_groups = set(train_rows[group_hash_col].astype(str))
    leaked = eval_groups.intersection(train_groups)

    detail = members[members[group_hash_col].astype(str).isin(leaked)].copy()
    detail["in_eval"] = detail["episode_index"].isin(eval_ids)
    detail["in_train"] = detail["episode_index"].isin(train_ids)
    return leaked, detail


def evaluate_manifests(
    members: pd.DataFrame,
    eval_manifest: dict[str, Any],
    training: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_ids = {int(x) for x in eval_manifest.get("episode_ids", [])}
    if not eval_ids:
        raise ValueError("fixed eval manifest has no episode_ids")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    for variant, manifest in training.items():
        train_ids = {int(x) for x in manifest.get("episode_ids", [])}
        exact_groups, exact_detail = _group_leakage(
            members, eval_ids, train_ids, "exact_group_hash"
        )
        action_groups, action_detail = _group_leakage(
            members, eval_ids, train_ids, "action_group_hash"
        )
        exact_eval_eps = set(exact_detail.loc[exact_detail["in_eval"], "episode_index"].astype(int))
        exact_train_eps = set(exact_detail.loc[exact_detail["in_train"], "episode_index"].astype(int))
        action_eval_eps = set(action_detail.loc[action_detail["in_eval"], "episode_index"].astype(int))
        action_train_eps = set(action_detail.loc[action_detail["in_train"], "episode_index"].astype(int))
        summary_rows.append(
            {
                "variant": variant,
                "eval_episode_count": len(eval_ids),
                "train_episode_count": len(train_ids),
                "exact_leakage_group_count": len(exact_groups),
                "exact_eval_episode_count": len(exact_eval_eps),
                "exact_train_episode_count": len(exact_train_eps),
                "action_leakage_group_count": len(action_groups),
                "action_eval_episode_count": len(action_eval_eps),
                "action_train_episode_count": len(action_train_eps),
                "exact_gate_pass": len(exact_groups) == 0,
            }
        )
        for kind, detail in (("exact", exact_detail), ("action", action_detail)):
            if detail.empty:
                continue
            d = detail.copy()
            d.insert(0, "variant", variant)
            d.insert(1, "leakage_kind", kind)
            detail_rows.append(d)

    summary = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)
    detail = (
        pd.concat(detail_rows, ignore_index=True)
        if detail_rows
        else pd.DataFrame(
            columns=[
                "variant",
                "leakage_kind",
                "episode_index",
                "task_index",
                "frames",
                "exact_group_id",
                "action_group_id",
                "in_eval",
                "in_train",
            ]
        )
    )
    return summary, detail


def analyze(
    *,
    root: Path,
    manifests_dir: Path,
    out_dir: Path,
    round_decimals: int = 6,
    metrics_csv: Path | None = None,
) -> dict[str, Any]:
    root = resolve_dataset_root(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = read_json(root / "meta" / "info.json")

    members = build_group_members(root, round_decimals)
    quality = load_optional_quality(metrics_csv)
    if quality is not None:
        quality["episode_index"] = pd.to_numeric(quality["episode_index"], errors="raise").astype(int)
        members = members.merge(quality, how="left", on="episode_index", validate="one_to_one")

    exact_summary = group_summary(members, "exact_group_hash", "exact_group_id")
    action_summary = group_summary(members, "action_group_hash", "action_group_id")
    eval_manifest, training = read_manifest_files(manifests_dir)
    leakage, detail = evaluate_manifests(members, eval_manifest, training)

    members.to_csv(out_dir / "trajectory_group_members.csv", index=False)
    exact_summary.to_csv(out_dir / "exact_trajectory_groups.csv", index=False)
    action_summary.to_csv(out_dir / "action_trajectory_groups.csv", index=False)
    leakage.to_csv(out_dir / "manifest_leakage_report.csv", index=False)
    detail.to_csv(out_dir / "manifest_leakage_members.csv", index=False)

    duplicate_exact = exact_summary[exact_summary["group_size"] > 1]
    duplicate_action = action_summary[action_summary["group_size"] > 1]

    review_stats: dict[str, Any] | None = None
    if "quality_review_candidate" in members.columns:
        review_mask = members["quality_review_candidate"].fillna(False).astype(bool)
        review = members[review_mask]
        review_stats = {
            "review_episode_count": int(len(review)),
            "review_in_duplicate_exact_group_count": int((review["exact_group_size"] > 1).sum()),
            "review_in_duplicate_action_group_count": int((review["action_group_size"] > 1).sum()),
        }

    exact_gate_pass = bool(leakage["exact_gate_pass"].all()) if not leakage.empty else True
    report = {
        "schema_version": 1,
        "dataset_root": str(root),
        "dataset_declared_episodes": info.get("total_episodes"),
        "dataset_declared_frames": info.get("total_frames"),
        "dataset_codebase_version": info.get("codebase_version"),
        "round_decimals": round_decimals,
        "episode_count_hashed": int(len(members)),
        "exact_group_count": int(len(exact_summary)),
        "duplicate_exact_group_count": int(len(duplicate_exact)),
        "episodes_in_duplicate_exact_groups": int(duplicate_exact["group_size"].sum()),
        "action_group_count": int(len(action_summary)),
        "duplicate_action_group_count": int(len(duplicate_action)),
        "episodes_in_duplicate_action_groups": int(duplicate_action["group_size"].sum()),
        "quality_review_crosscheck": review_stats,
        "variants": leakage.to_dict(orient="records"),
        "trajectory_leakage_gate": "PASS" if exact_gate_pass else "FAIL",
        "gate_definition": "PASS iff no exact state+action trajectory group overlaps fixed eval and any training variant",
        "action_group_note": "Action-only overlap is a softer control-equivalence signal and is reported but does not fail the exact gate.",
    }
    (out_dir / "trajectory_group_leakage_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifests-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--round-decimals", type=int, default=6)
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument(
        "--fail-on-exact-leakage",
        action="store_true",
        help="exit non-zero after writing reports when exact trajectory leakage is present",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(
        root=args.root,
        manifests_dir=args.manifests_dir,
        out_dir=args.out,
        round_decimals=args.round_decimals,
        metrics_csv=args.metrics_csv,
    )
    if args.fail_on_exact_leakage and report["trajectory_leakage_gate"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
