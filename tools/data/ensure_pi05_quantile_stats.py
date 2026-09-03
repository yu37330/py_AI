#!/usr/bin/env python3
"""Ensure q01/q99 (plus q10/q50/q90) exist for π0.5 STATE/ACTION stats.

LeRobot π0.5 uses QUANTILES normalization for observation.state and action.
Some v3 datasets, including the public LIBERO-plus proxy used by PARC experiments,
ship only min/max/mean/std/count in meta/stats.json. This utility derives exact
per-dimension quantiles from the local parquet data and merges them into the
local stats.json. The original remote dataset is not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_FEATURES = ("observation.state", "action")
QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))


def _as_2d(column: pa.ChunkedArray) -> np.ndarray:
    arr = column.combine_chunks()
    typ = arr.type
    if pa.types.is_fixed_size_list(typ):
        width = typ.list_size
        values = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float64)
        return values.reshape(len(arr), width)
    if pa.types.is_list(typ) or pa.types.is_large_list(typ):
        offsets = np.asarray(arr.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        lengths = np.diff(offsets)
        if len(lengths) == 0:
            return np.empty((0, 0), dtype=np.float64)
        if not np.all(lengths == lengths[0]):
            raise ValueError(f"variable-length feature cannot be normalized dimension-wise: {typ}")
        width = int(lengths[0])
        values = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float64)
        return values.reshape(len(arr), width)
    values = np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float64)
    return values.reshape(len(arr), -1) if values.ndim == 1 else values


def _complete(stats: dict, features: tuple[str, ...]) -> bool:
    return all(
        feature in stats and "q01" in stats[feature] and "q99" in stats[feature]
        for feature in features
    )


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        fh.write(text)
        tmp = Path(fh.name)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    ap.add_argument("--report", type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")

    features = tuple(args.features)
    stats = json.loads(stats_path.read_text())
    if _complete(stats, features) and not args.force:
        report = {
            "status": "already_complete",
            "dataset_root": str(root),
            "features": list(features),
            "parquet_file_count": len(parquet_files),
            "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        }
        if args.report:
            _atomic_json_write(args.report, report)
        print("PI05 Quantile Stats Gate: PASS (already present)", flush=True)
        return 0

    chunks: dict[str, list[np.ndarray]] = {feature: [] for feature in features}
    rows = 0
    print(f"quantile scan: {len(parquet_files)} parquet file(s)", flush=True)
    for i, path in enumerate(parquet_files, 1):
        table = pq.read_table(path, columns=list(features))
        rows += table.num_rows
        for feature in features:
            chunks[feature].append(_as_2d(table[feature]))
        print(f"quantile scan {i}/{len(parquet_files)} rows={rows}: {path.name}", flush=True)

    derived: dict[str, dict[str, list[float]]] = {}
    for feature in features:
        values = np.concatenate(chunks[feature], axis=0)
        if values.shape[0] != rows:
            raise AssertionError((feature, values.shape[0], rows))
        qvals = np.quantile(values, [q for _, q in QUANTILES], axis=0, method="linear")
        derived[feature] = {}
        stats.setdefault(feature, {})
        for idx, (name, _) in enumerate(QUANTILES):
            vals = qvals[idx].astype(float).tolist()
            stats[feature][name] = vals
            derived[feature][name] = vals
        print(f"quantiles ready: {feature} shape={values.shape}", flush=True)

    _atomic_json_write(stats_path, stats)
    reread = json.loads(stats_path.read_text())
    if not _complete(reread, features):
        raise RuntimeError("q01/q99 were not persisted correctly")

    report = {
        "status": "augmented",
        "dataset_root": str(root),
        "features": list(features),
        "rows": rows,
        "parquet_file_count": len(parquet_files),
        "derived": derived,
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }
    if args.report:
        _atomic_json_write(args.report, report)
    print("PI05 Quantile Stats Gate: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
