#!/usr/bin/env python3
"""Build a checksum-verified organizer handoff archive from persistent M3 evidence.

Run this on the Colab/Drive side before switching to the organizer GPU. The
archive intentionally contains only immutable/small prerequisite evidence plus
the exact Notebook73 forward/equal-data smoke checkpoints required by Notebook74.
The full training dataset is not included; it is reconstructed separately on
the organizer NVMe scratch area.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"
TRAINING_SCHEDULE_SEED = 20260906
MODELS = ("pi05", "smolvla", "openvla_oft")
ARTIFACT_MARKER = "model-benchmark-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    default_root = os.environ.get(
        "PARC_PERSIST_ROOT",
        os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"),
    )
    p.add_argument("--source-root", type=Path, default=Path(default_root))
    p.add_argument("--archive", type=Path, required=True)
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.is_symlink():
        raise RuntimeError(f"handoff refuses symlink: {path}")
    return path


def _checkpoint_from_result(source_root: Path, result: dict[str, Any]) -> Path:
    raw = str(result.get("checkpoint_ref") or "")
    if not raw:
        raise RuntimeError("checkpoint_ref missing")
    original = Path(raw).expanduser()
    if original.exists():
        checkpoint = original.resolve()
    else:
        parts = original.parts
        try:
            marker_index = parts.index(ARTIFACT_MARKER)
        except ValueError as exc:
            raise RuntimeError(f"checkpoint_ref cannot be safely relocated: {original}") from exc
        checkpoint = source_root.joinpath(*parts[marker_index:]).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    try:
        checkpoint.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(f"checkpoint is outside source root: {checkpoint}") from exc
    return checkpoint


def _collect_tree(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"handoff refuses symlink: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise RuntimeError(f"checkpoint tree is empty: {root}")
    return files


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    archive.parent.mkdir(parents=True, exist_ok=True)

    required = [
        source_root / "model-benchmark-v1/m3-training-smoke-v1/m3_training_smoke_summary.json",
        source_root / "model-benchmark-v1/m3_batch_probe_summary.json",
        source_root / f"model-benchmark-v1/m3-schedules-v1/m3_equal_data_seed-{TRAINING_SCHEDULE_SEED}.json",
        source_root / "openvla-streaming-selected-v1/streaming_bridge_contract.json",
        source_root
        / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json",
    ]
    required = [_require_file(path) for path in required]

    smoke = _load(required[0])
    if smoke.get("status") != "PASS" or smoke.get("stage") != "M3_A100_training_smoke":
        raise RuntimeError("Notebook73 summary is not PASS")
    if smoke.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("Notebook73 D10 mismatch")
    if smoke.get("benchmark_training_started") is not False:
        raise RuntimeError("Notebook73 unexpectedly claims benchmark training started")

    manifest_path = required[-1]
    if _sha256(manifest_path) != D10_MANIFEST_SHA256:
        raise RuntimeError("D10 manifest file SHA mismatch")
    manifest_data = _load(manifest_path)
    if manifest_data.get("episode_ids_sha256") != D10_HASH:
        raise RuntimeError("D10 manifest episode hash mismatch")

    all_files: set[Path] = set(required)
    checkpoint_roots: dict[str, str] = {}
    for model in MODELS:
        result_path = (
            source_root
            / "model-benchmark-v1/m3-training-smoke-v1/forward/equal_data"
            / f"seed-{TRAINING_SCHEDULE_SEED}"
            / model
            / "train_result.json"
        )
        _require_file(result_path)
        result = _load(result_path)
        expected = {
            "status": "PASS",
            "model": model,
            "mode": "smoke",
            "order": "forward",
            "track": "equal_data",
            "selected_episode_ids_sha256": D10_HASH,
            "sampling_seed": TRAINING_SCHEDULE_SEED,
            "effective_batch_size": 32,
            "samples_consumed": 64,
            "optimizer_updates": 2,
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise RuntimeError(f"Notebook73 result mismatch {result_path}: {key}")
        checkpoint = _checkpoint_from_result(source_root, result)
        checkpoint_roots[model] = str(checkpoint.relative_to(source_root))
        all_files.add(result_path)
        all_files.update(_collect_tree(checkpoint))

    entries: list[dict[str, Any]] = []
    for path in sorted(all_files, key=lambda p: str(p.relative_to(source_root))):
        relative = path.relative_to(source_root)
        entries.append(
            {
                "path": str(relative),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    handoff = {
        "schema_version": 1,
        "stage": "M3_organizer_handoff_bundle",
        "status": "READY_FOR_TRANSFER",
        "selected_episode_ids_sha256": D10_HASH,
        "d10_manifest_sha256": D10_MANIFEST_SHA256,
        "training_schedule_seed": TRAINING_SCHEDULE_SEED,
        "checkpoint_scope": "notebook73_forward_equal_data_smoke_only",
        "checkpoint_roots": checkpoint_roots,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
        "files": entries,
        "dataset_included": False,
        "source_training_results_mutated": False,
        "benchmark_training_started": False,
    }
    manifest_bytes = (json.dumps(handoff, indent=2, sort_keys=True) + "\n").encode("utf-8")

    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing handoff archive: {archive}")
    with tarfile.open(archive, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted(all_files, key=lambda p: str(p.relative_to(source_root))):
            tf.add(path, arcname=str(path.relative_to(source_root)), recursive=False)
        info = tarfile.TarInfo("organizer-handoff-v1/handoff_manifest.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        info.mtime = 0
        import io

        tf.addfile(info, io.BytesIO(manifest_bytes))

    archive_sha = _sha256(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive": str(archive),
                "archive_sha256": archive_sha,
                "archive_bytes": archive.stat().st_size,
                "file_count": len(entries),
                "selected_episode_ids_sha256": D10_HASH,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("=== M3 ORGANIZER HANDOFF BUNDLE: READY ===", flush=True)
    print(json.dumps({"archive": str(archive), "sha256": archive_sha, "sidecar": str(sidecar)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
