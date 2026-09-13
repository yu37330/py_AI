#!/usr/bin/env python3
"""Verify the checksum manifest after restoring the M3 organizer handoff bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"


def parse_args() -> argparse.Namespace:
    default_root = os.environ.get(
        "PARC_PERSIST_ROOT",
        os.environ.get("PARC_DRIVE_ROOT", str(Path.home() / "data/parc2026-cache")),
    )
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(default_root))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--evidence-out", type=Path)
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


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else root / "organizer-handoff-v1/handoff_manifest.json"
    )
    evidence_out = (
        args.evidence_out.expanduser().resolve()
        if args.evidence_out
        else root / "model-benchmark-v1/m3-organizer-migration-v1/handoff_integrity.json"
    )
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    data = _load(manifest)
    expected = {
        "status": "READY_FOR_TRANSFER",
        "stage": "M3_organizer_handoff_bundle",
        "selected_episode_ids_sha256": D10_HASH,
        "d10_manifest_sha256": D10_MANIFEST_SHA256,
        "checkpoint_scope": "notebook73_forward_equal_data_smoke_only",
        "dataset_included": False,
        "source_training_results_mutated": False,
        "benchmark_training_started": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"handoff manifest mismatch: {key}={data.get(key)!r} != {value!r}")

    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("handoff manifest has no file inventory")

    verified_bytes = 0
    verified_files = 0
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("invalid handoff file record")
        rel = Path(str(item.get("path") or ""))
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe handoff path: {rel}")
        path = root / rel
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"handoff file missing/invalid: {path}")
        expected_size = int(item.get("size", -1))
        if path.stat().st_size != expected_size:
            raise RuntimeError(
                f"handoff size mismatch: {rel} {path.stat().st_size} != {expected_size}"
            )
        expected_sha = str(item.get("sha256") or "")
        got_sha = _sha256(path)
        if got_sha != expected_sha:
            raise RuntimeError(f"handoff SHA mismatch: {rel} {got_sha} != {expected_sha}")
        verified_bytes += expected_size
        verified_files += 1

    if verified_files != int(data.get("file_count", -1)):
        raise RuntimeError("handoff file_count mismatch")
    if verified_bytes != int(data.get("total_bytes", -1)):
        raise RuntimeError("handoff total_bytes mismatch")

    evidence = {
        "schema_version": 1,
        "stage": "M3_organizer_handoff_integrity",
        "status": "PASS",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "selected_episode_ids_sha256": D10_HASH,
        "d10_manifest_sha256": D10_MANIFEST_SHA256,
        "verified_file_count": verified_files,
        "verified_bytes": verified_bytes,
        "dataset_included": False,
        "source_training_results_mutated": False,
        "benchmark_training_started": False,
    }
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    evidence_out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== M3 ORGANIZER HANDOFF INTEGRITY: PASS ===", flush=True)
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
