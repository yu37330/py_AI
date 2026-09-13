#!/usr/bin/env python3
"""Safely restore and verify an M3 organizer handoff archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def parse_args() -> argparse.Namespace:
    default_root = os.environ.get("PARC_PERSIST_ROOT", str(Path.home() / "data/parc2026-cache"))
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path(default_root))
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(name: str) -> Path:
    rel = Path(name)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    return rel


def main() -> int:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    root = args.root.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_sha = str(metadata.get("archive_sha256") or "")
    if not expected_sha:
        raise RuntimeError("archive_sha256 missing from sidecar")
    got_sha = _sha256(archive)
    if got_sha != expected_sha:
        raise RuntimeError(f"handoff archive SHA mismatch: {got_sha} != {expected_sha}")
    if archive.stat().st_size != int(metadata.get("archive_bytes", -1)):
        raise RuntimeError("handoff archive size mismatch")

    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError("handoff archive is empty")
        for member in members:
            rel = _safe_rel(member.name)
            if member.issym() or member.islnk():
                raise RuntimeError(f"handoff archive link is forbidden: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported handoff archive member: {member.name}")
            target = root / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing restored artifact: {target}; use a clean persistent root"
                )
            source = tf.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".partial")
            with tmp.open("wb") as out:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(chunk)
            tmp.chmod(member.mode & 0o777)
            tmp.replace(target)

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    verifier = repo / "tools/benchmark/verify_m3_organizer_handoff.py"
    subprocess.run(
        [sys.executable, "-u", str(verifier), "--root", str(root)],
        cwd=str(repo),
        env=os.environ.copy(),
        check=True,
    )
    print("=== M3 ORGANIZER HANDOFF RESTORE: PASS ===", flush=True)
    print(json.dumps({"archive": str(archive), "archive_sha256": got_sha, "root": str(root)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
