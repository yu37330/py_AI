#!/usr/bin/env python3
"""Create safe local/Drive headroom before M3 simulator evaluation.

Only disposable package-manager caches are removed. Hugging Face caches,
training checkpoints, D10 data/manifests, simulator evidence, and merged-model
artifacts are never deleted here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

GIB = 1024**3
LOCAL_MIN_FREE_GIB = 24.0
DRIVE_MIN_FREE_GIB = 20.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("pre-setup", "pre-eval"), required=True)
    return p.parse_args()


def _usage(path: Path) -> dict[str, float | str]:
    u = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gib": round(u.total / GIB, 2),
        "used_gib": round(u.used / GIB, 2),
        "free_gib": round(u.free / GIB, 2),
    }


def _best_effort(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    print(f"rc={proc.returncode}", flush=True)


def _du(path: Path) -> str:
    if not path.exists():
        return "missing"
    proc = subprocess.run(
        ["du", "-sh", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return (proc.stdout or f"rc={proc.returncode}").strip()


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 disk headroom preflight requires PARC_M3_EXECUTE=1")

    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026")).resolve()
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache")).resolve()
    if not Path("/content").exists():
        raise RuntimeError("/content is unavailable")
    if not drive.exists():
        raise RuntimeError(f"Drive root is unavailable: {drive}")

    before_local = _usage(Path("/content"))
    before_drive = _usage(drive)
    print(json.dumps({"phase": args.phase, "before": {"local": before_local, "drive": before_drive}}, indent=2), flush=True)

    # These caches are fully reconstructible and are not benchmark evidence.
    # Do not touch ~/.cache/huggingface: model snapshots can be reused by the
    # sequential evaluators and deleting them would create another large download.
    if shutil.which("uv"):
        _best_effort(["uv", "cache", "clean"])
    _best_effort([sys.executable, "-m", "pip", "cache", "purge"])
    if os.geteuid() == 0 and shutil.which("apt-get"):
        _best_effort(["apt-get", "clean"])

    # Remove only empty/reconstructible package caches if commands left them.
    for cache in (Path.home() / ".cache/pip", Path.home() / ".cache/uv"):
        try:
            if cache.exists():
                shutil.rmtree(cache)
                print(f"removed disposable cache: {cache}", flush=True)
        except OSError as exc:
            print(f"WARNING: could not remove disposable cache {cache}: {exc}", flush=True)

    after_local = _usage(Path("/content"))
    after_drive = _usage(drive)
    details = {
        "parc_root": _du(root),
        "hf_cache": _du(Path.home() / ".cache/huggingface"),
        "drive_benchmark": _du(drive / "model-benchmark-v1"),
    }
    print(json.dumps({"phase": args.phase, "after": {"local": after_local, "drive": after_drive}, "details": details}, indent=2), flush=True)

    local_free = float(after_local["free_gib"])
    drive_free = float(after_drive["free_gib"])
    if local_free < LOCAL_MIN_FREE_GIB:
        raise RuntimeError(
            f"insufficient local /content headroom after safe cache cleanup: {local_free:.2f} GiB free; "
            f"need >= {LOCAL_MIN_FREE_GIB:.0f} GiB before three-model simulator evaluation"
        )
    if drive_free < DRIVE_MIN_FREE_GIB:
        raise RuntimeError(
            f"insufficient Google Drive headroom for OpenVLA evaluation materialization: {drive_free:.2f} GiB free; "
            f"need >= {DRIVE_MIN_FREE_GIB:.0f} GiB. Do not delete D10/checkpoints; free unrelated Drive space instead."
        )

    print("=== M3 DISK HEADROOM: PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
