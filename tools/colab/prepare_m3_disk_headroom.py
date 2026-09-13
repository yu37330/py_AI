#!/usr/bin/env python3
"""Create safe scratch/persistent-storage headroom before M3 evaluation.

Only disposable package-manager caches are removed. Hugging Face caches,
training checkpoints, D10 data/manifests, simulator evidence, and merged-model
artifacts are never deleted here.

Colab remains the default. Organizer environments can override the checked
filesystems with PARC_LOCAL_SCRATCH_ROOT and PARC_PERSIST_ROOT while keeping the
legacy PARC_ROOT / PARC_DRIVE_ROOT contract for downstream tools.
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
PERSIST_MIN_FREE_GIB = 20.0


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


def _resolve_roots() -> tuple[Path, Path, Path]:
    parc_root = Path(os.environ.get("PARC_ROOT", "/content/parc2026")).expanduser().resolve()
    local_root = Path(
        os.environ.get("PARC_LOCAL_SCRATCH_ROOT", str(parc_root))
    ).expanduser().resolve()
    persist_root = Path(
        os.environ.get(
            "PARC_PERSIST_ROOT",
            os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"),
        )
    ).expanduser().resolve()
    return parc_root, local_root, persist_root


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 disk headroom preflight requires PARC_M3_EXECUTE=1")

    parc_root, local_root, persist_root = _resolve_roots()
    if not local_root.exists():
        raise RuntimeError(f"local scratch root is unavailable: {local_root}")
    if not persist_root.exists():
        raise RuntimeError(f"persistent root is unavailable: {persist_root}")

    before_local = _usage(local_root)
    before_persist = _usage(persist_root)
    print(
        json.dumps(
            {"phase": args.phase, "before": {"local": before_local, "persistent": before_persist}},
            indent=2,
        ),
        flush=True,
    )

    # ここで削除するのは再生成可能なpackage-manager cacheだけ。
    # Hugging Face cacheは大きな再downloadを避けるため触らない。
    if shutil.which("uv"):
        _best_effort(["uv", "cache", "clean"])
    _best_effort([sys.executable, "-m", "pip", "cache", "purge"])
    if os.geteuid() == 0 and shutil.which("apt-get"):
        _best_effort(["apt-get", "clean"])

    for cache in (Path.home() / ".cache/pip", Path.home() / ".cache/uv"):
        try:
            if cache.exists():
                shutil.rmtree(cache)
                print(f"removed disposable cache: {cache}", flush=True)
        except OSError as exc:
            print(f"WARNING: could not remove disposable cache {cache}: {exc}", flush=True)

    after_local = _usage(local_root)
    after_persist = _usage(persist_root)
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser()
    details = {
        "parc_root": _du(parc_root),
        "hf_cache": _du(hf_home),
        "persistent_benchmark": _du(persist_root / "model-benchmark-v1"),
    }
    print(
        json.dumps(
            {
                "phase": args.phase,
                "after": {"local": after_local, "persistent": after_persist},
                "details": details,
            },
            indent=2,
        ),
        flush=True,
    )

    local_free = float(after_local["free_gib"])
    persist_free = float(after_persist["free_gib"])
    if local_free < LOCAL_MIN_FREE_GIB:
        raise RuntimeError(
            f"insufficient local scratch headroom after safe cache cleanup: {local_free:.2f} GiB free at "
            f"{local_root}; need >= {LOCAL_MIN_FREE_GIB:.0f} GiB before three-model simulator evaluation"
        )
    if persist_free < PERSIST_MIN_FREE_GIB:
        raise RuntimeError(
            f"insufficient persistent-storage headroom for OpenVLA evaluation materialization: "
            f"{persist_free:.2f} GiB free at {persist_root}; need >= {PERSIST_MIN_FREE_GIB:.0f} GiB. "
            "Do not delete D10/checkpoints/evidence; free unrelated persistent storage instead."
        )

    print("=== M3 DISK HEADROOM: PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
