#!/usr/bin/env python3
"""Write a small immutable run manifest for Colab/organizer experiments.

Stdlib-only on purpose: this should work before model-specific environments are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--dataset-manifest", type=Path, required=True)
    ap.add_argument("--eval-split", type=Path, required=True)
    ap.add_argument("--command", default="")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    git_sha = run(["git", "-C", str(args.repo), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(args.repo), "status", "--porcelain"])

    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"sha": git_sha, "dirty": bool(dirty)},
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": gpu,
        },
        "inputs": {
            "model_config": str(args.model_config),
            "model_config_sha256": sha256(args.model_config),
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_manifest_sha256": sha256(args.dataset_manifest),
            "eval_split": str(args.eval_split),
            "eval_split_sha256": sha256(args.eval_split),
        },
        "command": args.command,
        "env": {
            key: os.environ[key]
            for key in ["CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE", "WANDB_MODE"]
            if key in os.environ
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
