#!/usr/bin/env python3
"""Minimal GPU smoke for pi0.5 subset training.

This intentionally skips the expensive group-aware manifest/static-quality gates.
Its only purpose is to verify that the already CPU-validated non-contiguous
subset can pass through the real pi0.5 LoRA training path on GPU with GA=8.

It runs exactly one effective optimizer step (8 micro-steps at BS=1), writes a
checkpoint, and exits. It is not a dataset-quality experiment and must not be
used for model selection.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def cmd(args, *, cwd=None, env=None):
    print(">>>", shlex.join([str(x) for x in args]), flush=True)
    child_env = (env or os.environ).copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(args, cwd=cwd, env=child_env, check=True)


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", root / "py_AI"))
    dataset_root = Path(os.environ["PI05_DATASET_ROOT"])
    dataset_id = os.environ.get("PI05_DATASET_REPO_ID", "lerobot/libero_plus")
    pi05_dir = repo / "examples/pi05_libero_finetune"

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for pi0.5 / PaliGemma access")
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(dataset_root / "meta" / "info.json")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found; GPU runtime is required")
    gpu = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    if not gpu:
        raise RuntimeError("no NVIDIA GPU detected")
    print("GPU:", gpu, flush=True)

    root.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    # π0.5 uses QUANTILES normalization for observation.state/action.  The public
    # LIBERO-plus proxy can ship only min/max/mean/std/count, so guarantee q01/q99
    # before model loading.  The utility is idempotent and exits quickly when the
    # quantiles are already present (e.g. after restoring a prepared Drive cache).
    quantile_report = root / "outputs" / "pi05_gpu_subset_smoke_quantile_gate.json"
    cmd([
        sys.executable,
        str(repo / "tools/data/ensure_pi05_quantile_stats.py"),
        "--root", str(dataset_root),
        "--report", str(quantile_report),
    ])
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    for feature in ("observation.state", "action"):
        if "q01" not in stats.get(feature, {}) or "q99" not in stats.get(feature, {}):
            raise RuntimeError(f"missing q01/q99 after quantile gate: {feature}")
    print("PI05 Quantile Stats Gate: PASS", flush=True)

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    total_episodes = int(info["total_episodes"])
    last = total_episodes - 1
    subset_eps = sorted(set([0, min(100, last), max(0, last - 1), last]))
    if len(subset_eps) < 2:
        raise RuntimeError(f"dataset too small for non-contiguous subset smoke: {total_episodes}")

    manifest_path = root / "outputs" / "pi05_gpu_subset_smoke_manifest.json"
    ids_bytes = json.dumps(subset_eps, separators=(",", ":")).encode()
    manifest = {
        "variant": "GPU_SUBSET_SMOKE",
        "dataset_id": dataset_id,
        "episode_ids": subset_eps,
        "episode_ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "summary": {"episode_count": len(subset_eps)},
        "screening_only": True,
        "note": "Infrastructure smoke only; never use for model selection.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("subset episodes:", subset_eps, flush=True)
    print("manifest:", manifest_path, flush=True)

    if shutil.which("uv") is None:
        cmd([sys.executable, "-m", "pip", "install", "-q", "uv"])
    cmd(["uv", "python", "install", "3.10"])
    py310 = subprocess.check_output(["uv", "python", "find", "3.10"], text=True).strip()

    data_root = root / "cache" / "pi05-gpu-subset-smoke"
    lerobot_root = root / "vendor" / "lerobot-pi05-gpu-subset-smoke"
    data_root.mkdir(parents=True, exist_ok=True)
    setup_env = os.environ.copy()
    setup_env.update({
        "PYTHON": py310,
        "DATA_ROOT": str(data_root),
        "LEROBOT_ROOT": str(lerobot_root),
        "INSTALL_FFMPEG": "0",
    })
    cmd(["bash", "scripts/setup_train.sh"], cwd=pi05_dir, env=setup_env)

    run_name = "pi05_gpu_subset_smoke"
    run_env = os.environ.copy()
    run_env.update({
        "DATA_ROOT": str(data_root),
        "LEROBOT_ROOT": str(lerobot_root),
        "ABLATION_MANIFEST": str(manifest_path),
        "PI05_DATASET_ROOT": str(dataset_root),
        "PI05_DATASET_REPO_ID": dataset_id,
        "PI05_VIDEO_BACKEND": "pyav",
        "ABLATION_BS": "1",
        "ABLATION_GA": "8",
        "ABLATION_STEPS": "1",
        "ABLATION_SEED": "1000",
        "RUN_NAME": run_name,
        "HF_TOKEN": os.environ["HF_TOKEN"],
    })
    cmd(
        ["bash", "-lc", "source env_train.sh && bash scripts/cheap_ablation_pi05.sh"],
        cwd=pi05_dir,
        env=run_env,
    )

    out_dir = data_root / "pi05-ft-outputs" / run_name
    summary = out_dir / "cheap_ablation_summary.json"
    checkpoint = out_dir / "checkpoints" / "last" / "pretrained_model"
    if not summary.exists():
        raise RuntimeError(f"summary missing: {summary}")
    if not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint missing: {checkpoint}")

    result = json.loads(summary.read_text())
    if int(result.get("optimizer_steps", -1)) != 1:
        raise RuntimeError(f"unexpected optimizer_steps: {result.get('optimizer_steps')}")
    if int(result.get("grad_accum", -1)) != 8:
        raise RuntimeError(f"unexpected grad_accum: {result.get('grad_accum')}")

    print(summary.read_text(), flush=True)
    print("PI05 GPU SUBSET SMOKE: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
