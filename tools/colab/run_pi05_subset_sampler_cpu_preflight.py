#!/usr/bin/env python3
"""CPU-only preflight for LeRobot episode-subset sampler behavior.

This intentionally does not load pi0.5 weights and does not require a GPU.
It verifies the exact failure class seen in cheap ablation: a filtered episode
subset must use relative row indices when EpisodeAwareSampler is active.

Expected env:
  PY_AI_REPO=/path/to/py_AI
  PI05_DATASET_ROOT=/path/to/local/lerobot dataset
  PI05_DATASET_REPO_ID=lerobot/libero_plus (optional)
  PARC_ROOT=/content/parc2026 (optional)
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


def cmd(args, *, cwd=None, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(args, cwd=cwd, env=env, check=True)


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", root / "py_AI"))
    dataset_root = Path(os.environ["PI05_DATASET_ROOT"])
    dataset_id = os.environ.get("PI05_DATASET_REPO_ID", "lerobot/libero_plus")
    pi05_dir = repo / "examples/pi05_libero_finetune"

    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(dataset_root / "meta" / "info.json")

    print("=== CPU subset-sampler preflight ===", flush=True)
    print("dataset:", dataset_id, flush=True)
    print("root:", dataset_root, flush=True)

    if shutil.which("uv") is None:
        cmd([sys.executable, "-m", "pip", "install", "-q", "uv"])
    cmd(["uv", "python", "install", "3.10"])
    py310 = subprocess.check_output(["uv", "python", "find", "3.10"], text=True).strip()

    data_root = root / "cache" / "pi05-cpu-preflight"
    lerobot_root = root / "vendor" / "lerobot-pi05-cpu-preflight"
    data_root.mkdir(parents=True, exist_ok=True)
    setup_env = os.environ.copy()
    setup_env.update({
        "PYTHON": py310,
        "DATA_ROOT": str(data_root),
        "LEROBOT_ROOT": str(lerobot_root),
        "INSTALL_FFMPEG": "0",
    })
    cmd(["bash", "scripts/setup_train.sh"], cwd=pi05_dir, env=setup_env)

    venv_py = lerobot_root / ".venv" / "bin" / "python"
    probe = r'''
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
import torch
import os

root = os.environ["PI05_DATASET_ROOT"]
repo_id = os.environ.get("PI05_DATASET_REPO_ID", "lerobot/libero_plus")

# Deliberately non-contiguous subset including a late episode. This reproduces
# the absolute-vs-relative mismatch without loading the full training subset.
base = LeRobotDataset(repo_id, root=root, episodes=None, video_backend="pyav")
last = base.meta.total_episodes - 1
subset_eps = sorted(set([0, min(100, last), max(0, last - 1), last]))

ds = LeRobotDataset(repo_id, root=root, episodes=subset_eps, video_backend="pyav")
mapping = getattr(ds, "_absolute_to_relative_idx", None)
assert mapping is not None and mapping, "subset absolute->relative mapping missing"

sampler = EpisodeAwareSampler(
    ds.meta.episodes["dataset_from_index"],
    ds.meta.episodes["dataset_to_index"],
    episode_indices_to_use=ds.episodes,
    index_mapping=mapping,
    drop_n_last_frames=49,
    shuffle=False,
)
assert len(sampler) > 0
mx = max(sampler.indices)
mn = min(sampler.indices)
assert mn >= 0
assert mx < len(ds), (mx, len(ds))

# Exercise actual dataset indexing through DataLoader on CPU. batch_size=1 and
# num_workers=0 keep failures deterministic and easy to diagnose.
dl = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0)
batch = next(iter(dl))
assert "action" in batch
print("subset episodes:", subset_eps)
print("filtered frames:", len(ds))
print("sampler range:", mn, mx)
print("first batch action shape:", tuple(batch["action"].shape))
print("SUBSET SAMPLER CPU GATE: PASS")
'''

    probe_env = os.environ.copy()
    probe_env.update({
        "PI05_DATASET_ROOT": str(dataset_root),
        "PI05_DATASET_REPO_ID": dataset_id,
        "PYTHONUNBUFFERED": "1",
    })
    cmd([str(venv_py), "-c", probe], env=probe_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
