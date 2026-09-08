#!/usr/bin/env python3
"""M3 runner source/config preflight. This script never starts model training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


SMOLVLA_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"


def out(*args) -> None:
    print(*args, flush=True)


def run(args, *, cwd: Path | None = None) -> None:
    out(">>>", " ".join(str(x) for x in args))
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def checkout_exact(path: Path, url: str, ref: str, required: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q", str(path)])
        run(["git", "-C", str(path), "remote", "add", "origin", url])
    run(["git", "-C", str(path), "fetch", "-q", "--depth", "1", "origin", ref])
    run(["git", "-C", str(path), "checkout", "-q", "--force", "FETCH_HEAD"])
    got = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if got != ref:
        raise RuntimeError(f"source pin mismatch for {url}: {got} != {ref}")
    required_path = path / required
    if not required_path.is_file():
        raise FileNotFoundError(required_path)
    return got


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", root / "py_AI_m3_preflight"))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    if not (repo / ".git").exists():
        raise FileNotFoundError(repo)

    protocol = drive / "model-benchmark-v1/comparison_protocol.json"
    manifest = drive / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json"
    streaming = drive / "openvla-streaming-selected-v1/streaming_bridge_contract.json"
    out_dir = drive / "model-benchmark-v1"
    result_path = out_dir / "m3_runner_preflight.json"
    source_path = out_dir / "m3_source_checkout.json"
    for path in (protocol, manifest, streaming):
        if not path.is_file():
            raise FileNotFoundError(path)

    validator = repo / "tools/benchmark/validate_m3_runner_preflight.py"
    run([
        sys.executable,
        "-u",
        validator,
        "--repo-root",
        repo,
        "--protocol",
        protocol,
        "--manifest",
        manifest,
        "--streaming-contract",
        streaming,
        "--out",
        result_path,
    ], cwd=repo)

    vendor = root / "vendor"
    sources = {
        "smolvla": {
            "url": "https://github.com/huggingface/lerobot.git",
            "ref": SMOLVLA_REF,
            "path": vendor / "lerobot-smolvla-m3",
            "required": "src/lerobot/scripts/lerobot_train.py",
        },
        "openvla_oft": {
            "url": "https://github.com/small-zeng/openvla-oft.git",
            "ref": OPENVLA_REF,
            "path": vendor / "openvla-oft-m3",
            "required": "vla-scripts/finetune.py",
        },
    }
    checked = {}
    for name, cfg in sources.items():
        got = checkout_exact(cfg["path"], cfg["url"], cfg["ref"], cfg["required"])
        checked[name] = {
            "url": cfg["url"],
            "revision": got,
            "training_entry": cfg["required"],
            "path": str(cfg["path"]),
            "status": "PASS",
        }
        out(f"{name}: {got} :: {cfg['required']} PASS")

    source_result = {
        "schema_version": 1,
        "stage": "M3_source_checkout",
        "status": "PASS",
        "training_started": False,
        "sources": checked,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(source_result, indent=2) + "\n", encoding="utf-8")

    preflight = json.loads(result_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "READY_FOR_BATCH_PROBE" or preflight.get("training_started") is not False:
        raise RuntimeError(f"unexpected preflight result: {preflight}")
    out("=== M3 RUNNER PREFLIGHT COMPLETE ===")
    out("Status: READY_FOR_BATCH_PROBE")
    out("Training started: False")
    out("Next: A100 per-model batch/forward-backward smoke; do not start M3 benchmark training yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
