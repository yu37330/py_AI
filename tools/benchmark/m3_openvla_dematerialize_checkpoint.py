#!/usr/bin/env python3
"""Remove only ephemeral merged OpenVLA weights after simulator evaluation.

The persistent M3 artifact remains LoRA + action head + proprio projector + D10
statistics. This bounds Drive usage while allowing the evaluator to materialize
a complete 7B checkpoint just-in-time. No training state or D10 data is removed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("OpenVLA checkpoint dematerialization requires PARC_M3_EXECUTE=1")
    checkpoint = args.checkpoint_root.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not (checkpoint / "lora_adapter").is_dir():
        raise FileNotFoundError("persistent OpenVLA LoRA adapter missing; refusing cleanup")
    if not (checkpoint / "dataset_statistics.json").is_file():
        raise FileNotFoundError("persistent OpenVLA D10 statistics missing; refusing cleanup")
    if not any(checkpoint.glob("action_head--*checkpoint.pt")):
        raise FileNotFoundError("persistent OpenVLA action-head checkpoint missing; refusing cleanup")
    if not any(checkpoint.glob("proprio_projector--*checkpoint.pt")):
        raise FileNotFoundError("persistent OpenVLA proprio-projector checkpoint missing; refusing cleanup")

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("model") != "openvla_oft":
        raise ValueError("OpenVLA training result is not PASS")
    if Path(str(result.get("checkpoint_ref"))).resolve() != checkpoint:
        raise ValueError("training result/checkpoint cleanup path mismatch")

    targets = list(checkpoint.glob("model*.safetensors"))
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        targets.append(index)
    # config/generation config are tiny and harmless; keep them so the root
    # retains model metadata for diagnostics. Removing the weights alone makes
    # the root intentionally non-evaluator-ready and re-finalizable.
    removed_bytes = 0
    removed = []
    for path in targets:
        if path.is_file():
            removed_bytes += path.stat().st_size
            removed.append(path.name)
            path.unlink()

    if any(checkpoint.glob("model*.safetensors")) or (checkpoint / "model.safetensors.index.json").exists():
        raise RuntimeError("merged OpenVLA weights remain after dematerialization")
    if not (checkpoint / "lora_adapter").is_dir():
        raise RuntimeError("LoRA adapter disappeared during dematerialization")

    result["checkpoint_eval_ready"] = False
    result["merged_checkpoint_dematerialized"] = True
    result["merged_checkpoint_removed_bytes"] = int(removed_bytes)
    result["persistent_checkpoint_form"] = "lora_components_and_d10_stats"
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_eval_ready": False,
        "removed_files": removed,
        "removed_bytes": removed_bytes,
        "persistent_checkpoint_form": "lora_components_and_d10_stats",
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
