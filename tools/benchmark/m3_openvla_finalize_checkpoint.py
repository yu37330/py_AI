#!/usr/bin/env python3
"""Merge an M3 OpenVLA LoRA checkpoint into an evaluator-loadable root.

This is intentionally a post-training operation. It must run after the frozen
train-loop timer has stopped, so model merge/serialization cannot affect the
M3 equal-wall metric. Component checkpoints and D10 normalization stats remain
next to the merged VLA root for the pinned OpenVLA-OFT evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"
BASE_MODEL = "openvla/openvla-7b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    return p.parse_args()


def _verify_source(root: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SOURCE_REF:
        raise RuntimeError(f"OpenVLA-OFT source mismatch during checkpoint finalization: {head}")


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("OpenVLA checkpoint finalization requires PARC_M3_EXECUTE=1")
    _verify_source(args.source_root)
    checkpoint = args.checkpoint_root.resolve()
    adapter = checkpoint / "lora_adapter"
    stats = checkpoint / "dataset_statistics.json"
    if not adapter.is_dir():
        raise FileNotFoundError(f"OpenVLA LoRA adapter missing: {adapter}")
    if not stats.is_file():
        raise FileNotFoundError(f"OpenVLA D10 statistics missing: {stats}")
    if not any(checkpoint.glob("action_head--*checkpoint.pt")):
        raise FileNotFoundError("OpenVLA action-head checkpoint missing")
    if not any(checkpoint.glob("proprio_projector--*checkpoint.pt")):
        raise FileNotFoundError("OpenVLA proprio-projector checkpoint missing")

    sys.path.insert(0, str(args.source_root))
    import torch  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForVision2Seq  # noqa: PLC0415

    base_path = snapshot_download(repo_id=BASE_MODEL)
    base = AutoModelForVision2Seq.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(checkpoint, safe_serialization=True)

    # The upstream evaluator imports local OpenVLA auto classes, but config.json
    # must still identify this root as a complete model checkpoint.
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("merged OpenVLA checkpoint did not write config.json")
    model_files = list(checkpoint.glob("model*.safetensors"))
    if not model_files:
        raise FileNotFoundError("merged OpenVLA checkpoint did not write model safetensors")

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("model") != "openvla_oft":
        raise ValueError("OpenVLA training result is not PASS")
    if Path(str(result.get("checkpoint_ref"))).resolve() != checkpoint:
        raise ValueError("training result/checkpoint finalization path mismatch")
    result["checkpoint_eval_ready"] = True
    result["lora_merged_for_evaluation"] = True
    result["merge_excluded_from_train_wall_time"] = True
    result["merged_model_file_count"] = len(model_files)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_eval_ready": True,
        "lora_merged_for_evaluation": True,
        "merge_excluded_from_train_wall_time": True,
        "model_files": len(model_files),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
