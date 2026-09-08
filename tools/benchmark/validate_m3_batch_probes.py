#!/usr/bin/env python3
"""Validate all three A100 batch probes and freeze batch/GA for M3 runner implementation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
EXPECTED_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
    "openvla_oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    selected: dict[str, dict] = {}
    for model, source_ref in EXPECTED_REFS.items():
        path = args.probe_dir / f"{model}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            raise RuntimeError(f"{model} batch probe is not PASS: {result}")
        if result.get("model") != model or result.get("source_ref") != source_ref:
            raise RuntimeError(f"{model} source identity mismatch: {result}")
        if result.get("selected_episode_ids_sha256") != EXPECTED_HASH:
            raise RuntimeError(f"{model} selected dataset hash mismatch")
        if result.get("probe_only") is not True or result.get("benchmark_training_started") is not False:
            raise RuntimeError(f"{model} probe/training boundary mismatch")
        bs = int(result.get("selected_micro_batch", 0))
        ga = int(result.get("gradient_accumulation", 0))
        eff = int(result.get("effective_batch_size", 0))
        if bs <= 0 or ga <= 0 or eff != 32 or bs * ga != 32:
            raise RuntimeError(f"{model} effective-batch contract mismatch: bs={bs} ga={ga} eff={eff}")
        if int(result.get("optimizer_steps", -1)) != 1:
            raise RuntimeError(f"{model} probe did not perform exactly one optimizer step")
        if result.get("loss") is None:
            raise RuntimeError(f"{model} probe did not report a loss")
        selected[model] = {
            "source_ref": source_ref,
            "micro_batch": bs,
            "gradient_accumulation": ga,
            "effective_batch_size": 32,
            "peak_vram_mib": int(result.get("peak_vram_mib", 0) or 0),
            "probe_loss": float(result["loss"]),
            "gpu_name": result.get("gpu_name"),
            "gpu_vram_mib": int(result.get("gpu_vram_mib", 0) or 0),
        }

    summary = {
        "schema_version": 1,
        "stage": "M3_A100_batch_probe_controller",
        "status": "READY_FOR_M3_RUNNER_IMPLEMENTATION",
        "selected_dataset_variant": "V2_SQRT_BALANCED_RAW",
        "selected_episode_ids_sha256": EXPECTED_HASH,
        "effective_batch_size": 32,
        "equal_data_protocol": {
            "sample_budget": 4800,
            "optimizer_updates": 150,
            "samples_per_optimizer_update": 32,
        },
        "models": selected,
        "probe_only": True,
        "benchmark_training_started": False,
        "checkpoint_promoted": False,
        "next_step": "implement the guarded 4800-sample and 1800-second M3 runners using these frozen per-model micro-batch/GA values",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("=== 72d M3 BATCH PROBE CONTROLLER: READY_FOR_M3_RUNNER_IMPLEMENTATION ===", flush=True)
    print("M3 benchmark training started: False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
