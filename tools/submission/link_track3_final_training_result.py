#!/usr/bin/env python3
"""Bind a completed M3-style training result to the immutable Track3 final-run config.

This is a provenance operation only. It does not launch or modify training and
does not alter checkpoint bytes. The linked result is what final simulator
evaluation and artifact freeze must consume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
EFFECTIVE_BATCH = 32
EQUAL_DATA_SAMPLES = 4800
EQUAL_DATA_UPDATES = 150
EQUAL_WALL_SEC = 1800.0


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json_without_self(config: dict[str, Any]) -> str:
    payload = dict(config)
    payload.pop("final_run_config_sha256", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def validate_final_run_config(path: Path) -> dict[str, Any]:
    cfg = load_object(path)
    if cfg.get("status") != "READY_FOR_FINAL_RUN":
        raise ValueError("final-run config is not READY_FOR_FINAL_RUN")
    if cfg.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("final-run config D10 mismatch")
    expected_hash = sha256_json_without_self(cfg)
    if cfg.get("final_run_config_sha256") != expected_hash:
        raise ValueError("final-run config self hash mismatch")
    candidate = str(cfg.get("candidate") or "")
    if candidate not in {"pi05", "smolvla", "openvla_oft"}:
        raise ValueError("final-run config candidate invalid")
    if not str(cfg.get("source_ref") or "").strip():
        raise ValueError("final-run config source_ref missing")
    budget = cfg.get("budget")
    if not isinstance(budget, dict) or budget.get("track") not in {"equal_data", "equal_wall"}:
        raise ValueError("final-run config budget invalid")
    if int(budget.get("effective_batch_size", -1)) != EFFECTIVE_BATCH:
        raise ValueError("final-run config effective batch mismatch")
    return cfg


def link_result(*, config_path: Path, raw_result_path: Path) -> dict[str, Any]:
    cfg = validate_final_run_config(config_path)
    result = load_object(raw_result_path)
    if result.get("status") != "PASS":
        raise ValueError("raw final training result must be PASS")
    if result.get("model") != cfg["candidate"]:
        raise ValueError("final training candidate mismatch")
    if result.get("source_ref") != cfg["source_ref"]:
        raise ValueError("final training source_ref differs from promoted/frozen source")
    if result.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("final training D10 mismatch")
    if int(result.get("effective_batch_size", -1)) != EFFECTIVE_BATCH:
        raise ValueError("final training effective batch mismatch")
    if not str(result.get("checkpoint_ref") or "").strip():
        raise ValueError("final training checkpoint_ref missing")
    if result.get("mode") not in {"benchmark", "final"}:
        raise ValueError("final training must come from benchmark/final execution, not smoke")

    samples = result.get("samples_consumed")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("final training must record actual samples_consumed")
    updates = result.get("optimizer_updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("final training must record optimizer_updates")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("final training metrics missing")
    wall = metrics.get("train_wall_time")
    peak = metrics.get("peak_train_vram")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or float(wall) < 0:
        raise ValueError("final training train_wall_time invalid")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or float(peak) < 0:
        raise ValueError("final training peak_train_vram invalid")

    budget = cfg["budget"]
    if result.get("track") != budget["track"]:
        raise ValueError("final training track differs from final-run config")
    if budget["track"] == "equal_data":
        if samples != EQUAL_DATA_SAMPLES:
            raise ValueError(f"final equal-data run must consume exactly {EQUAL_DATA_SAMPLES} samples")
        if updates != EQUAL_DATA_UPDATES:
            raise ValueError(f"final equal-data run must contain exactly {EQUAL_DATA_UPDATES} optimizer updates")
    else:
        if float(wall) + 1e-9 < EQUAL_WALL_SEC:
            raise ValueError(f"final equal-wall run must reach at least {EQUAL_WALL_SEC} sec")
        if samples <= 0 or samples % EFFECTIVE_BATCH != 0:
            raise ValueError("final equal-wall samples must end at an effective-batch boundary")

    linked = dict(result)
    linked["stage"] = "Track3_final_candidate_training"
    linked["final_candidate_run"] = True
    linked["final_run_config_ref"] = str(config_path.resolve())
    linked["final_run_config_sha256"] = cfg["final_run_config_sha256"]
    linked["promotion_summary_sha256"] = cfg["promotion_summary_sha256"]
    linked["recipe_sha256"] = cfg["recipe"]["sha256"]
    linked["model_registry_sha256"] = cfg["model_registry"]["sha256"]
    linked["training_loss_used_for_selection"] = False
    linked["linked_result_sha256"] = hashlib.sha256(canonical_json(linked)).hexdigest()
    return linked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--final-run-config", type=Path, required=True)
    p.add_argument("--raw-training-result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    linked = link_result(config_path=args.final_run_config, raw_result_path=args.raw_training_result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(linked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": linked["status"],
        "stage": linked["stage"],
        "candidate": linked["model"],
        "track": linked["track"],
        "samples_consumed": linked["samples_consumed"],
        "train_wall_time": linked["metrics"]["train_wall_time"],
        "checkpoint_ref": linked["checkpoint_ref"],
        "final_run_config_sha256": linked["final_run_config_sha256"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
