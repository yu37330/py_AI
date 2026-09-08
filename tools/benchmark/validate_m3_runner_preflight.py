#!/usr/bin/env python3
"""Validate M3 runner provenance before any model training is allowed to start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_VARIANT = "V2_SQRT_BALANCED_RAW"
EXPECTED_DATASET_ID = "lerobot/libero_plus"
EXPECTED_DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
EXPECTED_EPISODES = 10758
EXPECTED_FRAMES = 1620614
EXPECTED_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
EXPECTED_SAMPLE_BUDGET = 4800
EXPECTED_TRAIN_LOOP_SEC = 1800
EXPECTED_SMOLVLA_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
EXPECTED_OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def model_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(m["id"]): m for m in registry.get("models", [])}


def validate_static(repo: Path) -> dict[str, Any]:
    registry = read_json(repo / "experiments/model_registry_v1.json")
    budget = read_json(repo / "experiments/plans/model_benchmark_budget_v1.json")
    runner = read_json(repo / "experiments/plans/model_benchmark_runner_v1.json")
    models = model_map(registry)

    require(budget.get("status") == "FROZEN", "M3 benchmark budget is not FROZEN")
    require(budget.get("selected_dataset_variant") == EXPECTED_VARIANT, "budget variant mismatch")
    require(budget["equal_data_exposure"]["sample_budget"] == EXPECTED_SAMPLE_BUDGET, "sample budget mismatch")
    require(budget["equal_wall_time"]["train_loop_sec"] == EXPECTED_TRAIN_LOOP_SEC, "wall-time budget mismatch")
    require(budget["equal_data_exposure"].get("same_episode_pool") is True, "same_episode_pool must be true")
    require(budget["equal_data_exposure"].get("same_sampling_policy") is True, "same_sampling_policy must be true")
    require(budget["promotion"].get("training_loss_alone_is_not_sufficient") is True, "promotion contract weakened")

    require(runner.get("status") == "FROZEN_PREFLIGHT", "runner preflight contract not frozen")
    selected = runner["selected_dataset"]
    require(selected.get("variant") == EXPECTED_VARIANT, "runner variant mismatch")
    require(selected.get("repo_id") == EXPECTED_DATASET_ID, "runner dataset id mismatch")
    require(selected.get("revision") == EXPECTED_DATASET_REVISION, "runner dataset revision mismatch")
    require(selected.get("episode_count") == EXPECTED_EPISODES, "runner episode count mismatch")
    require(selected.get("frame_count") == EXPECTED_FRAMES, "runner frame count mismatch")
    require(selected.get("episode_ids_sha256") == EXPECTED_HASH, "runner dataset hash mismatch")
    require(runner["tracks"]["equal_data_exposure"]["sample_budget"] == EXPECTED_SAMPLE_BUDGET, "runner sample budget mismatch")
    require(runner["tracks"]["equal_wall_time"]["train_loop_sec"] == EXPECTED_TRAIN_LOOP_SEC, "runner wall-time mismatch")
    guard = runner["execution_guard"]
    require(guard.get("default_mode") == "preflight", "runner must default to preflight")
    require(guard.get("preflight_starts_training") is False, "preflight must not train")
    require(guard.get("execute_environment_variable") == "PARC_M3_EXECUTE", "execute guard variable mismatch")
    require(guard.get("execute_environment_value") == "1", "execute guard value mismatch")

    require(models["pi05"].get("framework_revision") == "v0.4.4", "pi05 registry pin mismatch")
    require(models["smolvla"].get("upstream_revision") == EXPECTED_SMOLVLA_REF, "SmolVLA registry pin mismatch")
    require(models["openvla_oft"].get("upstream_revision") == EXPECTED_OPENVLA_REF, "OpenVLA registry pin mismatch")
    require(runner["models"]["smolvla"]["source_ref"] == EXPECTED_SMOLVLA_REF, "SmolVLA runner pin mismatch")
    require(runner["models"]["openvla_oft"]["source_ref"] == EXPECTED_OPENVLA_REF, "OpenVLA runner pin mismatch")
    require(runner["models"]["openvla_oft"]["required_bridge_type"] == "lerobot_streaming", "OpenVLA bridge type mismatch")

    smol_text = (repo / "experiments/configs/smolvla.yaml").read_text(encoding="utf-8")
    openvla_text = (repo / "experiments/configs/openvla_oft.yaml").read_text(encoding="utf-8")
    require(EXPECTED_SMOLVLA_REF in smol_text, "SmolVLA YAML is not pinned to registry")
    require("optimizer_lr: 1.0e-4" in smol_text, "SmolVLA native optimizer preset missing")
    require("TODO_" not in smol_text, "SmolVLA YAML still contains TODOs")
    require(EXPECTED_OPENVLA_REF in openvla_text, "OpenVLA YAML is not pinned to registry")
    require("https://github.com/small-zeng/openvla-oft.git" in openvla_text, "OpenVLA YAML source repo mismatch")
    require("learning_rate: 5.0e-4" in openvla_text and "lora_r: 32" in openvla_text, "OpenVLA native preset missing")
    require("TODO_" not in openvla_text, "OpenVLA YAML still contains TODOs")

    return {"registry": registry, "budget": budget, "runner": runner}


def validate_runtime(protocol: dict[str, Any], manifest: dict[str, Any], streaming: dict[str, Any]) -> None:
    require(protocol.get("status") == "READY_FOR_RUNNER", "Notebook 70 controller is not READY_FOR_RUNNER")
    require(protocol.get("selected_dataset_variant") == EXPECTED_VARIANT, "controller variant mismatch")
    require(protocol.get("selected_episode_count") == EXPECTED_EPISODES, "controller episode count mismatch")
    require(protocol.get("selected_frame_count") == EXPECTED_FRAMES, "controller frame count mismatch")
    require(protocol.get("selected_episode_ids_sha256") == EXPECTED_HASH, "controller dataset hash mismatch")
    require(protocol.get("openvla_bridge_type") == "lerobot_streaming", "controller did not select streaming OpenVLA bridge")
    require(protocol["equal_data_exposure"]["sample_budget"] == EXPECTED_SAMPLE_BUDGET, "controller sample budget mismatch")
    require(protocol["equal_wall_time"]["a100_train_loop_sec"] == EXPECTED_TRAIN_LOOP_SEC, "controller wall-time mismatch")
    require(protocol.get("blocked_reasons") == [], "controller has blocked reasons")

    require(manifest.get("schema_version") == 2 and manifest.get("group_aware") is True, "manifest must be group-aware schema v2")
    require(manifest.get("variant") == EXPECTED_VARIANT, "manifest variant mismatch")
    require(manifest.get("episode_ids_sha256") == EXPECTED_HASH, "manifest hash mismatch")
    require(manifest.get("summary", {}).get("episode_count") == EXPECTED_EPISODES, "manifest episode count mismatch")
    require(manifest.get("summary", {}).get("frame_count") == EXPECTED_FRAMES, "manifest frame count mismatch")

    require(streaming.get("status") == "PASS", "69c streaming contract is not PASS")
    require(streaming.get("bridge_type") == "lerobot_streaming", "69c bridge type mismatch")
    require(streaming.get("selected_dataset_variant") == EXPECTED_VARIANT, "69c variant mismatch")
    require(streaming.get("source_dataset_id") == EXPECTED_DATASET_ID, "69c dataset id mismatch")
    require(streaming.get("source_dataset_revision") == EXPECTED_DATASET_REVISION, "69c dataset revision mismatch")
    require(streaming.get("source_episode_ids_sha256") == EXPECTED_HASH, "69c dataset hash mismatch")
    require(streaming.get("source_episode_count") == EXPECTED_EPISODES, "69c episode count mismatch")
    require(streaming.get("source_frame_count") == EXPECTED_FRAMES, "69c frame count mismatch")
    require(streaming.get("openvla_oft_revision") == EXPECTED_OPENVLA_REF, "69c OpenVLA revision mismatch")
    oc = streaming.get("openvla_contract", {})
    require(oc.get("action_dim") == 7 and oc.get("proprio_dim") == 8 and oc.get("action_chunk") == 8, "69c OpenVLA shape contract mismatch")
    require(oc.get("normalization_type") == "bounds_q99", "69c normalization mismatch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--protocol", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--streaming-contract", type=Path)
    p.add_argument("--out", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    validate_static(args.repo_root)
    print("[1/2] static registry / config / budget / runner contract: PASS", flush=True)

    runtime_paths = (args.protocol, args.manifest, args.streaming_contract)
    if any(runtime_paths) and not all(runtime_paths):
        raise RuntimeError("--protocol, --manifest, and --streaming-contract must be supplied together")
    if all(runtime_paths):
        validate_runtime(read_json(args.protocol), read_json(args.manifest), read_json(args.streaming_contract))
        print("[2/2] Notebook 70 + fixed manifest + 69c streaming runtime contracts: PASS", flush=True)
    else:
        print("[2/2] runtime contracts: SKIPPED (static validation only)", flush=True)

    result = {
        "schema_version": 1,
        "stage": "M3_runner_preflight",
        "status": "READY_FOR_BATCH_PROBE",
        "training_started": False,
        "selected_dataset_variant": EXPECTED_VARIANT,
        "selected_episode_ids_sha256": EXPECTED_HASH,
        "sample_budget": EXPECTED_SAMPLE_BUDGET,
        "train_loop_sec": EXPECTED_TRAIN_LOOP_SEC,
        "smolvla_revision": EXPECTED_SMOLVLA_REF,
        "openvla_oft_revision": EXPECTED_OPENVLA_REF,
        "next_step": "Run per-model A100 forward/backward batch probes; do not start M3 benchmark training yet.",
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("=== M3 RUNNER PREFLIGHT: READY_FOR_BATCH_PROBE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
