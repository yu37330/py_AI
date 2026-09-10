#!/usr/bin/env python3
"""Framework adapter contracts for the guarded PARC2026 M3 benchmark.

This module is deliberately pre-execution.  It turns the validated 72d batch
summary plus the frozen deterministic sampling policy into framework-specific
run specifications, but never starts training by itself.  The worker layer may
only consume these validated specs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.benchmark.m3_runner_core import (
    D10_HASH,
    D10_VARIANT,
    EFFECTIVE_BATCH,
    EQUAL_DATA_BUDGET,
    EQUAL_WALL_SEC,
    MODELS,
    ORDERS,
    TRACKS,
    load_batch_probe_summary,
    result_path,
)
from tools.benchmark.m3_sampling_schedule import POLICY, SEED_SET, sha256_json

EXPECTED_SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
    "openvla_oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}


@dataclass(frozen=True)
class AdapterDefinition:
    model: str
    framework: str
    source_ref: str
    worker: str
    base_checkpoint: str
    schedule_transport: str
    notes: tuple[str, ...]


ADAPTERS: dict[str, AdapterDefinition] = {
    "pi05": AdapterDefinition(
        model="pi05",
        framework="lerobot_v044",
        source_ref=EXPECTED_SOURCE_REFS["pi05"],
        worker="tools/benchmark/workers/run_m3_pi05.py",
        base_checkpoint="lerobot/pi05_libero_base",
        schedule_transport="canonical_lerobot_index_sampler",
        notes=(
            "LoRA r16",
            "framework steps must not be used as sample accounting",
        ),
    ),
    "smolvla": AdapterDefinition(
        model="smolvla",
        framework="lerobot_pinned",
        source_ref=EXPECTED_SOURCE_REFS["smolvla"],
        worker="tools/benchmark/workers/run_m3_smolvla.py",
        base_checkpoint="lerobot/smolvla_base",
        schedule_transport="canonical_lerobot_index_sampler",
        notes=(
            "native LeRobot micro-step semantics",
            "actual examples consumed remain the accounting source of truth",
        ),
    ),
    "openvla_oft": AdapterDefinition(
        model="openvla_oft",
        framework="openvla_oft_pinned",
        source_ref=EXPECTED_SOURCE_REFS["openvla_oft"],
        worker="tools/benchmark/workers/run_m3_openvla_oft.py",
        base_checkpoint="openvla/openvla-7b",
        schedule_transport="canonical_lerobot_stream_references",
        notes=(
            "LoRA r32 / L1 / 2 images / proprio",
            "image augmentation must be enabled in real M3 training",
            "never materialize full RLDS",
        ),
    ),
}


def load_equal_data_schedule(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "FROZEN":
        raise ValueError("equal-data schedule must be FROZEN")
    if data.get("policy") != POLICY:
        raise ValueError("equal-data sampling policy mismatch")
    if data.get("selected_dataset_variant") != D10_VARIANT:
        raise ValueError("equal-data dataset variant mismatch")
    if data.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("equal-data D10 hash mismatch")
    if int(data.get("sample_budget", -1)) != EQUAL_DATA_BUDGET:
        raise ValueError("equal-data sample budget mismatch")
    if int(data.get("seed", -1)) not in SEED_SET:
        raise ValueError("equal-data seed is outside the frozen seed set")
    refs = data.get("references")
    if not isinstance(refs, list) or len(refs) != EQUAL_DATA_BUDGET:
        raise ValueError("equal-data schedule must contain exactly 4,800 references")
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            raise ValueError(f"schedule reference {index} is not an object")
        if int(ref.get("sample_index", -1)) != index:
            raise ValueError(f"schedule sample_index drift at {index}")
        if int(ref.get("episode_id", -1)) < 0 or int(ref.get("timestep", -1)) < 0:
            raise ValueError(f"invalid schedule reference at {index}: {ref}")
    if data.get("references_sha256") != sha256_json(refs):
        raise ValueError("equal-data reference hash mismatch")
    schedule_copy = dict(data)
    claimed = schedule_copy.pop("schedule_sha256", None)
    if claimed != sha256_json(schedule_copy):
        raise ValueError("equal-data schedule SHA mismatch")
    return data


def build_equal_wall_stream_descriptor(*, seed: int) -> dict[str, Any]:
    if int(seed) not in SEED_SET:
        raise ValueError("equal-wall seed is outside the frozen seed set")
    descriptor = {
        "schema_version": 1,
        "stage": "M3_equal_wall_sampling_stream",
        "status": "FROZEN",
        "policy": POLICY,
        "seed": int(seed),
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_HASH,
        "start_sample_index": 0,
        "materialized": False,
        "stop_rule": "first_safe_optimizer_boundary_at_or_after_1800_train_loop_seconds",
        "train_loop_sec": EQUAL_WALL_SEC,
    }
    descriptor["schedule_sha256"] = sha256_json(descriptor)
    return descriptor


def build_adapter_plan(
    *,
    batch_summary_path: Path,
    equal_data_schedule_path: Path,
    results_root: Path,
    seed: int,
) -> dict[str, Any]:
    summary, batches = load_batch_probe_summary(Path(batch_summary_path))
    equal_data = load_equal_data_schedule(Path(equal_data_schedule_path))
    if int(equal_data["seed"]) != int(seed):
        raise ValueError("equal-data schedule seed does not match requested seed")
    equal_wall = build_equal_wall_stream_descriptor(seed=int(seed))

    runs: list[dict[str, Any]] = []
    for order in ("forward", "reverse"):
        for track in TRACKS:
            for sequence_index, model in enumerate(ORDERS[order]):
                adapter = ADAPTERS[model]
                batch = batches[model]
                if adapter.source_ref != batch["source_ref"]:
                    raise ValueError(f"adapter/source mismatch for {model}")
                schedule = equal_data if track == "equal_data" else equal_wall
                runs.append(
                    {
                        "order": order,
                        "track": track,
                        "model": model,
                        "sequence_index": sequence_index,
                        "framework": adapter.framework,
                        "source_ref": adapter.source_ref,
                        "base_checkpoint": adapter.base_checkpoint,
                        "worker": adapter.worker,
                        "schedule_transport": adapter.schedule_transport,
                        "schedule_sha256": schedule["schedule_sha256"],
                        "seed": int(seed),
                        "micro_batch": int(batch["micro_batch"]),
                        "gradient_accumulation": int(batch["gradient_accumulation"]),
                        "effective_batch_size": EFFECTIVE_BATCH,
                        "sample_budget": EQUAL_DATA_BUDGET if track == "equal_data" else None,
                        "train_loop_sec": EQUAL_WALL_SEC if track == "equal_wall" else None,
                        "result_path": str(
                            result_path(
                                Path(results_root), order=order, track=track, model=model
                            )
                        ),
                        "execution_guard": "PARC_M3_EXECUTE=1",
                        "notes": list(adapter.notes),
                    }
                )

    return {
        "schema_version": 1,
        "stage": "M3_training_adapter_preflight",
        "status": "READY_FOR_GUARDED_SMOKE_IMPLEMENTATION",
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_HASH,
        "seed": int(seed),
        "batch_probe_summary_status": summary["status"],
        "equal_data_schedule": str(equal_data_schedule_path),
        "equal_data_schedule_sha256": equal_data["schedule_sha256"],
        "equal_wall_stream": equal_wall,
        "orders": {name: list(models) for name, models in ORDERS.items()},
        "runs": runs,
        "run_count": len(runs),
        "training_started": False,
        "automatic_benchmark_start": False,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-summary", type=Path, required=True)
    p.add_argument("--equal-data-schedule", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=SEED_SET, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_adapter_plan(
        batch_summary_path=args.batch_summary,
        equal_data_schedule_path=args.equal_data_schedule,
        results_root=args.results_root,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": plan["status"],
                "run_count": plan["run_count"],
                "training_started": plan["training_started"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
