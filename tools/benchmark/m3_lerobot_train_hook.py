#!/usr/bin/env python3
"""Runtime hook consumed by exact-revision LeRobot M3 trainer patches.

The patched trainers only call the small ActiveRuntime API below.  All D10
schedule validation and actual raw-batch evidence lives here so π0.5 and
SmolVLA share identical sample-order semantics.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tools.benchmark.m3_lerobot_schedule_runtime import (
    CanonicalReferenceSampler,
    M3ScheduleEvidence,
)
from tools.benchmark.m3_sampling_schedule import (
    iter_references,
    load_d10_episode_lengths,
)
from tools.benchmark.m3_training_adapters import load_equal_data_schedule

SPEC_ENV = "PARC_M3_RUNTIME_SPEC"
_ACTIVE: "ActiveRuntime | None" = None


@dataclass
class ActiveRuntime:
    spec: dict[str, Any]
    sampler: CanonicalReferenceSampler
    evidence: M3ScheduleEvidence
    evidence_path: Path

    def before_batch_fetch(self) -> None:
        self.evidence.before_first_batch_fetch()

    def observe_raw_batch(self, batch: dict[str, Any]) -> None:
        self.evidence.observe_raw_batch(batch)

    def optimizer_boundary(self, *, sync_gradients: bool) -> bool:
        return self.evidence.optimizer_boundary(sync_gradients=bool(sync_gradients))

    def finalize(self) -> dict[str, Any]:
        payload = self.evidence.finalize()
        payload.update(
            {
                "schema_version": 1,
                "stage": "M3_lerobot_runtime_evidence",
                "status": "PASS",
                "model": self.spec["model"],
                "order": self.spec["order"],
                "seed": int(self.spec["seed"]),
                "schedule_sha256": self.spec["schedule_sha256"],
                "result_path": self.spec["result_path"],
            }
        )
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload


def _equal_data_refs(spec: dict[str, Any]) -> tuple[Iterator[dict[str, Any]], int]:
    schedule_path = Path(spec["equal_data_schedule"])
    schedule = load_equal_data_schedule(schedule_path)
    if schedule["schedule_sha256"] != spec["schedule_sha256"]:
        raise ValueError("runtime equal-data schedule SHA does not match run spec")
    if int(schedule["seed"]) != int(spec["seed"]):
        raise ValueError("runtime equal-data seed mismatch")
    refs = schedule["references"]
    return iter(refs), len(refs)


def _equal_wall_refs(spec: dict[str, Any]) -> tuple[Iterator[dict[str, Any]], int]:
    manifest_path = Path(spec["manifest_path"])
    dataset_root = Path(spec["dataset_root"])
    episode_ids, episode_lengths = load_d10_episode_lengths(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    refs = iter_references(
        episode_ids=episode_ids,
        episode_lengths=episode_lengths,
        seed=int(spec["seed"]),
        start_index=0,
    )
    # DataLoader/Accelerate may query sampler length even though equal-wall is a
    # time-bounded stream. One billion references is a non-materialized practical
    # infinity for a 30-minute single-A100 screening run.
    return refs, 1_000_000_000


def load_runtime_spec(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "model",
        "order",
        "track",
        "seed",
        "micro_batch",
        "gradient_accumulation",
        "effective_batch_size",
        "schedule_sha256",
        "manifest_path",
        "dataset_root",
        "result_path",
        "evidence_path",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"M3 runtime spec missing fields: {missing}")
    if data["model"] not in {"pi05", "smolvla"}:
        raise ValueError(f"LeRobot runtime cannot serve model {data['model']}")
    if data["order"] not in {"forward", "reverse"}:
        raise ValueError("invalid M3 order")
    if data["track"] not in {"equal_data", "equal_wall"}:
        raise ValueError("invalid M3 track")
    if int(data["effective_batch_size"]) != 32:
        raise ValueError("M3 effective batch drift")
    if int(data["micro_batch"]) * int(data["gradient_accumulation"]) != 32:
        raise ValueError("M3 micro-batch/GA mismatch")
    if data["track"] == "equal_data" and not data.get("equal_data_schedule"):
        raise ValueError("equal-data runtime requires the materialized canonical schedule")
    return data


def setup_runtime(dataset: Any, spec_path: str | Path | None = None) -> ActiveRuntime:
    global _ACTIVE
    raw = str(spec_path or os.environ.get(SPEC_ENV, "")).strip()
    if not raw:
        raise RuntimeError(f"{SPEC_ENV} is required for guarded M3 LeRobot training")
    spec = load_runtime_spec(Path(raw))
    if spec["track"] == "equal_data":
        refs_for_sampler, finite_length = _equal_data_refs(spec)
        refs_for_evidence, _ = _equal_data_refs(spec)
    else:
        refs_for_sampler, finite_length = _equal_wall_refs(spec)
        refs_for_evidence, _ = _equal_wall_refs(spec)
    sampler = CanonicalReferenceSampler(
        dataset,
        refs_for_sampler,
        finite_length=finite_length,
    )
    evidence = M3ScheduleEvidence(
        track=spec["track"],
        micro_batch=int(spec["micro_batch"]),
        gradient_accumulation=int(spec["gradient_accumulation"]),
        expected_references=refs_for_evidence,
        wall_target_sec=float(spec.get("train_loop_sec") or 1800.0),
    )
    _ACTIVE = ActiveRuntime(
        spec=spec,
        sampler=sampler,
        evidence=evidence,
        evidence_path=Path(spec["evidence_path"]),
    )
    return _ACTIVE


def get_active_runtime() -> ActiveRuntime | None:
    return _ACTIVE
