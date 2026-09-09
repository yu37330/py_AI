#!/usr/bin/env python3
"""Model-independent accounting/orchestration core for PARC2026 M3.

This module does not train a model by itself.  Framework-specific adapters plug
into these accounting primitives so sample/time semantics stay identical across
π0.5, SmolVLA and OpenVLA-OFT.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

D10_VARIANT = "V2_SQRT_BALANCED_RAW"
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
MODELS = ("pi05", "smolvla", "openvla_oft")
FORWARD = MODELS
REVERSE = tuple(reversed(MODELS))
ORDERS = {"forward": FORWARD, "reverse": REVERSE}
TRACKS = ("equal_data", "equal_wall")
EFFECTIVE_BATCH = 32
EQUAL_DATA_BUDGET = 4800
EQUAL_DATA_OPTIMIZER_UPDATES = 150
EQUAL_WALL_SEC = 1800.0
EXECUTE_ENV = "PARC_M3_EXECUTE"
EXECUTE_VALUE = "1"
BATCH_SUMMARY_STATUS = "READY_FOR_M3_RUNNER_IMPLEMENTATION"
EXPECTED_SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
    "openvla_oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}


def require_execution_guard(mode: str, env: dict[str, str] | None = None) -> None:
    if mode == "preflight":
        return
    if mode not in {"smoke", "benchmark"}:
        raise ValueError(f"unsupported runner mode: {mode}")
    values = os.environ if env is None else env
    if values.get(EXECUTE_ENV) != EXECUTE_VALUE:
        raise RuntimeError(
            f"{mode} requires explicit {EXECUTE_ENV}={EXECUTE_VALUE}; training not started"
        )


@dataclass
class EqualDataCounter:
    sample_budget: int = EQUAL_DATA_BUDGET
    consumed_samples: int = 0
    optimizer_updates: int = 0

    def before_batch(self, batch_examples: int) -> None:
        batch_examples = int(batch_examples)
        if batch_examples <= 0:
            raise ValueError("batch_examples must be > 0")
        if self.consumed_samples + batch_examples > self.sample_budget:
            raise RuntimeError(
                "equal-data sample overshoot blocked: "
                f"consumed={self.consumed_samples} next={batch_examples} "
                f"budget={self.sample_budget}"
            )

    def after_batch(self, batch_examples: int) -> None:
        self.before_batch(batch_examples)
        self.consumed_samples += int(batch_examples)

    def on_optimizer_step(self) -> None:
        self.optimizer_updates += 1

    @property
    def complete(self) -> bool:
        return self.consumed_samples == self.sample_budget

    def assert_complete(self) -> None:
        if not self.complete:
            raise RuntimeError(
                f"equal-data run incomplete: {self.consumed_samples}/{self.sample_budget} samples"
            )


@dataclass
class EqualWallTimer:
    target_sec: float = EQUAL_WALL_SEC
    clock: Callable[[], float] = time.monotonic
    started_at: float | None = None
    stopped_at: float | None = None

    def before_first_batch_fetch(self) -> None:
        if self.started_at is not None:
            raise RuntimeError("equal-wall timer already started")
        self.started_at = float(self.clock())

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None else float(self.clock())
        return max(0.0, end - self.started_at)

    def optimizer_boundary_should_stop(self) -> bool:
        if self.started_at is None:
            raise RuntimeError("equal-wall timer must start before first batch fetch")
        if self.stopped_at is not None:
            return True
        if self.elapsed() >= self.target_sec:
            self.stopped_at = float(self.clock())
            return True
        return False

    def assert_stopped_at_or_after_target(self) -> None:
        if self.stopped_at is None:
            raise RuntimeError("equal-wall timer has not stopped at an optimizer boundary")
        if self.elapsed() < self.target_sec:
            raise RuntimeError(
                f"equal-wall stopped too early: {self.elapsed():.6f} < {self.target_sec}"
            )


def validate_batch_probe_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if summary.get("status") != BATCH_SUMMARY_STATUS:
        raise ValueError(
            f"batch summary status must be {BATCH_SUMMARY_STATUS}: {summary.get('status')}"
        )
    if summary.get("selected_dataset_variant") != D10_VARIANT:
        raise ValueError("batch summary dataset variant mismatch")
    if summary.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("batch summary D10 hash mismatch")
    if int(summary.get("effective_batch_size", -1)) != EFFECTIVE_BATCH:
        raise ValueError("batch summary effective batch mismatch")
    protocol = summary.get("equal_data_protocol") or {}
    if int(protocol.get("sample_budget", -1)) != EQUAL_DATA_BUDGET:
        raise ValueError("batch summary sample budget mismatch")
    if int(protocol.get("optimizer_updates", -1)) != EQUAL_DATA_OPTIMIZER_UPDATES:
        raise ValueError("batch summary optimizer update count mismatch")
    if summary.get("benchmark_training_started") is not False:
        raise ValueError("batch-probe summary claims benchmark training started")

    models = summary.get("models")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise ValueError("batch summary must contain exactly the three M3 models")
    normalized: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        config = models[model]
        if not isinstance(config, dict):
            raise ValueError(f"batch config is not an object: {model}")
        if config.get("source_ref") != EXPECTED_SOURCE_REFS[model]:
            raise ValueError(f"source ref mismatch for {model}")
        micro = int(config.get("micro_batch", 0))
        ga = int(config.get("gradient_accumulation", 0))
        eff = int(config.get("effective_batch_size", 0))
        if micro <= 0 or ga <= 0 or eff != EFFECTIVE_BATCH or micro * ga != eff:
            raise ValueError(
                f"invalid batch config for {model}: micro={micro} ga={ga} eff={eff}"
            )
        if EQUAL_DATA_BUDGET % eff != 0:
            raise ValueError(f"equal-data budget cannot divide effective batch for {model}")
        normalized[model] = {
            "source_ref": config["source_ref"],
            "micro_batch": micro,
            "gradient_accumulation": ga,
            "effective_batch_size": eff,
        }
    return normalized


def load_batch_probe_summary(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("batch-probe summary must be a JSON object")
    return data, validate_batch_probe_summary(data)


def build_run_matrix() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for order_name in ("forward", "reverse"):
        for track in TRACKS:
            for sequence_index, model in enumerate(ORDERS[order_name]):
                rows.append(
                    {
                        "order": order_name,
                        "track": track,
                        "model": model,
                        "sequence_index": str(sequence_index),
                    }
                )
    return rows


def assert_reverse_is_exact_inverse() -> None:
    if REVERSE != tuple(reversed(FORWARD)):
        raise RuntimeError("reverse execution order is not exact inverse of forward")
    if set(REVERSE) != set(FORWARD):
        raise RuntimeError("forward/reverse model sets differ")


def build_execution_plan(
    *,
    batch_summary: dict[str, Any],
    equal_data_schedule_sha256: str,
    equal_wall_schedule_sha256: str,
    mode: str = "preflight",
) -> dict[str, Any]:
    require_execution_guard(mode)
    assert_reverse_is_exact_inverse()
    model_batches = validate_batch_probe_summary(batch_summary)
    schedule_hashes = {
        "equal_data": str(equal_data_schedule_sha256).strip(),
        "equal_wall": str(equal_wall_schedule_sha256).strip(),
    }
    if not all(schedule_hashes.values()):
        raise ValueError("both equal-data and equal-wall sampling schedule SHA values are required")
    rows = build_run_matrix()
    for row in rows:
        row["sampling_schedule_sha256"] = schedule_hashes[row["track"]]
        row["micro_batch"] = str(model_batches[row["model"]]["micro_batch"])
        row["gradient_accumulation"] = str(
            model_batches[row["model"]]["gradient_accumulation"]
        )
        row["effective_batch_size"] = str(EFFECTIVE_BATCH)
    return {
        "schema_version": 1,
        "stage": "M3_guarded_execution_plan",
        "status": "READY_FOR_SMOKE" if mode == "preflight" else "EXECUTION_EXPLICITLY_ENABLED",
        "mode": mode,
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_HASH,
        "equal_data_sample_budget": EQUAL_DATA_BUDGET,
        "equal_wall_train_loop_sec": EQUAL_WALL_SEC,
        "orders": {name: list(models) for name, models in ORDERS.items()},
        "sampling_schedule_sha256": schedule_hashes,
        "model_batches": model_batches,
        "run_matrix": rows,
        "benchmark_training_started": False,
        "automatic_full_run": False,
    }


def result_path(root: Path, *, order: str, track: str, model: str) -> Path:
    if order not in ORDERS or track not in TRACKS or model not in MODELS:
        raise ValueError(f"invalid result identity: {order}/{track}/{model}")
    return Path(root) / "runs" / order / track / model / "result.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--equal-data-schedule-sha256", required=True)
    parser.add_argument("--equal-wall-schedule-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "benchmark"), default="preflight")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data, _ = load_batch_probe_summary(args.batch_summary)
    plan = build_execution_plan(
        batch_summary=data,
        equal_data_schedule_sha256=args.equal_data_schedule_sha256,
        equal_wall_schedule_sha256=args.equal_wall_schedule_sha256,
        mode=args.mode,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": plan["status"],
        "mode": plan["mode"],
        "run_count": len(plan["run_matrix"]),
        "benchmark_training_started": plan["benchmark_training_started"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
