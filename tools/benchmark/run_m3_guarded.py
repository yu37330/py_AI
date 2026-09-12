#!/usr/bin/env python3
"""Guarded sequence orchestrator for PARC2026 M3 training.

The orchestrator executes exactly one requested order/track at a time. It never
starts training unless `PARC_M3_EXECUTE=1` is present, never selects a dataset,
and never infers batch sizes: all per-model micro-batch/gradient-accumulation
values come from the 72d-bound adapter plan.

Smoke mode is deliberately small and always uses the canonical equal-data
prefix (64 samples / 2 effective optimizer updates per model). Benchmark mode
runs one explicitly selected fair-comparison track. Running forward does not
automatically start reverse, and vice versa.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from tools.benchmark.m3_runner_core import D10_HASH, D10_VARIANT, require_execution_guard
from tools.benchmark.m3_training_adapters import ADAPTERS
from tools.benchmark.workers.m3_worker_common import (
    load_adapter_plan,
    load_manifest_episode_ids,
    select_run_spec,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--adapter-plan", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--streaming-contract", type=Path, required=True)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--track", choices=("equal_data", "equal_wall"), default="equal_data")
    p.add_argument(
        "--resume-completed",
        action="store_true",
        help="Skip only already-PASS worker results whose immutable run identity still matches.",
    )
    return p.parse_args()


def _sequence(plan: dict[str, Any], order: str) -> list[str]:
    value = plan.get("orders", {}).get(order)
    if not isinstance(value, list) or set(value) != set(ADAPTERS) or len(value) != 3:
        raise ValueError(f"invalid adapter-plan sequence for {order}: {value}")
    expected = (
        ["pi05", "smolvla", "openvla_oft"]
        if order == "forward"
        else ["openvla_oft", "smolvla", "pi05"]
    )
    if value != expected:
        raise ValueError(f"adapter-plan {order} sequence drift: {value}")
    return value


def _worker_result_path(
    *, run_root: Path, run_spec: dict[str, Any], mode: str, order: str, model: str
) -> Path:
    if mode == "smoke":
        return Path(run_root) / "smoke" / order / model / "training_result.json"
    return Path(run_spec["result_path"]).parent / "training_result.json"


def _completed_matches(
    path: Path, *, run: dict[str, Any], mode: str, order: str, track: str, model: str
) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    expected_track = "equal_data" if mode == "smoke" else track
    return (
        result.get("status") == "PASS"
        and result.get("mode") == mode
        and result.get("model") == model
        and result.get("order") == order
        and result.get("track") == expected_track
        and result.get("source_ref") == run["source_ref"]
        and result.get("selected_dataset_variant") == D10_VARIANT
        and result.get("selected_episode_ids_sha256") == D10_HASH
        and result.get("sampling_schedule_sha256") == run["schedule_sha256"]
        and int(result.get("micro_batch", -1)) == int(run["micro_batch"])
        and int(result.get("gradient_accumulation", -1)) == int(run["gradient_accumulation"])
        and int(result.get("effective_batch_size", -1)) == 32
    )


def _worker_module(model: str) -> str:
    path = ADAPTERS[model].worker
    if not path.endswith(".py"):
        raise ValueError(f"worker path is not a Python module: {path}")
    return path[:-3].replace("/", ".")


def _worker_command(
    *,
    repo: Path,
    model: str,
    adapter_plan: Path,
    manifest: Path,
    dataset_root: Path,
    streaming_contract: Path,
    work_root: Path,
    run_root: Path,
    mode: str,
    order: str,
    track: str,
) -> list[str]:
    worker = repo / ADAPTERS[model].worker
    if not worker.is_file():
        raise FileNotFoundError(worker)
    effective_track = "equal_data" if mode == "smoke" else track
    cmd = [
        sys.executable,
        "-u",
        "-m",
        _worker_module(model),
        "--repo-root",
        str(repo),
        "--adapter-plan",
        str(adapter_plan),
        "--manifest",
        str(manifest),
        "--dataset-root",
        str(dataset_root),
        "--work-root",
        str(work_root),
        "--run-root",
        str(run_root),
        "--order",
        order,
        "--track",
        effective_track,
        "--mode",
        mode,
    ]
    if model == "openvla_oft":
        cmd.extend(["--streaming-contract", str(streaming_contract)])
    return cmd


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    require_execution_guard(args.mode)
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("PARC_M3_EXECUTE=1 is required; training not started")
    if args.mode == "smoke" and args.track != "equal_data":
        raise ValueError("M3 smoke is fixed to the equal-data canonical prefix")

    plan = load_adapter_plan(args.adapter_plan)
    load_manifest_episode_ids(args.manifest)
    sequence = _sequence(plan, args.order)
    effective_track = "equal_data" if args.mode == "smoke" else args.track

    contract = json.loads(args.streaming_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming contract is not PASS")
    if contract.get("source_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("69c streaming contract D10 hash mismatch")
    if contract.get("storage_policy", {}).get("full_rlds_materialized") is not False:
        raise RuntimeError("M3 orchestrator refuses a full-RLDS contract")

    orchestration_dir = (
        args.run_root / "smoke" / args.order
        if args.mode == "smoke"
        else args.run_root / "runs" / args.order / args.track
    )
    status_path = orchestration_dir / "orchestrator_status.json"
    log_path = orchestration_dir / "orchestrator.log"
    orchestration_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M3_guarded_sequence",
        "status": "RUNNING",
        "mode": args.mode,
        "order": args.order,
        "track": effective_track,
        "sequence": sequence,
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_HASH,
        "execution_guard_verified": True,
        "automatic_reverse_start": False,
        "automatic_other_track_start": False,
        "completed_models": [],
        "skipped_verified_models": [],
        "current_model": None,
        "started_at_unix": time.time(),
        "benchmark_training_started": args.mode == "benchmark",
    }
    write_json(status_path, state)

    try:
        for sequence_index, model in enumerate(sequence):
            run = select_run_spec(
                plan,
                model=model,
                order=args.order,
                track=effective_track,
            )
            if int(run.get("sequence_index", -1)) != sequence_index:
                raise RuntimeError(
                    f"adapter-plan sequence index mismatch: {model} expected={sequence_index} "
                    f"actual={run.get('sequence_index')}"
                )
            result_path = _worker_result_path(
                run_root=args.run_root,
                run_spec=run,
                mode=args.mode,
                order=args.order,
                model=model,
            )
            if args.resume_completed and _completed_matches(
                result_path,
                run=run,
                mode=args.mode,
                order=args.order,
                track=effective_track,
                model=model,
            ):
                state["skipped_verified_models"].append(model)
                write_json(status_path, state)
                print(f"[m3] verified PASS exists; skipping {model}: {result_path}", flush=True)
                continue

            state["current_model"] = model
            state["current_sequence_index"] = sequence_index
            state["current_worker"] = ADAPTERS[model].worker
            write_json(status_path, state)
            cmd = _worker_command(
                repo=repo,
                model=model,
                adapter_plan=args.adapter_plan,
                manifest=args.manifest,
                dataset_root=args.dataset_root,
                streaming_contract=args.streaming_contract,
                work_root=args.work_root,
                run_root=args.run_root,
                mode=args.mode,
                order=args.order,
                track=effective_track,
            )
            print(
                f"=== M3 {args.mode.upper()} {args.order}/{effective_track}: "
                f"{sequence_index + 1}/3 {model} ===",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"\n=== {time.time():.6f} {args.mode} {args.order}/{effective_track} "
                    f"{sequence_index + 1}/3 {model} ===\n"
                )
                fh.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo),
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    fh.write(line)
                    fh.flush()
                rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"M3 worker failed: model={model} rc={rc}")
            if not _completed_matches(
                result_path,
                run=run,
                mode=args.mode,
                order=args.order,
                track=effective_track,
                model=model,
            ):
                raise RuntimeError(
                    f"worker returned 0 but immutable PASS result validation failed: {result_path}"
                )
            state["completed_models"].append(model)
            write_json(status_path, state)

        state["status"] = "PASS"
        state["current_model"] = None
        state["completed_at_unix"] = time.time()
        state["next_step"] = (
            "run the opposite-order guarded smoke explicitly"
            if args.mode == "smoke"
            else "run simulator evaluation for this completed training track; do not promote from loss"
        )
        write_json(status_path, state)
        print("=== M3 GUARDED SEQUENCE: PASS ===", flush=True)
        print(json.dumps(state, indent=2), flush=True)
        return 0
    except Exception as exc:
        state["status"] = "FAILED"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
        state["failed_at_unix"] = time.time()
        write_json(status_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
