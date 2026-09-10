#!/usr/bin/env python3
"""Canonical live M3 LIBERO evaluation entry with bounded OpenVLA storage.

For OpenVLA-OFT, the persistent training artifact is LoRA/components/D10 stats.
This wrapper materializes merged 7B weights only immediately before evaluation
and removes only those merged weights afterwards (unless explicitly retained).
π0.5 and SmolVLA pass straight through to m3_libero_executor.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

EXECUTE_ENV = "PARC_M3_EXECUTE"
KEEP_MERGED_ENV = "PARC_M3_KEEP_MERGED_OPENVLA"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--training-result", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--plan-out", type=Path, required=True)
    p.add_argument("--summary-out", type=Path)
    p.add_argument("--parc-root", type=Path, default=Path("/content/parc2026"))
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--mode", choices=("preflight", "smoke", "benchmark"), default="preflight")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def _lifecycle_commands(args: argparse.Namespace, training: dict[str, Any]) -> tuple[list[str], list[str]]:
    repo = args.repo_root.resolve()
    source_root = args.parc_root / "vendor/openvla-oft-m3"
    finalizer = repo / "tools/benchmark/m3_openvla_finalize_checkpoint.py"
    dematerializer = repo / "tools/benchmark/m3_openvla_dematerialize_checkpoint.py"
    if not finalizer.is_file():
        raise FileNotFoundError(
            f"OpenVLA evaluator lifecycle requires #54 finalizer to be present: {finalizer}"
        )
    if not dematerializer.is_file():
        raise FileNotFoundError(
            f"OpenVLA evaluator lifecycle requires #54 dematerializer to be present: {dematerializer}"
        )
    checkpoint = Path(str(training.get("checkpoint_ref") or ""))
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"OpenVLA checkpoint root missing: {checkpoint}")
    finalize = [
        str(args.parc_root / "venv-openvla-oft-m3/bin/python"),
        "-u",
        str(finalizer),
        "--source-root",
        str(source_root),
        "--checkpoint-root",
        str(checkpoint),
        "--result",
        str(args.training_result),
    ]
    cleanup = [
        str(args.parc_root / "venv-openvla-oft-m3/bin/python"),
        "-u",
        str(dematerializer),
        "--checkpoint-root",
        str(checkpoint),
        "--result",
        str(args.training_result),
    ]
    return finalize, cleanup


def _executor_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(args.repo_root / "tools/benchmark/m3_libero_executor.py"),
        "--training-result",
        str(args.training_result),
        "--output-root",
        str(args.output_root),
        "--plan-out",
        str(args.plan_out),
        "--parc-root",
        str(args.parc_root),
        "--repo-root",
        str(args.repo_root),
        "--mode",
        args.mode,
    ]
    if args.summary_out is not None:
        cmd.extend(["--summary-out", str(args.summary_out)])
    if args.execute:
        cmd.append("--execute")
    return cmd


def main() -> int:
    args = parse_args()
    if args.execute and args.mode == "preflight":
        raise RuntimeError("preflight cannot execute simulator evaluation")
    if args.execute and os.environ.get(EXECUTE_ENV) != "1":
        raise RuntimeError(f"live evaluation requires explicit {EXECUTE_ENV}=1")

    training = _load(args.training_result)
    model = training.get("model")
    if model not in {"pi05", "smolvla", "openvla_oft"}:
        raise ValueError(f"unknown M3 model: {model}")

    materialized_here = False
    cleanup_command: list[str] | None = None
    env = os.environ.copy()
    if args.execute and model == "openvla_oft" and training.get("checkpoint_eval_ready") is not True:
        finalize_command, cleanup_command = _lifecycle_commands(args, training)
        subprocess.run(finalize_command, cwd=str(args.repo_root), env=env, check=True)
        finalized = _load(args.training_result)
        if finalized.get("checkpoint_eval_ready") is not True:
            raise RuntimeError("OpenVLA finalizer did not produce eval-ready checkpoint")
        materialized_here = True

    executor_rc = 0
    cleanup_rc = 0
    try:
        completed = subprocess.run(
            _executor_command(args),
            cwd=str(args.repo_root),
            env=env,
            check=False,
        )
        executor_rc = int(completed.returncode)
    finally:
        if (
            materialized_here
            and cleanup_command is not None
            and os.environ.get(KEEP_MERGED_ENV) != "1"
        ):
            cleanup = subprocess.run(
                cleanup_command,
                cwd=str(args.repo_root),
                env=env,
                check=False,
            )
            cleanup_rc = int(cleanup.returncode)

    lifecycle = {
        "schema_version": 1,
        "stage": "M3_LIBERO_evaluation_lifecycle",
        "status": "PASS" if executor_rc == 0 and cleanup_rc == 0 else "FAILED",
        "model": model,
        "mode": args.mode,
        "execute": bool(args.execute),
        "openvla_materialized_for_this_evaluation": materialized_here,
        "openvla_keep_merged_requested": os.environ.get(KEEP_MERGED_ENV) == "1",
        "openvla_dematerialized_after_evaluation": bool(
            materialized_here and os.environ.get(KEEP_MERGED_ENV) != "1" and cleanup_rc == 0
        ),
        "executor_returncode": executor_rc,
        "cleanup_returncode": cleanup_rc,
        "automatic_upload": False,
    }
    lifecycle_path = args.output_root / str(model) / "evaluation_lifecycle.json"
    lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
    lifecycle_path.write_text(json.dumps(lifecycle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lifecycle, indent=2), flush=True)
    if executor_rc != 0:
        return executor_rc
    if cleanup_rc != 0:
        return cleanup_rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
