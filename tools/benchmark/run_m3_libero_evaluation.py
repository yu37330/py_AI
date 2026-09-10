#!/usr/bin/env python3
"""Canonical live M3 LIBERO evaluation entry with bounded OpenVLA storage.

For OpenVLA-OFT, the persistent M3 artifact is LoRA/components/D10 stats.
Merged 7B weights are required only while the upstream evaluator is running.
Unless `PARC_M3_KEEP_MERGED_OPENVLA=1` is explicitly set, this wrapper removes
those merged weights after evaluation even when they were materialized by an
earlier training/finalization step. π0.5 and SmolVLA pass straight through.
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


def _lifecycle_paths(args: argparse.Namespace, training: dict[str, Any]) -> dict[str, Path]:
    checkpoint = Path(str(training.get("checkpoint_ref") or ""))
    return {
        "source_root": args.parc_root / "vendor/openvla-oft-m3",
        "python": args.parc_root / "venv-openvla-oft-m3/bin/python",
        "finalizer": args.repo_root.resolve() / "tools/benchmark/m3_openvla_finalize_checkpoint.py",
        "dematerializer": args.repo_root.resolve() / "tools/benchmark/m3_openvla_dematerialize_checkpoint.py",
        "checkpoint": checkpoint,
    }


def _openvla_lifecycle_blockers(args: argparse.Namespace, training: dict[str, Any]) -> list[str]:
    if training.get("model") != "openvla_oft":
        return []
    paths = _lifecycle_paths(args, training)
    blockers: list[str] = []
    for label in ("finalizer", "dematerializer", "python"):
        if not paths[label].is_file():
            blockers.append(f"OpenVLA lifecycle {label} missing: {paths[label]}")
    if not paths["source_root"].is_dir():
        blockers.append(f"OpenVLA pinned source root missing: {paths['source_root']}")
    if not paths["checkpoint"].is_dir():
        blockers.append(f"OpenVLA persistent checkpoint root missing: {paths['checkpoint']}")
    return blockers


def _lifecycle_commands(args: argparse.Namespace, training: dict[str, Any]) -> tuple[list[str], list[str]]:
    blockers = _openvla_lifecycle_blockers(args, training)
    if blockers:
        raise RuntimeError("; ".join(blockers))
    paths = _lifecycle_paths(args, training)
    finalize = [
        str(paths["python"]),
        "-u",
        str(paths["finalizer"]),
        "--source-root",
        str(paths["source_root"]),
        "--checkpoint-root",
        str(paths["checkpoint"]),
        "--result",
        str(args.training_result),
    ]
    cleanup = [
        str(paths["python"]),
        "-u",
        str(paths["dematerializer"]),
        "--checkpoint-root",
        str(paths["checkpoint"]),
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


def _write_lifecycle(
    args: argparse.Namespace,
    *,
    model: str,
    status: str,
    blockers: list[str],
    materialized_here: bool,
    merged_present_before: bool,
    executor_rc: int | None,
    cleanup_rc: int | None,
) -> Path:
    keep = os.environ.get(KEEP_MERGED_ENV) == "1"
    lifecycle = {
        "schema_version": 3,
        "stage": "M3_LIBERO_evaluation_lifecycle",
        "status": status,
        "model": model,
        "mode": args.mode,
        "execute": bool(args.execute),
        "blockers": blockers,
        "openvla_merged_present_before_evaluation": merged_present_before,
        "openvla_materialized_for_this_evaluation": materialized_here,
        "openvla_keep_merged_requested": keep,
        "openvla_dematerialized_after_evaluation": bool(
            model == "openvla_oft"
            and args.execute
            and not keep
            and cleanup_rc == 0
        ),
        "executor_returncode": executor_rc,
        "cleanup_returncode": cleanup_rc,
        "automatic_upload": False,
    }
    path = args.output_root / str(model) / "evaluation_lifecycle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lifecycle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lifecycle, indent=2), flush=True)
    return path


def main() -> int:
    args = parse_args()
    if args.execute and args.mode == "preflight":
        raise RuntimeError("preflight cannot execute simulator evaluation")
    if args.execute and os.environ.get(EXECUTE_ENV) != "1":
        raise RuntimeError(f"live evaluation requires explicit {EXECUTE_ENV}=1")

    training = _load(args.training_result)
    model = str(training.get("model") or "")
    if model not in {"pi05", "smolvla", "openvla_oft"}:
        raise ValueError(f"unknown M3 model: {model}")
    merged_present_before = bool(model == "openvla_oft" and training.get("checkpoint_eval_ready") is True)

    lifecycle_blockers = _openvla_lifecycle_blockers(args, training)
    if lifecycle_blockers:
        _write_lifecycle(
            args,
            model=model,
            status="BLOCKED",
            blockers=lifecycle_blockers,
            materialized_here=False,
            merged_present_before=merged_present_before,
            executor_rc=None,
            cleanup_rc=None,
        )
        return 2

    materialized_here = False
    cleanup_command: list[str] | None = None
    env = os.environ.copy()
    if model == "openvla_oft":
        finalize_command, cleanup_command = _lifecycle_commands(args, training)
        if args.execute and training.get("checkpoint_eval_ready") is not True:
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
            args.execute
            and model == "openvla_oft"
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

    status = "PASS" if executor_rc == 0 and cleanup_rc == 0 else "FAILED"
    _write_lifecycle(
        args,
        model=model,
        status=status,
        blockers=[],
        materialized_here=materialized_here,
        merged_present_before=merged_present_before,
        executor_rc=executor_rc,
        cleanup_rc=cleanup_rc,
    )
    if executor_rc != 0:
        return executor_rc
    if cleanup_rc != 0:
        return cleanup_rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
