#!/usr/bin/env python3
"""Build and run guarded LIBERO evaluation jobs for M3 checkpoints.

LeRobot-family checkpoints use the pinned LeRobot evaluator. OpenVLA-OFT uses
its pinned upstream LIBERO evaluator.  This module owns provenance, seed/task
expansion, command construction, log locations and fail-closed execution.
Metric normalization remains delegated to m3_evaluation_metrics.py after the
instrumented episode-record shim writes per-episode evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tools.benchmark.m3_runner_core import D10_HASH, EXECUTE_ENV, EXECUTE_VALUE, require_execution_guard

SEEDS = (20260906, 20260907)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
    "openvla_oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}


@dataclass(frozen=True)
class EvalRuntime:
    model: str
    source_ref: str
    command_root: str
    evaluator: str


def default_eval_runtimes(root: Path) -> dict[str, EvalRuntime]:
    root = Path(root)
    return {
        "pi05": EvalRuntime(
            "pi05", SOURCE_REFS["pi05"],
            str(root / "vendor/lerobot-pi05-m3-probe"),
            str(root / "vendor/lerobot-pi05-m3-probe/.venv/bin/lerobot-eval"),
        ),
        "smolvla": EvalRuntime(
            "smolvla", SOURCE_REFS["smolvla"],
            str(root / "vendor/lerobot-smolvla-m3"),
            str(root / "venv-smolvla-m3/bin/lerobot-eval"),
        ),
        "openvla_oft": EvalRuntime(
            "openvla_oft", SOURCE_REFS["openvla_oft"],
            str(root / "vendor/openvla-oft-m3"),
            str(root / "venv-openvla-oft-m3/bin/python"),
        ),
    }


def validate_training_result(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("training result D10 hash mismatch")
    model = data.get("model")
    if model not in SOURCE_REFS:
        raise ValueError(f"unknown model in training result: {model}")
    if data.get("source_ref") != SOURCE_REFS[model]:
        raise ValueError("training result source_ref mismatch")
    if not str(data.get("checkpoint_ref") or "").strip():
        raise ValueError("training result checkpoint_ref missing")
    return data


def build_lerobot_eval_command(
    *, runtime: EvalRuntime, checkpoint: str, seed: int, output_dir: Path, smoke: bool
) -> list[str]:
    episodes = 1 if smoke else 10
    cmd = [
        runtime.evaluator,
        f"--policy.path={checkpoint}",
        "--env.type=libero",
        f"--env.task={','.join(SUITES)}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={episodes}",
        "--env.max_parallel_tasks=1",
        f"--seed={int(seed)}",
        f"--output_dir={output_dir}",
    ]
    if runtime.model == "pi05":
        cmd.append("--policy.n_action_steps=10")
    return cmd


def build_openvla_eval_commands(
    *, runtime: EvalRuntime, checkpoint: str, seed: int, output_dir: Path, smoke: bool
) -> list[list[str]]:
    trials = 1 if smoke else 10
    commands = []
    for suite in SUITES:
        commands.append([
            runtime.evaluator,
            str(Path(runtime.command_root) / "experiments/robot/libero/run_libero_eval.py"),
            "--pretrained_checkpoint", checkpoint,
            "--task_suite_name", suite,
            "--num_trials_per_task", str(trials),
            "--seed", str(int(seed)),
            "--center_crop", "True",
        ])
    return commands


def build_jobs(
    *, training_result: dict[str, Any], root: Path, output_root: Path, smoke: bool
) -> list[dict[str, Any]]:
    model = training_result["model"]
    runtime = default_eval_runtimes(root)[model]
    checkpoint = str(training_result["checkpoint_ref"])
    jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = Path(output_root) / model / f"seed-{seed}"
        if model in {"pi05", "smolvla"}:
            jobs.append({
                "model": model,
                "seed": seed,
                "suite": "all_standard",
                "runtime": asdict(runtime),
                "output_dir": str(seed_dir),
                "command": build_lerobot_eval_command(
                    runtime=runtime, checkpoint=checkpoint, seed=seed, output_dir=seed_dir, smoke=smoke
                ),
            })
        else:
            for suite, command in zip(
                SUITES,
                build_openvla_eval_commands(
                    runtime=runtime, checkpoint=checkpoint, seed=seed, output_dir=seed_dir, smoke=smoke
                ),
                strict=True,
            ):
                jobs.append({
                    "model": model,
                    "seed": seed,
                    "suite": suite,
                    "runtime": asdict(runtime),
                    "output_dir": str(seed_dir / suite),
                    "command": command,
                })
    return jobs


def execute_jobs(jobs: list[dict[str, Any]], *, mode: str) -> None:
    require_execution_guard(mode)
    if mode == "preflight":
        raise RuntimeError("preflight never executes simulator jobs")
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")
    for job in jobs:
        out = Path(job["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        log = out / "eval.log"
        with log.open("w", encoding="utf-8") as fh:
            subprocess.run(
                job["command"],
                cwd=job["runtime"]["command_root"],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=True,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--training-result", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--plan-out", type=Path, required=True)
    p.add_argument("--parc-root", type=Path, default=Path("/content/parc2026"))
    p.add_argument("--mode", choices=("preflight", "smoke", "benchmark"), default="preflight")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    training = validate_training_result(args.training_result)
    require_execution_guard(args.mode)
    jobs = build_jobs(
        training_result=training,
        root=args.parc_root,
        output_root=args.output_root,
        smoke=args.mode == "smoke",
    )
    plan = {
        "schema_version": 1,
        "stage": "M3_LIBERO_simulator_executor",
        "status": "READY_FOR_EVAL_EXECUTION",
        "model": training["model"],
        "checkpoint_ref": training["checkpoint_ref"],
        "source_ref": training["source_ref"],
        "selected_episode_ids_sha256": D10_HASH,
        "seed_set": list(SEEDS),
        "suites": list(SUITES),
        "mode": args.mode,
        "execution_guard": {"environment_variable": EXECUTE_ENV, "required_value": EXECUTE_VALUE},
        "jobs": jobs,
        "simulator_started": False,
        "normalization_target": "tools/benchmark/m3_evaluation_metrics.py",
        "instrumentation_target": "tools/benchmark/m3_eval_instrumentation.py",
    }
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "jobs": len(jobs), "plan": str(args.plan_out)}, indent=2))
    if args.execute:
        execute_jobs(jobs, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
