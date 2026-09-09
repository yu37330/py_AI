#!/usr/bin/env python3
"""Build and run guarded, instrumented LIBERO evaluation jobs for M3 checkpoints."""
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
    python: str


def default_eval_runtimes(root: Path) -> dict[str, EvalRuntime]:
    root = Path(root)
    return {
        "pi05": EvalRuntime(
            "pi05", SOURCE_REFS["pi05"],
            str(root / "vendor/lerobot-pi05-m3-probe"),
            str(root / "vendor/lerobot-pi05-m3-probe/.venv/bin/python"),
        ),
        "smolvla": EvalRuntime(
            "smolvla", SOURCE_REFS["smolvla"],
            str(root / "vendor/lerobot-smolvla-m3"),
            str(root / "venv-smolvla-m3/bin/python"),
        ),
        "openvla_oft": EvalRuntime(
            "openvla_oft", SOURCE_REFS["openvla_oft"],
            str(root / "vendor/openvla-oft-m3"),
            str(root / "venv-openvla-oft-m3/bin/python"),
        ),
    }


def validate_training_result(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise ValueError("training result must be PASS before simulator evaluation")
    if data.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("training result D10 hash mismatch")
    model = data.get("model")
    if model not in SOURCE_REFS:
        raise ValueError(f"unknown model in training result: {model}")
    if data.get("source_ref") != SOURCE_REFS[model]:
        raise ValueError("training result source_ref mismatch")
    if not str(data.get("checkpoint_ref") or "").strip():
        raise ValueError("training result checkpoint_ref missing")
    if data.get("order") not in {"forward", "reverse"}:
        raise ValueError("training result order missing/invalid")
    if data.get("track") not in {"equal_data", "equal_wall"}:
        raise ValueError("training result track missing/invalid")
    return data


def build_jobs(
    *,
    training_result: dict[str, Any],
    training_result_path: Path,
    root: Path,
    repo_root: Path,
    output_root: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    model = training_result["model"]
    runtime = default_eval_runtimes(root)[model]
    trials = 1 if smoke else 10
    jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for suite in SUITES:
            out = Path(output_root) / model / f"seed-{seed}" / suite
            records_out = out / "episode_records.json"
            if model in {"pi05", "smolvla"}:
                entry = repo_root / "tools/benchmark/m3_lerobot_eval_entry.py"
                command = [
                    runtime.python,
                    "-u",
                    str(entry),
                    "--repo-root",
                    str(repo_root),
                    "--training-result",
                    str(training_result_path),
                    "--model",
                    model,
                    "--suite",
                    suite,
                    "--seed",
                    str(seed),
                    "--episodes",
                    str(trials),
                    "--output-dir",
                    str(out),
                    "--records-out",
                    str(records_out),
                ]
            else:
                entry = repo_root / "tools/benchmark/m3_openvla_eval_entry.py"
                command = [
                    runtime.python,
                    "-u",
                    str(entry),
                    "--repo-root",
                    str(repo_root),
                    "--source-root",
                    runtime.command_root,
                    "--training-result",
                    str(training_result_path),
                    "--suite",
                    suite,
                    "--seed",
                    str(seed),
                    "--trials-per-task",
                    str(trials),
                    "--output-dir",
                    str(out),
                    "--records-out",
                    str(records_out),
                ]
            jobs.append({
                "model": model,
                "seed": seed,
                "suite": suite,
                "runtime": asdict(runtime),
                "output_dir": str(out),
                "records_out": str(records_out),
                "command": command,
            })
    return jobs


def execute_jobs(jobs: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    require_execution_guard(mode)
    if mode == "preflight":
        raise RuntimeError("preflight never executes simulator jobs")
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")
    records: list[dict[str, Any]] = []
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
        payload = json.loads(Path(job["records_out"]).read_text(encoding="utf-8"))
        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise RuntimeError(f"evaluation job produced no episode records: {job['records_out']}")
        records.extend(episodes)
    return records


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


def main() -> int:
    args = parse_args()
    training = validate_training_result(args.training_result)
    require_execution_guard(args.mode)
    jobs = build_jobs(
        training_result=training,
        training_result_path=args.training_result,
        root=args.parc_root,
        repo_root=args.repo_root,
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
        "max_steps_per_episode": 300,
        "mode": args.mode,
        "execution_guard": {"environment_variable": EXECUTE_ENV, "required_value": EXECUTE_VALUE},
        "jobs": jobs,
        "simulator_started": False,
        "normalization_target": "tools/benchmark/m3_evaluation_metrics.py",
    }
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "jobs": len(jobs), "plan": str(args.plan_out)}, indent=2))
    if args.execute:
        records = execute_jobs(jobs, mode=args.mode)
        combined_path = Path(args.output_root) / training["model"] / "episode_records_all.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps({"episodes": records}, indent=2) + "\n", encoding="utf-8")

        from tools.benchmark.m3_evaluation_metrics import aggregate_episode_records  # noqa: PLC0415

        summary = aggregate_episode_records(
            records,
            training_result=training,
            training_result_ref=str(args.training_result),
        )
        summary_out = args.summary_out or (Path(args.output_root) / training["model"] / "m3_evaluation_summary.json")
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": summary["status"],
            "model": summary["model"],
            "episode_count": summary["episode_count"],
            "metrics": summary["metrics"],
            "summary": str(summary_out),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
