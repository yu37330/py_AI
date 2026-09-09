#!/usr/bin/env python3
"""Instrument pinned LeRobot LIBERO evaluation and emit M3 episode records.

The upstream evaluator owns environment/policy preprocessing.  We wrap only its
``rollout`` function, preserving policy/environment behavior while measuring
steps, duration, synchronized policy inference latency and peak inference VRAM.
Evaluation batch size is forced to 1 so one rollout maps unambiguously to one
M3 episode record.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--training-result", type=Path, required=True)
    p.add_argument("--model", choices=("pi05", "smolvla"), required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--records-out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root))
    from tools.benchmark.m3_eval_instrumentation import (  # noqa: PLC0415
        build_episode_record,
        instrument_select_action,
    )
    from tools.benchmark.m3_libero_executor import validate_training_result  # noqa: PLC0415

    training = validate_training_result(args.training_result)
    if training["model"] != args.model:
        raise ValueError("training-result model mismatch")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    module = importlib.import_module("lerobot.scripts.lerobot_eval")
    original_rollout = module.rollout
    records: list[dict] = []

    def instrumented_rollout(*rollout_args, **rollout_kwargs):
        env = rollout_kwargs.get("env")
        policy = rollout_kwargs.get("policy")
        actual_seeds = rollout_kwargs.get("seeds")
        if env is None and rollout_args:
            env = rollout_args[0]
        if policy is None and len(rollout_args) > 1:
            policy = rollout_args[1]
        try:
            descriptions = list(env.call("task_description"))
        except Exception:
            descriptions = [args.suite] * int(getattr(env, "num_envs", 1))

        started = time.perf_counter()
        with instrument_select_action(policy) as inference:
            rollout_data = original_rollout(*rollout_args, **rollout_kwargs)
        duration = time.perf_counter() - started

        import torch  # noqa: PLC0415

        done = rollout_data["done"].to(torch.bool)
        success_tensor = rollout_data["success"].to(torch.bool)
        batch = int(done.shape[0])
        if batch != 1:
            raise RuntimeError("M3 evaluator requires eval.batch_size=1 for episode instrumentation")
        for i in range(batch):
            done_row = done[i]
            done_indices = torch.nonzero(done_row, as_tuple=False).flatten()
            steps = int(done_indices[0].item() + 1) if len(done_indices) else int(done_row.shape[0])
            steps = min(300, steps)
            success = bool(success_tensor[i, :steps].any().item())
            actual_seed = None
            if actual_seeds is not None and i < len(actual_seeds):
                actual_seed = int(actual_seeds[i])
            record = build_episode_record(
                model=args.model,
                checkpoint_ref=str(training["checkpoint_ref"]),
                source_ref=str(training["source_ref"]),
                order=str(training["order"]),
                track=str(training["track"]),
                seed=args.seed,
                task_id=f"{args.suite}:{descriptions[i]}",
                success=success,
                steps=steps,
                episode_duration_sec=duration / batch,
                mean_inference_latency_ms=inference.mean_latency_ms,
                peak_inference_vram_mib=inference.peak_vram_mib,
            )
            record["environment_seed"] = actual_seed
            records.append(record)
        return rollout_data

    module.rollout = instrumented_rollout
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cli = [
        "lerobot-eval",
        f"--policy.path={training['checkpoint_ref']}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        "--env.episode_length=300",
        "--env.init_states=true",
        "--eval.batch_size=1",
        f"--eval.n_episodes={args.episodes}",
        "--env.max_parallel_tasks=1",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir / 'upstream'}",
    ]
    if args.model == "pi05":
        cli.append("--policy.n_action_steps=10")
    old_argv = sys.argv
    try:
        sys.argv = cli
        module.main()
    finally:
        sys.argv = old_argv
        module.rollout = original_rollout

    if not records:
        raise RuntimeError("upstream LeRobot evaluator produced no instrumented episode records")
    args.records_out.parent.mkdir(parents=True, exist_ok=True)
    args.records_out.write_text(json.dumps({"episodes": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "model": args.model,
        "suite": args.suite,
        "run_seed": args.seed,
        "episode_count": len(records),
        "records": str(args.records_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
