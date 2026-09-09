#!/usr/bin/env python3
"""Instrument pinned OpenVLA-OFT LIBERO evaluation and emit M3 episode records."""
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
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--training-result", type=Path, required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--trials-per-task", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--records-out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials_per_task <= 0:
        raise ValueError("trials-per-task must be positive")
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(args.source_root))
    from tools.benchmark.m3_eval_instrumentation import build_episode_record  # noqa: PLC0415
    from tools.benchmark.m3_libero_executor import validate_training_result  # noqa: PLC0415

    training = validate_training_result(args.training_result)
    if training["model"] != "openvla_oft":
        raise ValueError("OpenVLA evaluator requires openvla_oft training result")

    module = importlib.import_module("experiments.robot.libero.run_libero_eval")
    original_run_episode = module.run_episode
    original_get_action = module.get_action
    original_save_video = module.save_rollout_video
    records: list[dict] = []
    current_latencies: list[float] = []

    import torch  # noqa: PLC0415

    def timed_get_action(*a, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_get_action(*a, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        current_latencies.append((time.perf_counter() - started) * 1000.0)
        return result

    module.get_action = timed_get_action
    # M3 evidence does not require replay videos; suppressing their encoding
    # saves submission-week wall time without changing the simulator rollout.
    module.save_rollout_video = lambda *a, **kw: None
    for key in list(module.TASK_MAX_STEPS):
        module.TASK_MAX_STEPS[key] = 300

    def instrumented_run_episode(
        cfg,
        env,
        task_description,
        model,
        resize_size,
        processor=None,
        action_head=None,
        proprio_projector=None,
        noisy_action_projector=None,
        initial_state=None,
        log_file=None,
    ):
        current_latencies.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        success, replay_images = original_run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = float(torch.cuda.max_memory_allocated() / (1024**2))
        else:
            peak = 0.0
        duration = time.perf_counter() - started
        steps = min(300, len(replay_images))
        mean_latency = sum(current_latencies) / len(current_latencies) if current_latencies else 0.0
        records.append(
            build_episode_record(
                model="openvla_oft",
                checkpoint_ref=str(training["checkpoint_ref"]),
                source_ref=str(training["source_ref"]),
                order=str(training["order"]),
                track=str(training["track"]),
                seed=args.seed,
                task_id=f"{args.suite}:{task_description}",
                success=bool(success),
                steps=steps,
                episode_duration_sec=duration,
                mean_inference_latency_ms=mean_latency,
                peak_inference_vram_mib=peak,
            )
        )
        return success, replay_images

    module.run_episode = instrumented_run_episode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cli = [
        "run_libero_eval.py",
        "--pretrained_checkpoint",
        str(training["checkpoint_ref"]),
        "--task_suite_name",
        args.suite,
        "--num_trials_per_task",
        str(args.trials_per_task),
        "--seed",
        str(args.seed),
        "--center_crop",
        "True",
        "--local_log_dir",
        str(args.output_dir / "upstream_logs"),
        "--use_wandb",
        "False",
    ]
    old_argv = sys.argv
    try:
        sys.argv = cli
        module.eval_libero()
    finally:
        sys.argv = old_argv
        module.run_episode = original_run_episode
        module.get_action = original_get_action
        module.save_rollout_video = original_save_video

    expected = 10 * args.trials_per_task
    if len(records) != expected:
        raise RuntimeError(f"OpenVLA episode record count mismatch: {len(records)} != {expected}")
    args.records_out.parent.mkdir(parents=True, exist_ok=True)
    args.records_out.write_text(json.dumps({"episodes": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "model": "openvla_oft",
        "suite": args.suite,
        "run_seed": args.seed,
        "episode_count": len(records),
        "records": str(args.records_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
