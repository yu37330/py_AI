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


def _validate_checkpoint(checkpoint: Path, suites: tuple[str, ...]) -> None:
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("OpenVLA M3 checkpoint is not a merged evaluator-loadable model: config.json missing")
    if not list(checkpoint.glob("model*.safetensors")):
        raise FileNotFoundError("OpenVLA M3 checkpoint is not a merged evaluator-loadable model: safetensors missing")
    if not list(checkpoint.glob("action_head--*checkpoint.pt")):
        raise FileNotFoundError("OpenVLA M3 checkpoint action head missing")
    if not list(checkpoint.glob("proprio_projector--*checkpoint.pt")):
        raise FileNotFoundError("OpenVLA M3 checkpoint proprio projector missing")
    path = checkpoint / "dataset_statistics.json"
    if not path.is_file():
        raise FileNotFoundError(f"OpenVLA M3 checkpoint missing normalization stats: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for suite in suites:
        stats = payload.get(suite)
        if not isinstance(stats, dict):
            raise ValueError(f"OpenVLA checkpoint missing D10 normalization alias: {suite}")
        for feature in ("action", "proprio"):
            feature_stats = stats.get(feature)
            if not isinstance(feature_stats, dict):
                raise ValueError(f"OpenVLA checkpoint missing {suite}.{feature} stats")
            for key in ("min", "max", "q01", "q99"):
                if key not in feature_stats:
                    raise ValueError(f"OpenVLA checkpoint missing {suite}.{feature}.{key}")


def main() -> int:
    args = parse_args()
    if args.trials_per_task <= 0:
        raise ValueError("trials-per-task must be positive")
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(args.source_root))
    from tools.benchmark.m3_eval_instrumentation import build_episode_record  # noqa: PLC0415
    from tools.benchmark.m3_eval_runtime_guard import validate_runtime  # noqa: PLC0415
    from tools.benchmark.m3_libero_executor import (  # noqa: PLC0415
        SEEDS,
        SUITES,
        validate_training_result,
    )

    training = validate_training_result(args.training_result)
    if training["model"] != "openvla_oft":
        raise ValueError("OpenVLA evaluator requires openvla_oft training result")
    if training.get("checkpoint_eval_ready") is not True:
        raise ValueError("OpenVLA training result is not marked checkpoint_eval_ready")
    if training.get("lora_merged_for_evaluation") is not True:
        raise ValueError("OpenVLA LoRA checkpoint was not finalized for evaluation")
    if args.suite not in SUITES:
        raise ValueError(f"unexpected LIBERO suite: {args.suite}")
    if args.seed not in SEEDS:
        raise ValueError(f"unexpected M3 evaluation seed: {args.seed}")
    runtime = validate_runtime("openvla_oft", args.source_root)
    checkpoint = Path(str(training["checkpoint_ref"])).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"OpenVLA M3 checkpoint directory missing: {checkpoint}")
    _validate_checkpoint(checkpoint, SUITES)

    module = importlib.import_module("experiments.robot.libero.run_libero_eval")
    original_run_episode = module.run_episode
    original_get_action = module.get_action
    original_save_video = module.save_rollout_video
    original_log_message = module.log_message
    original_task_max_steps = dict(module.TASK_MAX_STEPS)
    records: list[dict] = []
    current_latencies: list[float] = []
    current_errors: list[str] = []

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

    def fail_closed_log(message: str, log_file=None):
        text = str(message)
        original_log_message(text, log_file)
        if text.startswith("Episode error:"):
            current_errors.append(text)

    module.get_action = timed_get_action
    module.log_message = fail_closed_log
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
        current_errors.clear()
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
        if current_errors:
            raise RuntimeError(current_errors[-1])
        if not current_latencies:
            raise RuntimeError(
                "OpenVLA episode completed without a model inference call; refusing silent simulator/model failure"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = float(torch.cuda.max_memory_allocated() / (1024**2))
        else:
            peak = 0.0
        duration = time.perf_counter() - started
        steps = len(replay_images)
        if not 0 < steps <= 300:
            raise RuntimeError(f"OpenVLA evaluator violated frozen 300-step boundary: {steps}")
        mean_latency = sum(current_latencies) / len(current_latencies)
        record = build_episode_record(
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
        record["gpu_name"] = runtime["gpu_name"]
        record["gpu_vram_mib"] = runtime["gpu_vram_mib"]
        record["inference_calls"] = len(current_latencies)
        records.append(record)
        return success, replay_images

    module.run_episode = instrumented_run_episode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cli = [
        "run_libero_eval.py",
        "--pretrained_checkpoint",
        str(checkpoint),
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
        module.log_message = original_log_message
        module.TASK_MAX_STEPS.clear()
        module.TASK_MAX_STEPS.update(original_task_max_steps)

    expected = 10 * args.trials_per_task
    if len(records) != expected:
        raise RuntimeError(f"OpenVLA episode record count mismatch: {len(records)} != {expected}")
    args.records_out.parent.mkdir(parents=True, exist_ok=True)
    args.records_out.write_text(
        json.dumps({"runtime": runtime, "episodes": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "model": "openvla_oft",
        "suite": args.suite,
        "run_seed": args.seed,
        "episode_count": len(records),
        "gpu_name": runtime["gpu_name"],
        "records": str(args.records_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
