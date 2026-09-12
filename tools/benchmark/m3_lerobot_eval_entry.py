#!/usr/bin/env python3
"""Instrument pinned LeRobot LIBERO evaluation and emit M3 episode records.

The upstream evaluator owns environment/policy preprocessing. We wrap only its
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

ENV_VISUAL_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
CANONICAL_D10_VISUAL_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--training-result", type=Path, required=True)
    p.add_argument("--model", choices=("pi05", "smolvla"), required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--records-out", type=Path, required=True)
    return p.parse_args()


def _checkpoint_visual_features(checkpoint: Path) -> tuple[str, ...]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LeRobot checkpoint config missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    input_features = payload.get("input_features")
    if not isinstance(input_features, dict):
        raise RuntimeError(f"LeRobot checkpoint input_features missing/invalid: {config_path}")
    visuals = tuple(str(key) for key in input_features if str(key).startswith("observation.images."))
    if len(visuals) != 2:
        raise RuntimeError(
            f"M3 LeRobot evaluator requires exactly two visual inputs; got {list(visuals)} from {config_path}"
        )
    return visuals


def _camera_rename_map(checkpoint: Path) -> dict[str, str]:
    """Resolve the frozen LIBERO env camera keys against checkpoint feature names.

    hf-libero emits ``image`` (agent/front) and ``image2`` (wrist). M3 D10
    training checkpoints use ``front`` and ``wrist``. If a checkpoint already
    expects the env-native names, no map is required. Any other layout is
    rejected rather than guessed so evaluation cannot silently feed cameras to
    the wrong policy inputs.
    """
    visuals = _checkpoint_visual_features(checkpoint)
    expected = set(visuals)
    source = set(ENV_VISUAL_KEYS)
    canonical = set(CANONICAL_D10_VISUAL_KEYS)
    if expected == source:
        return {}
    if expected == canonical:
        return {
            ENV_VISUAL_KEYS[0]: CANONICAL_D10_VISUAL_KEYS[0],
            ENV_VISUAL_KEYS[1]: CANONICAL_D10_VISUAL_KEYS[1],
        }
    raise RuntimeError(
        "unsupported checkpoint visual feature layout for M3 LIBERO evaluation: "
        f"{list(visuals)}; expected either {list(ENV_VISUAL_KEYS)} or {list(CANONICAL_D10_VISUAL_KEYS)}"
    )


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root))
    from tools.benchmark.m3_eval_instrumentation import (  # noqa: PLC0415
        build_episode_record,
        instrument_select_action,
    )
    from tools.benchmark.m3_eval_runtime_guard import validate_runtime  # noqa: PLC0415
    from tools.benchmark.m3_libero_executor import (  # noqa: PLC0415
        SEEDS,
        SUITES,
        validate_training_result,
    )

    training = validate_training_result(args.training_result)
    if training["model"] != args.model:
        raise ValueError("training-result model mismatch")
    if args.suite not in SUITES:
        raise ValueError(f"unexpected LIBERO suite: {args.suite}")
    if args.seed not in SEEDS:
        raise ValueError(f"unexpected M3 evaluation seed: {args.seed}")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    runtime = validate_runtime(args.model, args.source_root)

    checkpoint = Path(str(training["checkpoint_ref"])).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"LeRobot checkpoint missing: {checkpoint}")
    visual_features = _checkpoint_visual_features(checkpoint)
    rename_map = _camera_rename_map(checkpoint)
    print(
        json.dumps(
            {
                "model": args.model,
                "checkpoint_visual_features": list(visual_features),
                "camera_rename_map": rename_map,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    module = importlib.import_module("lerobot.scripts.lerobot_eval")
    original_rollout = module.rollout
    original_eval_policy_all = getattr(module, "eval_policy_all", None)
    records: list[dict] = []

    def instrumented_rollout(*rollout_args, **rollout_kwargs):
        env = rollout_kwargs.get("env")
        policy = rollout_kwargs.get("policy")
        actual_seeds = rollout_kwargs.get("seeds")
        if env is None and rollout_args:
            env = rollout_args[0]
        if policy is None and len(rollout_args) > 1:
            policy = rollout_args[1]
        if env is None or policy is None:
            raise RuntimeError("cannot instrument LeRobot rollout without env/policy")
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
            if steps > 300:
                raise RuntimeError(f"LeRobot evaluator exceeded frozen 300-step boundary: {steps}")
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
            record["gpu_name"] = runtime["gpu_name"]
            record["gpu_vram_mib"] = runtime["gpu_vram_mib"]
            records.append(record)
        return rollout_data

    def no_video_eval_policy_all(*eval_args, **eval_kwargs):
        """Keep M3 smoke/screening evaluation headless and storage-bounded."""
        if original_eval_policy_all is None:
            raise RuntimeError("pinned LeRobot evaluator is missing eval_policy_all")
        eval_kwargs["max_episodes_rendered"] = 0
        eval_kwargs["videos_dir"] = None
        if "recording_dir" in eval_kwargs:
            eval_kwargs["recording_dir"] = None
        return original_eval_policy_all(*eval_args, **eval_kwargs)

    module.rollout = instrumented_rollout
    if original_eval_policy_all is not None:
        module.eval_policy_all = no_video_eval_policy_all
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cli = [
        "lerobot-eval",
        f"--policy.path={training['checkpoint_ref']}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        "--env.episode_length=300",
        "--env.init_states=true",
        # Do not pass --env.hard_reset here. π0.5 is pinned to LeRobot v0.4.4,
        # whose LiberoEnv config has no hard_reset field. hf-libero/robosuite
        # already defaults hard_reset=True; the newer SmolVLA LiberoEnv does too.
        "--eval.batch_size=1",
        f"--eval.n_episodes={args.episodes}",
        "--env.max_parallel_tasks=1",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir / 'upstream'}",
    ]
    if rename_map:
        cli.append(f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}")
    if args.model == "pi05":
        cli.append("--policy.n_action_steps=10")
    old_argv = sys.argv
    try:
        sys.argv = cli
        module.main()
    finally:
        sys.argv = old_argv
        module.rollout = original_rollout
        if original_eval_policy_all is not None:
            module.eval_policy_all = original_eval_policy_all

    expected = 10 * args.episodes
    if len(records) != expected:
        raise RuntimeError(
            f"LeRobot episode record count mismatch for {args.suite}: {len(records)} != {expected}"
        )
    args.records_out.parent.mkdir(parents=True, exist_ok=True)
    args.records_out.write_text(
        json.dumps({"runtime": runtime, "episodes": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "model": args.model,
        "suite": args.suite,
        "run_seed": args.seed,
        "episode_count": len(records),
        "gpu_name": runtime["gpu_name"],
        "records": str(args.records_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
