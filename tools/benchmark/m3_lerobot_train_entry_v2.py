#!/usr/bin/env python3
"""Execute π0.5 / SmolVLA M3 training against the canonical sample stream.

Equal-data consumes exactly 4,800 examples (64 in smoke mode). Equal-wall uses
an effectively-unbounded counter sampler and stops at the first optimizer
boundary at/after the wall-clock target. The final equal-wall checkpoint is
saved only *after* the timer has stopped, so checkpoint I/O is excluded from
the frozen M3 wall-time metric.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
}
SEEDS = (20260906, 20260907)
EFFECTIVE_BATCH = 32
EQUAL_DATA_BUDGET = 4800
EQUAL_WALL_SEC = 1800.0
SMOKE_EQUAL_DATA_SAMPLES = 64
SMOKE_WALL_SEC = 5.0
HUGE_STEPS = 1_000_000_000


class _EqualWallStop(RuntimeError):
    """Internal control-flow exception raised only after a safe optimizer boundary."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--model", choices=("pi05", "smolvla"), required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--micro-batch", type=int, required=True)
    p.add_argument("--grad-accum", type=int, required=True)
    p.add_argument("--track", choices=("equal_data", "equal_wall"), required=True)
    p.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result-out", type=Path, required=True)
    p.add_argument("--wall-target-sec", type=float)
    return p.parse_args()


def _run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _verify_source(args: argparse.Namespace) -> None:
    root = args.source_root.resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"pinned LeRobot checkout missing .git: {root}")
    head = _run_text(["git", "-C", str(root), "rev-parse", "HEAD"])
    if args.model == "smolvla":
        if head != SOURCE_REFS["smolvla"]:
            raise RuntimeError(f"SmolVLA source HEAD mismatch: {head}")
    else:
        tagged = _run_text(["git", "-C", str(root), "rev-parse", "v0.4.4"])
        if head != tagged:
            raise RuntimeError(f"pi0.5 source is not pinned to v0.4.4: head={head} tag={tagged}")
        patched = root / "src/lerobot/scripts/lerobot_train.py"
        if "LEROBOT_GRAD_ACCUM" not in patched.read_text(encoding="utf-8"):
            raise RuntimeError("pi0.5 source is missing the required M3 grad-accum patch")


def _checkpoint_ref(output_dir: Path) -> str:
    for candidate in (
        output_dir / "checkpoints/last/pretrained_model",
        output_dir / "checkpoints/last",
    ):
        if candidate.exists():
            return str(candidate.resolve())
    candidates = sorted(
        (output_dir / "checkpoints").glob("*/pretrained_model")
        if (output_dir / "checkpoints").is_dir()
        else []
    )
    if candidates:
        return str(candidates[-1].resolve())
    raise FileNotFoundError(f"no final checkpoint found under {output_dir}")


def _episode_ids_json(manifest: dict[str, Any]) -> str:
    return json.dumps([int(x) for x in manifest["episode_ids"]], separators=(",", ":"))


def _wall_target(args: argparse.Namespace) -> float:
    if args.track != "equal_wall":
        if args.wall_target_sec is not None:
            raise ValueError("--wall-target-sec is valid only for equal_wall")
        return 0.0
    if args.mode == "benchmark":
        value = EQUAL_WALL_SEC if args.wall_target_sec is None else float(args.wall_target_sec)
        if abs(value - EQUAL_WALL_SEC) > 1e-9:
            raise ValueError("benchmark equal-wall target is frozen at exactly 1800 seconds")
        return value
    value = SMOKE_WALL_SEC if args.wall_target_sec is None else float(args.wall_target_sec)
    if not 0 < value <= 60:
        raise ValueError("smoke equal-wall target must be within (0,60] seconds")
    return value


def _pi05_args(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    seed: int,
    steps: int,
    save_checkpoint: bool,
) -> list[str]:
    return [
        "lerobot-train",
        "--policy.type=pi05",
        "--policy.pretrained_path=lerobot/pi05_libero_base",
        "--policy.dtype=bfloat16",
        "--policy.n_obs_steps=1",
        "--policy.n_action_steps=10",
        "--policy.optimizer_lr=5e-5",
        "--policy.num_inference_steps=10",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--peft.method_type=LORA",
        "--peft.r=16",
        "--dataset.repo_id=lerobot/libero_plus",
        f"--dataset.root={args.dataset_root}",
        "--dataset.video_backend=pyav",
        f"--dataset.episodes={_episode_ids_json(manifest)}",
        f"--batch_size={args.micro_batch}",
        f"--steps={int(steps)}",
        "--num_workers=0",
        f"--save_checkpoint={'true' if save_checkpoint else 'false'}",
        f"--save_freq={int(steps)}",
        "--eval_freq=999999999",
        "--log_freq=1",
        f"--output_dir={args.output_dir}",
        f"--job_name=m3_pi05_{args.order}_{args.track}_{args.mode}_seed{seed}",
        f"--seed={seed}",
        "--wandb.enable=false",
    ]


def _smolvla_args(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    seed: int,
    steps: int,
    save_checkpoint: bool,
) -> list[str]:
    return [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--dataset.repo_id=lerobot/libero_plus",
        f"--dataset.root={args.dataset_root}",
        "--dataset.video_backend=pyav",
        "--dataset.return_uint8=true",
        f"--dataset.episodes={_episode_ids_json(manifest)}",
        f"--batch_size={args.micro_batch}",
        f"--accelerator.gradient_accumulation.steps={args.grad_accum}",
        f"--steps={int(steps)}",
        "--num_workers=0",
        f"--save_checkpoint={'true' if save_checkpoint else 'false'}",
        f"--save_freq={int(steps)}",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--ema.enable=false",
        "--log_freq=1",
        f"--output_dir={args.output_dir}",
        f"--job_name=m3_smolvla_{args.order}_{args.track}_{args.mode}_seed{seed}",
        f"--seed={seed}",
        "--wandb.enable=false",
    ]


def _manual_wall_checkpoint(trainer, state: dict[str, Any], args: argparse.Namespace) -> Path:
    required = ("cfg", "policy", "optimizer", "accelerator", "preprocessor", "postprocessor")
    missing = [key for key in required if state.get(key) is None]
    if missing:
        raise RuntimeError(f"cannot save equal-wall checkpoint; missing runtime state: {missing}")
    step = int(state["optimizer_updates"] if args.model == "pi05" else state["micro_steps"])
    if step <= 0:
        raise RuntimeError("cannot save checkpoint before an optimizer step")
    checkpoint_dir = trainer.get_step_checkpoint_dir(args.output_dir, step, step)
    kwargs = {
        "checkpoint_dir": checkpoint_dir,
        "step": step,
        "cfg": state["cfg"],
        "optimizer": state["optimizer"],
        "scheduler": state.get("scheduler"),
        "preprocessor": state["preprocessor"],
        "postprocessor": state["postprocessor"],
    }
    if args.model == "pi05":
        kwargs["policy"] = state["accelerator"].unwrap_model(state["policy"])
    else:
        kwargs["policy"] = state["policy"]
        kwargs["accelerator"] = state["accelerator"]
    trainer.save_checkpoint(**kwargs)
    trainer.update_last_checkpoint(checkpoint_dir)
    state["accelerator"].wait_for_everyone()
    return checkpoint_dir


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 training requires explicit PARC_M3_EXECUTE=1")
    if args.micro_batch <= 0 or args.grad_accum <= 0 or args.micro_batch * args.grad_accum != EFFECTIVE_BATCH:
        raise ValueError("M3 effective batch must equal 32")
    _verify_source(args)
    target_wall = _wall_target(args)

    sys.path.insert(0, str(args.repo_root))
    from tools.benchmark.m3_equal_wall_schedule import (  # noqa: PLC0415
        make_counter_stream_sampler,
        stream_descriptor,
        verify_materialized_prefix,
    )
    from tools.benchmark.m3_scheduled_data import (  # noqa: PLC0415
        dataset_absolute_to_relative_map,
        load_schedule,
        make_fixed_index_sampler,
        references_to_lerobot_indices,
    )
    from tools.data.openvla_lerobot_streaming import load_episode_metadata, load_manifest  # noqa: PLC0415

    manifest = load_manifest(args.manifest)
    if manifest.get("episode_ids_sha256") != D10_HASH:
        raise ValueError("manifest D10 mismatch")
    schedule = load_schedule(args.schedule, require_equal_data=True)
    seed = int(schedule.get("seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"unexpected M3 seed: {seed}")
    ids = [int(x) for x in manifest["episode_ids"]]
    metadata = load_episode_metadata(args.dataset_root, ids)
    lengths = [int(metadata.loc[episode_id]["length"]) for episode_id in ids]
    starts = [int(metadata.loc[episode_id]["dataset_from_index"]) for episode_id in ids]
    verify_materialized_prefix(schedule, episode_ids=ids, episode_lengths=lengths)
    wall_descriptor = stream_descriptor(seed=seed) if args.track == "equal_wall" else None

    fixed_references = list(schedule["references"])
    if args.track == "equal_data" and args.mode == "smoke":
        fixed_references = fixed_references[:SMOKE_EQUAL_DATA_SAMPLES]
    if args.track == "equal_data":
        if len(fixed_references) % EFFECTIVE_BATCH != 0:
            raise ValueError("equal-data reference count must divide effective batch exactly")
        expected_micro_steps = len(fixed_references) // args.micro_batch
        expected_optimizer_updates = len(fixed_references) // EFFECTIVE_BATCH
    else:
        expected_micro_steps = None
        expected_optimizer_updates = None

    import torch  # noqa: PLC0415
    import accelerate.data_loader as accelerate_dl  # noqa: PLC0415

    trainer = importlib.import_module("lerobot.scripts.lerobot_train")
    original_dataloader = torch.utils.data.DataLoader
    original_shard_iter = accelerate_dl.DataLoaderShard.__iter__
    original_update = trainer.update_policy
    original_make_pp = trainer.make_pre_post_processors
    original_make_dataset = getattr(trainer, "make_dataset", None)
    original_make_train_eval = getattr(trainer, "make_train_eval_datasets", None)

    state: dict[str, Any] = {
        "cfg": None,
        "policy": None,
        "optimizer": None,
        "scheduler": None,
        "accelerator": None,
        "preprocessor": None,
        "postprocessor": None,
        "started_at": None,
        "stopped_at": None,
        "micro_steps": 0,
        "optimizer_updates": 0,
        "samples_consumed": 0,
        "last_loss": None,
        "peak_train_vram_mib": 0.0,
        "sampler_injected": False,
        "resolved_index_sha256": None,
    }

    def capture_cfg(fn):
        def wrapped(cfg, *fn_args, **fn_kwargs):
            state["cfg"] = cfg
            return fn(cfg, *fn_args, **fn_kwargs)
        return wrapped

    if original_make_dataset is not None:
        trainer.make_dataset = capture_cfg(original_make_dataset)
    if original_make_train_eval is not None:
        trainer.make_train_eval_datasets = capture_cfg(original_make_train_eval)

    def capture_processors(*pp_args, **pp_kwargs):
        pre, post = original_make_pp(*pp_args, **pp_kwargs)
        state["preprocessor"] = pre
        state["postprocessor"] = post
        return pre, post

    trainer.make_pre_post_processors = capture_processors

    class CanonicalScheduleDataLoader(original_dataloader):
        def __init__(self, dataset, *dl_args, **dl_kwargs):
            is_training_dataset = (
                not state["sampler_injected"]
                and hasattr(dataset, "meta")
                and getattr(dataset, "episodes", None) is not None
            )
            if is_training_dataset:
                absolute_map = dataset_absolute_to_relative_map(dataset)
                if args.track == "equal_data":
                    relative = references_to_lerobot_indices(
                        fixed_references,
                        episode_metadata=metadata,
                        absolute_to_relative_idx=absolute_map,
                    )
                    state["resolved_index_sha256"] = hashlib.sha256(
                        json.dumps(relative, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    sampler = make_fixed_index_sampler(relative)
                else:
                    sampler = make_counter_stream_sampler(
                        episode_ids=ids,
                        episode_lengths=lengths,
                        dataset_from_indices=starts,
                        absolute_to_relative_idx=absolute_map,
                        seed=seed,
                    )
                    state["resolved_index_sha256"] = wall_descriptor["stream_sha256"]
                dl_kwargs["sampler"] = sampler
                dl_kwargs["shuffle"] = False
                state["sampler_injected"] = True
            super().__init__(dataset, *dl_args, **dl_kwargs)

    def timed_shard_iter(self):
        if state["started_at"] is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            state["started_at"] = time.perf_counter()
        yield from original_shard_iter(self)

    def capture_stop() -> None:
        if state["stopped_at"] is not None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            state["peak_train_vram_mib"] = float(torch.cuda.max_memory_allocated() / (1024**2))
        state["stopped_at"] = time.perf_counter()

    def timed_update(*uargs, **ukwargs):
        result = original_update(*uargs, **ukwargs)
        state["micro_steps"] += 1
        state["samples_consumed"] += int(args.micro_batch)
        state["policy"] = uargs[1]
        state["optimizer"] = uargs[3]
        state["accelerator"] = ukwargs.get("accelerator")
        state["scheduler"] = ukwargs.get("lr_scheduler")
        accelerator = state["accelerator"]
        if accelerator is None:
            raise RuntimeError("trainer update did not expose Accelerator")
        tracker = result[0]
        loss_obj = getattr(tracker, "loss", None)
        if loss_obj is not None:
            for candidate in (loss_obj, getattr(loss_obj, "val", None), getattr(loss_obj, "avg", None)):
                if candidate is None:
                    continue
                try:
                    state["last_loss"] = float(candidate)
                    break
                except (TypeError, ValueError):
                    continue
        if accelerator.sync_gradients:
            state["optimizer_updates"] += 1
            if args.track == "equal_wall":
                if state["started_at"] is None:
                    raise RuntimeError("equal-wall timer did not start before batch fetch")
                elapsed = time.perf_counter() - float(state["started_at"])
                if elapsed >= target_wall:
                    capture_stop()
                    raise _EqualWallStop()
        if args.track == "equal_data" and expected_micro_steps is not None:
            if state["micro_steps"] == expected_micro_steps:
                capture_stop()
        return result

    torch.utils.data.DataLoader = CanonicalScheduleDataLoader
    accelerate_dl.DataLoaderShard.__iter__ = timed_shard_iter
    trainer.update_policy = timed_update

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "pi05":
        os.environ["LEROBOT_GRAD_ACCUM"] = str(args.grad_accum)
        steps = expected_optimizer_updates if args.track == "equal_data" else HUGE_STEPS
        trainer_argv = _pi05_args(
            args, manifest, seed=seed, steps=int(steps), save_checkpoint=args.track == "equal_data"
        )
    else:
        steps = expected_micro_steps if args.track == "equal_data" else HUGE_STEPS
        trainer_argv = _smolvla_args(
            args, manifest, seed=seed, steps=int(steps), save_checkpoint=args.track == "equal_data"
        )

    old_argv = sys.argv
    wall_stopped = False
    try:
        sys.argv = trainer_argv
        try:
            trainer.train()
        except _EqualWallStop:
            if args.track != "equal_wall":
                raise
            wall_stopped = True
    finally:
        sys.argv = old_argv
        trainer.update_policy = original_update
        trainer.make_pre_post_processors = original_make_pp
        if original_make_dataset is not None:
            trainer.make_dataset = original_make_dataset
        if original_make_train_eval is not None:
            trainer.make_train_eval_datasets = original_make_train_eval
        torch.utils.data.DataLoader = original_dataloader
        accelerate_dl.DataLoaderShard.__iter__ = original_shard_iter

    if not state["sampler_injected"]:
        raise RuntimeError("canonical M3 sampler was never injected")
    if state["started_at"] is None:
        raise RuntimeError("train-loop timer was never started")
    if args.track == "equal_data":
        if state["samples_consumed"] != len(fixed_references):
            raise RuntimeError(
                f"equal-data sample mismatch: {state['samples_consumed']} != {len(fixed_references)}"
            )
        if state["optimizer_updates"] != expected_optimizer_updates:
            raise RuntimeError(
                f"equal-data optimizer-update mismatch: {state['optimizer_updates']} != {expected_optimizer_updates}"
            )
        if state["stopped_at"] is None:
            capture_stop()
    else:
        if not wall_stopped or state["stopped_at"] is None:
            raise RuntimeError("equal-wall run did not stop at a safe optimizer boundary")
        elapsed = float(state["stopped_at"] - state["started_at"])
        if elapsed + 1e-9 < target_wall:
            raise RuntimeError(f"equal-wall stopped early: {elapsed:.6f} < {target_wall:.6f}")
        if state["samples_consumed"] % EFFECTIVE_BATCH != 0:
            raise RuntimeError("equal-wall stopped away from an effective-batch optimizer boundary")
        # Timer is already frozen. Checkpoint I/O below is excluded by construction.
        _manual_wall_checkpoint(trainer, state, args)

    checkpoint_ref = _checkpoint_ref(args.output_dir)
    train_wall = float(state["stopped_at"] - state["started_at"])
    result = {
        "schema_version": 2,
        "stage": "M3_model_training",
        "status": "PASS",
        "model": args.model,
        "source_ref": SOURCE_REFS[args.model],
        "checkpoint_ref": checkpoint_ref,
        "order": args.order,
        "track": args.track,
        "mode": args.mode,
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": seed,
        "sampling_policy": schedule["policy"],
        "sampling_schedule_sha256": (
            schedule.get("schedule_sha256") if args.track == "equal_data" else wall_descriptor["stream_sha256"]
        ),
        "resolved_index_sha256": state["resolved_index_sha256"],
        "samples_consumed": int(state["samples_consumed"]),
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": EFFECTIVE_BATCH,
        "micro_steps": int(state["micro_steps"]),
        "optimizer_updates": int(state["optimizer_updates"]),
        "wall_target_sec": target_wall if args.track == "equal_wall" else None,
        "timer_boundary": "immediately_before_first_train_batch_fetch_to_safe_optimizer_boundary",
        "checkpoint_save_excluded_from_train_wall_time": args.track == "equal_wall",
        "metrics": {
            "train_wall_time": train_wall,
            "peak_train_vram": float(state["peak_train_vram_mib"]),
        },
        "training_loss_best_effort": state["last_loss"],
        "canonical_schedule_sampler": True,
        "native_random_sampler_used": False,
        "benchmark_training_started": args.mode == "benchmark",
    }
    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    args.result_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
