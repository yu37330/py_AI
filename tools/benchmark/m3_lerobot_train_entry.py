#!/usr/bin/env python3
"""Execute π0.5/SmolVLA equal-data M3 training on the canonical fixed schedule.

This entry point deliberately reuses each pinned LeRobot trainer and only
replaces its training sampler.  The fixed sampler is injected before
``accelerator.prepare`` so Accelerate shards/handles it normally without ever
falling back to shuffle/EpisodeAwareSampler.

Equal-wall is intentionally not implemented here yet; a separate stop hook is
required so the final optimizer-boundary checkpoint is saved without polluting
the 1800-second train-loop timer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
}


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
    p.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result-out", type=Path, required=True)
    return p.parse_args()


def _checkpoint_ref(output_dir: Path) -> str:
    candidates = [
        output_dir / "checkpoints/last/pretrained_model",
        output_dir / "checkpoints/last",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    checkpoints = sorted(
        (output_dir / "checkpoints").glob("*/pretrained_model")
        if (output_dir / "checkpoints").is_dir()
        else []
    )
    if checkpoints:
        return str(checkpoints[-1])
    raise FileNotFoundError(f"no final checkpoint found under {output_dir}")


def _episode_ids_json(manifest: dict[str, Any]) -> str:
    return json.dumps([int(x) for x in manifest["episode_ids"]], separators=(",", ":"))


def _pi05_trainer_args(args: argparse.Namespace, manifest: dict[str, Any], optimizer_updates: int) -> list[str]:
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
        f"--steps={optimizer_updates}",
        "--num_workers=0",
        f"--save_freq={optimizer_updates}",
        "--eval_freq=999999999",
        "--log_freq=1",
        f"--output_dir={args.output_dir}",
        f"--job_name=m3_{args.model}_{args.order}_equal_data_{args.mode}",
        "--seed=1000",
        "--wandb.enable=false",
    ]


def _smolvla_trainer_args(args: argparse.Namespace, manifest: dict[str, Any], micro_steps: int) -> list[str]:
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
        f"--steps={micro_steps}",
        "--num_workers=0",
        "--save_checkpoint=true",
        f"--save_freq={micro_steps}",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--log_freq=1",
        f"--output_dir={args.output_dir}",
        f"--job_name=m3_{args.model}_{args.order}_equal_data_{args.mode}",
        "--seed=1000",
        "--wandb.enable=false",
    ]


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 training requires explicit PARC_M3_EXECUTE=1")
    if args.micro_batch * args.grad_accum != 32:
        raise ValueError("effective batch must be 32")
    sys.path.insert(0, str(args.repo_root))
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
    references = list(schedule["references"])
    if args.mode == "smoke":
        references = references[:64]
    if len(references) % args.micro_batch != 0:
        raise ValueError("scheduled example count must divide micro-batch exactly")
    micro_steps = len(references) // args.micro_batch
    if micro_steps % args.grad_accum != 0:
        raise ValueError("scheduled micro-step count must divide gradient accumulation")
    optimizer_updates = micro_steps // args.grad_accum
    metadata = load_episode_metadata(args.dataset_root, [int(x) for x in manifest["episode_ids"]])

    import torch  # noqa: PLC0415
    import accelerate.data_loader as accelerate_dl  # noqa: PLC0415

    original_dataloader = torch.utils.data.DataLoader
    original_shard_iter = accelerate_dl.DataLoaderShard.__iter__
    timing: dict[str, Any] = {
        "started_at": None,
        "stopped_at": None,
        "update_calls": 0,
        "last_loss": None,
        "peak_train_vram_mib": 0.0,
        "sampler_injected": False,
        "resolved_index_sha256": None,
    }

    class FixedScheduleDataLoader(original_dataloader):
        def __init__(self, dataset, *dl_args, **dl_kwargs):
            is_training_dataset = (
                not timing["sampler_injected"]
                and hasattr(dataset, "meta")
                and getattr(dataset, "episodes", None) is not None
            )
            if is_training_dataset:
                relative = references_to_lerobot_indices(
                    references,
                    episode_metadata=metadata,
                    absolute_to_relative_idx=dataset_absolute_to_relative_map(dataset),
                )
                timing["resolved_index_sha256"] = hashlib.sha256(
                    json.dumps(relative, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                dl_kwargs["sampler"] = make_fixed_index_sampler(relative)
                dl_kwargs["shuffle"] = False
                timing["sampler_injected"] = True
            super().__init__(dataset, *dl_args, **dl_kwargs)

    def timed_shard_iter(self):
        if timing["started_at"] is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            timing["started_at"] = time.perf_counter()
        yield from original_shard_iter(self)

    torch.utils.data.DataLoader = FixedScheduleDataLoader
    accelerate_dl.DataLoaderShard.__iter__ = timed_shard_iter

    trainer = importlib.import_module("lerobot.scripts.lerobot_train")
    original_update = trainer.update_policy

    def timed_update(*uargs, **ukwargs):
        result = original_update(*uargs, **ukwargs)
        timing["update_calls"] += 1
        tracker = result[0]
        loss_obj = getattr(tracker, "loss", None)
        if loss_obj is not None:
            try:
                timing["last_loss"] = float(loss_obj)
            except (TypeError, ValueError):
                try:
                    timing["last_loss"] = float(loss_obj.val)
                except Exception:
                    pass
        if timing["update_calls"] == micro_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                timing["peak_train_vram_mib"] = float(torch.cuda.max_memory_allocated() / (1024**2))
            timing["stopped_at"] = time.perf_counter()
        return result

    trainer.update_policy = timed_update

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "pi05":
        patched_source = Path(args.source_root) / "src/lerobot/scripts/lerobot_train.py"
        if "LEROBOT_GRAD_ACCUM" not in patched_source.read_text(encoding="utf-8"):
            raise RuntimeError("pi05 source is missing required grad-accum patch")
        os.environ["LEROBOT_GRAD_ACCUM"] = str(args.grad_accum)
        trainer_argv = _pi05_trainer_args(args, manifest, optimizer_updates)
    else:
        trainer_argv = _smolvla_trainer_args(args, manifest, micro_steps)

    old_argv = sys.argv
    try:
        sys.argv = trainer_argv
        trainer.train()
    finally:
        sys.argv = old_argv
        trainer.update_policy = original_update
        torch.utils.data.DataLoader = original_dataloader
        accelerate_dl.DataLoaderShard.__iter__ = original_shard_iter

    if not timing["sampler_injected"]:
        raise RuntimeError("fixed M3 sampler was never injected")
    if timing["update_calls"] != micro_steps:
        raise RuntimeError(
            f"trainer update call mismatch: {timing['update_calls']} != {micro_steps}"
        )
    if timing["started_at"] is None or timing["stopped_at"] is None:
        raise RuntimeError("train-loop timer was not captured")
    checkpoint_ref = _checkpoint_ref(args.output_dir)
    result = {
        "schema_version": 1,
        "stage": "M3_model_training",
        "status": "PASS",
        "model": args.model,
        "source_ref": SOURCE_REFS[args.model],
        "checkpoint_ref": checkpoint_ref,
        "order": args.order,
        "track": "equal_data",
        "mode": args.mode,
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_schedule_sha256": schedule.get("schedule_sha256"),
        "sampling_seed": schedule.get("seed"),
        "resolved_index_sha256": timing["resolved_index_sha256"],
        "samples_consumed": len(references),
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": 32,
        "micro_steps": micro_steps,
        "optimizer_updates": optimizer_updates,
        "seed_set": [20260906, 20260907],
        "metrics": {
            "train_wall_time": float(timing["stopped_at"] - timing["started_at"]),
            "peak_train_vram": float(timing["peak_train_vram_mib"]),
        },
        "training_loss_best_effort": timing["last_loss"],
        "fixed_schedule_sampler": True,
        "native_random_sampler_used": False,
        "benchmark_training_started": True,
    }
    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    args.result_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
