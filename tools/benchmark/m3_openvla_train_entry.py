#!/usr/bin/env python3
"""Execute OpenVLA-OFT M3 training on the validated 69c LeRobot stream.

The model recipe matches the pinned OFT LIBERO recipe while data access remains
the exact selected LeRobot pool. Equal-data consumes exactly 4,800 canonical
references. Equal-wall continues the same counter stream and stops only at a
gradient-accumulation optimizer boundary at/after 1,800 train-loop seconds.
Checkpoint I/O occurs after the timer is frozen.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
SOURCE_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"
SEEDS = (20260906, 20260907)
EFFECTIVE_BATCH = 32
EQUAL_DATA_BUDGET = 4800
EQUAL_WALL_SEC = 1800.0
SMOKE_EQUAL_DATA_SAMPLES = 64
SMOKE_WALL_SEC = 5.0
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--streaming-contract", type=Path, required=True)
    p.add_argument("--micro-batch", type=int, required=True)
    p.add_argument("--grad-accum", type=int, required=True)
    p.add_argument("--track", choices=("equal_data", "equal_wall"), required=True)
    p.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result-out", type=Path, required=True)
    p.add_argument("--wall-target-sec", type=float)
    p.add_argument("--stream-chunk-size", type=int, default=256)
    return p.parse_args()


def _load_finetune_module(root: Path):
    path = root / "vla-scripts/finetune.py"
    spec = importlib.util.spec_from_file_location("parc_m3_openvla_oft_finetune", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned finetune module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_source(root: Path) -> None:
    root = root.resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"OpenVLA-OFT checkout missing .git: {root}")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SOURCE_REF:
        raise RuntimeError(f"OpenVLA-OFT source HEAD mismatch: {head} != {SOURCE_REF}")


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


def _checkpoint_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    """Expose the one global D10 normalization under each evaluator suite key."""
    required = ("action", "proprio")
    for key in required:
        if not isinstance(statistics.get(key), dict):
            raise ValueError(f"69c statistics missing {key}")
    return {suite: copy.deepcopy(statistics) for suite in SUITES}


def _sync_pinned_model_logic(
    source_root: Path,
    vla_path: str,
    update_auto_map,
    check_model_logic_mismatch,
) -> None:
    """Sync the pinned OFT HF model implementation into the downloaded checkpoint.

    Upstream ``check_model_logic_mismatch`` discovers its source files through
    ``./prismatic``.  Notebook 72c invokes its probe with ``cwd=source_root``;
    Notebook 73 historically inherited the py_AI repository cwd instead, so the
    sync silently found no model files and Transformers loaded stale checkpoint
    logic without the multi-image vision-backbone methods.  Keep the cwd switch
    local to the upstream helper and restore it before continuing.
    """
    source_root = Path(source_root).resolve()
    source_hf = source_root / "prismatic/extern/hf"
    required = ("modeling_prismatic.py", "configuration_prismatic.py")
    missing_source = [name for name in required if not (source_hf / name).is_file()]
    if missing_source:
        raise FileNotFoundError(
            f"pinned OpenVLA model-logic files missing under {source_hf}: {missing_source}"
        )

    previous_cwd = Path.cwd()
    try:
        os.chdir(source_root)
        update_auto_map(vla_path)
        check_model_logic_mismatch(vla_path)
    finally:
        os.chdir(previous_cwd)

    checkpoint_root = Path(vla_path)
    missing_synced = [name for name in required if not (checkpoint_root / name).is_file()]
    if missing_synced:
        raise RuntimeError(
            f"OpenVLA checkpoint model-logic sync incomplete at {checkpoint_root}: {missing_synced}"
        )


def main() -> int:
    args = parse_args()
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 training requires explicit PARC_M3_EXECUTE=1")
    if args.micro_batch <= 0 or args.grad_accum <= 0 or args.micro_batch * args.grad_accum != EFFECTIVE_BATCH:
        raise ValueError("M3 effective batch must equal 32")
    if args.stream_chunk_size < args.micro_batch:
        raise ValueError("OpenVLA stream chunk must be at least one micro-batch")
    _verify_source(args.source_root)
    target_wall = _wall_target(args)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(args.repo_root / "tools/data"))

    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from accelerate import PartialState  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from torch.optim import AdamW  # noqa: PLC0415
    from torch.optim.lr_scheduler import MultiStepLR  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: PLC0415

    from experiments.robot.openvla_utils import (  # noqa: PLC0415
        check_model_logic_mismatch,
        update_auto_map,
    )
    from prismatic.models.action_heads import L1RegressionActionHead  # noqa: PLC0415
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: PLC0415
    from prismatic.models.projectors import ProprioProjector  # noqa: PLC0415
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: PLC0415
    from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: PLC0415
    from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM  # noqa: PLC0415
    from prismatic.vla.datasets import RLDSBatchTransform  # noqa: PLC0415

    from tools.benchmark.m3_equal_wall_schedule import (  # noqa: PLC0415
        stream_descriptor,
        verify_materialized_prefix,
    )
    from tools.benchmark.m3_scheduled_data import load_schedule  # noqa: PLC0415
    from openvla_lerobot_streaming import load_episode_metadata, load_manifest  # noqa: PLC0415
    from openvla_scheduled_stream import (  # noqa: PLC0415
        make_counter_stream_iterable_dataset,
        make_finite_scheduled_iterable_dataset,
    )

    oft = _load_finetune_module(args.source_root)
    state = PartialState()
    device_id = state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    contract = json.loads(args.streaming_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming contract is not PASS lerobot_streaming")
    if contract.get("source_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("69c streaming contract D10 mismatch")
    statistics = contract.get("dataset_statistics")
    if not isinstance(statistics, dict) or int(statistics.get("num_transitions", 0)) <= 0:
        raise RuntimeError("69c streaming contract lacks selected-pool statistics")

    manifest = load_manifest(args.manifest)
    if manifest.get("episode_ids_sha256") != D10_HASH:
        raise RuntimeError("D10 manifest mismatch")
    schedule = load_schedule(args.schedule, require_equal_data=True)
    seed = int(schedule.get("seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"unexpected M3 seed: {seed}")
    episode_ids = [int(x) for x in manifest["episode_ids"]]
    metadata = load_episode_metadata(args.dataset_root, episode_ids)
    episode_lengths = [int(metadata.loc[episode_id]["length"]) for episode_id in episode_ids]
    verify_materialized_prefix(
        schedule, episode_ids=episode_ids, episode_lengths=episode_lengths
    )
    wall_descriptor = stream_descriptor(seed=seed) if args.track == "equal_wall" else None

    vla_path = snapshot_download(repo_id="openvla/openvla-7b")
    if state.is_main_process:
        _sync_pinned_model_logic(
            args.source_root,
            vla_path,
            update_auto_map,
            check_model_logic_mismatch,
        )
    dist.barrier()

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    required_vision_methods = (
        "set_num_images_in_input",
        "get_num_images_in_input",
        "get_num_patches",
    )
    missing_vision_methods = [
        name for name in required_vision_methods if not hasattr(vla.vision_backbone, name)
    ]
    if missing_vision_methods:
        raise RuntimeError(
            "OpenVLA loaded stale/incompatible vision-backbone logic after sync; "
            f"missing methods: {missing_vision_methods}"
        )
    vla.vision_backbone.set_num_images_in_input(2)
    if int(vla.vision_backbone.get_num_images_in_input()) != 2:
        raise RuntimeError("OpenVLA vision backbone did not accept two-image M3 configuration")
    vla = get_peft_model(
        vla,
        LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        ),
    )
    vla = oft.wrap_ddp(vla, device_id, find_unused=True)

    cfg = oft.FinetuneConfig(
        vla_path=vla_path,
        run_root_dir=args.output_dir.parent,
        run_id_override=args.output_dir.name,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        batch_size=args.micro_batch,
        learning_rate=5e-4,
        num_steps_before_decay=100000,
        grad_accumulation_steps=args.grad_accum,
        use_val_set=False,
        image_aug=True,
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        merge_lora_during_training=False,
        save_latest_checkpoint_only=True,
    )
    proprio_projector = oft.init_module(
        ProprioProjector,
        "proprio_projector",
        cfg,
        device_id,
        {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
    )
    action_head = oft.init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )
    num_patches = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
        + 1
    )

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    if args.track == "equal_data":
        references = list(schedule["references"])
        if args.mode == "smoke":
            references = references[:SMOKE_EQUAL_DATA_SAMPLES]
        if len(references) % EFFECTIVE_BATCH != 0:
            raise ValueError("OpenVLA equal-data references must divide effective batch 32")
        dataset = make_finite_scheduled_iterable_dataset(
            root=args.dataset_root,
            references=references,
            statistics=statistics,
            batch_transform=batch_transform,
            image_aug=True,
            schedule_seed=seed,
            chunk_size=args.stream_chunk_size,
        )
        target_samples = len(references)
    else:
        dataset = make_counter_stream_iterable_dataset(
            root=args.dataset_root,
            episode_ids=episode_ids,
            episode_lengths=episode_lengths,
            statistics=statistics,
            batch_transform=batch_transform,
            image_aug=True,
            schedule_seed=seed,
            chunk_size=args.stream_chunk_size,
        )
        target_samples = None
    # Upstream checkpoint serialization expects a mapping of normalization keys
    # to per-dataset statistics. Training still uses the one global 69c stats.
    dataset.dataset_statistics = _checkpoint_statistics(statistics)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    trainable = [p for p in vla.parameters() if p.requires_grad]
    trainable += [p for p in action_head.parameters() if p.requires_grad]
    trainable += [p for p in proprio_projector.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100000], gamma=0.1)
    optimizer.zero_grad()
    vla.train()
    action_head.train()
    proprio_projector.train()

    micro_steps = 0
    optimizer_updates = 0
    samples_consumed = 0
    losses: list[float] = []
    iterator = iter(dataloader)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device_id)
    started = time.perf_counter()  # immediately before first train-loop batch fetch
    stopped: float | None = None
    peak_train_vram = 0.0

    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            if args.track == "equal_wall":
                raise RuntimeError("equal-wall OpenVLA stream ended unexpectedly")
            break
        batch_size = int(batch["actions"].shape[0])
        if batch_size != args.micro_batch:
            raise RuntimeError(
                f"partial OpenVLA micro-batch would violate exact accounting: {batch_size} != {args.micro_batch}"
            )
        loss, metrics = oft.run_forward_pass(
            vla=vla,
            action_head=action_head,
            noisy_action_projector=None,
            proprio_projector=proprio_projector,
            batch=batch,
            action_tokenizer=action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_proprio=True,
            use_film=False,
            num_patches=num_patches,
        )
        (loss / args.grad_accum).backward()
        micro_steps += 1
        samples_consumed += batch_size
        losses.append(float(metrics["loss_value"]))

        if micro_steps % args.grad_accum != 0:
            continue
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        optimizer_updates += 1

        should_stop = False
        if args.track == "equal_data":
            if target_samples is None:
                raise RuntimeError("equal-data target missing")
            if samples_consumed > target_samples:
                raise RuntimeError("OpenVLA equal-data sample overshoot")
            should_stop = samples_consumed == target_samples
        else:
            should_stop = (time.perf_counter() - started) >= target_wall
        if should_stop:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_train_vram = float(torch.cuda.max_memory_allocated(device_id) / (1024**2))
            stopped = time.perf_counter()
            break

    if stopped is None:
        if args.track == "equal_data" and target_samples == samples_consumed:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_train_vram = float(torch.cuda.max_memory_allocated(device_id) / (1024**2))
            stopped = time.perf_counter()
        else:
            raise RuntimeError("OpenVLA M3 loop did not stop at a valid boundary")
    if samples_consumed % EFFECTIVE_BATCH != 0:
        raise RuntimeError("OpenVLA M3 loop stopped outside an effective-batch boundary")
    if args.track == "equal_data":
        expected = EQUAL_DATA_BUDGET if args.mode == "benchmark" else SMOKE_EQUAL_DATA_SAMPLES
        if samples_consumed != expected:
            raise RuntimeError(f"OpenVLA equal-data sample mismatch: {samples_consumed} != {expected}")
    else:
        elapsed = stopped - started
        if elapsed + 1e-9 < target_wall:
            raise RuntimeError(f"OpenVLA equal-wall stopped early: {elapsed:.6f} < {target_wall:.6f}")

    # The metric timer is frozen above. Serialization/merging is excluded.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oft.save_training_checkpoint(
        cfg,
        args.output_dir,
        optimizer_updates,
        vla,
        processor,
        proprio_projector,
        None,
        action_head,
        dataset,
        state,
    )
    dist.barrier()

    checkpoint_ref = str(args.output_dir.resolve())
    result = {
        "schema_version": 2,
        "stage": "M3_model_training",
        "status": "PASS",
        "model": "openvla_oft",
        "source_ref": SOURCE_REF,
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
        "samples_consumed": samples_consumed,
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": EFFECTIVE_BATCH,
        "micro_steps": micro_steps,
        "optimizer_updates": optimizer_updates,
        "wall_target_sec": target_wall if args.track == "equal_wall" else None,
        "timer_boundary": "immediately_before_first_train_batch_fetch_to_safe_optimizer_boundary",
        "checkpoint_save_excluded_from_train_wall_time": True,
        "metrics": {
            "train_wall_time": float(stopped - started),
            "peak_train_vram": peak_train_vram,
        },
        "training_loss_best_effort": losses[-1] if losses else None,
        "mean_micro_loss_best_effort": sum(losses) / len(losses) if losses else None,
        "dataset_bridge": "lerobot_streaming",
        "full_rlds_materialization": False,
        "modified_libero_rlds_used": False,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "image_aug_applied": True,
        "checkpoint_statistics_scope": "global_D10_stats_aliased_to_each_LIBERO_suite_for_inference_unnormalization",
        "benchmark_training_started": args.mode == "benchmark",
    }
    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    args.result_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if state.is_main_process:
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())