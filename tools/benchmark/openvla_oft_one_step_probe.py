#!/usr/bin/env python3
"""Single-process DDP OpenVLA-OFT one-optimizer-step probe on the validated 69c stream.

This is intentionally not the M3 benchmark runner. It exercises the pinned OFT
model/LoRA/L1/proprio/two-camera path against exact-selected LeRobot streaming
samples, performs one effective-batch optimizer update, writes a local probe
result, and exits without saving or promoting a checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--openvla-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--streaming-contract", type=Path, required=True)
    p.add_argument("--micro-batch", type=int, required=True)
    p.add_argument("--grad-accum", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_finetune_module(root: Path):
    path = root / "vla-scripts/finetune.py"
    spec = importlib.util.spec_from_file_location("parc_openvla_oft_finetune", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    if args.micro_batch * args.grad_accum != 32:
        raise ValueError("OpenVLA probe effective batch must equal 32")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    sys.path.insert(0, str(args.openvla_root))
    sys.path.insert(0, str(args.repo_root / "tools/data"))

    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from accelerate import PartialState  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from torch.optim import AdamW  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: PLC0415

    from experiments.robot.openvla_utils import (  # noqa: PLC0415
        check_model_logic_mismatch,
        update_auto_map,
    )
    from prismatic.models.action_heads import L1RegressionActionHead  # noqa: PLC0415
    from prismatic.models.projectors import ProprioProjector  # noqa: PLC0415
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: PLC0415
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: PLC0415
    from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: PLC0415
    from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM  # noqa: PLC0415
    from prismatic.vla.datasets import RLDSBatchTransform  # noqa: PLC0415

    from openvla_lerobot_streaming import load_manifest  # noqa: PLC0415
    from openvla_streaming_training_adapter import make_torch_iterable_dataset  # noqa: PLC0415

    oft = load_finetune_module(args.openvla_root)
    state = PartialState()
    device_id = state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    contract = json.loads(args.streaming_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming contract is not PASS")
    statistics = contract.get("dataset_statistics")
    if not isinstance(statistics, dict) or int(statistics.get("num_transitions", 0)) <= 0:
        raise RuntimeError("69c streaming contract lacks selected-pool statistics")
    manifest = load_manifest(args.manifest)

    vla_path = snapshot_download(repo_id="openvla/openvla-7b")
    if state.is_main_process:
        update_auto_map(vla_path)
        check_model_logic_mismatch(vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    vla.vision_backbone.set_num_images_in_input(2)
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_cfg)
    vla = oft.wrap_ddp(vla, device_id, find_unused=True)

    cfg = oft.FinetuneConfig(
        vla_path=vla_path,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        batch_size=args.micro_batch,
        learning_rate=5e-4,
        grad_accumulation_steps=args.grad_accum,
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        merge_lora_during_training=False,
        image_aug=True,
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
    dataset = make_torch_iterable_dataset(
        args.dataset_root,
        manifest,
        statistics,
        batch_transform,
        max_episodes=8,
    )
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
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device_id)
    vla.train()
    action_head.train()
    proprio_projector.train()

    losses: list[float] = []
    started = time.perf_counter()
    iterator = iter(dataloader)
    for micro_step in range(args.grad_accum):
        batch = next(iterator)
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
        losses.append(float(metrics["loss_value"]))
        if state.is_main_process:
            print(
                f"[openvla-probe] micro_step={micro_step + 1}/{args.grad_accum} loss_value={losses[-1]:.8f}",
                flush=True,
            )
    optimizer.step()
    optimizer.zero_grad()
    dist.barrier()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device_id) / (1024**2))

    result = {
        "status": "PASS",
        "model": "openvla_oft",
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": args.micro_batch * args.grad_accum,
        "optimizer_steps": 1,
        "loss": losses[-1] if losses else None,
        "mean_micro_loss": sum(losses) / len(losses) if losses else None,
        "torch_peak_allocated_mib": peak,
        "train_step_elapsed_sec": elapsed,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "image_aug_configured_for_m3": True,
        "image_aug_applied_in_probe": False,
        "note": "Probe keeps validated streaming sample shapes and model recipe; stochastic image augmentation is added in the M3 runner implementation, not this memory-fit probe.",
        "checkpoint_saved": False,
        "benchmark_training_started": False,
    }
    if state.is_main_process:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
