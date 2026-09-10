#!/usr/bin/env python3
"""Inner single-GPU OpenVLA-OFT trainer for guarded PARC2026 M3 runs.

Run only via `torchrun --nproc-per-node=1` from `run_m3_openvla_oft.py`.
The outer worker establishes the exact pinned runtime and execution guard.  This
inner loop consumes the framework-independent canonical sample stream directly,
uses the validated 69c normalization/streaming contract, and saves an unmerged
LoRA checkpoint after the measured training loop stops.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--openvla-root", type=Path, required=True)
    p.add_argument("--runtime-spec", type=Path, required=True)
    p.add_argument("--streaming-contract", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    return p.parse_args()


def load_finetune_module(root: Path):
    path = root / "vla-scripts/finetune.py"
    spec = importlib.util.spec_from_file_location("parc_m3_openvla_oft_finetune", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_runtime(runtime: dict[str, Any]) -> None:
    if runtime.get("model") != "openvla_oft":
        raise ValueError("OpenVLA inner worker received the wrong model runtime")
    if runtime.get("mode") not in {"smoke", "benchmark"}:
        raise ValueError("invalid M3 runtime mode")
    if runtime.get("track") not in {"equal_data", "equal_wall"}:
        raise ValueError("invalid M3 runtime track")
    bs = int(runtime.get("micro_batch", 0))
    ga = int(runtime.get("gradient_accumulation", 0))
    if bs <= 0 or ga <= 0 or bs * ga != 32:
        raise ValueError(f"OpenVLA effective batch drift: bs={bs} ga={ga}")
    if runtime.get("selected_dataset_variant") != "V2_SQRT_BALANCED_RAW":
        raise ValueError("OpenVLA runtime dataset variant mismatch")
    if runtime.get("selected_episode_ids_sha256") != (
        "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
    ):
        raise ValueError("OpenVLA runtime D10 hash mismatch")
    if runtime.get("source_ref") != "e4287e94541f459edc4feabc4e181f537cd569a8":
        raise ValueError("OpenVLA runtime source pin mismatch")
    if runtime.get("execution_guard_verified") is not True:
        raise RuntimeError("OpenVLA runtime execution guard was not verified")
    if runtime["mode"] == "benchmark" and runtime["track"] == "equal_data":
        if int(runtime["sample_target"]) != 4800 or int(runtime["optimizer_target"]) != 150:
            raise ValueError("benchmark equal-data target must remain exactly 4800/150")
    if runtime["mode"] == "benchmark" and runtime["track"] == "equal_wall":
        if float(runtime.get("train_loop_sec") or 0.0) != 1800.0:
            raise ValueError("benchmark equal-wall target must remain exactly 1800 sec")


def _reference_iterator(runtime: dict[str, Any]) -> Iterator[dict[str, int]]:
    from tools.benchmark.m3_sampling_schedule import iter_references, load_d10_episode_lengths
    from tools.benchmark.m3_training_adapters import (
        build_equal_wall_stream_descriptor,
        load_equal_data_schedule,
    )

    if runtime["track"] == "equal_data":
        schedule = load_equal_data_schedule(Path(runtime["equal_data_schedule"]))
        if schedule["schedule_sha256"] != runtime["schedule_sha256"]:
            raise ValueError("OpenVLA equal-data schedule SHA mismatch")
        if int(schedule["seed"]) != int(runtime["seed"]):
            raise ValueError("OpenVLA equal-data schedule seed mismatch")
        target = int(runtime["sample_target"])
        if not 0 < target <= len(schedule["references"]):
            raise ValueError("OpenVLA equal-data sample target exceeds schedule")
        yield from schedule["references"][:target]
        return

    descriptor = build_equal_wall_stream_descriptor(seed=int(runtime["seed"]))
    if descriptor["schedule_sha256"] != runtime["schedule_sha256"]:
        raise ValueError("OpenVLA equal-wall stream descriptor SHA mismatch")
    episode_ids, lengths = load_d10_episode_lengths(
        manifest_path=Path(runtime["manifest_path"]),
        dataset_root=Path(runtime["dataset_root"]),
    )
    yield from iter_references(
        episode_ids=episode_ids,
        episode_lengths=lengths,
        seed=int(runtime["seed"]),
        start_index=0,
    )


def _take_refs(iterator: Iterator[dict[str, int]], count: int) -> list[dict[str, int]]:
    refs: list[dict[str, int]] = []
    for _ in range(int(count)):
        try:
            refs.append(next(iterator))
        except StopIteration as exc:
            raise RuntimeError("canonical OpenVLA schedule ended before a complete micro-batch") from exc
    return refs


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    openvla_root = args.openvla_root.resolve()
    runtime = json.loads(args.runtime_spec.read_text(encoding="utf-8"))
    _validate_runtime(runtime)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    sys.path.insert(0, str(openvla_root))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tools/data"))

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from accelerate import PartialState  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from torch.optim import AdamW  # noqa: PLC0415
    from torch.optim.lr_scheduler import MultiStepLR  # noqa: PLC0415
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
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics  # noqa: PLC0415

    from tools.data.openvla_lerobot_streaming import load_manifest  # noqa: PLC0415
    from tools.data.openvla_scheduled_training_source import (  # noqa: PLC0415
        ScheduledOpenVLASource,
        transform_scheduled_batch,
    )

    oft = load_finetune_module(openvla_root)
    state = PartialState()
    device_id = state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    contract = json.loads(args.streaming_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming contract is not PASS")
    if contract.get("source_episode_ids_sha256") != runtime["selected_episode_ids_sha256"]:
        raise RuntimeError("69c contract D10 hash mismatch")
    if contract.get("openvla_oft_revision") != runtime["source_ref"]:
        raise RuntimeError("69c OpenVLA source revision mismatch")
    if contract.get("storage_policy", {}).get("full_rlds_materialized") is not False:
        raise RuntimeError("M3 OpenVLA worker refuses a full-RLDS materialization contract")
    statistics = contract.get("dataset_statistics")
    if not isinstance(statistics, dict) or int(statistics.get("num_transitions", 0)) != 1620614:
        raise RuntimeError("69c contract lacks exact selected-pool statistics")
    manifest = load_manifest(Path(runtime["manifest_path"]))

    seed = int(runtime["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        batch_size=int(runtime["micro_batch"]),
        learning_rate=5e-4,
        grad_accumulation_steps=int(runtime["gradient_accumulation"]),
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        merge_lora_during_training=False,
        image_aug=True,
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
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    source = ScheduledOpenVLASource(
        Path(runtime["dataset_root"]),
        manifest,
        statistics,
    )

    trainable = [p for p in vla.parameters() if p.requires_grad]
    trainable += [p for p in action_head.parameters() if p.requires_grad]
    trainable += [p for p in proprio_projector.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100_000], gamma=0.1)
    optimizer.zero_grad(set_to_none=True)
    vla.train()
    action_head.train()
    proprio_projector.train()

    refs_iter = _reference_iterator(runtime)
    bs = int(runtime["micro_batch"])
    ga = int(runtime["gradient_accumulation"])
    samples_consumed = 0
    optimizer_updates = 0
    losses: list[float] = []
    first_ref: dict[str, int] | None = None
    last_ref: dict[str, int] | None = None
    started: float | None = None
    stopped: float | None = None

    while True:
        for micro_step in range(ga):
            if runtime["track"] == "equal_data":
                target = int(runtime["sample_target"])
                if samples_consumed + bs > target:
                    raise RuntimeError(
                        f"OpenVLA equal-data overshoot blocked: {samples_consumed}+{bs}>{target}"
                    )
            if started is None:
                torch.cuda.reset_peak_memory_stats(device_id)
                started = time.perf_counter()
            refs = _take_refs(refs_iter, bs)
            transformed, identities = transform_scheduled_batch(
                source,
                refs,
                batch_transform,
                run_seed=seed,
                image_aug=True,
            )
            expected = [(int(r["episode_id"]), int(r["timestep"])) for r in refs]
            if identities != expected:
                raise RuntimeError(
                    f"OpenVLA canonical batch identity mismatch: expected={expected} actual={identities}"
                )
            batch = collator(transformed)
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
            (loss / ga).backward()
            loss_value = float(metrics["loss_value"])
            losses.append(loss_value)
            samples_consumed += len(refs)
            if first_ref is None:
                first_ref = dict(refs[0])
            last_ref = dict(refs[-1])
            if state.is_main_process:
                print(
                    f"[openvla-m3] update={optimizer_updates + 1} micro={micro_step + 1}/{ga} "
                    f"samples={samples_consumed} loss={loss_value:.8f}",
                    flush=True,
                )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        optimizer_updates += 1
        elapsed = time.perf_counter() - started

        stop = False
        if runtime["track"] == "equal_data":
            sample_target = int(runtime["sample_target"])
            optimizer_target = int(runtime["optimizer_target"])
            if samples_consumed == sample_target:
                if optimizer_updates != optimizer_target:
                    raise RuntimeError(
                        f"OpenVLA optimizer target mismatch at sample boundary: "
                        f"{optimizer_updates}/{optimizer_target}"
                    )
                stop = True
            elif optimizer_updates >= optimizer_target:
                raise RuntimeError("OpenVLA optimizer target reached before equal-data sample target")
        else:
            wall_target = float(runtime["train_loop_sec"])
            if elapsed >= wall_target:
                stop = True

        if stop:
            stopped = time.perf_counter()
            break

    if started is None or stopped is None:
        raise RuntimeError("OpenVLA training loop did not establish timer boundaries")
    train_wall = stopped - started
    if runtime["track"] == "equal_wall" and train_wall < float(runtime["train_loop_sec"]):
        raise RuntimeError("OpenVLA equal-wall loop stopped before target")
    peak_mib = int(torch.cuda.max_memory_allocated(device_id) / (1024**2))
    dist.barrier()

    checkpoint_dir = args.checkpoint_dir.resolve()
    if state.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = checkpoint_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)
        torch.save(
            proprio_projector.state_dict(),
            checkpoint_dir / "proprio_projector--latest_checkpoint.pt",
        )
        torch.save(
            action_head.state_dict(),
            checkpoint_dir / "action_head--latest_checkpoint.pt",
        )
        save_dataset_statistics(statistics, checkpoint_dir)
        checkpoint_manifest = {
            "schema_version": 1,
            "stage": "M3_openvla_oft_checkpoint",
            "status": "PASS",
            "source_ref": runtime["source_ref"],
            "base_checkpoint": "openvla/openvla-7b",
            "lora_rank": 32,
            "use_l1_regression": True,
            "use_proprio": True,
            "num_images_in_input": 2,
            "image_aug": True,
            "merge_lora_during_training": False,
            "optimizer_updates": optimizer_updates,
            "samples_consumed": samples_consumed,
            "sampling_schedule_sha256": runtime["schedule_sha256"],
        }
        (checkpoint_dir / "m3_checkpoint_manifest.json").write_text(
            json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
        )
    dist.barrier()

    evidence = {
        "schema_version": 1,
        "stage": "M3_openvla_oft_runtime_evidence",
        "status": "PASS",
        "mode": runtime["mode"],
        "model": "openvla_oft",
        "order": runtime["order"],
        "track": runtime["track"],
        "seed": seed,
        "source_ref": runtime["source_ref"],
        "selected_dataset_variant": runtime["selected_dataset_variant"],
        "selected_episode_ids_sha256": runtime["selected_episode_ids_sha256"],
        "sampling_schedule_sha256": runtime["schedule_sha256"],
        "micro_batch": bs,
        "gradient_accumulation": ga,
        "effective_batch_size": bs * ga,
        "consumed_samples": samples_consumed,
        "optimizer_updates": optimizer_updates,
        "train_wall_time_sec": train_wall,
        "peak_train_vram_mib": peak_mib,
        "canonical_sample_order_verified": True,
        "sample_mismatch_count": 0,
        "image_augmentation_applied": True,
        "image_augmentation_recipe": "openvla_oft_rlds_image_aug_v1",
        "first_reference": first_ref,
        "last_reference": last_ref,
        "final_loss": losses[-1] if losses else None,
        "mean_last20_loss": sum(losses[-20:]) / len(losses[-20:]) if losses else None,
        "checkpoint": str(checkpoint_dir),
        "full_rlds_materialized": False,
    }
    if state.is_main_process:
        evidence_path = Path(runtime["evidence_path"])
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2), flush=True)
        print("=== M3 OPENVLA-OFT TRAINING WORKER: PASS ===", flush=True)
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
