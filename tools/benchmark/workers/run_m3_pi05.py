#!/usr/bin/env python3
"""Guarded π0.5 worker for PARC2026 M3 smoke/equal-data/equal-wall runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

from tools.benchmark.workers.m3_worker_common import (
    build_runtime_spec,
    load_adapter_plan,
    load_manifest_episode_ids,
    require_a100,
    select_run_spec,
    write_json,
)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _ensure_uv() -> None:
    if shutil.which("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"])


def _setup_pi05(repo: Path, work_root: Path) -> tuple[Path, Path]:
    _ensure_uv()
    _run(["uv", "python", "install", "3.10"])
    py310 = subprocess.check_output(["uv", "python", "find", "3.10"], text=True).strip()
    source = work_root / "vendor/lerobot-pi05-m3"
    if (source / ".git").is_dir():
        _run(["git", "-C", str(source), "reset", "--hard"])
    setup = repo / "examples/pi05_libero_finetune/scripts/setup_train.sh"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": py310,
            "LEROBOT_ROOT": str(source),
            "DATA_ROOT": str(work_root / "cache/m3-pi05"),
            "INSTALL_FFMPEG": "0",
        }
    )
    _run(["bash", str(setup)], cwd=setup.parent.parent, env=env)
    trainer = source / "src/lerobot/scripts/lerobot_train.py"
    _run(
        [
            sys.executable,
            str(repo / "tools/benchmark/patch_m3_lerobot_train.py"),
            "--trainer",
            str(trainer),
            "--kind",
            "pi05_v044",
        ],
        cwd=repo,
    )
    exe = source / ".venv/bin/lerobot-train"
    if not exe.is_file():
        raise FileNotFoundError(exe)
    return source, exe


def _loss_from_log(path: Path) -> float | None:
    if not path.is_file():
        return None
    values: list[float] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in (
        r"\bloss[=: ]+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)",
        r"\btrain_loss[=: ]+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)",
    ):
        values.extend(float(x) for x in re.findall(pattern, text))
    return values[-1] if values else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--adapter-plan", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--track", choices=("equal_data", "equal_wall"), required=True)
    p.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    gpu_name, gpu_vram = require_a100()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for π0.5 M3 worker")
    plan = load_adapter_plan(args.adapter_plan)
    run = select_run_spec(plan, model="pi05", order=args.order, track=args.track)
    episode_ids = load_manifest_episode_ids(args.manifest)

    if args.mode == "smoke":
        run_dir = args.run_root / "smoke" / args.order / "pi05"
    else:
        run_dir = Path(run["result_path"]).parent
    train_dir = run_dir / "train"
    runtime_path = run_dir / "runtime_spec.json"
    evidence_path = run_dir / "training_evidence.json"
    result_path = run_dir / "training_result.json"
    log_path = run_dir / "train.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime = build_runtime_spec(
        plan=plan,
        run=run,
        mode=args.mode,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        evidence_path=evidence_path,
        result_path=result_path,
    )
    write_json(runtime_path, runtime)
    write_json(
        run_dir / "status.json",
        {
            "status": "SETUP",
            "model": "pi05",
            "mode": args.mode,
            "order": args.order,
            "track": args.track,
            "benchmark_training_started": False,
        },
    )

    try:
        _, exe = _setup_pi05(repo, args.work_root)
        if train_dir.exists():
            shutil.rmtree(train_dir)
        ga = int(run["gradient_accumulation"])
        bs = int(run["micro_batch"])
        if args.mode == "smoke" or args.track == "equal_data":
            max_optimizer_steps = int(runtime["optimizer_target"])
        else:
            max_optimizer_steps = 1_000_000
        episodes_json = json.dumps(episode_ids, separators=(",", ":"))
        cmd = [
            str(exe),
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
            f"--dataset.episodes={episodes_json}",
            f"--batch_size={bs}",
            f"--steps={max_optimizer_steps}",
            "--num_workers=0",
            "--save_freq=999999999",
            "--eval_freq=999999999",
            "--log_freq=1",
            f"--output_dir={train_dir}",
            f"--job_name=m3_{args.mode}_{args.order}_{args.track}_pi05",
            f"--seed={int(run['seed'])}",
            "--wandb.enable=false",
        ]
        env = os.environ.copy()
        env.update(
            {
                "PARC_M3_RUNTIME_SPEC": str(runtime_path),
                "LEROBOT_GRAD_ACCUM": str(ga),
                "PYTHONPATH": str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "HF_HOME": str(args.work_root / "cache/huggingface-m3"),
                "WANDB_DISABLED": "true",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        write_json(
            run_dir / "status.json",
            {
                "status": "TRAINING",
                "model": "pi05",
                "mode": args.mode,
                "order": args.order,
                "track": args.track,
                "benchmark_training_started": args.mode == "benchmark",
            },
        )
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"π0.5 trainer failed rc={proc.returncode}; log={log_path}")
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") != "PASS" or evidence.get("canonical_sample_order_verified") is not True:
            raise RuntimeError(f"π0.5 runtime evidence is not PASS: {evidence}")
        checkpoint = train_dir / "checkpoints/last/pretrained_model"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        result = {
            "schema_version": 1,
            "stage": "M3_training_worker",
            "status": "PASS",
            "mode": args.mode,
            "model": "pi05",
            "order": args.order,
            "track": args.track,
            "source_ref": run["source_ref"],
            "selected_dataset_variant": runtime["selected_dataset_variant"],
            "selected_episode_ids_sha256": runtime["selected_episode_ids_sha256"],
            "sampling_schedule_sha256": runtime["schedule_sha256"],
            "seed": int(run["seed"]),
            "micro_batch": bs,
            "gradient_accumulation": ga,
            "effective_batch_size": 32,
            "samples_consumed": int(evidence["consumed_samples"]),
            "optimizer_updates": int(evidence["optimizer_updates"]),
            "train_wall_time": float(evidence["train_wall_time_sec"]),
            "peak_train_vram": int(evidence.get("peak_train_vram_mib", 0)),
            "final_logged_loss_best_effort": _loss_from_log(log_path),
            "checkpoint": str(checkpoint),
            "training_evidence": str(evidence_path),
            "gpu_name": gpu_name,
            "gpu_vram_mib": gpu_vram,
            "simulator_evaluation_pending": True,
        }
        write_json(result_path, result)
        write_json(run_dir / "status.json", {**result, "result_path": str(result_path)})
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        write_json(
            run_dir / "status.json",
            {
                "status": "FAILED",
                "model": "pi05",
                "mode": args.mode,
                "order": args.order,
                "track": args.track,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "log_path": str(log_path),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
