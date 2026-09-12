#!/usr/bin/env python3
"""Guarded SmolVLA worker for PARC2026 M3 smoke/equal-data/equal-wall runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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

SMOL_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
SMOL_URL = "https://github.com/huggingface/lerobot.git"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _checkout_exact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").is_dir():
        shutil.rmtree(path, ignore_errors=True)
        _run(["git", "init", "-q", str(path)])
        _run(["git", "-C", str(path), "remote", "add", "origin", SMOL_URL])
    _run(["git", "-C", str(path), "fetch", "-q", "--depth", "1", "origin", SMOL_REF])
    _run(["git", "-C", str(path), "checkout", "-q", "--force", "FETCH_HEAD"])
    _run(["git", "-C", str(path), "reset", "--hard", SMOL_REF])
    got = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if got != SMOL_REF:
        raise RuntimeError(f"SmolVLA source pin mismatch: {got}")


def _setup(repo: Path, work_root: Path) -> tuple[Path, Path]:
    if shutil.which("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    source = work_root / "vendor/lerobot-smolvla-m3"
    _checkout_exact(source)
    _run(["uv", "python", "install", "3.12"])
    venv = work_root / "venv-smolvla-m3"
    if not (venv / "bin/python").is_file():
        _run(["uv", "venv", "--python", "3.12", str(venv)])
    python_bin = venv / "bin/python"
    train_bin = venv / "bin/lerobot-train"
    marker = venv / ".m3_smolvla_ref"
    if not train_bin.is_file() or not marker.is_file() or marker.read_text().strip() != SMOL_REF:
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_bin),
                "-e",
                f"{source}[training,smolvla]",
            ]
        )
        marker.write_text(SMOL_REF + "\n", encoding="utf-8")
    version = subprocess.check_output(
        [str(python_bin), "-c", "import sys,lerobot; print(sys.version.split()[0]); print(lerobot.__version__)"],
        text=True,
    ).strip().splitlines()
    if not version or not version[0].startswith("3.12"):
        raise RuntimeError(f"SmolVLA worker must use Python 3.12: {version}")
    trainer = source / "src/lerobot/scripts/lerobot_train.py"
    _run(
        [
            sys.executable,
            str(repo / "tools/benchmark/patch_m3_lerobot_train.py"),
            "--trainer",
            str(trainer),
            "--kind",
            "smolvla_pinned",
        ],
        cwd=repo,
    )
    return source, train_bin


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
        raise RuntimeError("HF_TOKEN is required for SmolVLA M3 worker")
    plan = load_adapter_plan(args.adapter_plan)
    run = select_run_spec(plan, model="smolvla", order=args.order, track=args.track)
    episode_ids = load_manifest_episode_ids(args.manifest)

    run_dir = (
        args.run_root / "smoke" / args.order / "smolvla"
        if args.mode == "smoke"
        else Path(run["result_path"]).parent
    )
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

    try:
        source, train_bin = _setup(repo, args.work_root)
        if train_dir.exists():
            shutil.rmtree(train_dir)
        bs = int(run["micro_batch"])
        ga = int(run["gradient_accumulation"])
        if args.mode == "smoke" or args.track == "equal_data":
            native_steps = int(runtime["optimizer_target"]) * ga
        else:
            native_steps = 1_000_000_000
        episodes_json = json.dumps(episode_ids, separators=(",", ":"))
        cmd = [
            str(train_bin),
            "--policy.path=lerobot/smolvla_base",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            "--policy.input_features=null",
            "--policy.output_features=null",
            "--dataset.repo_id=lerobot/libero_plus",
            f"--dataset.root={args.dataset_root}",
            "--dataset.video_backend=pyav",
            "--dataset.return_uint8=true",
            f"--dataset.episodes={episodes_json}",
            f"--batch_size={bs}",
            f"--accelerator.gradient_accumulation.steps={ga}",
            f"--steps={native_steps}",
            "--num_workers=0",
            "--save_checkpoint=true",
            "--save_freq=999999999",
            "--env_eval_freq=0",
            "--eval_steps=0",
            "--log_freq=1",
            f"--output_dir={train_dir}",
            f"--job_name=m3_{args.mode}_{args.order}_{args.track}_smolvla",
            f"--seed={int(run['seed'])}",
            "--wandb.enable=false",
        ]
        env = os.environ.copy()
        env.update(
            {
                "PARC_M3_RUNTIME_SPEC": str(runtime_path),
                "PYTHONPATH": str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "HF_HOME": str(args.work_root / "cache/huggingface-smolvla-m3"),
                "WANDB_DISABLED": "true",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        write_json(
            run_dir / "status.json",
            {
                "status": "TRAINING",
                "model": "smolvla",
                "mode": args.mode,
                "order": args.order,
                "track": args.track,
                "benchmark_training_started": args.mode == "benchmark",
            },
        )
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                cmd, cwd=str(source), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True
            )
        if proc.returncode != 0:
            raise RuntimeError(f"SmolVLA trainer failed rc={proc.returncode}; log={log_path}")
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") != "PASS" or evidence.get("canonical_sample_order_verified") is not True:
            raise RuntimeError(f"SmolVLA runtime evidence is not PASS: {evidence}")
        checkpoint = train_dir / "checkpoints/last/pretrained_model"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        result = {
            "schema_version": 1,
            "stage": "M3_training_worker",
            "status": "PASS",
            "mode": args.mode,
            "model": "smolvla",
            "order": args.order,
            "track": args.track,
            "source_ref": SMOL_REF,
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
                "model": "smolvla",
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
