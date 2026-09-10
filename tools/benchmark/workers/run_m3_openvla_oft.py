#!/usr/bin/env python3
"""Guarded OpenVLA-OFT worker for PARC2026 M3 smoke/benchmark runs."""
from __future__ import annotations

import argparse
import importlib
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

OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"
WANDB_VERSION = "0.16.6"
PROTOBUF_VERSION = "3.20.3"
PANDAS_VERSION = "2.2.3"
PYARROW_VERSION = "17.0.0"
AV_VERSION = "12.3.0"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> None:
    print("+", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _setup_openvla(repo: Path, work_root: Path, run_dir: Path) -> tuple[Path, Path]:
    """Reuse the exact 72c source/environment preparation and compatibility pins."""
    sys.path.insert(0, str(repo / "tools/colab"))
    probe = importlib.import_module("run_m3_openvla_batch_probe")
    source = work_root / "vendor/openvla-oft-m3"
    setup_log = run_dir / "setup.log"
    setup_status = run_dir / "setup_status.json"
    setup_log.write_text("=== M3 OpenVLA-OFT setup ===\n", encoding="utf-8")

    probe.checkout_exact(source, status_path=setup_status, setup_log=setup_log)
    venv = probe.ensure_env(work_root, source, status_path=setup_status, setup_log=setup_log)
    python_bin = venv / "bin/python"
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)

    # These are the compatibility pins that made the 72c streaming probe
    # environment pass. Keep them explicit for the actual M3 worker as well.
    if shutil.which("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"], log=setup_log)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            f"wandb=={WANDB_VERSION}",
            f"protobuf=={PROTOBUF_VERSION}",
            f"pandas=={PANDAS_VERSION}",
            f"pyarrow=={PYARROW_VERSION}",
            f"av=={AV_VERSION}",
        ],
        log=setup_log,
    )
    verify = (
        "import importlib.metadata as im,sys; "
        "import torch,flash_attn,tensorflow_datasets,dlimp,wandb,pandas,pyarrow,av; "
        "print(sys.version.split()[0]); "
        "print('torch=' + torch.__version__); "
        "print('wandb=' + im.version('wandb')); "
        "print('protobuf=' + im.version('protobuf')); "
        "print('pandas=' + im.version('pandas')); "
        "print('pyarrow=' + im.version('pyarrow')); "
        "print('av=' + im.version('av')); "
        "print('m3_openvla_runtime_ok')"
    )
    text = subprocess.check_output([str(python_bin), "-c", verify], text=True)
    with setup_log.open("a", encoding="utf-8") as fh:
        fh.write(text)
    expected = (
        "3.10",
        f"wandb={WANDB_VERSION}",
        f"protobuf={PROTOBUF_VERSION}",
        f"pandas={PANDAS_VERSION}",
        f"pyarrow={PYARROW_VERSION}",
        f"av={AV_VERSION}",
        "m3_openvla_runtime_ok",
    )
    if any(token not in text for token in expected):
        raise RuntimeError(f"OpenVLA M3 environment verification failed: {text}")
    got = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if got != OPENVLA_REF:
        raise RuntimeError(f"OpenVLA M3 source pin mismatch: {got}")
    write_json(
        setup_status,
        {
            "status": "PASS",
            "stage": "M3_openvla_oft_setup",
            "source_ref": got,
            "python": "3.10",
            "wandb": WANDB_VERSION,
            "protobuf": PROTOBUF_VERSION,
            "pandas": PANDAS_VERSION,
            "pyarrow": PYARROW_VERSION,
            "av": AV_VERSION,
            "log": str(setup_log),
        },
    )
    return source, python_bin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--adapter-plan", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--streaming-contract", type=Path, required=True)
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
        raise RuntimeError("HF_TOKEN is required for OpenVLA-OFT M3 worker")
    plan = load_adapter_plan(args.adapter_plan)
    run = select_run_spec(plan, model="openvla_oft", order=args.order, track=args.track)
    if run.get("source_ref") != OPENVLA_REF:
        raise RuntimeError("OpenVLA adapter-plan source pin mismatch")
    load_manifest_episode_ids(args.manifest)

    run_dir = (
        args.run_root / "smoke" / args.order / "openvla_oft"
        if args.mode == "smoke"
        else Path(run["result_path"]).parent
    )
    runtime_path = run_dir / "runtime_spec.json"
    evidence_path = run_dir / "training_evidence.json"
    result_path = run_dir / "training_result.json"
    checkpoint_dir = run_dir / "checkpoint"
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
            "model": "openvla_oft",
            "mode": args.mode,
            "order": args.order,
            "track": args.track,
            "benchmark_training_started": False,
        },
    )

    try:
        source, python_bin = _setup_openvla(repo, args.work_root, run_dir)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        inner = repo / "tools/benchmark/workers/openvla_oft_train_inner.py"
        if not inner.is_file():
            raise FileNotFoundError(inner)
        cmd = [
            str(python_bin),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            str(inner),
            "--repo-root",
            str(repo),
            "--openvla-root",
            str(source),
            "--runtime-spec",
            str(runtime_path),
            "--streaming-contract",
            str(args.streaming_contract),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
        env = os.environ.copy()
        env.update(
            {
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "HF_HOME": str(args.work_root / "cache/huggingface-openvla-m3"),
                "TRANSFORMERS_CACHE": str(args.work_root / "cache/huggingface-openvla-m3/transformers"),
                "WANDB_MODE": "disabled",
                "WANDB_DISABLED": "true",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(repo) + os.pathsep + str(source) + os.pathsep + env.get("PYTHONPATH", ""),
            }
        )
        write_json(
            run_dir / "status.json",
            {
                "status": "TRAINING",
                "model": "openvla_oft",
                "mode": args.mode,
                "order": args.order,
                "track": args.track,
                "benchmark_training_started": args.mode == "benchmark",
            },
        )
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                cmd,
                cwd=str(source),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if proc.returncode != 0:
            tail = ""
            if log_path.is_file():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
                print("=== OPENVLA M3 FAILURE TAIL ===", flush=True)
                print(tail, flush=True)
            raise RuntimeError(f"OpenVLA-OFT trainer failed rc={proc.returncode}; log={log_path}\n{tail}")
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") != "PASS":
            raise RuntimeError(f"OpenVLA training evidence is not PASS: {evidence}")
        if evidence.get("canonical_sample_order_verified") is not True:
            raise RuntimeError("OpenVLA canonical sample order was not verified")
        if evidence.get("image_augmentation_applied") is not True:
            raise RuntimeError("OpenVLA M3 image augmentation was not applied")
        if evidence.get("full_rlds_materialized") is not False:
            raise RuntimeError("OpenVLA M3 unexpectedly materialized full RLDS")
        checkpoint_manifest = checkpoint_dir / "m3_checkpoint_manifest.json"
        if not checkpoint_manifest.is_file():
            raise FileNotFoundError(checkpoint_manifest)

        result = {
            "schema_version": 1,
            "stage": "M3_training_worker",
            "status": "PASS",
            "mode": args.mode,
            "model": "openvla_oft",
            "order": args.order,
            "track": args.track,
            "source_ref": OPENVLA_REF,
            "selected_dataset_variant": runtime["selected_dataset_variant"],
            "selected_episode_ids_sha256": runtime["selected_episode_ids_sha256"],
            "sampling_schedule_sha256": runtime["schedule_sha256"],
            "seed": int(run["seed"]),
            "micro_batch": int(run["micro_batch"]),
            "gradient_accumulation": int(run["gradient_accumulation"]),
            "effective_batch_size": 32,
            "samples_consumed": int(evidence["consumed_samples"]),
            "optimizer_updates": int(evidence["optimizer_updates"]),
            "train_wall_time": float(evidence["train_wall_time_sec"]),
            "peak_train_vram": int(evidence["peak_train_vram_mib"]),
            "final_logged_loss_best_effort": evidence.get("final_loss"),
            "checkpoint": str(checkpoint_dir),
            "training_evidence": str(evidence_path),
            "image_augmentation_applied": True,
            "full_rlds_materialized": False,
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
                "model": "openvla_oft",
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
