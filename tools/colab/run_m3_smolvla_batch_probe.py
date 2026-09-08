#!/usr/bin/env python3
"""Probe SmolVLA micro-batch size on A100 with one effective-batch-32 optimizer step."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SMOL_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
SMOL_URL = "https://github.com/huggingface/lerobot.git"


def checkout_exact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").is_dir():
        shutil.rmtree(path, ignore_errors=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", SMOL_URL], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "-q", "--depth", "1", "origin", SMOL_REF], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", "--force", "FETCH_HEAD"], check=True)
    got = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if got != SMOL_REF:
        raise RuntimeError(f"SmolVLA source pin mismatch: {got}")


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", str(root / "py_AI_m3_probe")))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    sys.path.insert(0, str(repo / "tools/benchmark"))
    from m3_batch_probe_common import (  # noqa: PLC0415
        TARGET_EFFECTIVE_BATCH,
        candidate_result,
        gpu_info,
        grad_accum_for,
        load_gate_paths,
        run_logged,
        write_result,
    )

    gpu_name, gpu_vram = gpu_info()
    _, manifest_path, _ = load_gate_paths(drive)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes_json = json.dumps([int(x) for x in manifest["episode_ids"]], separators=(",", ":"))
    dataset_root = Path(
        os.environ.get(
            "PARC_DRIVE_DATASET",
            str(drive / "datasets/lerobot_libero_plus_v3_train"),
        )
    )
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(dataset_root / "meta/info.json")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for SmolVLA probe")
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)

    source = root / "vendor/lerobot-smolvla-m3"
    checkout_exact(source)
    subprocess.run(["uv", "python", "install", "3.12"], check=True)
    venv = root / "venv-smolvla-m3"
    if not (venv / "bin/python").is_file():
        subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True)
    python_bin = venv / "bin/python"
    train_bin = venv / "bin/lerobot-train"
    marker = venv / ".m3_smolvla_ref"
    if not train_bin.is_file() or not marker.is_file() or marker.read_text().strip() != SMOL_REF:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_bin),
                "-e",
                f"{source}[training,smolvla]",
            ],
            check=True,
        )
        marker.write_text(SMOL_REF + "\n", encoding="utf-8")
    version = subprocess.check_output(
        [str(python_bin), "-c", "import sys,lerobot; print(sys.version.split()[0]); print(lerobot.__version__)"],
        text=True,
    ).strip().splitlines()
    if not version or not version[0].startswith("3.12"):
        raise RuntimeError(f"SmolVLA probe must use Python 3.12: {version}")

    cache = root / "cache/m3-smolvla-batch-probe"
    log_root = drive / "model-benchmark-v1/batch-probes/logs/smolvla"
    out_root = cache / "outputs"
    result_path = drive / "model-benchmark-v1/batch-probes/smolvla.json"
    candidates = [32, 16, 8, 4, 2, 1]
    trials = []
    selected = None
    for micro_batch in candidates:
        ga = grad_accum_for(micro_batch)
        run_name = f"m3_smolvla_probe_bs{micro_batch}_ga{ga}"
        out_dir = out_root / run_name
        shutil.rmtree(out_dir, ignore_errors=True)
        cmd = [
            str(train_bin),
            "--policy.path=lerobot/smolvla_base",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            # The base SmolVLA checkpoint was trained with an ALOHA feature schema
            # (camera1/2/3, state6, action6). For LIBERO fine-tuning, keep the
            # pretrained weights but infer the concrete input/output feature schema
            # from the selected LIBERO dataset (front/wrist, state8, action7).
            "--policy.input_features=null",
            "--policy.output_features=null",
            "--dataset.repo_id=lerobot/libero_plus",
            f"--dataset.root={dataset_root}",
            "--dataset.video_backend=pyav",
            "--dataset.return_uint8=true",
            f"--dataset.episodes={episodes_json}",
            f"--batch_size={micro_batch}",
            f"--accelerator.gradient_accumulation.steps={ga}",
            f"--steps={ga}",
            "--num_workers=0",
            "--save_checkpoint=false",
            "--env_eval_freq=0",
            "--eval_steps=0",
            "--log_freq=1",
            f"--output_dir={out_dir}",
            f"--job_name={run_name}",
            "--seed=1000",
            "--wandb.enable=false",
        ]
        env = {
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "HF_HOME": str(root / "cache/huggingface-smolvla-m3"),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        log_path = log_root / f"{run_name}.log"
        rc, text, elapsed, peak = run_logged(
            cmd,
            cwd=source,
            env=env,
            log_path=log_path,
        )
        trial = candidate_result(
            micro_batch=micro_batch,
            rc=rc,
            text=text,
            elapsed_sec=elapsed,
            peak_vram=peak,
        )
        trial["micro_steps"] = ga
        trial["optimizer_steps"] = 1 if rc == 0 else 0
        trial["log_path"] = str(log_path)
        trials.append(trial)
        shutil.rmtree(out_dir, ignore_errors=True)
        if rc == 0:
            selected = trial
            break
        if trial["status"] != "OOM":
            write_result(
                result_path,
                {
                    "status": "FAILED",
                    "model": "smolvla",
                    "source_ref": SMOL_REF,
                    "python": version[0] if version else None,
                    "gpu_name": gpu_name,
                    "gpu_vram_mib": gpu_vram,
                    "failing_trial": trial,
                    "trials": trials,
                    "error_tail": text[-8000:],
                },
            )
            raise RuntimeError(f"SmolVLA probe failed for non-OOM reason: {trial}; log={log_path}")

    if selected is None:
        write_result(
            result_path,
            {
                "status": "FAILED_NO_FIT",
                "model": "smolvla",
                "source_ref": SMOL_REF,
                "python": version[0] if version else None,
                "gpu_name": gpu_name,
                "gpu_vram_mib": gpu_vram,
                "trials": trials,
            },
        )
        raise RuntimeError("SmolVLA did not fit even at micro_batch=1")

    write_result(
        result_path,
        {
            "status": "PASS",
            "model": "smolvla",
            "source_ref": SMOL_REF,
            "python": version[0],
            "gpu_name": gpu_name,
            "gpu_vram_mib": gpu_vram,
            "selected_micro_batch": selected["micro_batch"],
            "gradient_accumulation": selected["gradient_accumulation"],
            "effective_batch_size": TARGET_EFFECTIVE_BATCH,
            "peak_vram_mib": selected["peak_vram_mib"],
            "loss": selected["loss"],
            "optimizer_steps": 1,
            "trials": trials,
        },
    )
    print("=== 72b SMOLVLA BATCH PROBE: PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
