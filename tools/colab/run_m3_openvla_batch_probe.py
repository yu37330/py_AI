#!/usr/bin/env python3
"""Probe OpenVLA-OFT micro-batch size on A100 using the validated selected-pool stream."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"
OPENVLA_URL = "https://github.com/small-zeng/openvla-oft.git"


def checkout_exact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").is_dir():
        shutil.rmtree(path, ignore_errors=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", OPENVLA_URL], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "-q", "--depth", "1", "origin", OPENVLA_REF], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", "--force", "FETCH_HEAD"], check=True)
    got = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if got != OPENVLA_REF:
        raise RuntimeError(f"OpenVLA-OFT source pin mismatch: {got}")


def ensure_env(root: Path, source: Path) -> Path:
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(["uv", "python", "install", "3.10"], check=True)
    venv = root / "venv-openvla-oft-m3"
    if not (venv / "bin/python").is_file():
        subprocess.run(["uv", "venv", "--python", "3.10", str(venv)], check=True)
    python_bin = venv / "bin/python"
    marker = venv / ".m3_openvla_ref"
    if not marker.is_file() or marker.read_text().strip() != OPENVLA_REF:
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python_bin), "-e", str(source)],
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python_bin), "packaging", "ninja"],
            check=True,
        )
        # uv-created venvs are not guaranteed to expose pip as a module. FlashAttention's
        # upstream recipe explicitly needs a traditional pip PEP-517 build with
        # --no-build-isolation, so seed pip before following that pinned recipe.
        subprocess.run([str(python_bin), "-m", "ensurepip", "--upgrade"], check=True)
        subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "flash-attn==2.5.5",
                "--no-build-isolation",
            ],
            check=True,
        )
        marker.write_text(OPENVLA_REF + "\n", encoding="utf-8")
    check = subprocess.check_output(
        [
            str(python_bin),
            "-c",
            "import sys,torch; print(sys.version.split()[0]); print(torch.__version__); import flash_attn; print('flash_attn_ok')",
        ],
        text=True,
    ).strip().splitlines()
    if not check or not check[0].startswith("3.10"):
        raise RuntimeError(f"OpenVLA-OFT probe must use Python 3.10: {check}")
    return venv


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
    _, manifest, streaming = load_gate_paths(drive)
    dataset_root = Path(
        os.environ.get(
            "PARC_DRIVE_DATASET",
            str(drive / "datasets/lerobot_libero_plus_v3_train"),
        )
    )
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(dataset_root / "meta/info.json")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for OpenVLA-OFT probe")

    source = root / "vendor/openvla-oft-m3"
    checkout_exact(source)
    venv = ensure_env(root, source)
    python_bin = venv / "bin/python"
    cache = root / "cache/m3-openvla-batch-probe"
    log_root = cache / "logs"
    trial_root = cache / "trials"
    candidates = [8, 4, 2, 1]
    trials = []
    selected = None
    inner = repo / "tools/benchmark/openvla_oft_one_step_probe.py"
    if not inner.is_file():
        raise FileNotFoundError(inner)

    for micro_batch in candidates:
        ga = grad_accum_for(micro_batch)
        run_name = f"m3_openvla_probe_bs{micro_batch}_ga{ga}"
        trial_json = trial_root / f"{run_name}.json"
        trial_json.unlink(missing_ok=True)
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
            "--dataset-root",
            str(dataset_root),
            "--manifest",
            str(manifest),
            "--streaming-contract",
            str(streaming),
            "--micro-batch",
            str(micro_batch),
            "--grad-accum",
            str(ga),
            "--out",
            str(trial_json),
        ]
        env = {
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "HF_HOME": str(root / "cache/huggingface-openvla-m3"),
            "TRANSFORMERS_CACHE": str(root / "cache/huggingface-openvla-m3/transformers"),
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
        rc, text, elapsed, sampled_peak = run_logged(
            cmd,
            cwd=source,
            env=env,
            log_path=log_root / f"{run_name}.log",
        )
        inner_result = None
        if rc == 0:
            if not trial_json.is_file():
                raise RuntimeError(f"OpenVLA probe returned 0 but result is missing: {trial_json}")
            inner_result = json.loads(trial_json.read_text(encoding="utf-8"))
            if int(inner_result.get("optimizer_steps", -1)) != 1:
                raise RuntimeError(f"OpenVLA optimizer step mismatch: {inner_result}")
            if int(inner_result.get("effective_batch_size", -1)) != TARGET_EFFECTIVE_BATCH:
                raise RuntimeError(f"OpenVLA effective batch mismatch: {inner_result}")
        peak = sampled_peak
        loss = None
        if inner_result:
            peak = max(peak, int(inner_result.get("torch_peak_allocated_mib", 0) or 0))
            loss = inner_result.get("loss")
        trial = candidate_result(
            micro_batch=micro_batch,
            rc=rc,
            text=text,
            elapsed_sec=elapsed,
            peak_vram=peak,
        )
        if loss is not None:
            trial["loss"] = loss
        trial["optimizer_steps"] = 1 if rc == 0 else 0
        trials.append(trial)
        if rc == 0:
            selected = trial
            break
        if trial["status"] != "OOM":
            raise RuntimeError(
                f"OpenVLA-OFT probe failed for non-OOM reason: {trial}; log={log_root / (run_name + '.log')}"
            )

    result_path = drive / "model-benchmark-v1/batch-probes/openvla_oft.json"
    if selected is None:
        write_result(
            result_path,
            {
                "status": "FAILED_NO_FIT",
                "model": "openvla_oft",
                "source_ref": OPENVLA_REF,
                "gpu_name": gpu_name,
                "gpu_vram_mib": gpu_vram,
                "trials": trials,
            },
        )
        raise RuntimeError("OpenVLA-OFT did not fit even at micro_batch=1")

    write_result(
        result_path,
        {
            "status": "PASS",
            "model": "openvla_oft",
            "source_ref": OPENVLA_REF,
            "python": "3.10",
            "gpu_name": gpu_name,
            "gpu_vram_mib": gpu_vram,
            "selected_micro_batch": selected["micro_batch"],
            "gradient_accumulation": selected["gradient_accumulation"],
            "effective_batch_size": TARGET_EFFECTIVE_BATCH,
            "peak_vram_mib": selected["peak_vram_mib"],
            "loss": selected["loss"],
            "optimizer_steps": 1,
            "recipe": {
                "lora_r": 32,
                "learning_rate": 0.0005,
                "use_l1_regression": True,
                "use_diffusion": False,
                "use_film": False,
                "num_images_in_input": 2,
                "use_proprio": True,
                "image_aug_m3": True,
                "image_aug_probe": False,
            },
            "trials": trials,
        },
    )
    print("=== 72c OPENVLA-OFT BATCH PROBE: PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
