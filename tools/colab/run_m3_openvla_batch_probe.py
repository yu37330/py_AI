#!/usr/bin/env python3
"""Probe OpenVLA-OFT micro-batch size on A100 using the validated selected-pool stream."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any

OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"
OPENVLA_URL = "https://github.com/small-zeng/openvla-oft.git"
TF_METADATA_VERSION = "1.16.1"
PROTOBUF_VERSION = "3.20.3"


def _write_status(
    path: Path,
    *,
    status: str,
    stage: str,
    setup_log: Path,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M3_OpenVLA_A100_batch_probe",
        "status": status,
        "current_stage": stage,
        "source_ref": OPENVLA_REF,
        "setup_log": str(setup_log),
        "benchmark_training_started": False,
    }
    if error is not None:
        payload["error"] = error
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _run_stage(
    stage: str,
    cmd: list[str],
    *,
    status_path: Path,
    setup_log: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    _write_status(
        status_path,
        status="RUNNING",
        stage=stage,
        setup_log=setup_log,
        extra={"command": cmd},
    )
    setup_log.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[72c:{stage}] >>> {' '.join(cmd)}", flush=True)
    lines: list[str] = []
    with setup_log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {stage} ===\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            fh.write(line + "\n")
            fh.flush()
            print(line, flush=True)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{stage} failed (rc={rc}); full log: {setup_log}")
    _write_status(status_path, status="PASS", stage=stage, setup_log=setup_log)
    return "\n".join(lines)


def checkout_exact(path: Path, *, status_path: Path, setup_log: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").is_dir():
        shutil.rmtree(path, ignore_errors=True)
        _run_stage(
            "source_git_init",
            ["git", "init", "-q", str(path)],
            status_path=status_path,
            setup_log=setup_log,
        )
        _run_stage(
            "source_git_remote",
            ["git", "-C", str(path), "remote", "add", "origin", OPENVLA_URL],
            status_path=status_path,
            setup_log=setup_log,
        )
    _run_stage(
        "source_fetch",
        ["git", "-C", str(path), "fetch", "-q", "--depth", "1", "origin", OPENVLA_REF],
        status_path=status_path,
        setup_log=setup_log,
    )
    _run_stage(
        "source_checkout",
        ["git", "-C", str(path), "checkout", "-q", "--force", "FETCH_HEAD"],
        status_path=status_path,
        setup_log=setup_log,
    )
    got = _run_stage(
        "source_verify",
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        status_path=status_path,
        setup_log=setup_log,
    ).strip().splitlines()[-1]
    if got != OPENVLA_REF:
        raise RuntimeError(f"OpenVLA-OFT source pin mismatch: {got}")


def ensure_env(root: Path, source: Path, *, status_path: Path, setup_log: Path) -> Path:
    if shutil.which("uv") is None:
        _run_stage(
            "install_uv",
            [sys.executable, "-m", "pip", "install", "-q", "uv"],
            status_path=status_path,
            setup_log=setup_log,
        )
    _run_stage(
        "install_python_310",
        ["uv", "python", "install", "3.10"],
        status_path=status_path,
        setup_log=setup_log,
    )
    venv = root / "venv-openvla-oft-m3"
    if not (venv / "bin/python").is_file():
        _run_stage(
            "create_openvla_venv",
            ["uv", "venv", "--python", "3.10", str(venv)],
            status_path=status_path,
            setup_log=setup_log,
        )
    python_bin = venv / "bin/python"
    marker = venv / ".m3_openvla_ref"
    if not marker.is_file() or marker.read_text().strip() != OPENVLA_REF:
        _run_stage(
            "install_openvla_editable",
            ["uv", "pip", "install", "--python", str(python_bin), "-e", str(source)],
            status_path=status_path,
            setup_log=setup_log,
        )
        _run_stage(
            "install_build_helpers",
            ["uv", "pip", "install", "--python", str(python_bin), "packaging", "ninja"],
            status_path=status_path,
            setup_log=setup_log,
        )
        _run_stage(
            "ensure_pip",
            [str(python_bin), "-m", "ensurepip", "--upgrade"],
            status_path=status_path,
            setup_log=setup_log,
        )
        _run_stage(
            "install_flash_attn",
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "flash-attn==2.5.5",
                "--no-build-isolation",
            ],
            status_path=status_path,
            setup_log=setup_log,
        )
        marker.write_text(OPENVLA_REF + "\n", encoding="utf-8")

    # OpenVLA's unbounded TFDS dependency can resolve to a future tensorflow-metadata
    # release whose generated protobuf code is incompatible with TensorFlow 2.15's
    # protobuf<5 constraint. Keep the Python 3.10 environment on the compatible line.
    _run_stage(
        "pin_tf_metadata_protobuf",
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            f"tensorflow-metadata=={TF_METADATA_VERSION}",
            f"protobuf=={PROTOBUF_VERSION}",
        ],
        status_path=status_path,
        setup_log=setup_log,
    )

    check = _run_stage(
        "verify_openvla_env",
        [
            str(python_bin),
            "-c",
            (
                "import importlib.metadata as im,sys,torch; "
                "print(sys.version.split()[0]); print(torch.__version__); "
                "import flash_attn; print('flash_attn_ok'); "
                "import tensorflow_datasets as tfds; import tensorflow_metadata; import dlimp; "
                "print('tensorflow-datasets=' + im.version('tensorflow-datasets')); "
                "print('tensorflow-metadata=' + im.version('tensorflow-metadata')); "
                "print('protobuf=' + im.version('protobuf')); print('tfds_dlimp_ok')"
            ),
        ],
        status_path=status_path,
        setup_log=setup_log,
    ).strip().splitlines()
    if not any(line.startswith("3.10") for line in check):
        raise RuntimeError(f"OpenVLA-OFT probe must use Python 3.10: {check}")
    if f"tensorflow-metadata={TF_METADATA_VERSION}" not in check:
        raise RuntimeError(f"tensorflow-metadata pin mismatch: {check}")
    if f"protobuf={PROTOBUF_VERSION}" not in check:
        raise RuntimeError(f"protobuf pin mismatch: {check}")
    if "tfds_dlimp_ok" not in check:
        raise RuntimeError(f"TFDS/dlimp import verification failed: {check}")
    return venv


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", str(root / "py_AI_m3_probe")))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    log_root = drive / "model-benchmark-v1/batch-probes/logs/openvla_oft"
    setup_log = log_root / "setup.log"
    status_path = drive / "model-benchmark-v1/batch-probes/openvla_oft_status.json"
    setup_log.parent.mkdir(parents=True, exist_ok=True)
    setup_log.write_text("=== 72c OpenVLA-OFT setup diagnostics ===\n", encoding="utf-8")
    stage = "bootstrap"

    try:
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

        stage = "preflight_gpu"
        _write_status(status_path, status="RUNNING", stage=stage, setup_log=setup_log)
        gpu_name, gpu_vram = gpu_info()
        stage = "preflight_contracts"
        _write_status(status_path, status="RUNNING", stage=stage, setup_log=setup_log)
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
        stage = "source_checkout"
        checkout_exact(source, status_path=status_path, setup_log=setup_log)
        stage = "environment_setup"
        venv = ensure_env(root, source, status_path=status_path, setup_log=setup_log)
        python_bin = venv / "bin/python"
        cache = root / "cache/m3-openvla-batch-probe"
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
            stage = f"probe_bs{micro_batch}_ga{ga}"
            _write_status(
                status_path,
                status="RUNNING",
                stage=stage,
                setup_log=setup_log,
                extra={"micro_batch": micro_batch, "gradient_accumulation": ga},
            )
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
            log_path = log_root / f"{run_name}.log"
            rc, text, elapsed, sampled_peak = run_logged(
                cmd,
                cwd=source,
                env=env,
                log_path=log_path,
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
            trial["log_path"] = str(log_path)
            trials.append(trial)
            if rc == 0:
                selected = trial
                break
            if trial["status"] != "OOM":
                raise RuntimeError(f"OpenVLA-OFT probe failed for non-OOM reason: {trial}; log={log_path}")

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
        _write_status(
            status_path,
            status="PASS",
            stage="complete",
            setup_log=setup_log,
            extra={"result": str(result_path)},
        )
        print("=== 72c OPENVLA-OFT BATCH PROBE: PASS ===", flush=True)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        payload = _write_status(
            status_path,
            status="FAILED",
            stage=stage,
            setup_log=setup_log,
            error=f"{type(exc).__name__}: {exc}",
            extra={"traceback": tb},
        )
        print("=== 72c DIAGNOSTICS ===", flush=True)
        print(json.dumps(payload, indent=2), flush=True)
        if setup_log.is_file():
            lines = setup_log.read_text(encoding="utf-8", errors="replace").splitlines()
            print("=== 72c SETUP LOG TAIL ===", flush=True)
            print("\n".join(lines[-200:]), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())