#!/usr/bin/env python3
"""Rebuild the three pinned M3 training runtimes in a fresh Colab VM.

This is setup-only. It does not rerun 72a/72b/72c probes and never starts
M3 smoke or benchmark training. Colab `/content` is ephemeral, so Notebook 73
must be able to reconstruct the exact source checkouts/venvs from persistent
72d/69c evidence without relying on a previous notebook session.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

SMOL_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
SMOL_URL = "https://github.com/huggingface/lerobot.git"
OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"


def _write_status(path: Path, *, status: str, stage: str, error: str | None = None) -> None:
    payload = {
        "schema_version": 1,
        "stage": "M3_training_runtime_setup",
        "status": status,
        "current_stage": stage,
        "benchmark_training_started": False,
        "probes_rerun": False,
    }
    if error:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(cmd: list[str], *, log: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    child = os.environ.copy()
    if env:
        child.update(env)
    child.setdefault("PYTHONUNBUFFERED", "1")
    print(">>>", " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=child,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            print(line, end="", flush=True)
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}; log={log}")


def _ensure_uv(log: Path) -> None:
    if shutil.which("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"], log=log)


def _prepare_pi05(root: Path, repo: Path, log: Path) -> None:
    _run(["uv", "python", "install", "3.10"], log=log)
    py310 = subprocess.check_output(["uv", "python", "find", "3.10"], text=True).strip()
    setup = repo / "examples/pi05_libero_finetune/scripts/setup_train.sh"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    source = root / "vendor/lerobot-pi05-m3-probe"
    env = {
        "PYTHON": py310,
        "DATA_ROOT": str(root / "cache/m3-pi05-batch-probe"),
        "LEROBOT_ROOT": str(source),
        "INSTALL_FFMPEG": "0",
    }
    _run(["bash", "scripts/setup_train.sh"], log=log, cwd=setup.parent.parent, env=env)
    python_bin = source / ".venv/bin/python"
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    train_py = source / "src/lerobot/scripts/lerobot_train.py"
    if "LEROBOT_GRAD_ACCUM" not in train_py.read_text(encoding="utf-8"):
        raise RuntimeError("pi0.5 grad-accum patch missing after setup")


def _checkout_smol(source: Path, log: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").is_dir():
        shutil.rmtree(source, ignore_errors=True)
        _run(["git", "init", "-q", str(source)], log=log)
        _run(["git", "-C", str(source), "remote", "add", "origin", SMOL_URL], log=log)
    _run(["git", "-C", str(source), "fetch", "-q", "--depth", "1", "origin", SMOL_REF], log=log)
    _run(["git", "-C", str(source), "checkout", "-q", "--force", "FETCH_HEAD"], log=log)
    got = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if got != SMOL_REF:
        raise RuntimeError(f"SmolVLA source pin mismatch: {got}")


def _prepare_smolvla(root: Path, log: Path) -> None:
    source = root / "vendor/lerobot-smolvla-m3"
    _checkout_smol(source, log)
    _run(["uv", "python", "install", "3.12"], log=log)
    venv = root / "venv-smolvla-m3"
    if not (venv / "bin/python").is_file():
        _run(["uv", "venv", "--python", "3.12", str(venv)], log=log)
    python_bin = venv / "bin/python"
    marker = venv / ".m3_smolvla_ref"
    train_bin = venv / "bin/lerobot-train"
    if not train_bin.is_file() or not marker.is_file() or marker.read_text().strip() != SMOL_REF:
        _run(
            ["uv", "pip", "install", "--python", str(python_bin), "-e", f"{source}[training,smolvla]"],
            log=log,
        )
        marker.write_text(SMOL_REF + "\n", encoding="utf-8")
    check = subprocess.check_output(
        [str(python_bin), "-c", "import sys,lerobot; print(sys.version.split()[0]); print(lerobot.__version__)"],
        text=True,
    ).splitlines()
    if not check or not check[0].startswith("3.12"):
        raise RuntimeError(f"SmolVLA runtime Python mismatch: {check}")


def _prepare_openvla(root: Path, repo: Path, drive: Path, log: Path) -> None:
    helper = repo / "tools/colab/prepare_m3_openvla_wandb_compat.py"
    if not helper.is_file():
        raise FileNotFoundError(helper)
    env = {
        "PARC_ROOT": str(root),
        "PY_AI_REPO": str(repo),
        "PARC_DRIVE_ROOT": str(drive),
    }
    _run([sys.executable, "-u", str(helper)], log=log, cwd=repo, env=env)
    python_bin = root / "venv-openvla-oft-m3/bin/python"
    source = root / "vendor/openvla-oft-m3"
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    got = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if got != OPENVLA_REF:
        raise RuntimeError(f"OpenVLA source pin mismatch: {got}")


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026")).resolve()
    repo = Path(os.environ.get("PY_AI_REPO", root / "py_AI_m3_training_smoke")).resolve()
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache")).resolve()
    setup_root = drive / "model-benchmark-v1/m3-training-smoke-v1/runtime-setup"
    log = setup_root / "runtime_setup.log"
    status = setup_root / "runtime_setup_status.json"
    setup_root.mkdir(parents=True, exist_ok=True)
    log.write_text("=== M3 training runtime setup ===\n", encoding="utf-8")

    stage = "preflight"
    try:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is required for M3 runtime setup")
        sys.path.insert(0, str(repo))
        from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
        from tools.benchmark.m3_model_adapters import default_runtimes, runtime_preflight  # noqa: PLC0415

        gpu_name, gpu_vram = gpu_info()
        print(f"M3 runtime setup GPU: {gpu_name} {gpu_vram} MiB", flush=True)
        _ensure_uv(log)

        stage = "pi05"
        _write_status(status, status="RUNNING", stage=stage)
        _prepare_pi05(root, repo, log)

        stage = "smolvla"
        _write_status(status, status="RUNNING", stage=stage)
        _prepare_smolvla(root, log)

        stage = "openvla_oft"
        _write_status(status, status="RUNNING", stage=stage)
        _prepare_openvla(root, repo, drive, log)

        stage = "runtime_preflight"
        _write_status(status, status="RUNNING", stage=stage)
        blockers: list[str] = []
        for runtime in default_runtimes(root, repo).values():
            blockers.extend(runtime_preflight(runtime, repo=repo, strict_files=True))
        if blockers:
            raise RuntimeError(f"M3 runtime blockers after setup: {blockers}")

        _write_status(status, status="PASS", stage="complete")
        print("=== M3 TRAINING RUNTIME SETUP: PASS ===", flush=True)
        print("Setup only. 72 probes were not rerun; benchmark training has NOT started.", flush=True)
        return 0
    except Exception as exc:
        _write_status(status, status="FAILED", stage=stage, error=f"{type(exc).__name__}: {exc}")
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n=== FAILURE ===\n" + traceback.format_exc() + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
