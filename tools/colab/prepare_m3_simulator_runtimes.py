#!/usr/bin/env python3
"""Prepare the three pinned M3 runtimes for Notebook 74 simulator smoke.

This is setup-only. It reuses Notebook 73 training-smoke evidence, reconstructs
missing model runtimes without rerunning 72 probes, and adds only the LIBERO
runtime dependencies required for evaluation. It never starts benchmark
training, promotion, or the final 800-episode evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
LIBERO_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
HF_LIBERO_VERSION = "0.1.3"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_status(path: Path, *, status: str, stage: str, error: str | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M3_simulator_runtime_setup",
        "status": status,
        "current_stage": stage,
        "selected_episode_ids_sha256": D10_HASH,
        "probes_rerun": False,
        "benchmark_training_started": False,
        "simulator_started": False,
        "promotion_started": False,
        "final_800_episode_evaluation_started": False,
    }
    if error:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(
    cmd: list[str],
    *,
    log: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    child = os.environ.copy()
    if env:
        child.update(env)
    child.setdefault("PYTHONUNBUFFERED", "1")
    child.setdefault("MUJOCO_GL", "egl")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(">>>", " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
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


def _system_libs_ready() -> bool:
    try:
        text = subprocess.check_output(["bash", "-lc", "ldconfig -p 2>/dev/null"], text=True)
    except Exception:
        return False
    return all(token in text for token in ("libEGL.so", "libGL.so", "libMagickWand"))


def _prepare_system_libs(log: Path) -> None:
    if _system_libs_ready():
        print("simulator system libraries already present", flush=True)
        return
    if os.geteuid() != 0 and shutil.which("sudo") is None:
        raise RuntimeError("simulator system libraries missing and sudo/root is unavailable")
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    _run(prefix + ["apt-get", "update", "-qq"], log=log)
    _run(
        prefix
        + [
            "env",
            "DEBIAN_FRONTEND=noninteractive",
            "apt-get",
            "install",
            "-y",
            "-qq",
            "--no-install-recommends",
            "libmagickwand-dev",
            "libosmesa6",
            "libosmesa6-dev",
            "libgl1",
            "libglfw3",
            "libglew-dev",
            "libegl1",
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "libglib2.0-0",
        ],
        log=log,
    )


def _runtime_paths(root: Path) -> dict[str, tuple[Path, Path]]:
    return {
        "pi05": (
            root / "vendor/lerobot-pi05-m3-probe",
            root / "vendor/lerobot-pi05-m3-probe/.venv/bin/python",
        ),
        "smolvla": (
            root / "vendor/lerobot-smolvla-m3",
            root / "venv-smolvla-m3/bin/python",
        ),
        "openvla_oft": (
            root / "vendor/openvla-oft-m3",
            root / "venv-openvla-oft-m3/bin/python",
        ),
    }


def _training_runtimes_missing(root: Path) -> bool:
    for source, python_bin in _runtime_paths(root).values():
        if not source.is_dir() or not python_bin.is_file():
            return True
    return False


def _prepare_training_runtimes(root: Path, repo: Path, drive: Path, log: Path) -> None:
    if not _training_runtimes_missing(root):
        print("Notebook 73 model runtimes already present; bootstrap skipped", flush=True)
        return
    helper = repo / "tools/colab/prepare_m3_training_runtimes.py"
    if not helper.is_file():
        raise FileNotFoundError(helper)
    env = {
        "PARC_ROOT": str(root),
        "PY_AI_REPO": str(repo),
        "PARC_DRIVE_ROOT": str(drive),
    }
    _run([sys.executable, "-u", str(helper)], log=log, cwd=repo, env=env)


def _install_lerobot_libero(root: Path, log: Path) -> None:
    _ensure_uv(log)
    for model in ("pi05", "smolvla"):
        source, python_bin = _runtime_paths(root)[model]
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_bin),
                f"hf-libero=={HF_LIBERO_VERSION}",
            ],
            log=log,
        )
        _run(
            [
                str(python_bin),
                "-c",
                (
                    "import importlib.metadata as im; import lerobot; import libero; "
                    "import lerobot.envs.libero; "
                    "print('lerobot=' + str(lerobot.__version__)); "
                    "print('hf-libero=' + im.version('hf-libero')); "
                    "print('libero_import_ok')"
                ),
            ],
            log=log,
            cwd=source,
        )


def _checkout_openvla_libero(root: Path, log: Path) -> Path:
    source = root / "vendor/libero-openvla-m3"
    if not (source / ".git").is_dir():
        shutil.rmtree(source, ignore_errors=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q", str(source)], log=log)
        _run(["git", "-C", str(source), "remote", "add", "origin", LIBERO_URL], log=log)
    _run(["git", "-C", str(source), "fetch", "-q", "--depth", "1", "origin", LIBERO_REF], log=log)
    _run(["git", "-C", str(source), "checkout", "-q", "--force", "FETCH_HEAD"], log=log)
    got = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if got != LIBERO_REF:
        raise RuntimeError(f"OpenVLA LIBERO source pin mismatch: {got}")
    return source


def _write_libero_config(source: Path) -> None:
    package_root = source / "libero/libero"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    cfg = Path.home() / ".libero/config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "\n".join(
            [
                f"benchmark_root: {package_root}",
                f"bddl_files: {package_root / 'bddl_files'}",
                f"init_states: {package_root / 'init_files'}",
                f"datasets: {package_root / 'datasets'}",
                f"assets: {package_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _install_openvla_libero(root: Path, log: Path) -> None:
    _ensure_uv(log)
    openvla_source, python_bin = _runtime_paths(root)["openvla_oft"]
    libero_source = _checkout_openvla_libero(root, log)
    _write_libero_config(libero_source)
    _run(
        ["uv", "pip", "install", "--python", str(python_bin), "-e", str(libero_source)],
        log=log,
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "robosuite==1.4.1",
            "gym==0.25.2",
            "bddl==3.6.0",
            "easydict==1.13",
            "cloudpickle==3.1.2",
            "imageio[ffmpeg]",
        ],
        log=log,
    )
    env = {
        "PYTHONPATH": str(openvla_source),
        "MUJOCO_GL": "egl",
    }
    _run(
        [
            str(python_bin),
            "-c",
            (
                "import libero, robosuite; "
                "import experiments.robot.libero.run_libero_eval; "
                "print('openvla_libero_import_ok')"
            ),
        ],
        log=log,
        cwd=openvla_source,
        env=env,
    )


def _validate_training_smoke(drive: Path) -> None:
    path = drive / "model-benchmark-v1/m3-training-smoke-v1/m3_training_smoke_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = _load(path)
    if data.get("status") != "PASS" or data.get("stage") != "M3_A100_training_smoke":
        raise RuntimeError("Notebook 73 training smoke summary is not PASS")
    if data.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("Notebook 73 training smoke D10 mismatch")
    if data.get("benchmark_training_started") is not False:
        raise RuntimeError("Notebook 73 smoke unexpectedly claims benchmark training started")


def main() -> int:
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("simulator runtime setup requires explicit PARC_M3_EXECUTE=1")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for simulator runtime setup")

    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026")).resolve()
    repo = Path(os.environ.get("PY_AI_REPO", root / "py_AI_m3_minimal_sim_smoke")).resolve()
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache")).resolve()
    out = drive / "model-benchmark-v1/m3-simulator-minimal-smoke-v1/runtime-setup"
    log = out / "runtime_setup.log"
    status = out / "runtime_setup_status.json"
    out.mkdir(parents=True, exist_ok=True)
    log.write_text("=== M3 simulator runtime setup ===\n", encoding="utf-8")

    stage = "training_smoke_gate"
    try:
        _write_status(status, status="RUNNING", stage=stage)
        _validate_training_smoke(drive)

        sys.path.insert(0, str(repo))
        from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
        from tools.benchmark.m3_eval_runtime_guard import verify_source_revision  # noqa: PLC0415

        gpu_name, gpu_vram_mib = gpu_info()
        if "A100" not in gpu_name or gpu_vram_mib < 38000:
            raise RuntimeError(f"Notebook 74 requires one A100 >=38000 MiB, got {gpu_name} {gpu_vram_mib}")

        stage = "training_runtime_bootstrap"
        _write_status(status, status="RUNNING", stage=stage)
        _prepare_training_runtimes(root, repo, drive, log)

        stage = "system_libs"
        _write_status(status, status="RUNNING", stage=stage)
        _prepare_system_libs(log)

        stage = "lerobot_libero"
        _write_status(status, status="RUNNING", stage=stage)
        _install_lerobot_libero(root, log)

        stage = "openvla_libero"
        _write_status(status, status="RUNNING", stage=stage)
        _install_openvla_libero(root, log)

        stage = "source_pins"
        _write_status(status, status="RUNNING", stage=stage)
        for model, (source, _python_bin) in _runtime_paths(root).items():
            verify_source_revision(model, source)

        _write_status(status, status="PASS", stage="complete")
        print("=== M3 SIMULATOR RUNTIME SETUP: PASS ===", flush=True)
        print(
            "Setup only. 72 probes were not rerun; benchmark training/promotion/final evaluation have NOT started.",
            flush=True,
        )
        return 0
    except Exception as exc:
        _write_status(status, status="FAILED", stage=stage, error=f"{type(exc).__name__}: {exc}")
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n=== FAILURE ===\n" + traceback.format_exc() + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
