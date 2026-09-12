#!/usr/bin/env python3
"""Pin the MuJoCo version required by the frozen LIBERO/robosuite runtimes.

Notebook 74 uses robosuite 1.4.x through hf-libero/OpenVLA. Fresh resolution of
hf-libero currently pulls the newest MuJoCo, but robosuite 1.4.x predates the
breaking MuJoCo 3.10+ API changes. Keep all three evaluator environments on the
same known-compatible MuJoCo release before any simulator episode is started.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

MUJOCO_VERSION = "3.3.1"
RUNTIMES = {
    "pi05": "vendor/lerobot-pi05-m3-probe/.venv/bin/python",
    "smolvla": "venv-smolvla-m3/bin/python",
    "openvla_oft": "venv-openvla-oft-m3/bin/python",
}


def _run(cmd: list[str], *, env: dict[str, str]) -> str:
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    print(proc.stdout, end="", flush=True)
    return proc.stdout


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026")).resolve()
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required before MuJoCo compatibility pinning")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    results: dict[str, dict[str, str]] = {}

    for model, rel_python in RUNTIMES.items():
        python_bin = root / rel_python
        if not python_bin.is_file():
            raise FileNotFoundError(python_bin)
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_bin),
                f"mujoco=={MUJOCO_VERSION}",
            ],
            env=env,
        )
        out = _run(
            [
                str(python_bin),
                "-c",
                (
                    "import importlib.metadata as im; import mujoco, robosuite; "
                    f"assert mujoco.__version__ == '{MUJOCO_VERSION}', mujoco.__version__; "
                    "print('mujoco=' + mujoco.__version__); "
                    "print('robosuite=' + im.version('robosuite')); "
                    "print('mujoco_robosuite_import_ok')"
                ),
            ],
            env=env,
        )
        versions = {}
        for line in out.splitlines():
            if "=" in line and line.split("=", 1)[0] in {"mujoco", "robosuite"}:
                key, value = line.split("=", 1)
                versions[key] = value
        results[model] = versions

    evidence = {
        "schema_version": 1,
        "stage": "M3_mujoco_compat",
        "status": "PASS",
        "mujoco_version": MUJOCO_VERSION,
        "runtimes": results,
        "benchmark_training_started": False,
        "simulator_episode_started": False,
        "promotion_started": False,
        "final_800_episode_evaluation_started": False,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
