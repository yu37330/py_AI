#!/usr/bin/env python3
"""Prepare the pinned OpenVLA-OFT probe environment with a protobuf-compatible W&B runtime."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

WANDB_VERSION = "0.16.6"
PROTOBUF_VERSION = "3.20.3"


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", str(root / "py_AI_m3_probe")))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    log_root = drive / "model-benchmark-v1/batch-probes/logs/openvla_oft"
    setup_log = log_root / "setup.log"
    status_path = drive / "model-benchmark-v1/batch-probes/openvla_oft_status.json"

    sys.path.insert(0, str(repo / "tools/colab"))
    probe = importlib.import_module("run_m3_openvla_batch_probe")

    source = root / "vendor/openvla-oft-m3"
    probe.checkout_exact(source, status_path=status_path, setup_log=setup_log)
    venv = probe.ensure_env(root, source, status_path=status_path, setup_log=setup_log)
    python_bin = venv / "bin/python"

    # Upstream leaves wandb unbounded. In 2026 this resolves to a newer W&B
    # build whose generated telemetry protobufs are incompatible with the
    # protobuf 3.20.x line required by the TensorFlow 2.15 / TFDS runtime.
    probe._run_stage(
        "pin_wandb_protobuf_compat",
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            f"wandb=={WANDB_VERSION}",
            f"protobuf=={PROTOBUF_VERSION}",
        ],
        status_path=status_path,
        setup_log=setup_log,
    )

    check = probe._run_stage(
        "verify_wandb_compat",
        [
            str(python_bin),
            "-c",
            (
                "import importlib.metadata as im; import wandb; "
                "print('wandb=' + im.version('wandb')); "
                "print('protobuf=' + im.version('protobuf')); "
                "print('wandb_import_ok')"
            ),
        ],
        status_path=status_path,
        setup_log=setup_log,
    ).strip().splitlines()

    if f"wandb={WANDB_VERSION}" not in check:
        raise RuntimeError(f"wandb pin mismatch: {check}")
    if f"protobuf={PROTOBUF_VERSION}" not in check:
        raise RuntimeError(f"protobuf pin mismatch after wandb repair: {check}")
    if "wandb_import_ok" not in check:
        raise RuntimeError(f"wandb import verification failed: {check}")

    print(
        f"72c W&B compatibility ready: wandb={WANDB_VERSION} protobuf={PROTOBUF_VERSION}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
