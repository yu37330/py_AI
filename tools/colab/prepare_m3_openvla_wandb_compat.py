#!/usr/bin/env python3
"""Prepare the pinned OpenVLA-OFT probe runtime for W&B and streaming data access."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

WANDB_VERSION = "0.16.6"
PROTOBUF_VERSION = "3.20.3"
PANDAS_VERSION = "2.2.3"
PYARROW_VERSION = "17.0.0"
AV_VERSION = "12.3.0"


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

    # The validated 69c streaming adapter reads LeRobot metadata/parquet/video
    # directly. OpenVLA-OFT itself does not depend on pandas/pyarrow/PyAV, so
    # these packages must be added explicitly to the isolated training venv.
    # Pin 2024-era versions to avoid another dependency drift during submission.
    probe._run_stage(
        "pin_streaming_data_runtime",
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            f"pandas=={PANDAS_VERSION}",
            f"pyarrow=={PYARROW_VERSION}",
            f"av=={AV_VERSION}",
        ],
        status_path=status_path,
        setup_log=setup_log,
    )

    check = probe._run_stage(
        "verify_wandb_streaming_compat",
        [
            str(python_bin),
            "-c",
            (
                "import importlib.metadata as im; "
                "import wandb, pandas, pyarrow, av; "
                "print('wandb=' + im.version('wandb')); "
                "print('protobuf=' + im.version('protobuf')); "
                "print('pandas=' + im.version('pandas')); "
                "print('pyarrow=' + im.version('pyarrow')); "
                "print('av=' + im.version('av')); "
                "print('wandb_streaming_import_ok')"
            ),
        ],
        status_path=status_path,
        setup_log=setup_log,
    ).strip().splitlines()

    expected = {
        "wandb": WANDB_VERSION,
        "protobuf": PROTOBUF_VERSION,
        "pandas": PANDAS_VERSION,
        "pyarrow": PYARROW_VERSION,
        "av": AV_VERSION,
    }
    for package, version in expected.items():
        if f"{package}={version}" not in check:
            raise RuntimeError(f"{package} pin mismatch: {check}")
    if "wandb_streaming_import_ok" not in check:
        raise RuntimeError(f"W&B/streaming import verification failed: {check}")

    print(
        "72c runtime compatibility ready: "
        f"wandb={WANDB_VERSION} protobuf={PROTOBUF_VERSION} "
        f"pandas={PANDAS_VERSION} pyarrow={PYARROW_VERSION} av={AV_VERSION}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
