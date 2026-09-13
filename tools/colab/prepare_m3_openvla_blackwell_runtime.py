#!/usr/bin/env python3
"""Prepare the pinned OpenVLA-OFT source on a Blackwell-capable PyTorch runtime.

The upstream pinned source declares torch==2.2.0 and the historical A100 setup
adds flash-attn==2.5.5. Those binaries predate Blackwell support. For the
organizer RTX PRO 6000 Blackwell path, preserve the exact OpenVLA source commit
and model logic while overriding only the hardware runtime to PyTorch 2.7.1 +
CUDA 12.8. The M3 code does not request flash_attention_2, so the historical
FlashAttention custom extension is intentionally absent on this profile.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"
OPENVLA_URL = "https://github.com/small-zeng/openvla-oft.git"
TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
TORCHAUDIO_VERSION = "2.7.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
WANDB_VERSION = "0.16.6"
TF_METADATA_VERSION = "1.16.1"
PROTOBUF_VERSION = "3.20.3"
PANDAS_VERSION = "2.2.3"
PYARROW_VERSION = "17.0.0"
AV_VERSION = "12.3.0"


def _run(cmd: list[str], *, log: Path, cwd: Path | None = None, check: bool = True) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(">>>", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.write(proc.stdout or "")
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}; log={log}")
    return proc.stdout or ""


def _checkout(source: Path, log: Path) -> None:
    if not (source / ".git").is_dir():
        shutil.rmtree(source, ignore_errors=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q", str(source)], log=log)
        _run(["git", "-C", str(source), "remote", "add", "origin", OPENVLA_URL], log=log)
    _run(["git", "-C", str(source), "fetch", "-q", "--depth", "1", "origin", OPENVLA_REF], log=log)
    _run(["git", "-C", str(source), "checkout", "-q", "--force", "FETCH_HEAD"], log=log)
    got = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if got != OPENVLA_REF:
        raise RuntimeError(f"OpenVLA source pin mismatch: {got} != {OPENVLA_REF}")


def main() -> int:
    if os.environ.get("PARC_M3_HARDWARE_PROFILE") != "organizer_rtx_pro_6000_blackwell":
        raise RuntimeError("Blackwell OpenVLA runtime requires organizer hardware profile")
    root = Path(os.environ.get("PARC_LOCAL_SCRATCH_ROOT", os.environ.get("PARC_ROOT", "/opt/dlami/nvme/parc2026"))).expanduser().resolve()
    persist = Path(os.environ.get("PARC_PERSIST_ROOT", os.environ.get("PARC_DRIVE_ROOT", str(Path.home() / "data/parc2026-cache")))).expanduser().resolve()
    source = root / "vendor/openvla-oft-m3"
    venv = root / "venv-openvla-oft-m3"
    log = persist / "model-benchmark-v1/m3-organizer-migration-v1/openvla_blackwell_runtime.log"
    status = persist / "model-benchmark-v1/m3-organizer-migration-v1/openvla_blackwell_runtime.json"
    stage = "bootstrap"
    try:
        if shutil.which("uv") is None:
            _run([sys.executable, "-m", "pip", "install", "-q", "uv"], log=log)
        stage = "source"
        _checkout(source, log)
        stage = "python"
        _run(["uv", "python", "install", "3.10"], log=log)
        if not (venv / "bin/python").is_file():
            _run(["uv", "venv", "--python", "3.10", str(venv)], log=log)
        python_bin = venv / "bin/python"

        # Install the exact source dependency graph first. This temporarily pulls
        # torch 2.2.0 as declared upstream; the next stage replaces only the
        # hardware runtime wheels while leaving the source commit untouched.
        stage = "source_dependencies"
        _run(["uv", "pip", "install", "--python", str(python_bin), "-e", str(source)], log=log)

        stage = "blackwell_torch"
        _run(
            [
                "uv", "pip", "install", "--python", str(python_bin),
                "--index-url", TORCH_INDEX,
                f"torch=={TORCH_VERSION}",
                f"torchvision=={TORCHVISION_VERSION}",
                f"torchaudio=={TORCHAUDIO_VERSION}",
            ],
            log=log,
        )

        # Historical flash-attn 2.5.5 custom kernels predate this hardware.
        # M3 OpenVLA loading does not request flash_attention_2, so keep the
        # Blackwell path on standard PyTorch attention rather than an unverified
        # custom kernel extension.
        stage = "remove_historical_flash_attn"
        _run(["uv", "pip", "uninstall", "--python", str(python_bin), "flash-attn"], log=log, check=False)

        stage = "compat_pins"
        _run(
            [
                "uv", "pip", "install", "--python", str(python_bin),
                f"tensorflow-metadata=={TF_METADATA_VERSION}",
                f"protobuf=={PROTOBUF_VERSION}",
                f"wandb=={WANDB_VERSION}",
                f"pandas=={PANDAS_VERSION}",
                f"pyarrow=={PYARROW_VERSION}",
                f"av=={AV_VERSION}",
            ],
            log=log,
        )

        stage = "verify"
        verify = (
            "import importlib.util,json,torch; "
            "cap=torch.cuda.get_device_capability(); "
            "arch=torch.cuda.get_arch_list(); "
            "x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16); y=(x@x).sum(); "
            "payload={'torch':torch.__version__,'cuda':torch.version.cuda,'capability':list(cap),"
            "'arch_list':arch,'device':torch.cuda.get_device_name(),'matmul':float(y)}; "
            "assert tuple(cap)==(12,0), payload; "
            "assert 'sm_120' in arch, payload; "
            "assert str(torch.version.cuda).startswith('12.8'), payload; "
            "assert importlib.util.find_spec('flash_attn') is None, 'historical flash_attn still installed'; "
            "import transformers,peft,tensorflow_datasets,pandas,pyarrow,av; print(json.dumps(payload))"
        )
        output = _run([str(python_bin), "-c", verify], log=log).strip().splitlines()
        payload = json.loads(output[-1])
        status_payload = {
            "schema_version": 1,
            "stage": "M3_OpenVLA_Blackwell_runtime",
            "status": "PASS",
            "source_ref": OPENVLA_REF,
            "hardware_profile": "organizer_rtx_pro_6000_blackwell",
            "torch": payload["torch"],
            "cuda": payload["cuda"],
            "capability": payload["capability"],
            "arch_list": payload["arch_list"],
            "device": payload["device"],
            "historical_flash_attn_installed": False,
            "source_dependency_torch_override": True,
            "benchmark_training_started": False,
        }
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("=== M3 OPENVLA BLACKWELL RUNTIME: PASS ===", flush=True)
        print(json.dumps(status_payload, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "M3_OpenVLA_Blackwell_runtime",
                    "status": "FAILED",
                    "current_stage": stage,
                    "source_ref": OPENVLA_REF,
                    "error": f"{type(exc).__name__}: {exc}",
                    "benchmark_training_started": False,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n=== FAILURE ===\n" + traceback.format_exc() + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
