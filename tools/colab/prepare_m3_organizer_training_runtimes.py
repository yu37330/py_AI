#!/usr/bin/env python3
"""Prepare all three M3 training runtimes for organizer RTX PRO Blackwell.

π0.5 and SmolVLA reuse the already-pinned source setup. OpenVLA uses a
Blackwell-specific hardware runtime because its pinned upstream metadata still
declares torch 2.2.0. This helper never reruns 72 probes or starts training.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HARDWARE_PROFILE = "organizer_rtx_pro_6000_blackwell"


def _load_base(repo: Path):
    path = repo / "tools/colab/prepare_m3_training_runtimes.py"
    spec = importlib.util.spec_from_file_location("parc_base_training_runtime_setup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base runtime helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_blackwell_python(python_bin: Path, model: str) -> dict:
    code = (
        "import json,torch; "
        "cap=torch.cuda.get_device_capability(); arch=torch.cuda.get_arch_list(); "
        "cuda=tuple(int(x) for x in str(torch.version.cuda).split('.')[:2]); "
        "x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16); y=(x@x).sum(); "
        "payload={'torch':torch.__version__,'cuda':torch.version.cuda,'capability':list(cap),"
        "'arch_list':arch,'device':torch.cuda.get_device_name(),'matmul':float(y)}; "
        "assert tuple(cap)==(12,0), payload; assert 'sm_120' in arch, payload; "
        "assert cuda >= (12,8), payload; print(json.dumps(payload))"
    )
    output = subprocess.check_output([str(python_bin), "-c", code], text=True).strip().splitlines()
    if not output:
        raise RuntimeError(f"empty Blackwell verification output for {model}")
    payload = json.loads(output[-1])
    payload["model"] = model
    return payload


def main() -> int:
    if os.environ.get("PARC_M3_HARDWARE_PROFILE") != HARDWARE_PROFILE:
        raise RuntimeError("organizer runtime setup requires Blackwell hardware profile")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    root = Path(os.environ.get("PARC_LOCAL_SCRATCH_ROOT", os.environ.get("PARC_ROOT", "/opt/dlami/nvme/parc2026"))).expanduser().resolve()
    persist = Path(os.environ.get("PARC_PERSIST_ROOT", os.environ.get("PARC_DRIVE_ROOT", str(Path.home() / "data/parc2026-cache")))).expanduser().resolve()
    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    log = persist / "model-benchmark-v1/m3-organizer-migration-v1/training_runtime_setup.log"
    status = persist / "model-benchmark-v1/m3-organizer-migration-v1/training_runtime_setup.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    stage = "bootstrap"
    try:
        base = _load_base(repo)
        stage = "uv"
        base._ensure_uv(log)
        stage = "pi05"
        base._prepare_pi05(root, repo, log)
        stage = "smolvla"
        base._prepare_smolvla(root, log)

        stage = "openvla_blackwell"
        subprocess.run(
            [sys.executable, "-u", str(repo / "tools/colab/prepare_m3_openvla_blackwell_runtime.py")],
            cwd=str(repo),
            env=os.environ.copy(),
            check=True,
        )

        stage = "source_runtime_preflight"
        sys.path.insert(0, str(repo))
        from tools.benchmark.m3_model_adapters import default_runtimes, runtime_preflight  # noqa: PLC0415

        blockers: list[str] = []
        runtimes = default_runtimes(root, repo)
        for runtime in runtimes.values():
            blockers.extend(runtime_preflight(runtime, repo=repo, strict_files=True))
        if blockers:
            raise RuntimeError(f"organizer M3 runtime blockers: {blockers}")

        stage = "blackwell_cuda_verify"
        checks = []
        for model, runtime in runtimes.items():
            checks.append(_verify_blackwell_python(Path(runtime.python), model))

        payload = {
            "schema_version": 1,
            "stage": "M3_organizer_training_runtime_setup",
            "status": "PASS",
            "hardware_profile": HARDWARE_PROFILE,
            "runtime_checks": checks,
            "probes_rerun": False,
            "benchmark_training_started": False,
        }
        status.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("=== M3 ORGANIZER TRAINING RUNTIMES: PASS ===", flush=True)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        status.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "M3_organizer_training_runtime_setup",
                    "status": "FAILED",
                    "current_stage": stage,
                    "hardware_profile": HARDWARE_PROFILE,
                    "error": f"{type(exc).__name__}: {exc}",
                    "probes_rerun": False,
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
