#!/usr/bin/env python3
"""Fail-closed hardware/source guard shared by M3 simulator evaluators."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

EXECUTE_ENV = "PARC_M3_EXECUTE"
EXECUTE_VALUE = "1"
MIN_A100_VRAM_MIB = 38_000
SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
    "openvla_oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}


def require_execution_opt_in() -> None:
    if os.environ.get(EXECUTE_ENV) != EXECUTE_VALUE:
        raise RuntimeError(
            f"M3 simulator evaluation requires explicit {EXECUTE_ENV}={EXECUTE_VALUE}"
        )


def require_a100() -> tuple[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"M3 evaluator requires exactly one GPU, got {len(output)}")
    name, memory = [piece.strip() for piece in output[0].split(",", 1)]
    memory_mib = int(float(memory))
    if "A100" not in name:
        raise RuntimeError(f"M3 simulator metrics require NVIDIA A100, got {name}")
    if memory_mib < MIN_A100_VRAM_MIB:
        raise RuntimeError(
            f"M3 evaluator requires >= {MIN_A100_VRAM_MIB} MiB A100 VRAM, got {memory_mib}"
        )
    return name, memory_mib


def verify_source_revision(model: str, source_root: Path) -> str:
    if model not in SOURCE_REFS:
        raise ValueError(f"unknown M3 model: {model}")
    root = Path(source_root).resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"pinned source checkout missing .git: {root}")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if model == "pi05":
        tag = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "v0.4.4"], text=True
        ).strip()
        if head != tag:
            raise RuntimeError(f"pi0.5 evaluator source is not pinned v0.4.4: {head}")
    elif head != SOURCE_REFS[model]:
        raise RuntimeError(
            f"{model} evaluator source mismatch: {head} != {SOURCE_REFS[model]}"
        )
    return head


def validate_runtime(model: str, source_root: Path) -> dict[str, object]:
    require_execution_opt_in()
    gpu_name, gpu_vram_mib = require_a100()
    head = verify_source_revision(model, source_root)
    return {
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "source_head": head,
        "source_ref": SOURCE_REFS[model],
    }
