#!/usr/bin/env python3
"""Hardware profile guard for PARC2026 M3 execution.

The goal is to preserve the validated A100 path while explicitly allowing the
organizer-provided RTX PRO 6000 Blackwell environment documented for PARC2026.
Do not silently accept smaller vGPU slices or unrelated GPUs.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    required_name_tokens: tuple[str, ...]
    min_vram_mib: int


PROFILES = {
    "colab_a100": HardwareProfile(
        name="colab_a100",
        required_name_tokens=("A100",),
        min_vram_mib=38_000,
    ),
    "organizer_rtx_pro_6000_blackwell": HardwareProfile(
        name="organizer_rtx_pro_6000_blackwell",
        required_name_tokens=("RTX PRO 6000", "Blackwell"),
        # The organizer documents RTX PRO 6000 Blackwell. NVIDIA's full device
        # has 96 GB; require >=90,000 MiB so a smaller vGPU/MIG slice fails closed.
        min_vram_mib=90_000,
    ),
}


def query_gpu_info() -> tuple[str, int]:
    """Return raw GPU identity without imposing the historical A100 contract.

    Historical 72 probe code keeps using m3_batch_probe_common.gpu_info(), which
    intentionally remains A100-only. Organizer wrappers must call this function
    first and then validate the explicitly selected hardware profile.
    """
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    total = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
    )
    return name, total


def selected_profile() -> HardwareProfile:
    value = os.environ.get("PARC_M3_HARDWARE_PROFILE", "colab_a100").strip()
    try:
        return PROFILES[value]
    except KeyError as exc:
        raise RuntimeError(
            f"unknown PARC_M3_HARDWARE_PROFILE={value!r}; allowed={sorted(PROFILES)}"
        ) from exc


def validate_hardware(gpu_name: str, gpu_vram_mib: int) -> HardwareProfile:
    profile = selected_profile()
    normalized = gpu_name.upper()
    missing = [token for token in profile.required_name_tokens if token.upper() not in normalized]
    if missing or gpu_vram_mib < profile.min_vram_mib:
        raise RuntimeError(
            "M3 hardware profile mismatch: "
            f"profile={profile.name} gpu={gpu_name!r} vram_mib={gpu_vram_mib} "
            f"required_tokens={profile.required_name_tokens} min_vram_mib={profile.min_vram_mib}"
        )
    return profile
