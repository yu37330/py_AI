#!/usr/bin/env python3
"""Read-only progress summary for PARC2026 organizer GPU M3 execution."""
from __future__ import annotations

import json
import os
from pathlib import Path


MODELS = ("pi05", "smolvla", "openvla_oft")
TRACKS = ("equal_data", "equal_wall")
ORDERS = ("forward", "reverse")


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "UNREADABLE"}
    return data if isinstance(data, dict) else {"status": "INVALID"}


def _flag(label: str, path: Path, *, accepted: tuple[str, ...] = ("PASS",)) -> bool:
    data = _load(path)
    status = str((data or {}).get("status", "MISSING"))
    ok = status in accepted
    mark = "[x]" if ok else "[ ]"
    print(f"{mark} {label:<24} {status:<20} {path}")
    return ok


def main() -> int:
    persist = Path(os.environ.get("PARC_PERSIST_ROOT", str(Path.home() / "data/parc2026-cache"))).expanduser().resolve()
    attempt = os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "organizer-1")
    smoke_attempt = os.environ.get("PARC_M3_SIM_SMOKE_ATTEMPT", "organizer-1")
    compat_attempt = os.environ.get("PARC_M3_TRAINING_COMPAT_ATTEMPT", "organizer-1")

    print("=== PARC2026 ORGANIZER M3 PROGRESS ===")
    print(f"persistent: {persist}")
    print(f"benchmark attempt: {attempt}")
    print()

    dataset = _flag(
        "dataset readiness",
        persist / "model-benchmark-v1/m3-organizer-migration-v1/dataset_readiness.json",
    )
    smoke = _flag(
        "74 simulator smoke",
        persist
        / f"model-benchmark-v1/m3-simulator-minimal-smoke-v1/attempt-{smoke_attempt}/m3_minimal_simulator_smoke_summary.json",
    )
    compat = _flag(
        "Blackwell compat",
        persist
        / f"model-benchmark-v1/m3-organizer-training-compat-v1/attempt-{compat_attempt}/m3_organizer_training_compat_summary.json",
    )

    training: dict[str, bool] = {}
    screening: dict[str, bool] = {}
    for order in ORDERS:
        training[order] = _flag(
            f"75 {order}",
            persist
            / f"model-benchmark-v1/m3-benchmark-v1/attempt-{attempt}/m3_training_{order}_summary.json",
        )
        screening[order] = _flag(
            f"76 {order} final",
            persist
            / f"model-benchmark-v1/m3-screening-eval-v1/attempt-{attempt}/{order}/m3_screening_{order}_summary.json",
        )

    promotion = _flag(
        "77 promotion",
        persist / f"model-benchmark-v1/m3-promotion-v1/attempt-{attempt}/m3_promotion_summary.json",
        accepted=("READY_FOR_PROMOTION",),
    )

    print()
    unit_done = 0
    unit_total = len(ORDERS) * len(TRACKS) * len(MODELS)
    for order in ORDERS:
        for track in TRACKS:
            for model in MODELS:
                path = (
                    persist
                    / f"model-benchmark-v1/m3-screening-eval-v1/attempt-{attempt}"
                    / order
                    / track
                    / model
                    / "m3_screening_evaluation_summary.json"
                )
                data = _load(path)
                if data and data.get("status") == "PASS" and int(data.get("episode_count", -1)) == 80:
                    unit_done += 1
    print(f"76 checkpoint units: {unit_done}/{unit_total} PASS")

    print()
    if promotion:
        next_step = "PR #68 final-candidate / 800-episode path"
    elif screening["forward"] and screening["reverse"]:
        next_step = "77 promotion"
    elif training["forward"] and training["reverse"]:
        next_step = "76 checkpoint units / order finalizers"
    elif smoke and compat and dataset and training["forward"]:
        next_step = "75 reverse (rebuild NVMe dataset/runtime first if this is a new session)"
    elif smoke and compat and dataset:
        next_step = "75 forward"
    elif smoke and dataset:
        next_step = "Blackwell training compatibility"
    elif smoke:
        next_step = "exact dataset on NVMe"
    else:
        next_step = "restore/verify handoff, then 74 simulator smoke"
    print(f"NEXT: {next_step}")
    print("Read-only command: no training/evaluation/upload was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
