#!/usr/bin/env python3
"""Run one M3 screening-evaluation order on organizer Blackwell.

The canonical 76 runner is kept unchanged. This wrapper reconstructs only
reproducible simulator runtimes on NVMe, validates organizer hardware and the
completed benchmark order, and stores all screening/promotion evidence under
the persistent root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
HARDWARE_PROFILE = "organizer_rtx_pro_6000_blackwell"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--attempt", default=os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "organizer-1"))
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _safe_attempt(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(ch not in allowed for ch in value):
        raise ValueError("attempt may contain only letters, digits, '-' and '_'")
    return value


def main() -> int:
    args = parse_args()
    attempt = _safe_attempt(args.attempt)
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("organizer screening requires explicit PARC_M3_EXECUTE=1")
    for key in ("PARC_LOCAL_SCRATCH_ROOT", "PARC_PERSIST_ROOT", "HF_TOKEN"):
        if not os.environ.get(key):
            raise RuntimeError(f"{key} is required")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    root = Path(os.environ["PARC_LOCAL_SCRATCH_ROOT"]).expanduser().resolve()
    persist = Path(os.environ["PARC_PERSIST_ROOT"]).expanduser().resolve()

    benchmark_summary = persist / f"model-benchmark-v1/m3-benchmark-v1/attempt-{attempt}/m3_training_{args.order}_summary.json"
    benchmark = _load(benchmark_summary)
    expected_benchmark = {
        "status": "PASS",
        "stage": "M3_training_order_benchmark",
        "order": args.order,
        "attempt": attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "result_count": 6,
        "ready_for_screening_evaluation": True,
    }
    for key, value in expected_benchmark.items():
        if benchmark.get(key) != value:
            raise RuntimeError(f"benchmark gate mismatch: {key}={benchmark.get(key)!r} != {value!r}")

    os.environ["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_hardware_guard import validate_hardware  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    profile = validate_hardware(gpu_name, gpu_vram_mib)

    env = os.environ.copy()
    env["PY_AI_REPO"] = str(repo)
    env["PARC_ROOT"] = str(root)
    env["PARC_DRIVE_ROOT"] = str(persist)
    env["PARC_LOCAL_SCRATCH_ROOT"] = str(root)
    env["PARC_PERSIST_ROOT"] = str(persist)
    env["PARC_M3_HARDWARE_PROFILE"] = HARDWARE_PROFILE
    env["PARC_M3_BENCHMARK_ATTEMPT"] = attempt
    env["LIBERO_CONFIG_PATH"] = str(root / "m3-libero-config/shared")
    env.setdefault("HF_HOME", str(root / "cache/huggingface"))
    env.setdefault("TORCH_HOME", str(root / "cache/torch"))
    env.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    env.setdefault("UV_CACHE_DIR", str(root / "cache/uv"))
    env.setdefault("PIP_CACHE_DIR", str(root / "cache/pip"))
    env.setdefault("TMPDIR", str(root / "tmp"))
    for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    env["MPLBACKEND"] = "Agg"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    helper_chain = [
        ("tools/benchmark/verify_m3_organizer_handoff.py", "--root", str(persist)),
        ("tools/colab/prepare_m3_disk_headroom.py", "--phase", "pre-setup"),
        ("tools/colab/prepare_m3_libero_noninteractive_config.py",),
        ("tools/colab/prepare_m3_simulator_runtimes.py",),
        ("tools/colab/prepare_m3_mujoco_compat.py",),
        ("tools/colab/prepare_m3_disk_headroom.py", "--phase", "pre-eval"),
    ]
    for parts in helper_chain:
        subprocess.run(
            [sys.executable, "-u", str(repo / parts[0]), *parts[1:]],
            cwd=str(repo),
            env=env,
            check=True,
        )

    runner = repo / "tools/colab/run_m3_screening_evaluation_order.py"
    subprocess.run(
        [sys.executable, "-u", str(runner), "--order", args.order, "--attempt", attempt],
        cwd=str(repo),
        env=env,
        check=True,
    )

    summary_path = persist / f"model-benchmark-v1/m3-screening-eval-v1/attempt-{attempt}/{args.order}/m3_screening_{args.order}_summary.json"
    summary = _load(summary_path)
    expected = {
        "status": "PASS",
        "stage": "M3_screening_evaluation_order",
        "order": args.order,
        "attempt": attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "checkpoint_count": 6,
        "total_episode_records": 480,
        "promotion_record_count": 6,
        "ready_for_forward_reverse_promotion": True,
        "final_800_episode_evaluation_started": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"screening summary mismatch: {key}={summary.get(key)!r} != {value!r}")

    handoff = {
        "schema_version": 1,
        "stage": "M3_organizer_screening_order_handoff",
        "status": "PASS",
        "hardware_profile": profile.name,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "order": args.order,
        "attempt": attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "summary": str(summary_path),
        "checkpoint_count": 6,
        "total_episode_records": 480,
        "ready_for_forward_reverse_promotion": True,
        "next_action": "parc-home-sync data-push before stopping the server",
    }
    out = persist / f"model-benchmark-v1/m3-organizer-migration-v1/screening-{args.order}-attempt-{attempt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"=== M3 ORGANIZER {args.order.upper()} SCREENING: PASS ===", flush=True)
    print(json.dumps(handoff, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
