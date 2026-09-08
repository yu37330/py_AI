#!/usr/bin/env python3
"""Shared helpers for M3 A100 one-optimizer-step batch probes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

EXPECTED_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
TARGET_EFFECTIVE_BATCH = 32


def out(*args: Any) -> None:
    print(*args, flush=True)


def gpu_info() -> tuple[str, int]:
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    total = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
    )
    if "A100" not in name or total < 38000:
        raise RuntimeError(f"M3 batch probe requires NVIDIA A100 with >=38 GiB VRAM; got {name} {total} MiB")
    return name, total


def peak_vram_mib() -> int:
    try:
        return int(
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0]
        )
    except Exception:
        return 0


def load_gate_paths(drive: Path) -> tuple[Path, Path, Path]:
    preflight = drive / "model-benchmark-v1/m3_runner_preflight.json"
    manifest = drive / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json"
    streaming = drive / "openvla-streaming-selected-v1/streaming_bridge_contract.json"
    for path in (preflight, manifest, streaming):
        if not path.is_file():
            raise FileNotFoundError(path)
    pf = json.loads(preflight.read_text(encoding="utf-8"))
    if pf.get("status") != "READY_FOR_BATCH_PROBE" or pf.get("training_started") is not False:
        raise RuntimeError(f"M3 preflight gate is not READY_FOR_BATCH_PROBE: {pf}")
    mf = json.loads(manifest.read_text(encoding="utf-8"))
    if mf.get("episode_ids_sha256") != EXPECTED_HASH:
        raise RuntimeError("selected manifest hash mismatch")
    sc = json.loads(streaming.read_text(encoding="utf-8"))
    if sc.get("status") != "PASS" or sc.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming bridge contract is not PASS")
    if sc.get("source_episode_ids_sha256") != EXPECTED_HASH:
        raise RuntimeError("69c streaming bridge hash mismatch")
    return preflight, manifest, streaming


def grad_accum_for(micro_batch: int) -> int:
    if micro_batch <= 0 or TARGET_EFFECTIVE_BATCH % micro_batch:
        raise ValueError(f"micro batch must divide {TARGET_EFFECTIVE_BATCH}: {micro_batch}")
    return TARGET_EFFECTIVE_BATCH // micro_batch


def classify_failure(text: str) -> str:
    low = text.lower()
    oom_tokens = (
        "cuda out of memory",
        "outofmemoryerror",
        "out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    )
    return "OOM" if any(token in low for token in oom_tokens) else "ERROR"


def extract_loss(text: str) -> float | None:
    patterns = [
        r"(?:loss_value|loss)[=: ]+([0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)",
        r"'loss'\s*:\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
    return None


def run_logged(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path,
) -> tuple[int, str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    out(">>>", " ".join(args))
    start = time.perf_counter()
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as fh:
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            fh.write(line + "\n")
            fh.flush()
            print(line, flush=True)
    rc = proc.wait()
    return rc, "\n".join(lines), time.perf_counter() - start


def write_result(path: Path, result: dict[str, Any]) -> None:
    result.setdefault("schema_version", 1)
    result.setdefault("probe_only", True)
    result.setdefault("benchmark_training_started", False)
    result.setdefault("selected_episode_ids_sha256", EXPECTED_HASH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    out(json.dumps(result, indent=2))


def candidate_result(
    *,
    micro_batch: int,
    rc: int,
    text: str,
    elapsed_sec: float,
    peak_vram: int,
) -> dict[str, Any]:
    return {
        "micro_batch": micro_batch,
        "gradient_accumulation": grad_accum_for(micro_batch),
        "effective_batch_size": TARGET_EFFECTIVE_BATCH,
        "return_code": rc,
        "status": "PASS" if rc == 0 else classify_failure(text),
        "loss": extract_loss(text),
        "elapsed_sec": elapsed_sec,
        "peak_vram_mib": peak_vram,
    }
