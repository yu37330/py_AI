#!/usr/bin/env python3
"""Instrumentation helpers for PARC2026 M3 LIBERO evaluation.

The helpers are deliberately policy-class preserving: ``instrument_select_action``
replaces the bound ``select_action`` method on the existing policy instance so
upstream evaluators that require a concrete PreTrainedPolicy keep working.
"""
from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

MAX_STEPS = 300


@dataclass
class InferenceRecorder:
    latencies_ms: list[float] = field(default_factory=list)
    peak_vram_mib: float = 0.0

    @property
    def mean_latency_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0


@contextlib.contextmanager
def instrument_select_action(policy: Any) -> Iterator[InferenceRecorder]:
    """Measure synchronized select_action latency without changing policy type."""
    import torch

    original = policy.select_action
    recorder = InferenceRecorder()
    cuda = torch.cuda.is_available()
    device = None
    if cuda:
        try:
            device = next(policy.parameters()).device
        except Exception:
            device = torch.device("cuda")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def wrapped(*args, **kwargs):
        if cuda:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = original(*args, **kwargs)
        if cuda:
            torch.cuda.synchronize(device)
        recorder.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if cuda:
            recorder.peak_vram_mib = max(
                recorder.peak_vram_mib,
                float(torch.cuda.max_memory_allocated(device) / (1024**2)),
            )
        return result

    policy.select_action = wrapped
    try:
        yield recorder
    finally:
        policy.select_action = original


def build_episode_record(
    *,
    model: str,
    checkpoint_ref: str,
    source_ref: str,
    order: str,
    track: str,
    seed: int,
    task_id: str,
    success: bool,
    steps: int,
    episode_duration_sec: float,
    mean_inference_latency_ms: float,
    peak_inference_vram_mib: float,
) -> dict[str, Any]:
    steps = int(steps)
    if not 0 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be within [0,{MAX_STEPS}]")
    if success and steps <= 0:
        raise ValueError("successful episode must have positive steps")
    return {
        "model": model,
        "checkpoint_ref": str(checkpoint_ref),
        "source_ref": str(source_ref),
        "order": order,
        "track": track,
        "seed": int(seed),
        "task_id": str(task_id),
        "success": bool(success),
        "steps": steps,
        "episode_duration_sec": float(episode_duration_sec),
        "mean_inference_latency_ms": float(mean_inference_latency_ms),
        "peak_inference_vram_mib": float(peak_inference_vram_mib),
    }
