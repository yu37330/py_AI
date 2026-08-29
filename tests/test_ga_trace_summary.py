"""Gradient-accumulation trace summary tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_ga_trace_passes_for_two_microsteps_per_update(tmp_path: Path):
    rows = [
        {"seq": 1, "event": "sitecustomize_loaded"},
        {"seq": 2, "event": "scheduler_step"},  # scheduler init
        {"seq": 3, "event": "backward", "sync_gradients": False},
        {"seq": 4, "event": "accelerated_optimizer_step_call", "sync_gradients": False},
        {"seq": 5, "event": "backward", "sync_gradients": True},
        {"seq": 6, "event": "accelerated_optimizer_step_call", "sync_gradients": True},
        {"seq": 7, "event": "underlying_optimizer_step", "optimizer": "AdamW"},
        {"seq": 8, "event": "scheduler_step"},
        {"seq": 9, "event": "backward", "sync_gradients": False},
        {"seq": 10, "event": "accelerated_optimizer_step_call", "sync_gradients": False},
        {"seq": 11, "event": "backward", "sync_gradients": True},
        {"seq": 12, "event": "accelerated_optimizer_step_call", "sync_gradients": True},
        {"seq": 13, "event": "underlying_optimizer_step", "optimizer": "AdamW"},
        {"seq": 14, "event": "scheduler_step"},
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            str(_ROOT / "tools/pi05/summarize_ga_trace.py"),
            str(trace),
            "--expected-ga",
            "2",
            "--expected-steps",
            "2",
            "--json-out",
            str(out),
        ],
        check=True,
    )

    summary = json.loads(out.read_text())
    assert summary["pass"] is True
    assert summary["observed"]["backward_calls"] == 4
    assert summary["observed"]["underlying_optimizer_steps"] == 2
    assert summary["observed"]["scheduler_steps_after_training_started"] == 2
