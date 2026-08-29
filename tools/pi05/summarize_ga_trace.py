#!/usr/bin/env python3
"""Summarize a PARC pi0.5 gradient-accumulation runtime trace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--expected-ga", type=int, required=True)
    ap.add_argument("--expected-steps", type=int, required=True)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    rows = load_rows(args.trace)
    counts = Counter(r.get("event") for r in rows)

    backward = [r for r in rows if r.get("event") == "backward"]
    accel_steps = [
        r for r in rows if r.get("event") == "accelerated_optimizer_step_call"
    ]
    real_steps = [r for r in rows if r.get("event") == "underlying_optimizer_step"]
    errors = [r for r in rows if r.get("event") == "instrumentation_error"]

    sync_backward = sum(bool(r.get("sync_gradients")) for r in backward)
    sync_accel_steps = sum(bool(r.get("sync_gradients")) for r in accel_steps)

    first_accel_seq = min((int(r["seq"]) for r in accel_steps), default=10**18)
    training_scheduler_steps = [
        r
        for r in rows
        if r.get("event") == "scheduler_step" and int(r.get("seq", 0)) > first_accel_seq
    ]

    expected_micro = args.expected_ga * args.expected_steps
    checks = {
        "backward_calls_eq_steps_x_ga": len(backward) == expected_micro,
        "accelerated_step_calls_eq_steps_x_ga": len(accel_steps) == expected_micro,
        "sync_backward_eq_steps": sync_backward == args.expected_steps,
        "sync_accelerated_steps_eq_steps": sync_accel_steps == args.expected_steps,
        "underlying_optimizer_steps_eq_steps": len(real_steps) == args.expected_steps,
        "training_scheduler_steps_eq_steps": len(training_scheduler_steps)
        == args.expected_steps,
        "no_instrumentation_errors": not errors,
    }

    summary = {
        "trace": str(args.trace),
        "expected_ga": args.expected_ga,
        "expected_optimizer_steps": args.expected_steps,
        "expected_micro_steps": expected_micro,
        "observed": {
            "backward_calls": len(backward),
            "backward_sync_true": sync_backward,
            "accelerated_optimizer_step_calls": len(accel_steps),
            "accelerated_optimizer_step_sync_true": sync_accel_steps,
            "underlying_optimizer_steps": len(real_steps),
            "scheduler_steps_after_training_started": len(training_scheduler_steps),
            "event_counts": dict(counts),
        },
        "checks": checks,
        "pass": all(checks.values()),
        "instrumentation_errors": errors,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
