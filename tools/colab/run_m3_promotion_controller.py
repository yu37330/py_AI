#!/usr/bin/env python3
"""Aggregate complete forward/reverse M3 screening evidence for promotion.

CPU-only controller. It requires all 12 screening records (2 orders x 2 tracks
x 3 models) for a normal complete comparison and delegates ranking to the
already-frozen `aggregate_m3_results.py` contract. It never trains, evaluates,
or uploads anything.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
ORDERS = ("forward", "reverse")
TRACKS = ("equal_data", "equal_wall")
MODELS = ("pi05", "smolvla", "openvla_oft")
EXPECTED_RECORDS = 12
SCREENING_EPISODES_PER_CHECKPOINT = 80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attempt", default=os.environ.get("PARC_M3_BENCHMARK_ATTEMPT", "1"))
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def main() -> int:
    args = parse_args()
    if not args.attempt or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in args.attempt):
        raise ValueError("--attempt may contain only letters, digits, '-' and '_'")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).resolve()
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    screening_root = drive / f"model-benchmark-v1/m3-screening-eval-v1/attempt-{args.attempt}"
    record_root = drive / f"model-benchmark-v1/m3-promotion-records-v1/attempt-{args.attempt}"
    out_root = drive / f"model-benchmark-v1/m3-promotion-v1/attempt-{args.attempt}"
    out_path = out_root / "m3_promotion_summary.json"
    inventory_path = out_root / "m3_promotion_input_inventory.json"

    sys.path.insert(0, str(repo))
    from tools.benchmark.aggregate_m3_results import aggregate  # noqa: PLC0415

    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for order in ORDERS:
        order_summary_path = screening_root / order / f"m3_screening_{order}_summary.json"
        if not order_summary_path.is_file():
            raise FileNotFoundError(order_summary_path)
        order_summary = _load(order_summary_path)
        if order_summary.get("status") != "PASS" or order_summary.get("order") != order:
            raise RuntimeError(f"screening order summary is not PASS: {order_summary_path}")
        if order_summary.get("selected_episode_ids_sha256") != D10_HASH:
            raise RuntimeError(f"screening order summary D10 mismatch: {order}")
        if int(order_summary.get("episodes_per_checkpoint", -1)) != SCREENING_EPISODES_PER_CHECKPOINT:
            raise RuntimeError(f"screening episode protocol drifted for {order}")
        if int(order_summary.get("promotion_record_count", -1)) != 6:
            raise RuntimeError(f"screening promotion record count mismatch for {order}")
        if order_summary.get("ready_for_forward_reverse_promotion") is not True:
            raise RuntimeError(f"screening order is not promotion-ready: {order}")

        for track in TRACKS:
            for model in MODELS:
                path = record_root / order / track / model / "promotion_record.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                record = _load(path)
                expected = {
                    "status": "PASS",
                    "stage": "M3_screening_promotion_record",
                    "model": model,
                    "order": order,
                    "track": track,
                    "selected_episode_ids_sha256": D10_HASH,
                    "evaluation_episode_count": SCREENING_EPISODES_PER_CHECKPOINT,
                }
                for key, value in expected.items():
                    if record.get(key) != value:
                        raise RuntimeError(
                            f"promotion input mismatch {path}: {key}={record.get(key)!r} != {value!r}"
                        )
                records.append(record)
                inventory.append({
                    "model": model,
                    "order": order,
                    "track": track,
                    "path": str(path.resolve()),
                    "promotion_record_sha256": record.get("promotion_record_sha256"),
                    "checkpoint_ref": record.get("checkpoint_ref"),
                    "evaluation_episode_count": record.get("evaluation_episode_count"),
                })

    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"promotion input count mismatch: {len(records)} != {EXPECTED_RECORDS}")
    identities = {(r["model"], r["order"], r["track"]) for r in records}
    if len(identities) != EXPECTED_RECORDS:
        raise RuntimeError("duplicate/missing forward/reverse promotion inputs")

    summary = aggregate(records)
    if summary.get("status") != "READY_FOR_PROMOTION":
        raise RuntimeError(f"promotion controller did not reach READY_FOR_PROMOTION: {summary.get('status')}")
    promoted = summary.get("promoted_models")
    if not isinstance(promoted, list) or not (1 <= len(promoted) <= 2):
        raise RuntimeError(f"invalid promotion set: {promoted}")
    if summary.get("primary_metric") != "simulator_success_rate":
        raise RuntimeError("promotion primary metric drifted")
    if summary.get("training_loss_used_for_promotion") is not False:
        raise RuntimeError("training loss was incorrectly used for promotion")

    out_root.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inventory = sorted(inventory, key=lambda r: (r["model"], r["order"], r["track"]))
    inventory_payload = {
        "schema_version": 1,
        "stage": "M3_forward_reverse_promotion_input_inventory",
        "status": "PASS",
        "attempt": args.attempt,
        "selected_episode_ids_sha256": D10_HASH,
        "record_count": len(inventory),
        "screening_episodes_per_checkpoint": SCREENING_EPISODES_PER_CHECKPOINT,
        "total_screening_episode_records": EXPECTED_RECORDS * SCREENING_EPISODES_PER_CHECKPOINT,
        "records": inventory,
        "automatic_training": False,
        "automatic_evaluation": False,
        "automatic_upload": False,
    }
    inventory_path.write_text(json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== 77 M3 FORWARD/REVERSE PROMOTION: READY ===", flush=True)
    print(json.dumps({
        "status": summary["status"],
        "ranking": summary.get("ranking"),
        "promoted_models": promoted,
        "primary_metric": summary["primary_metric"],
        "screening_records": EXPECTED_RECORDS,
        "screening_episodes_total": EXPECTED_RECORDS * SCREENING_EPISODES_PER_CHECKPOINT,
        "promotion_summary": str(out_path),
        "input_inventory": str(inventory_path),
        "next": "select final candidate; prepare immutable final-run config via PR #68",
        "automatic_final_training": False,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
