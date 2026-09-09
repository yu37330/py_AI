#!/usr/bin/env python3
"""Aggregate M3 forward/reverse benchmark evidence and select promotions.

This controller intentionally refuses to promote from training loss or from a
single execution direction.  A promotable model must provide the complete
forward/reverse x equal-data/equal-wall matrix with matching D10, sampling
schedule, seed and source provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

D10_VARIANT = "V2_SQRT_BALANCED_RAW"
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
MODELS = ("pi05", "smolvla", "openvla_oft")
ORDERS = ("forward", "reverse")
TRACKS = ("equal_data", "equal_wall")
SEED_SET = (20260906, 20260907)
EQUAL_DATA_SAMPLES = 4800
EQUAL_WALL_SEC = 1800.0
MAX_PROMOTIONS = 2
REQUIRED_METRICS = (
    "simulator_success_rate",
    "steps_to_success",
    "episode_duration",
    "inference_latency",
    "peak_inference_vram",
    "train_wall_time",
    "peak_train_vram",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def validate_pass_record(record: dict[str, Any]) -> dict[str, Any]:
    model = record.get("model")
    order = record.get("order")
    track = record.get("track")
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    if order not in ORDERS:
        raise ValueError(f"unknown order: {order}")
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track}")
    if record.get("status") != "PASS":
        raise ValueError("validate_pass_record expects status=PASS")
    if record.get("selected_dataset_variant") != D10_VARIANT:
        raise ValueError("selected dataset variant mismatch")
    if record.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("D10 episode hash mismatch")
    if not str(record.get("source_ref") or "").strip():
        raise ValueError("source_ref is required")
    schedule_hash = str(record.get("sampling_schedule_sha256") or "").strip()
    if not schedule_hash:
        raise ValueError("sampling_schedule_sha256 is required")
    seed_set = tuple(int(x) for x in record.get("seed_set", []))
    if seed_set != SEED_SET:
        raise ValueError(f"seed_set mismatch: {seed_set}")

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    normalized_metrics = {
        name: _number(metrics.get(name), f"metrics.{name}")
        for name in REQUIRED_METRICS
    }
    success = normalized_metrics["simulator_success_rate"]
    if not 0.0 <= success <= 1.0:
        raise ValueError("simulator_success_rate must be within [0,1]")
    for name in (
        "steps_to_success",
        "episode_duration",
        "inference_latency",
        "peak_inference_vram",
        "train_wall_time",
        "peak_train_vram",
    ):
        if normalized_metrics[name] < 0:
            raise ValueError(f"metrics.{name} must be >= 0")

    samples_consumed = int(record.get("samples_consumed", -1))
    if samples_consumed < 0:
        raise ValueError("samples_consumed must be >= 0")
    if track == "equal_data" and samples_consumed != EQUAL_DATA_SAMPLES:
        raise ValueError(
            f"equal_data must consume exactly {EQUAL_DATA_SAMPLES} samples"
        )
    if track == "equal_wall" and normalized_metrics["train_wall_time"] < EQUAL_WALL_SEC:
        raise ValueError(
            f"equal_wall train_wall_time must be >= {EQUAL_WALL_SEC} sec"
        )

    normalized = dict(record)
    normalized["seed_set"] = list(SEED_SET)
    normalized["sampling_schedule_sha256"] = schedule_hash
    normalized["samples_consumed"] = samples_consumed
    normalized["metrics"] = normalized_metrics
    return normalized


def validate_exclusion(record: dict[str, Any]) -> dict[str, Any]:
    model = record.get("model")
    if model not in MODELS:
        raise ValueError(f"unknown excluded model: {model}")
    if record.get("status") != "EXCLUDED_WITH_EVIDENCE":
        raise ValueError("validate_exclusion expects EXCLUDED_WITH_EVIDENCE")
    if record.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("excluded record D10 hash mismatch")
    evidence = str(record.get("evidence") or "").strip()
    if not evidence:
        raise ValueError("excluded model requires evidence")
    out = dict(record)
    out["evidence"] = evidence
    return out


def _mean(records: list[dict[str, Any]], metric: str) -> float:
    values = [float(record["metrics"][metric]) for record in records]
    return sum(values) / len(values)


def _model_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_combo = {(r["order"], r["track"]): r for r in records}
    expected = {(order, track) for order in ORDERS for track in TRACKS}
    if set(by_combo) != expected:
        missing = sorted(expected - set(by_combo))
        extra = sorted(set(by_combo) - expected)
        raise ValueError(f"incomplete result matrix: missing={missing}, extra={extra}")

    source_refs = {r["source_ref"] for r in records}
    if len(source_refs) != 1:
        raise ValueError(f"source_ref changed across runs: {sorted(source_refs)}")
    seed_sets = {tuple(r["seed_set"]) for r in records}
    if seed_sets != {SEED_SET}:
        raise ValueError(f"seed_set changed across runs: {seed_sets}")

    track_schedule_hashes: dict[str, str] = {}
    order_effect: dict[str, dict[str, float]] = {}
    track_metrics: dict[str, dict[str, float]] = {}
    for track in TRACKS:
        pair = [by_combo[(order, track)] for order in ORDERS]
        hashes = {r["sampling_schedule_sha256"] for r in pair}
        if len(hashes) != 1:
            raise ValueError(f"sampling schedule differs by order for {track}: {hashes}")
        track_schedule_hashes[track] = next(iter(hashes))
        track_metrics[track] = {
            metric: _mean(pair, metric) for metric in REQUIRED_METRICS
        }
        forward = by_combo[("forward", track)]
        reverse = by_combo[("reverse", track)]
        order_effect[track] = {
            metric: float(reverse["metrics"][metric])
            - float(forward["metrics"][metric])
            for metric in REQUIRED_METRICS
        }
        order_effect[track]["samples_consumed_delta"] = (
            int(reverse["samples_consumed"]) - int(forward["samples_consumed"])
        )

    overall = {metric: _mean(records, metric) for metric in REQUIRED_METRICS}
    return {
        "status": "PASS",
        "source_ref": next(iter(source_refs)),
        "schedule_hashes": track_schedule_hashes,
        "records": sorted(records, key=lambda r: (r["order"], r["track"])),
        "track_metrics": track_metrics,
        "overall_metrics": overall,
        "order_effect_reverse_minus_forward": order_effect,
    }


def aggregate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    pass_records: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    exclusions: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}

    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("every result record must be an object")
        status = raw.get("status")
        model = raw.get("model")
        if model not in MODELS:
            raise ValueError(f"unknown model: {model}")
        if status == "PASS":
            pass_records[model].append(validate_pass_record(raw))
        elif status == "EXCLUDED_WITH_EVIDENCE":
            exclusions[model] = validate_exclusion(raw)
        elif status == "FAILED":
            evidence = str(raw.get("evidence") or raw.get("error") or "").strip()
            if not evidence:
                raise ValueError("FAILED record requires evidence or error")
            failures[model].append(dict(raw))
        else:
            raise ValueError(f"unsupported status: {status}")

    model_summaries: dict[str, dict[str, Any]] = {}
    promotable: list[str] = []
    for model in MODELS:
        if model in exclusions:
            model_summaries[model] = {
                "status": "EXCLUDED_WITH_EVIDENCE",
                "evidence": exclusions[model]["evidence"],
            }
            continue
        if failures[model]:
            model_summaries[model] = {
                "status": "FAILED",
                "failures": failures[model],
            }
            continue
        if not pass_records[model]:
            model_summaries[model] = {
                "status": "MISSING",
                "reason": "no PASS/FAILED/EXCLUDED evidence supplied",
            }
            continue
        try:
            summary = _model_aggregate(pass_records[model])
        except ValueError as exc:
            model_summaries[model] = {
                "status": "INCOMPLETE",
                "reason": str(exc),
            }
            continue
        model_summaries[model] = summary
        promotable.append(model)

    def ranking_key(model: str):
        metrics = model_summaries[model]["overall_metrics"]
        return (
            -float(metrics["simulator_success_rate"]),
            float(metrics["steps_to_success"]),
            float(metrics["inference_latency"]),
            float(metrics["peak_inference_vram"]),
            model,
        )

    ranked = sorted(promotable, key=ranking_key)
    promoted = ranked[:MAX_PROMOTIONS]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M3_forward_reverse_promotion",
        "status": "READY_FOR_PROMOTION" if promoted else "NOT_READY",
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_HASH,
        "required_orders": list(ORDERS),
        "required_tracks": list(TRACKS),
        "seed_set": list(SEED_SET),
        "primary_metric": "simulator_success_rate",
        "training_loss_used_for_promotion": False,
        "max_promotions": MAX_PROMOTIONS,
        "model_summaries": model_summaries,
        "ranking": ranked,
        "promoted_models": promoted,
    }
    payload["summary_sha256"] = _sha256_json(payload)
    return payload


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict) and isinstance(data.get("records"), list):
            records.extend(data["records"])
        elif isinstance(data, dict):
            records.append(data)
        else:
            raise ValueError(f"unsupported result JSON shape: {path}")
    return records


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = aggregate(load_records(args.input))
    write_summary(args.out, summary)
    print(json.dumps({
        "status": summary["status"],
        "ranking": summary["ranking"],
        "promoted_models": summary["promoted_models"],
        "summary_sha256": summary["summary_sha256"],
        "out": str(args.out),
    }, indent=2))
    return 0 if summary["status"] == "READY_FOR_PROMOTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
