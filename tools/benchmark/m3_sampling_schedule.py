#!/usr/bin/env python3
"""Deterministic sample schedule for PARC2026 M3 model comparison.

The schedule is deliberately model/framework independent.  For a fixed D10
manifest, episode-length table, seed and sample index, every model resolves the
same `(episode_id, timestep)` training example.  This closes the fairness gap
between "same selected episode pool" and "same sampling policy".

No dataset copy is created.  Episode lengths are read from the existing
LeRobot v3 metadata and a counter-based SHA256 sampler maps global selected
frame positions back to manifest-ordered episodes.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

D10_VARIANT = "V2_SQRT_BALANCED_RAW"
D10_EPISODE_IDS_SHA256 = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_FRAME_COUNT = 1620614
POLICY = "uniform_selected_frames_with_replacement_v1"
POLICY_DOMAIN = "parc2026-m3-schedule-v1"
EQUAL_DATA_SAMPLE_BUDGET = 4800
SEED_SET = (20260906, 20260907)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _counter_position(*, seed: int, sample_index: int, frame_count: int) -> int:
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    payload = (
        f"{POLICY_DOMAIN}|{D10_EPISODE_IDS_SHA256}|{int(seed)}|{int(sample_index)}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % frame_count


def validate_episode_lengths(
    episode_ids: Sequence[int], episode_lengths: Sequence[int]
) -> tuple[list[int], list[int], int]:
    ids = [int(x) for x in episode_ids]
    lengths = [int(x) for x in episode_lengths]
    if len(ids) != 10758:
        raise ValueError(f"expected 10,758 selected episodes, got {len(ids)}")
    if len(ids) != len(set(ids)):
        raise ValueError("selected episode IDs must be unique")
    if len(lengths) != len(ids):
        raise ValueError("episode length count does not match selected episode count")
    if any(length <= 0 for length in lengths):
        raise ValueError("all selected episode lengths must be > 0")
    total = sum(lengths)
    if total != D10_FRAME_COUNT:
        raise ValueError(
            f"selected frame count mismatch: expected {D10_FRAME_COUNT}, got {total}"
        )
    return ids, lengths, total


def cumulative_stops(lengths: Sequence[int]) -> list[int]:
    stops: list[int] = []
    total = 0
    for length in lengths:
        total += int(length)
        stops.append(total)
    return stops


def resolve_global_position(
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    position: int,
) -> dict[str, int]:
    ids, lengths, total = validate_episode_lengths(episode_ids, episode_lengths)
    if not 0 <= int(position) < total:
        raise IndexError(position)
    stops = cumulative_stops(lengths)
    episode_offset = bisect.bisect_right(stops, int(position))
    episode_start = 0 if episode_offset == 0 else stops[episode_offset - 1]
    timestep = int(position) - episode_start
    return {
        "episode_id": ids[episode_offset],
        "timestep": timestep,
    }


def iter_references(
    *,
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    seed: int,
    start_index: int = 0,
) -> Iterator[dict[str, int]]:
    ids, lengths, total = validate_episode_lengths(episode_ids, episode_lengths)
    stops = cumulative_stops(lengths)
    sample_index = int(start_index)
    if sample_index < 0:
        raise ValueError("start_index must be >= 0")
    while True:
        position = _counter_position(
            seed=int(seed), sample_index=sample_index, frame_count=total
        )
        episode_offset = bisect.bisect_right(stops, position)
        episode_start = 0 if episode_offset == 0 else stops[episode_offset - 1]
        yield {
            "sample_index": sample_index,
            "episode_id": ids[episode_offset],
            "timestep": position - episode_start,
        }
        sample_index += 1


def build_equal_data_schedule(
    *,
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    seed: int,
    sample_budget: int = EQUAL_DATA_SAMPLE_BUDGET,
) -> dict[str, Any]:
    if int(sample_budget) != EQUAL_DATA_SAMPLE_BUDGET:
        raise ValueError(
            f"M3 equal-data sample budget is frozen at {EQUAL_DATA_SAMPLE_BUDGET}"
        )
    ids, lengths, total = validate_episode_lengths(episode_ids, episode_lengths)
    refs_iter = iter_references(
        episode_ids=ids,
        episode_lengths=lengths,
        seed=int(seed),
    )
    references = [next(refs_iter) for _ in range(int(sample_budget))]
    refs_hash = sha256_json(references)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M3_equal_data_sampling_schedule",
        "status": "FROZEN",
        "policy": POLICY,
        "seed": int(seed),
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_EPISODE_IDS_SHA256,
        "selected_frame_count": total,
        "sample_budget": int(sample_budget),
        "references": references,
        "references_sha256": refs_hash,
    }
    payload["schedule_sha256"] = sha256_json(payload)
    return payload


def load_d10_episode_lengths(
    *, manifest_path: Path, dataset_root: Path
) -> tuple[list[int], list[int]]:
    """Load manifest-ordered D10 episode IDs and exact LeRobot episode lengths."""
    # Reuse the already-validated selected-pool loader instead of reimplementing
    # D10 provenance checks here.
    from tools.data.openvla_lerobot_streaming import (  # noqa: PLC0415
        load_episode_metadata,
        load_manifest,
    )

    manifest = load_manifest(Path(manifest_path))
    if manifest.get("episode_ids_sha256") != D10_EPISODE_IDS_SHA256:
        raise ValueError("D10 episode hash mismatch")
    ids = [int(x) for x in manifest["episode_ids"]]
    metadata = load_episode_metadata(Path(dataset_root), ids)
    lengths = [int(metadata.loc[episode_id]["length"]) for episode_id in ids]
    validate_episode_lengths(ids, lengths)
    return ids, lengths


def write_schedule(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEED_SET, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ids, lengths = load_d10_episode_lengths(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
    )
    payload = build_equal_data_schedule(
        episode_ids=ids,
        episode_lengths=lengths,
        seed=args.seed,
    )
    write_schedule(args.out, payload)
    print(json.dumps({
        "status": "PASS",
        "seed": payload["seed"],
        "sample_budget": payload["sample_budget"],
        "references_sha256": payload["references_sha256"],
        "schedule_sha256": payload["schedule_sha256"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
