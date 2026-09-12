#!/usr/bin/env python3
"""Prepare the immutable final-run configuration for a promoted Track3 model.

This tool does not start training. It binds the promoted model to the frozen
candidate recipe, source revision, D10 identity, sampling policy, and one of the
already-frozen M3 budgets. The resulting JSON is the authority that a later
final training result must link back to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tools.submission.freeze_track3_final_candidate import (
    D10_HASH,
    D10_VARIANT,
    validate_promotion,
)

TRAINING_SCHEDULE_SEED = 20260906
EFFECTIVE_BATCH = 32
EQUAL_DATA_SAMPLES = 4800
EQUAL_DATA_UPDATES = 150
EQUAL_WALL_SEC = 1800.0
CONFIG_PATHS = {
    "pi05": "experiments/configs/pi05.yaml",
    "smolvla": "experiments/configs/smolvla.yaml",
    "openvla_oft": "experiments/configs/openvla_oft.yaml",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _promotion_source_ref(promotion: dict[str, Any], candidate: str) -> str:
    summary = promotion["model_summaries"][candidate]
    refs = {
        str(record.get("source_ref") or "").strip()
        for record in summary.get("records", [])
        if isinstance(record, dict)
    }
    refs.discard("")
    if len(refs) != 1:
        raise ValueError(f"promotion source_ref is not unique for {candidate}: {sorted(refs)}")
    return next(iter(refs))


def _registry_source_ref(repo_root: Path, candidate: str) -> tuple[str, Path]:
    path = repo_root / "experiments/model_registry_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError("model registry missing models")
    matches = [item for item in models if isinstance(item, dict) and item.get("id") == candidate]
    if len(matches) != 1:
        raise ValueError(f"model registry must contain exactly one {candidate} entry")
    item = matches[0]
    ref = str(item.get("upstream_revision") or item.get("framework_revision") or "").strip()
    if not ref:
        raise ValueError(f"model registry source ref missing for {candidate}")
    return ref, path


def _yaml_top_level_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.+?)\s*$")
    matches: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "\t")):
            continue
        match = pattern.match(line)
        if match:
            value = match.group(1).strip().strip('"\'')
            matches.append(value)
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"expected one top-level {key} in {path}")
    return matches[0]


def _budget(track: str) -> dict[str, Any]:
    if track == "equal_data":
        return {
            "track": "equal_data",
            "sample_budget": EQUAL_DATA_SAMPLES,
            "optimizer_updates": EQUAL_DATA_UPDATES,
            "effective_batch_size": EFFECTIVE_BATCH,
            "zero_overshoot": True,
        }
    if track == "equal_wall":
        return {
            "track": "equal_wall",
            "train_loop_sec": EQUAL_WALL_SEC,
            "effective_batch_size": EFFECTIVE_BATCH,
            "stop_at_first_safe_optimizer_boundary_at_or_after_target": True,
            "record_actual_samples_consumed": True,
        }
    raise ValueError(f"unsupported final-run track: {track}")


def build_final_run_config(
    *, repo_root: Path, promotion_path: Path, candidate: str, track: str
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    promotion_path = promotion_path.resolve()
    promotion = validate_promotion(promotion_path, candidate)
    promotion_ref = _promotion_source_ref(promotion, candidate)
    registry_ref, registry_path = _registry_source_ref(repo_root, candidate)
    recipe_path = repo_root / CONFIG_PATHS[candidate]
    if not recipe_path.is_file():
        raise FileNotFoundError(recipe_path)
    recipe_ref = _yaml_top_level_scalar(recipe_path, "source_ref")
    if not (promotion_ref == registry_ref == recipe_ref):
        raise ValueError(
            "final-run source provenance mismatch: "
            f"promotion={promotion_ref!r}, registry={registry_ref!r}, recipe={recipe_ref!r}"
        )

    config: dict[str, Any] = {
        "schema_version": 1,
        "stage": "Track3_final_candidate_run_config",
        "status": "READY_FOR_FINAL_RUN",
        "candidate": candidate,
        "source_ref": promotion_ref,
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_HASH,
        "promotion_summary_path": str(promotion_path),
        "promotion_summary_sha256": _sha256_file(promotion_path),
        "promotion_primary_metric": "simulator_success_rate",
        "training_loss_used_for_selection": False,
        "recipe": {
            "path": str(recipe_path.resolve()),
            "sha256": _sha256_file(recipe_path),
        },
        "model_registry": {
            "path": str(registry_path.resolve()),
            "sha256": _sha256_file(registry_path),
        },
        "sampling": {
            "policy": "uniform_selected_frames_with_replacement_v1",
            "training_schedule_seed": TRAINING_SCHEDULE_SEED,
            "native_random_sampler_allowed": False,
        },
        "budget": _budget(track),
        "batch_config_source": "m3_batch_probe_summary.json",
        "execution_guard": {"environment_variable": "PARC_M3_EXECUTE", "required_value": "1"},
        "automatic_training": False,
        "automatic_evaluation": False,
        "automatic_upload": False,
    }
    config["final_run_config_sha256"] = _sha256_bytes(_canonical_json(config))
    return config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--promotion", type=Path, required=True)
    p.add_argument("--candidate", choices=tuple(CONFIG_PATHS), required=True)
    p.add_argument("--track", choices=("equal_data", "equal_wall"), required=True)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = build_final_run_config(
        repo_root=args.repo_root,
        promotion_path=args.promotion,
        candidate=args.candidate,
        track=args.track,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": config["status"],
        "candidate": config["candidate"],
        "track": config["budget"]["track"],
        "source_ref": config["source_ref"],
        "final_run_config_sha256": config["final_run_config_sha256"],
        "out": str(args.out),
        "automatic_training": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
