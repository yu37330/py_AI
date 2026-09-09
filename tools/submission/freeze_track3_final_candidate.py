#!/usr/bin/env python3
"""Freeze the PARC2026 Track3 final candidate from promotion/evaluation evidence.

This is the deterministic bridge between M3 promotion and the already-merged
submission-integrity builder.  It does not choose a model from training loss and
does not launch a long run automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

D10_VARIANT = "V2_SQRT_BALANCED_RAW"
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"
MODELS = ("pi05", "smolvla", "openvla_oft")
EXECUTE_ENV = "PARC_M3_EXECUTE"
EXECUTE_VALUE = "1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {path}")
    for file in files:
        rel = file.relative_to(path).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        digest = sha256_file(file).encode("ascii")
        h.update(digest)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def validate_source_sha(source_sha: str) -> str:
    source_sha = str(source_sha).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be a full lowercase 40-hex Git commit SHA")
    return source_sha


def validate_promotion(path: Path, candidate: str) -> dict[str, Any]:
    promotion = load_json(path)
    if promotion.get("status") != "READY_FOR_PROMOTION":
        raise ValueError("promotion summary is not READY_FOR_PROMOTION")
    if promotion.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("promotion D10 hash mismatch")
    if promotion.get("primary_metric") != "simulator_success_rate":
        raise ValueError("promotion primary metric drifted")
    if promotion.get("training_loss_used_for_promotion") is not False:
        raise ValueError("promotion summary may not use training loss")
    promoted = promotion.get("promoted_models")
    if not isinstance(promoted, list) or not promoted:
        raise ValueError("promotion summary has no promoted models")
    if candidate not in promoted:
        raise ValueError(f"candidate {candidate} was not promoted: {promoted}")
    if len(promoted) > 2:
        raise ValueError("promotion summary exceeds max two models")
    return promotion


def validate_final_training(path: Path, candidate: str) -> dict[str, Any]:
    result = load_json(path)
    if result.get("status") != "PASS":
        raise ValueError("final training result must be PASS")
    if result.get("model") != candidate:
        raise ValueError("final training model mismatch")
    if result.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("final training D10 hash mismatch")
    if not str(result.get("checkpoint_ref") or "").strip():
        raise ValueError("final training result missing checkpoint_ref")
    if not str(result.get("source_ref") or "").strip():
        raise ValueError("final training result missing source_ref")
    return result


def validate_evaluation(path: Path, candidate: str, checkpoint_ref: str, source_ref: str) -> dict[str, Any]:
    evaluation = load_json(path)
    if evaluation.get("status") != "PASS":
        raise ValueError("final evaluation must be PASS")
    if evaluation.get("stage") != "M3_simulator_evaluation":
        raise ValueError("unexpected final evaluation stage")
    if evaluation.get("model") != candidate:
        raise ValueError("final evaluation model mismatch")
    if evaluation.get("checkpoint_ref") != checkpoint_ref:
        raise ValueError("final evaluation checkpoint mismatch")
    if evaluation.get("source_ref") != source_ref:
        raise ValueError("final evaluation source mismatch")
    if evaluation.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("final evaluation D10 hash mismatch")
    metrics = evaluation.get("metrics")
    required = {
        "simulator_success_rate",
        "steps_to_success",
        "episode_duration",
        "inference_latency",
        "peak_inference_vram",
        "train_wall_time",
        "peak_train_vram",
    }
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        raise ValueError("final evaluation metrics incomplete")
    return evaluation


def validate_manifest(path: Path) -> dict[str, Any]:
    if sha256_file(path) != D10_MANIFEST_SHA256:
        raise ValueError("D10 manifest file SHA mismatch")
    manifest = load_json(path)
    if manifest.get("variant") != D10_VARIANT:
        raise ValueError("D10 manifest variant mismatch")
    if manifest.get("episode_ids_sha256") != D10_HASH:
        raise ValueError("D10 manifest episode hash mismatch")
    return manifest


def build_selection_record(
    *, candidate: str, promotion: dict[str, Any], rationale: str
) -> dict[str, Any]:
    rationale = rationale.strip()
    if not rationale:
        raise ValueError("explicit final-candidate rationale is required")
    ranking = promotion.get("ranking") or []
    return {
        "schema_version": 1,
        "stage": "Track3_final_candidate_selection",
        "status": "SELECTED_FROM_M3_PROMOTION",
        "candidate": candidate,
        "promoted_models": list(promotion["promoted_models"]),
        "promotion_ranking": list(ranking),
        "primary_metric": "simulator_success_rate",
        "training_loss_used_for_selection": False,
        "rationale": rationale,
    }


def build_submission_spec(
    *,
    source_sha: str,
    manifest_path: Path,
    candidate: str,
    artifact_path: Path,
    artifact_sha256: str,
    config_path: Path,
    promotion_path: Path,
    evaluation_path: Path,
    reproduction_command: str,
    checkpoint_ref: str,
) -> dict[str, Any]:
    command = reproduction_command.strip()
    if not command:
        raise ValueError("reproduction command is required")
    return {
        "source_sha": source_sha,
        "dataset": {
            "variant": D10_VARIANT,
            "manifest_path": str(manifest_path),
            "manifest_sha256": D10_MANIFEST_SHA256,
            "episode_ids_sha256": D10_HASH,
        },
        "model": {
            "name": candidate,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "immutable_ref": checkpoint_ref,
            "config_path": str(config_path),
        },
        "benchmark": {"promotion_summary_path": str(promotion_path)},
        "evaluation": {"metrics_path": str(evaluation_path)},
        "reproduction": {"command": command},
    }


def freeze(
    *,
    candidate: str,
    source_sha: str,
    manifest_path: Path,
    promotion_path: Path,
    final_training_path: Path,
    evaluation_path: Path,
    artifact_path: Path,
    config_path: Path,
    reproduction_command: str,
    rationale: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if candidate not in MODELS:
        raise ValueError(f"unknown candidate: {candidate}")
    source_sha = validate_source_sha(source_sha)
    validate_manifest(manifest_path)
    promotion = validate_promotion(promotion_path, candidate)
    training = validate_final_training(final_training_path, candidate)
    evaluation = validate_evaluation(
        evaluation_path,
        candidate,
        str(training["checkpoint_ref"]),
        str(training["source_ref"]),
    )
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    artifact_sha = sha256_tree(artifact_path)
    config_sha = sha256_file(config_path)
    selection = build_selection_record(candidate=candidate, promotion=promotion, rationale=rationale)
    spec = build_submission_spec(
        source_sha=source_sha,
        manifest_path=manifest_path,
        candidate=candidate,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha,
        config_path=config_path,
        promotion_path=promotion_path,
        evaluation_path=evaluation_path,
        reproduction_command=reproduction_command,
        checkpoint_ref=str(training["checkpoint_ref"]),
    )
    freeze_manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "Track3_final_candidate_artifact_freeze",
        "status": "SUBMISSION_ARTIFACT_FROZEN",
        "candidate": candidate,
        "source_sha": source_sha,
        "source_ref": training["source_ref"],
        "checkpoint_ref": training["checkpoint_ref"],
        "selected_episode_ids_sha256": D10_HASH,
        "manifest_sha256": D10_MANIFEST_SHA256,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "promotion_summary_path": str(promotion_path),
        "promotion_summary_sha256": sha256_file(promotion_path),
        "final_training_result_path": str(final_training_path),
        "final_training_result_sha256": sha256_file(final_training_path),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "selection": selection,
        "reproduction_command": reproduction_command.strip(),
        "automatic_upload": False,
    }
    freeze_manifest["freeze_manifest_sha256"] = hashlib.sha256(canonical_json(freeze_manifest)).hexdigest()
    return selection, spec, freeze_manifest


def run_submission_builder(*, repo_root: Path, spec_path: Path, out_dir: Path, final: bool) -> int:
    builder = repo_root / "tools/submission/build_track3_submission_manifest.py"
    cmd = ["python", "-u", str(builder), "--spec", str(spec_path), "--out-dir", str(out_dir)]
    if final:
        cmd.append("--final")
    return subprocess.run(cmd, cwd=str(repo_root), check=False).returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", choices=MODELS, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--promotion", type=Path, required=True)
    p.add_argument("--final-training", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--reproduction-command", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--run-submission-builder", action="store_true")
    p.add_argument("--builder-final", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selection, spec, manifest = freeze(
        candidate=args.candidate,
        source_sha=args.source_sha,
        manifest_path=args.manifest,
        promotion_path=args.promotion,
        final_training_path=args.final_training,
        evaluation_path=args.evaluation,
        artifact_path=args.artifact,
        config_path=args.config,
        reproduction_command=args.reproduction_command,
        rationale=args.rationale,
    )
    selection_path = out_dir / "final_candidate_selection.json"
    spec_path = out_dir / "track3_submission_spec.json"
    manifest_path = out_dir / "submission_artifact_freeze.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "candidate": manifest["candidate"],
        "artifact_sha256": manifest["artifact_sha256"],
        "freeze_manifest_sha256": manifest["freeze_manifest_sha256"],
        "selection": str(selection_path),
        "submission_spec": str(spec_path),
        "freeze_manifest": str(manifest_path),
    }, indent=2))
    if args.run_submission_builder:
        rc = run_submission_builder(
            repo_root=args.repo_root,
            spec_path=spec_path,
            out_dir=out_dir / "submission-package",
            final=args.builder_final,
        )
        if rc != 0:
            raise SystemExit(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
