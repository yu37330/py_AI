#!/usr/bin/env python3
"""Canonical final Track3 freeze entry after promoted final training/evaluation.

This wrapper closes the provenance chain:
promotion -> final-run config -> linked training result -> 800-episode evaluation
-> immutable submission freeze. It never starts training/evaluation/upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.submission.freeze_track3_final_candidate import (
    D10_HASH,
    EXPECTED_FINAL_EVAL_EPISODES,
    canonical_json,
    freeze,
    load_json,
    run_submission_builder,
    sha256_file,
    validate_final_training,
    validate_promotion,
)
from tools.submission.link_track3_final_training_result import validate_final_run_config


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


def validate_chain(
    *,
    candidate: str,
    promotion_path: Path,
    final_run_config_path: Path,
    linked_training_path: Path,
    evaluation_path: Path,
    model_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    promotion = validate_promotion(promotion_path, candidate)
    final_cfg = validate_final_run_config(final_run_config_path)
    if final_cfg.get("candidate") != candidate:
        raise ValueError("final-run config candidate mismatch")
    if final_cfg.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("final-run config D10 mismatch")
    if final_cfg.get("promotion_summary_sha256") != sha256_file(promotion_path):
        raise ValueError("promotion summary changed after final-run config was prepared")
    promotion_ref = _promotion_source_ref(promotion, candidate)
    if final_cfg.get("source_ref") != promotion_ref:
        raise ValueError("final-run source_ref differs from promotion source_ref")
    recipe = final_cfg.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("final-run config recipe provenance missing")
    if Path(str(recipe.get("path") or "")).resolve() != model_config_path.resolve():
        raise ValueError("model config path differs from final-run recipe path")
    if recipe.get("sha256") != sha256_file(model_config_path):
        raise ValueError("model config changed after final-run config was prepared")

    training = validate_final_training(linked_training_path, candidate)
    if training.get("final_candidate_run") is not True:
        raise ValueError("final training result is not linked as a final candidate run")
    if training.get("final_run_config_sha256") != final_cfg.get("final_run_config_sha256"):
        raise ValueError("final training/result final-run config hash mismatch")
    config_ref = str(training.get("final_run_config_ref") or "").strip()
    if not config_ref or Path(config_ref).resolve() != final_run_config_path.resolve():
        raise ValueError("final training/result final-run config path mismatch")
    if training.get("source_ref") != promotion_ref:
        raise ValueError("final training source_ref differs from promotion source_ref")
    if training.get("promotion_summary_sha256") != final_cfg.get("promotion_summary_sha256"):
        raise ValueError("final training promotion provenance mismatch")
    if training.get("recipe_sha256") != recipe.get("sha256"):
        raise ValueError("final training recipe provenance mismatch")
    if not isinstance(training.get("samples_consumed"), int) or training["samples_consumed"] < 0:
        raise ValueError("final training actual samples_consumed missing")

    evaluation = load_json(evaluation_path)
    if int(evaluation.get("episode_count", -1)) != EXPECTED_FINAL_EVAL_EPISODES:
        raise ValueError("final evaluation must be the full 800-episode protocol")
    eval_training_ref = str(evaluation.get("training_result_ref") or "").strip()
    if not eval_training_ref or Path(eval_training_ref).resolve() != linked_training_path.resolve():
        raise ValueError("final evaluation does not link the immutable final training result")
    if evaluation.get("source_ref") != promotion_ref:
        raise ValueError("final evaluation source_ref differs from promoted source")
    if evaluation.get("checkpoint_ref") != training.get("checkpoint_ref"):
        raise ValueError("final evaluation checkpoint differs from final training checkpoint")
    return promotion, final_cfg, training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", choices=("pi05", "smolvla", "openvla_oft"), required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--promotion", type=Path, required=True)
    p.add_argument("--final-run-config", type=Path, required=True)
    p.add_argument("--final-training", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--reproduction-command", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--run-submission-builder", action="store_true")
    p.add_argument("--builder-final", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _, final_cfg, training = validate_chain(
        candidate=args.candidate,
        promotion_path=args.promotion,
        final_run_config_path=args.final_run_config,
        linked_training_path=args.final_training,
        evaluation_path=args.evaluation,
        model_config_path=args.model_config,
    )
    selection, spec, freeze_manifest = freeze(
        candidate=args.candidate,
        source_sha=args.source_sha,
        manifest_path=args.manifest,
        promotion_path=args.promotion,
        final_training_path=args.final_training,
        evaluation_path=args.evaluation,
        artifact_path=args.artifact,
        config_path=args.model_config,
        reproduction_command=args.reproduction_command,
        rationale=args.rationale,
    )

    final_run_block = {
        "config_path": str(args.final_run_config.resolve()),
        "config_sha256": final_cfg["final_run_config_sha256"],
        "track": final_cfg["budget"]["track"],
        "source_ref": final_cfg["source_ref"],
        "recipe_sha256": final_cfg["recipe"]["sha256"],
        "linked_training_result_path": str(args.final_training.resolve()),
        "linked_training_result_sha256": sha256_file(args.final_training),
        "samples_consumed": training["samples_consumed"],
        "optimizer_updates": training.get("optimizer_updates"),
        "train_wall_time": training["metrics"]["train_wall_time"],
        "checkpoint_ref": training["checkpoint_ref"],
    }
    spec["final_run"] = final_run_block
    freeze_manifest["final_run"] = final_run_block
    freeze_manifest["freeze_manifest_sha256"] = hashlib.sha256(canonical_json({
        k: v for k, v in freeze_manifest.items() if k != "freeze_manifest_sha256"
    })).hexdigest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.out_dir / "final_candidate_selection.json"
    spec_path = args.out_dir / "track3_submission_spec.json"
    manifest_path = args.out_dir / "submission_artifact_freeze.json"
    evidence_path = args.out_dir / "final_run_evidence.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "stage": "Track3_final_run_provenance",
        "status": "PASS",
        "candidate": args.candidate,
        "selected_episode_ids_sha256": D10_HASH,
        "final_run": final_run_block,
        "final_evaluation_episode_count": EXPECTED_FINAL_EVAL_EPISODES,
        "automatic_training": False,
        "automatic_evaluation": False,
        "automatic_upload": False,
    }
    evidence["evidence_sha256"] = hashlib.sha256(canonical_json(evidence)).hexdigest()
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": freeze_manifest["status"],
        "candidate": args.candidate,
        "final_run_config_sha256": final_cfg["final_run_config_sha256"],
        "checkpoint_ref": training["checkpoint_ref"],
        "submission_spec": str(spec_path),
        "freeze_manifest": str(manifest_path),
        "final_run_evidence": str(evidence_path),
    }, indent=2))

    if args.run_submission_builder:
        rc = run_submission_builder(
            repo_root=args.repo_root,
            spec_path=spec_path,
            out_dir=args.out_dir / "submission-package",
            final=args.builder_final,
        )
        if rc != 0:
            raise SystemExit(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
