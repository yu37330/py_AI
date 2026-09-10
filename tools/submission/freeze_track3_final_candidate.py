#!/usr/bin/env python3
"""Freeze the PARC2026 Track3 final candidate from complete M3 evidence.

Dry-run is intentionally fail-soft: before promotion/final artifacts exist it
returns NOT_READY with explicit blockers and never launches training/evaluation.
The real freeze path is fail-closed and requires complete comparison evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

D10_VARIANT = "V2_SQRT_BALANCED_RAW"
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"
MODELS = ("pi05", "smolvla", "openvla_oft")
ORDERS = ("forward", "reverse")
TRACKS = ("equal_data", "equal_wall")
SEEDS = (20260906, 20260907)
EXPECTED_FINAL_EVAL_EPISODES = 800
EXPECTED_PER_SEED_EPISODES = 400
MAX_STEPS = 300


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
        h.update(sha256_file(file).encode("ascii"))
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


def _validate_model_matrix(model: str, summary: dict[str, Any]) -> None:
    if summary.get("status") != "PASS":
        raise ValueError(f"promotion evidence for {model} is not PASS")
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError(f"promotion evidence for {model} has no records")
    expected = {(order, track) for order in ORDERS for track in TRACKS}
    observed: set[tuple[str, str]] = set()
    source_refs: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "PASS":
            raise ValueError(f"promotion record for {model} is not PASS")
        if record.get("model") != model:
            raise ValueError(f"promotion record model mismatch for {model}")
        combo = (str(record.get("order")), str(record.get("track")))
        if combo not in expected or combo in observed:
            raise ValueError(f"invalid/duplicate promotion cell for {model}: {combo}")
        observed.add(combo)
        if record.get("selected_episode_ids_sha256") != D10_HASH:
            raise ValueError(f"promotion D10 mismatch for {model}/{combo}")
        if tuple(int(x) for x in record.get("seed_set", [])) != SEEDS:
            raise ValueError(f"promotion seed set mismatch for {model}/{combo}")
        if not str(record.get("sampling_schedule_sha256") or "").strip():
            raise ValueError(f"promotion schedule hash missing for {model}/{combo}")
        source_ref = str(record.get("source_ref") or "").strip()
        if not source_ref:
            raise ValueError(f"promotion source ref missing for {model}/{combo}")
        source_refs.add(source_ref)
    if observed != expected:
        raise ValueError(f"promotion matrix incomplete for {model}: {sorted(expected - observed)}")
    if len(source_refs) != 1:
        raise ValueError(f"promotion source changed across matrix for {model}")


def validate_promotion(path: Path, candidate: str) -> dict[str, Any]:
    promotion = load_json(path)
    if promotion.get("status") != "READY_FOR_PROMOTION":
        raise ValueError("promotion summary is not READY_FOR_PROMOTION")
    if promotion.get("selected_dataset_variant") not in (None, D10_VARIANT):
        raise ValueError("promotion dataset variant mismatch")
    if promotion.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("promotion D10 hash mismatch")
    if promotion.get("primary_metric") != "simulator_success_rate":
        raise ValueError("promotion primary metric drifted")
    if promotion.get("training_loss_used_for_promotion") is not False:
        raise ValueError("promotion summary may not use training loss")
    if tuple(promotion.get("required_orders", ORDERS)) != ORDERS:
        raise ValueError("promotion required_orders drifted")
    if tuple(promotion.get("required_tracks", TRACKS)) != TRACKS:
        raise ValueError("promotion required_tracks drifted")
    if tuple(int(x) for x in promotion.get("seed_set", SEEDS)) != SEEDS:
        raise ValueError("promotion seed set drifted")

    promoted = promotion.get("promoted_models")
    if not isinstance(promoted, list) or not promoted:
        raise ValueError("promotion summary has no promoted models")
    if candidate not in promoted:
        raise ValueError(f"candidate {candidate} was not promoted: {promoted}")
    if len(promoted) > 2:
        raise ValueError("promotion summary exceeds max two models")

    summaries = promotion.get("model_summaries")
    if not isinstance(summaries, dict):
        raise ValueError("promotion summary missing model_summaries")
    for model in MODELS:
        summary = summaries.get(model)
        if not isinstance(summary, dict):
            raise ValueError(f"promotion summary missing model evidence: {model}")
        status = summary.get("status")
        if status == "PASS":
            _validate_model_matrix(model, summary)
        elif status == "EXCLUDED_WITH_EVIDENCE":
            evidence = str(summary.get("evidence") or "").strip()
            if not evidence:
                raise ValueError(f"excluded model lacks evidence: {model}")
            if model == candidate:
                raise ValueError("selected candidate cannot use exclusion evidence")
        else:
            raise ValueError(
                f"model {model} lacks complete forward/reverse promotion evidence or documented exclusion: {status}"
            )
    _validate_model_matrix(candidate, summaries[candidate])
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
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("final training result missing metrics")
    for key in ("train_wall_time", "peak_train_vram"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            raise ValueError(f"final training metric invalid: {key}")
    if candidate == "openvla_oft":
        if result.get("checkpoint_eval_ready") is not True:
            raise ValueError("OpenVLA final checkpoint is not evaluator-ready")
        if result.get("lora_merged_for_evaluation") is not True:
            raise ValueError("OpenVLA final LoRA was not merged for evaluation")
        if result.get("merge_excluded_from_train_wall_time") is not True:
            raise ValueError("OpenVLA merge timing provenance missing")
    return result


def validate_evaluation(
    path: Path,
    candidate: str,
    checkpoint_ref: str,
    source_ref: str,
    *,
    final_training_path: Path | None = None,
) -> dict[str, Any]:
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
    if tuple(int(x) for x in evaluation.get("seed_set", [])) != SEEDS:
        raise ValueError("final evaluation seed set mismatch")
    if int(evaluation.get("max_steps_per_episode", -1)) != MAX_STEPS:
        raise ValueError("final evaluation step cap mismatch")
    if int(evaluation.get("episode_count", -1)) != EXPECTED_FINAL_EVAL_EPISODES:
        raise ValueError(
            f"final evaluation must contain {EXPECTED_FINAL_EVAL_EPISODES} episodes; smoke evidence is insufficient"
        )
    per_seed = evaluation.get("per_seed")
    if not isinstance(per_seed, dict):
        raise ValueError("final evaluation missing per_seed evidence")
    for seed in SEEDS:
        seed_summary = per_seed.get(str(seed))
        if not isinstance(seed_summary, dict) or int(seed_summary.get("episode_count", -1)) != EXPECTED_PER_SEED_EPISODES:
            raise ValueError(f"final evaluation seed {seed} must contain {EXPECTED_PER_SEED_EPISODES} episodes")
    if final_training_path is not None:
        ref = str(evaluation.get("training_result_ref") or "").strip()
        if not ref:
            raise ValueError("final evaluation missing training_result_ref")
        if Path(ref).resolve() != Path(final_training_path).resolve():
            raise ValueError("final evaluation training-result linkage mismatch")

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
    for name in required:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            raise ValueError(f"final evaluation metric invalid: {name}")
    if not 0.0 <= float(metrics["simulator_success_rate"]) <= 1.0:
        raise ValueError("final simulator success rate must be within [0,1]")
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


def build_readiness_report(
    *,
    candidate: str | None,
    promotion_path: Path,
    source_sha: str | None = None,
    manifest_path: Path | None = None,
    final_training_path: Path | None = None,
    evaluation_path: Path | None = None,
    artifact_path: Path | None = None,
    config_path: Path | None = None,
    reproduction_command: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    promotion: dict[str, Any] | None = None
    if candidate not in MODELS:
        blockers.append("candidate:not_selected")
    if not promotion_path.is_file():
        blockers.append(f"promotion:missing:{promotion_path}")
    elif candidate in MODELS:
        try:
            promotion = validate_promotion(promotion_path, str(candidate))
        except Exception as exc:  # readiness must report, not raise
            blockers.append(f"promotion:{type(exc).__name__}:{exc}")
    else:
        try:
            raw = load_json(promotion_path)
            if raw.get("status") != "READY_FOR_PROMOTION":
                blockers.append(f"promotion:not_ready:{raw.get('status')}")
        except Exception as exc:
            blockers.append(f"promotion:{type(exc).__name__}:{exc}")

    checks: list[tuple[str, Any, Any]] = [
        ("source_sha", source_sha, lambda v: validate_source_sha(str(v))),
        ("manifest", manifest_path, lambda v: validate_manifest(Path(v))),
        ("final_training", final_training_path, lambda v: validate_final_training(Path(v), str(candidate))),
    ]
    training: dict[str, Any] | None = None
    for label, value, validator in checks:
        if value is None:
            blockers.append(f"{label}:missing")
            continue
        try:
            validated = validator(value)
            if label == "final_training":
                training = validated
        except Exception as exc:
            blockers.append(f"{label}:{type(exc).__name__}:{exc}")

    if evaluation_path is None:
        blockers.append("evaluation:missing")
    elif training is None:
        blockers.append("evaluation:blocked_by_final_training")
    else:
        try:
            validate_evaluation(
                Path(evaluation_path),
                str(candidate),
                str(training["checkpoint_ref"]),
                str(training["source_ref"]),
                final_training_path=Path(final_training_path),
            )
        except Exception as exc:
            blockers.append(f"evaluation:{type(exc).__name__}:{exc}")
    if artifact_path is None or not Path(artifact_path).exists():
        blockers.append("artifact:missing")
    if config_path is None or not Path(config_path).is_file():
        blockers.append("config:missing")
    if not str(reproduction_command or "").strip():
        blockers.append("reproduction_command:missing")
    if not str(rationale or "").strip():
        blockers.append("rationale:missing")
    return {
        "schema_version": 1,
        "stage": "Track3_final_candidate_freeze_readiness",
        "status": "READY_TO_FREEZE" if not blockers else "NOT_READY",
        "candidate": candidate,
        "selected_episode_ids_sha256": D10_HASH,
        "promotion_status": promotion.get("status") if promotion else None,
        "blockers": blockers,
        "automatic_training": False,
        "automatic_evaluation": False,
        "automatic_upload": False,
    }


def build_selection_record(*, candidate: str, promotion: dict[str, Any], rationale: str) -> dict[str, Any]:
    rationale = rationale.strip()
    if not rationale:
        raise ValueError("explicit final-candidate rationale is required")
    return {
        "schema_version": 1,
        "stage": "Track3_final_candidate_selection",
        "status": "SELECTED_FROM_M3_PROMOTION",
        "candidate": candidate,
        "promoted_models": list(promotion["promoted_models"]),
        "promotion_ranking": list(promotion.get("ranking") or []),
        "primary_metric": "simulator_success_rate",
        "training_loss_used_for_selection": False,
        "rationale": rationale,
    }


def build_submission_spec(
    *, source_sha: str, manifest_path: Path, candidate: str, artifact_path: Path,
    artifact_sha256: str, config_path: Path, promotion_path: Path, evaluation_path: Path,
    reproduction_command: str, checkpoint_ref: str,
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
    *, candidate: str, source_sha: str, manifest_path: Path, promotion_path: Path,
    final_training_path: Path, evaluation_path: Path, artifact_path: Path, config_path: Path,
    reproduction_command: str, rationale: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if candidate not in MODELS:
        raise ValueError(f"unknown candidate: {candidate}")
    source_sha = validate_source_sha(source_sha)
    validate_manifest(manifest_path)
    promotion = validate_promotion(promotion_path, candidate)
    training = validate_final_training(final_training_path, candidate)
    validate_evaluation(
        evaluation_path,
        candidate,
        str(training["checkpoint_ref"]),
        str(training["source_ref"]),
        final_training_path=final_training_path,
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
        "schema_version": 2,
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
        "final_evaluation_episode_count": EXPECTED_FINAL_EVAL_EPISODES,
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
    p.add_argument("--candidate", choices=MODELS)
    p.add_argument("--source-sha")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--promotion", type=Path, required=True)
    p.add_argument("--final-training", type=Path)
    p.add_argument("--evaluation", type=Path)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--reproduction-command")
    p.add_argument("--rationale")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-submission-builder", action="store_true")
    p.add_argument("--builder-final", action="store_true")
    return p.parse_args()


def _required(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} is required unless --dry-run is used")
    return value


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    readiness = build_readiness_report(
        candidate=args.candidate,
        promotion_path=args.promotion,
        source_sha=args.source_sha,
        manifest_path=args.manifest,
        final_training_path=args.final_training,
        evaluation_path=args.evaluation,
        artifact_path=args.artifact,
        config_path=args.config,
        reproduction_command=args.reproduction_command,
        rationale=args.rationale,
    )
    readiness_path = args.out_dir / "freeze_readiness.json"
    readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": readiness["status"], "blockers": readiness["blockers"], "readiness": str(readiness_path)}, indent=2))
        return 0
    if readiness["status"] != "READY_TO_FREEZE":
        raise RuntimeError(f"final candidate freeze is NOT_READY: {readiness['blockers']}")

    candidate = str(_required(args.candidate, "--candidate"))
    source_sha = str(_required(args.source_sha, "--source-sha"))
    manifest_path = Path(_required(args.manifest, "--manifest"))
    final_training_path = Path(_required(args.final_training, "--final-training"))
    evaluation_path = Path(_required(args.evaluation, "--evaluation"))
    artifact_path = Path(_required(args.artifact, "--artifact"))
    config_path = Path(_required(args.config, "--config"))
    reproduction_command = str(_required(args.reproduction_command, "--reproduction-command"))
    rationale = str(_required(args.rationale, "--rationale"))

    selection, spec, manifest = freeze(
        candidate=candidate,
        source_sha=source_sha,
        manifest_path=manifest_path,
        promotion_path=args.promotion,
        final_training_path=final_training_path,
        evaluation_path=evaluation_path,
        artifact_path=artifact_path,
        config_path=config_path,
        reproduction_command=reproduction_command,
        rationale=rationale,
    )
    selection_path = args.out_dir / "final_candidate_selection.json"
    spec_path = args.out_dir / "track3_submission_spec.json"
    manifest_path_out = args.out_dir / "submission_artifact_freeze.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "candidate": manifest["candidate"],
        "artifact_sha256": manifest["artifact_sha256"],
        "freeze_manifest_sha256": manifest["freeze_manifest_sha256"],
        "selection": str(selection_path),
        "submission_spec": str(spec_path),
        "freeze_manifest": str(manifest_path_out),
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
