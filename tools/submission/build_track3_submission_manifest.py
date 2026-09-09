#!/usr/bin/env python3
"""Build and validate PARC2026 Track3 submission integrity evidence.

This tool never uploads to Omnicampus.  It can be run early in dry-run mode to
surface missing submission blockers, then rerun in final mode after the model,
benchmark and evaluation artifacts are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

D10_VARIANT = "V2_SQRT_BALANCED_RAW"
D10_EPISODE_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_HASH = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"
REQUIRED_ORDERS = ("forward", "reverse")
REQUIRED_METRICS = (
    "simulator_success_rate",
    "steps_to_success",
    "episode_duration",
    "inference_latency",
    "peak_inference_vram",
    "train_wall_time",
    "peak_train_vram",
)
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int]:
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def artifact_digest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_file():
        return {
            "kind": "file",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if path.is_dir():
        digest, file_count = sha256_tree(path)
        return {
            "kind": "directory",
            "sha256": digest,
            "file_count": file_count,
        }
    raise FileNotFoundError(path)


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nested(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_path(value: Any, *, spec_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = spec_dir / path
    return path.resolve()


def _metric_object(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    if all(name in data for name in REQUIRED_METRICS):
        return data
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and all(name in metrics for name in REQUIRED_METRICS):
        return metrics
    overall = data.get("overall_metrics")
    if isinstance(overall, dict) and all(name in overall for name in REQUIRED_METRICS):
        return overall
    return None


def _validate_benchmark_summary(data: Any, blockers: list[str]) -> None:
    if not isinstance(data, dict):
        blockers.append("benchmark promotion summary must be a JSON object")
        return
    orders = tuple(data.get("required_orders", []))
    if set(orders) != set(REQUIRED_ORDERS):
        blockers.append("benchmark summary must contain forward + reverse order evidence")
    promoted = data.get("promoted_models")
    if not isinstance(promoted, list) or not promoted:
        blockers.append("benchmark summary has no promoted model")
    summaries = data.get("model_summaries")
    if not isinstance(summaries, dict):
        blockers.append("benchmark summary missing model_summaries")
        return
    for model, summary in summaries.items():
        if not isinstance(summary, dict):
            blockers.append(f"benchmark model summary malformed: {model}")
            continue
        status = summary.get("status")
        if status == "EXCLUDED_WITH_EVIDENCE" and not str(summary.get("evidence") or "").strip():
            blockers.append(f"excluded model lacks evidence: {model}")
        if status == "PASS":
            records = summary.get("records")
            if not isinstance(records, list):
                blockers.append(f"PASS model lacks forward/reverse records: {model}")
                continue
            record_orders = {record.get("order") for record in records if isinstance(record, dict)}
            if record_orders != set(REQUIRED_ORDERS):
                blockers.append(f"PASS model lacks both execution orders: {model}")


def validate_spec(spec: dict[str, Any], *, spec_path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    artifacts: dict[str, Any] = {}
    spec_dir = Path(spec_path).resolve().parent

    source_sha = str(spec.get("source_sha") or "").strip().lower()
    if not SHA40_RE.fullmatch(source_sha):
        blockers.append("source_sha must be an exact 40-character lowercase Git SHA")

    dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else {}
    if dataset.get("variant") not in (None, D10_VARIANT):
        blockers.append("dataset.variant changed from frozen D10 variant")
    if dataset.get("episode_ids_sha256") != D10_EPISODE_HASH:
        blockers.append("dataset.episode_ids_sha256 does not match frozen D10")
    if dataset.get("manifest_sha256") != D10_MANIFEST_HASH:
        blockers.append("dataset.manifest_sha256 does not match frozen D10 manifest")

    path_fields = {
        "dataset_manifest": "dataset.manifest_path",
        "model_artifact": "model.artifact_path",
        "model_config": "model.config_path",
        "promotion_summary": "benchmark.promotion_summary_path",
        "evaluation_metrics": "evaluation.metrics_path",
    }
    resolved: dict[str, Path] = {}
    for name, dotted in path_fields.items():
        path = _resolve_path(_nested(spec, dotted), spec_dir=spec_dir)
        if path is None:
            blockers.append(f"missing required path: {dotted}")
            continue
        if not path.exists():
            blockers.append(f"required artifact does not exist: {dotted}={path}")
            continue
        resolved[name] = path
        try:
            artifacts[name] = {
                "path": str(path),
                **artifact_digest(path),
            }
        except Exception as exc:  # fail closed, but keep a useful dry-run report
            blockers.append(f"cannot hash/read {dotted}: {type(exc).__name__}: {exc}")

    manifest_info = artifacts.get("dataset_manifest")
    if manifest_info and manifest_info.get("sha256") != D10_MANIFEST_HASH:
        blockers.append("actual D10 manifest file SHA256 does not match frozen manifest hash")

    model = spec.get("model") if isinstance(spec.get("model"), dict) else {}
    expected_model_sha = str(model.get("artifact_sha256") or "").strip().lower()
    actual_model_sha = (artifacts.get("model_artifact") or {}).get("sha256")
    if expected_model_sha and actual_model_sha and expected_model_sha != actual_model_sha:
        blockers.append("model.artifact_sha256 does not match actual model artifact")
    if not str(model.get("immutable_ref") or "").strip():
        warnings.append("model.immutable_ref is empty; final manifest will rely on artifact SHA only")

    if "promotion_summary" in resolved:
        try:
            _validate_benchmark_summary(_load_json(resolved["promotion_summary"]), blockers)
        except Exception as exc:
            blockers.append(f"cannot parse benchmark summary: {type(exc).__name__}: {exc}")

    if "evaluation_metrics" in resolved:
        try:
            evaluation_data = _load_json(resolved["evaluation_metrics"])
            metrics = _metric_object(evaluation_data)
            if metrics is None:
                blockers.append("evaluation metrics file does not contain all required metrics")
            else:
                for name in REQUIRED_METRICS:
                    value = metrics.get(name)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        blockers.append(f"evaluation metric is not numeric: {name}")
        except Exception as exc:
            blockers.append(f"cannot parse evaluation metrics: {type(exc).__name__}: {exc}")

    reproduction = spec.get("reproduction") if isinstance(spec.get("reproduction"), dict) else {}
    command = str(reproduction.get("command") or "").strip()
    if not command:
        blockers.append("reproduction.command is required")

    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "P0_Track3_submission_integrity",
        "status": "READY" if not blockers else "NOT_READY",
        "selected_dataset_variant": D10_VARIANT,
        "selected_episode_ids_sha256": D10_EPISODE_HASH,
        "selected_manifest_sha256": D10_MANIFEST_HASH,
        "source_sha": source_sha,
        "artifacts": artifacts,
        "blockers": blockers,
        "warnings": warnings,
        "required_metrics": list(REQUIRED_METRICS),
        "automatic_omnicampus_upload": False,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def build_final_manifest(spec: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "READY":
        raise ValueError("cannot build final manifest while integrity report is NOT_READY")
    return {
        "schema_version": 1,
        "status": "SUBMISSION_ARTIFACT_FROZEN",
        "source_sha": report["source_sha"],
        "dataset": {
            "variant": D10_VARIANT,
            "episode_ids_sha256": D10_EPISODE_HASH,
            "manifest_sha256": D10_MANIFEST_HASH,
        },
        "model": spec["model"],
        "benchmark": spec["benchmark"],
        "evaluation": spec["evaluation"],
        "reproduction": spec["reproduction"],
        "artifact_digests": report["artifacts"],
        "integrity_report_sha256": report["report_sha256"],
        "automatic_omnicampus_upload": False,
    }


def _copy_artifacts(report: dict[str, Any], target: Path) -> None:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, info in sorted(report["artifacts"].items()):
        source = Path(info["path"])
        destination = target / name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--copy-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = _load_json(args.spec)
    if not isinstance(spec, dict):
        raise ValueError("submission spec must be a JSON object")
    report = validate_spec(spec, spec_path=args.spec)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "track3_integrity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.final:
        if report["status"] != "READY":
            print(json.dumps(report, indent=2))
            return 2
        manifest = build_final_manifest(spec, report)
        manifest_path = args.out_dir / "track3_submission_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.copy_artifacts:
            _copy_artifacts(report, args.out_dir / "files")

    print(json.dumps({
        "status": report["status"],
        "blockers": report["blockers"],
        "warnings": report["warnings"],
        "report": str(report_path),
        "final_requested": bool(args.final),
        "automatic_omnicampus_upload": False,
    }, indent=2))
    # Dry-run is diagnostic and intentionally returns success even while NOT_READY.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
