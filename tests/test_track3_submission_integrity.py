import json
from pathlib import Path

import tools.submission.build_track3_submission_manifest as submission

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _ready_fixture(tmp_path, monkeypatch):
    manifest = tmp_path / "D10.json"
    manifest.write_text('{"fixture":"d10"}\n', encoding="utf-8")
    manifest_sha = submission.sha256_file(manifest)
    monkeypatch.setattr(submission, "D10_MANIFEST_HASH", manifest_sha)

    model = tmp_path / "adapter.bin"
    model.write_bytes(b"final-adapter-fixture")
    config = _write_json(tmp_path / "config.json", {"model": "pi05"})
    promotion = _write_json(
        tmp_path / "promotion.json",
        {
            "required_orders": ["forward", "reverse"],
            "promoted_models": ["pi05"],
            "model_summaries": {
                "pi05": {
                    "status": "PASS",
                    "records": [
                        {"order": "forward"},
                        {"order": "reverse"},
                    ],
                },
                "openvla_oft": {
                    "status": "EXCLUDED_WITH_EVIDENCE",
                    "evidence": "documented cut-rule exclusion",
                },
            },
        },
    )
    metrics = _write_json(
        tmp_path / "metrics.json",
        {
            "metrics": {
                "simulator_success_rate": 0.8,
                "steps_to_success": 90.0,
                "episode_duration": 4.5,
                "inference_latency": 0.04,
                "peak_inference_vram": 12000.0,
                "train_wall_time": 1801.0,
                "peak_train_vram": 30000.0,
            }
        },
    )
    spec_path = tmp_path / "spec.json"
    spec = {
        "source_sha": "a" * 40,
        "dataset": {
            "variant": "V2_SQRT_BALANCED_RAW",
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_sha,
            "episode_ids_sha256": submission.D10_EPISODE_HASH,
        },
        "model": {
            "name": "pi05",
            "artifact_path": str(model),
            "artifact_sha256": submission.sha256_file(model),
            "immutable_ref": "fixture-final-ref",
            "config_path": str(config),
        },
        "benchmark": {"promotion_summary_path": str(promotion)},
        "evaluation": {"metrics_path": str(metrics)},
        "reproduction": {"command": "python reproduce.py --frozen"},
    }
    _write_json(spec_path, spec)
    return spec_path, spec


def test_submission_contract_freezes_deadline_and_disables_auto_upload():
    plan = json.loads(
        (ROOT / "experiments/plans/track3_submission_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["official_deadline_jst"] == "2026-09-17T23:59:00+09:00"
    assert plan["internal_targets"]["artifact_freeze_date"] == "2026-09-15"
    assert plan["safety"]["automatic_omnicampus_upload"] is False
    assert plan["benchmark_evidence"]["required_orders"] == ["forward", "reverse"]


def test_dry_run_reports_missing_blockers_instead_of_guessing(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec = {
        "source_sha": "",
        "dataset": {},
        "model": {},
        "benchmark": {},
        "evaluation": {},
        "reproduction": {},
    }
    _write_json(spec_path, spec)
    report = submission.validate_spec(spec, spec_path=spec_path)
    assert report["status"] == "NOT_READY"
    assert report["blockers"]
    assert report["automatic_omnicampus_upload"] is False


def test_ready_spec_hashes_artifacts_and_builds_final_manifest(tmp_path, monkeypatch):
    spec_path, spec = _ready_fixture(tmp_path, monkeypatch)
    report = submission.validate_spec(spec, spec_path=spec_path)
    assert report["status"] == "READY"
    assert report["blockers"] == []
    assert report["artifacts"]["model_artifact"]["sha256"] == spec["model"]["artifact_sha256"]
    final = submission.build_final_manifest(spec, report)
    assert final["status"] == "SUBMISSION_ARTIFACT_FROZEN"
    assert final["automatic_omnicampus_upload"] is False
    assert final["dataset"]["episode_ids_sha256"] == submission.D10_EPISODE_HASH


def test_model_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    spec_path, spec = _ready_fixture(tmp_path, monkeypatch)
    spec["model"]["artifact_sha256"] = "0" * 64
    _write_json(spec_path, spec)
    report = submission.validate_spec(spec, spec_path=spec_path)
    assert report["status"] == "NOT_READY"
    assert any("model.artifact_sha256" in blocker for blocker in report["blockers"])


def test_benchmark_summary_without_reverse_is_rejected(tmp_path, monkeypatch):
    spec_path, spec = _ready_fixture(tmp_path, monkeypatch)
    promotion_path = Path(spec["benchmark"]["promotion_summary_path"])
    data = json.loads(promotion_path.read_text(encoding="utf-8"))
    data["required_orders"] = ["forward"]
    data["model_summaries"]["pi05"]["records"] = [{"order": "forward"}]
    _write_json(promotion_path, data)
    report = submission.validate_spec(spec, spec_path=spec_path)
    assert report["status"] == "NOT_READY"
    assert any("forward + reverse" in blocker for blocker in report["blockers"])


def test_exclusion_without_evidence_is_rejected(tmp_path, monkeypatch):
    spec_path, spec = _ready_fixture(tmp_path, monkeypatch)
    promotion_path = Path(spec["benchmark"]["promotion_summary_path"])
    data = json.loads(promotion_path.read_text(encoding="utf-8"))
    data["model_summaries"]["openvla_oft"]["evidence"] = ""
    _write_json(promotion_path, data)
    report = submission.validate_spec(spec, spec_path=spec_path)
    assert report["status"] == "NOT_READY"
    assert any("lacks evidence" in blocker for blocker in report["blockers"])


def test_final_manifest_cannot_be_built_from_not_ready_report(tmp_path):
    spec = {"model": {}, "benchmark": {}, "evaluation": {}, "reproduction": {}}
    report = {"status": "NOT_READY"}
    try:
        submission.build_final_manifest(spec, report)
    except ValueError as exc:
        assert "NOT_READY" in str(exc)
    else:
        raise AssertionError("NOT_READY report must fail closed")
