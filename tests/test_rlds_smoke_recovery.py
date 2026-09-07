"""CPU-only regression tests for the 69b recovery orchestration."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rlds_smoke_recovery", ROOT / "tools/colab/rlds_smoke_recovery.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture_tree(tmp_path):
    root, drive = tmp_path / "work", tmp_path / "drive"
    source = drive / "datasets/lerobot_libero_plus_v3_train"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text("{}")
    (source / ".parc_prefetch_complete.json").write_text("{}")
    runner.write_json(drive / "pi05-top2-tiebreak-v1/provisional_best_dataset_recipe.json", {"status": "DECIDED", "selected_variant": runner.VARIANT})
    ids = list(range(10758))
    manifest = {"schema_version": 2, "kind": "training_episode_manifest", "group_aware": True,
                "variant": runner.VARIANT, "dataset_id": runner.DATASET_ID, "dataset_revision": runner.REVISION,
                "episode_ids": ids, "episode_ids_sha256": runner.ids_hash(ids),
                "summary": {"episode_count": len(ids), "frame_count": 100000}}
    path = root / "outputs/dataset_ablation_manifests_v2_group_aware" / (runner.VARIANT + ".json")
    runner.write_json(path, manifest)
    args = SimpleNamespace(root=root, repo=tmp_path / "repo", drive=drive, force=False)
    return args, manifest


def good_report(tmp_path, manifest):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "dataset_info.json").write_text("{}")
    return {"status": "SMOKE_PASS", "selected_dataset_variant": runner.VARIANT,
            "source_dataset_id": runner.DATASET_ID, "source_dataset_revision": runner.REVISION,
            "source_episode_ids_sha256": manifest["episode_ids_sha256"], "source_frame_count": 100000,
            "converted_episode_ids": manifest["episode_ids"][:8], "converted_episode_count": 8,
            "converted_frames": 80, "tfds_builder_from_directory": "PASS", "prepared_dir": str(prepared),
            "raw_action_preserved": True, "raw_state_preserved": True, "no_noop_filter_applied": True,
            "bytes_per_frame": 1000, "projected_full_gib": 2.5}


def test_manifest_and_report_provenance(tmp_path):
    args, manifest = fixture_tree(tmp_path)
    path, loaded = runner.find_manifest(args.drive, args.root)
    assert loaded["episode_ids_sha256"] == manifest["episode_ids_sha256"]
    report = good_report(tmp_path, manifest)
    runner.validate_report(report, manifest, manifest["episode_ids"][:8])
    report["source_episode_ids_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.validate_report(report, manifest, manifest["episode_ids"][:8])
    runner.write_json(path, {**manifest, "episode_ids_sha256": "wrong"})
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.validate_manifest(path)


def test_dependencies_are_binary_only_and_pinned(tmp_path, monkeypatch):
    commands = []
    py = tmp_path / "venv-openvla-rlds/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(runner, "run_logged", lambda cmd, label, log_dir, **kw: commands.append((cmd, label)))
    assert runner.ensure_env(tmp_path, tmp_path / "logs") == py
    install = next(cmd for cmd, label in commands if label == "rlds-conversion-deps")
    assert install[install.index("--only-binary") + 1] == ":all:"
    assert "av==12.3.0" in install
    assert "av>=12,<15" not in install
    assert "tensorflow-cpu==2.17.1" in install


def test_failed_conversion_blocks_capacity_and_keeps_previous_artifacts(tmp_path):
    args, manifest = fixture_tree(tmp_path)
    out = args.drive / "openvla-rlds-selected-v1"
    old = {"status": "PASS", "decision": "old"}
    runner.write_json(out / "bridge_capacity_decision.json", old)
    runner.write_json(out / "bridge_smoke_report.json", {"status": "old"})
    with patch.object(runner, "ensure_env", return_value=Path(sys.executable)):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            runner.run(args, exec_command=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg")))
    assert runner.read_json(out / "bridge_smoke_status.json")["status"] == "FAILED"
    assert runner.read_json(out / "bridge_capacity_decision.json")["status"] == "BLOCKED"
    assert runner.read_json(out / "bridge_smoke_report.json") == {"status": "old"}
    assert any(runner.read_json(p) == old for p in (out / "history").glob("*.json"))
    assert (args.drive / "pi05-top2-tiebreak-v1/provisional_best_dataset_recipe.json").exists()


def test_missing_report_stops_without_capacity_success(tmp_path):
    args, _ = fixture_tree(tmp_path)
    with patch.object(runner, "ensure_env", return_value=Path(sys.executable)):
        with pytest.raises(RuntimeError, match="did not create"):
            runner.run(args, exec_command=lambda *a, **k: None)
    out = args.drive / "openvla-rlds-selected-v1"
    assert runner.read_json(out / "bridge_capacity_decision.json")["status"] == "BLOCKED"
    assert runner.read_json(out / "bridge_smoke_status.json")["status"] == "FAILED"


def test_verified_reuse_skips_conversion(tmp_path):
    args, manifest = fixture_tree(tmp_path)
    out = args.drive / "openvla-rlds-selected-v1"
    runner.write_json(out / "bridge_smoke_report.json", good_report(tmp_path, manifest))
    calls = []
    with patch.object(runner, "ensure_env", return_value=Path(sys.executable)):
        result = runner.run(args, exec_command=lambda cmd, label, log_dir, **kw: calls.append(label))
    assert calls == ["verify-existing-tfds"]
    assert result["decision"] == "MATERIALIZE_CANDIDATE"
    assert runner.read_json(out / "bridge_smoke_status.json")["status"] == "PASS"
    assert runner.read_json(out / "bridge_capacity_decision.json")["status"] == "PASS"


def test_capacity_gate_rejects_invalid_numbers():
    report = {"projected_full_gib": float("nan")}
    with pytest.raises(ValueError):
        runner.capacity_decision(report, {})
    report["projected_full_gib"] = 36.0
    report.update(converted_episode_count=8, converted_frames=80, bytes_per_frame=1000)
    assert runner.capacity_decision(report, {"episode_ids_sha256": "abc"})["decision"] == "STREAMING_BRIDGE_RECOMMENDED"


def test_subprocess_error_is_preserved_in_log(tmp_path):
    with pytest.raises(RuntimeError, match="rc=7"):
        runner.run_logged([sys.executable, "-c", "import sys; print('real failure', file=sys.stderr); sys.exit(7)"], "diagnostic", tmp_path, interval=0.01)
    assert "real failure" in (tmp_path / "diagnostic.log").read_text()


def test_notebook_has_one_parseable_code_cell():
    import ast
    nb = json.loads((ROOT / "colab/69b_openvla_selected_rlds_smoke.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 1
    ast.parse(code[0])
    assert "rlds_smoke_recovery.py" in code[0]
    assert "subprocess.run(cmd, check=True)" in code[0]
    assert "av>=12,<15" not in code[0]
