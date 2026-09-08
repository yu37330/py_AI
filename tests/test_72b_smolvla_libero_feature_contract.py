"""Regression contract for adapting pretrained SmolVLA to the LIBERO feature schema."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/colab/run_m3_smolvla_batch_probe.py"
NOTEBOOK = ROOT / "colab/72b_m3_smolvla_batch_probe.ipynb"


def test_72b_reinfers_pretrained_smolvla_features_from_libero_dataset():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"--policy.path=lerobot/smolvla_base"' in source
    assert '"--policy.input_features=null"' in source
    assert '"--policy.output_features=null"' in source
    assert "--rename_map" not in source
    assert 'f"--dataset.episodes={episodes_json}"' in source


def test_72b_persists_non_oom_failure_diagnostics():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"status": "FAILED"' in source
    assert '"failing_trial": trial' in source
    assert '"error_tail": text[-8000:]' in source
    assert 'trial["log_path"] = str(log_path)' in source


def test_72b_notebook_surfaces_child_diagnostics_on_failure():
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "=== 72b DIAGNOSTICS ===" in source
    assert "smolvla.json" in source
    assert "logs/smolvla" in source
    assert "lines[-250:]" in source
