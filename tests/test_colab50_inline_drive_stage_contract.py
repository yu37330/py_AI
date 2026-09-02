import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab" / "50_pi05_dataset_ablation.ipynb"
RUNNER = ROOT / "tools" / "colab" / "run_pi05_group_aware_ablation.py"


def notebook_text() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_colab50_inlines_drive_stage_and_blocks_a100_redownload():
    text = notebook_text()
    assert "DRIVE SOURCE GATE: PASS" in text
    assert "LOCAL STAGE GATE: PASS" in text
    assert ".parc_prefetch_complete.json" in text
    assert "48_prefetch_training_dataset_to_drive.ipynb" in text
    assert "hf_hub_download" not in text
    assert "snapshot_download" not in text


def test_colab50_keeps_long_run_safety_gate():
    text = notebook_text()
    assert "RUN_ABLATIONS=False" in text
    assert "run_pi05_group_aware_ablation.py" in text


def test_unified_runner_keeps_group_aware_and_ga_gates():
    text = RUNNER.read_text(encoding="utf-8")
    assert "dataset_ablation_manifests_v2_group_aware" in text
    assert "--fail-on-exact-leakage" in text
    assert "TRAINING MANIFEST SAFETY GATE: PASS" in text
    assert "--expected-ga" in text
    assert '"8"' in text
    assert "GA Gate: PASS" in text
