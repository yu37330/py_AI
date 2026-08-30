from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _code_text() -> str:
    nb = json.loads((ROOT / "colab" / "50_pi05_dataset_ablation.ipynb").read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_pi05_ablation_accepts_group_aware_v2_only():
    text = _code_text()
    assert "dataset_ablation_manifests_v2_group_aware" in text
    assert "build_group_aware_ablation_manifests.py" in text
    assert "schema_version') == 2" in text
    assert "group_aware') is True" in text
    assert "protected_eval_group_count" in text
    assert "protected_episode_count" in text


def test_pi05_ablation_requires_leakage_gate_before_training():
    text = _code_text()
    assert "check_trajectory_group_leakage.py" in text
    assert "--fail-on-exact-leakage" in text
    assert "trajectory_leakage_gate'] == 'PASS'" in text
    assert "exact_leakage_group_count" in text
    assert "action_leakage_group_count" in text
    assert "TRAINING MANIFEST SAFETY GATE: PASS" in text


def test_pi05_ablation_keeps_explicit_long_run_gate():
    text = _code_text()
    assert "RUN_ABLATIONS=False" in text
    assert "ABLATION_BS=4" in text
    assert "ABLATION_GA=8" in text
    assert "ABLATION_STEPS=150" in text
    assert "dataset_ablation_manifests_v1'\n" not in text.split("if RUN_ABLATIONS:", 1)[-1]
