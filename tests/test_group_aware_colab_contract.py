from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _code_text(name: str) -> str:
    nb = json.loads((ROOT / "colab" / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_group_aware_notebook_is_self_contained_and_fail_fast():
    text = _code_text("47_group_aware_ablation_manifests.ipynb")
    assert "Path('/content/parc2026')" in text
    assert "git','clone','https://github.com/yu37330/py_AI.git'" in text
    assert "lerobot/libero_plus" in text
    assert "build_group_aware_ablation_manifests.py" in text
    assert "check_trajectory_group_leakage.py" in text
    assert "dataset_ablation_manifests_v2_group_aware" in text
    assert "--fail-on-exact-leakage" in text
    assert "Group-aware Manifest Gate: PASS" in text
    assert "Trajectory Leakage Gate: PASS" in text
    assert "videos/" not in text
