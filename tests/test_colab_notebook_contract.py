"""Regression checks for the Colab execution contract.

The model/data notebooks must be runnable from a fresh Colab runtime without
requiring 00_a100_preflight.ipynb to have been executed in the same session.
"""

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


def test_pi05_notebook_is_self_contained_and_has_public_fallback():
    text = _code_text("10_pi05_smoke_ga.ipynb")
    assert "Path('/content/parc2026')" in text
    assert "git', 'clone', 'https://github.com/yu37330/py_AI.git'" in text
    assert "assert REPO.exists()" not in text
    assert "Sylvest/libero_plus_lerobot" in text
    assert "allow_patterns=['meta/*', 'data/*', 'videos/*']" in text


def test_inventory_notebook_is_self_contained_and_metadata_only_fallback():
    text = _code_text("20_dataset_inventory.ipynb")
    assert "Path('/content/parc2026')" in text
    assert "git', 'clone', 'https://github.com/yu37330/py_AI.git'" in text
    assert "assert REPO.exists()" not in text
    assert "Sylvest/libero_plus_lerobot" in text
    assert "meta/info.json" in text
    assert "meta/episodes.jsonl" in text
    assert "videos/*" not in text


def test_static_quality_notebook_uses_compact_v3_parquet_without_video():
    text = _code_text("30_static_quality_analyzer.ipynb")
    assert "Path('/content/parc2026')" in text
    assert "git','clone','https://github.com/yu37330/py_AI.git'" in text
    assert "lerobot/libero_plus" in text
    assert "HfApi" in text
    assert "hf_hub_download" in text
    assert "data/" in text and ".parquet" in text
    assert "snapshot_download" not in text
    assert "videos/*" not in text
    assert "static_quality_analyzer.py" in text
    assert "quality_review_candidates.csv" in text


def test_manifest_notebook_is_self_contained_and_freezes_eval_split():
    text = _code_text("40_dataset_ablation_manifests.ipynb")
    assert "Path('/content/parc2026')" in text
    assert "git','clone','https://github.com/yu37330/py_AI.git'" in text
    assert "lerobot/libero_plus" in text
    assert "static_quality_analyzer.py" in text
    assert "build_dataset_ablation_manifests.py" in text
    assert "--eval-per-task','2'" in text
    assert "Dataset Manifest Gate: PASS" in text
    assert "videos/" not in text


def test_pi05_dataset_ablation_notebook_has_safety_gate_and_native_episode_manifests():
    text = _code_text("50_pi05_dataset_ablation.ipynb")
    assert "Path('/content/parc2026')" in text
    assert "git','clone','https://github.com/yu37330/py_AI.git'" in text
    assert "lerobot/libero_plus" in text
    assert "build_dataset_ablation_manifests.py" in text
    assert "cheap_ablation_pi05.sh" in text
    assert "RUN_ABLATIONS=False" in text
    assert "ABLATION_BS=4" in text
    assert "ABLATION_GA=8" in text
    assert "ABLATION_STEPS=150" in text
    assert "SCREENING COMPLETE" in text
