from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "examples/pi05_libero_finetune/scripts/full_pi05_lora_manifest.sh"
RUNNER = ROOT / "tools/colab/run_pi05_full_train_manifest.py"
NOTEBOOK = ROOT / "colab/70_pi05_full_train_manifest_a100.ipynb"


def test_manifest_full_train_launcher_contract() -> None:
    text = LAUNCHER.read_text()
    assert "FULL_TRAIN_MANIFEST" in text
    assert "--dataset.episodes=$EPISODES_JSON" in text
    assert "schema_version" in text
    assert "group_aware" in text
    assert "_maybe_resume" in text
    assert "LEROBOT_GRAD_ACCUM" in text
    assert 'PI05_BS="${PI05_BS:-8}"' in text
    assert 'PI05_GA="${PI05_GA:-16}"' in text
    assert 'PI05_STEPS="${PI05_STEPS:-20000}"' in text
    assert 'PI05_SAVE_STEPS="${PI05_SAVE_STEPS:-500}"' in text


def test_full_train_colab_runner_safety_contract() -> None:
    text = RUNNER.read_text()
    assert "vram < 38000" in text
    assert "pi05-top2-tiebreak-v1/screening_summary.json" in text
    assert "run_pi05_group_aware_ablation.py" in text
    assert '"RUN_ABLATIONS": "false"' in text
    assert '"V1_MULTI": 13579' in text
    assert '"V2_SQRT": 10758' in text
    assert "checkpoint_last" in text
    assert "checkpoint_meta.json" in text
    assert "bs, ga = 8, 16" in text
    assert "bs, ga = 16, 8" in text


def test_full_train_notebook_is_valid_and_calls_runner() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    source = "".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "HF_TOKEN" in source
    assert "gpu_mem < 38000" in source
    assert "run_pi05_full_train_manifest.py" in source
    assert "SELECTED_VARIANT_OVERRIDE" in source
