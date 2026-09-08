"""Notebook 71 must stay pinned, one-cell, and training-free."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab/71_m3_runner_preflight.ipynb"
FIXED_PIN = "b64e7a2e7ec481e17d3de5a3c2aef5f18f53cfda"


def test_notebook71_is_pinned_preflight_only():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) == 1
    source = "".join(code_cells[0]["source"])
    ast.parse(source)
    assert FIXED_PIN in source
    assert "run_m3_runner_preflight.py" in source
    assert "=== 71 COMPLETE ===" in source
    assert "lerobot-train" not in source
    assert "finetune.py" not in source
    assert "accelerate launch" not in source
    assert "torchrun" not in source


def test_notebook71_docs_make_training_boundary_explicit():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
        for line in cell.get("source", [])
    )
    assert "trainingを開始しません" in markdown
    assert "READY_FOR_BATCH_PROBE" in markdown
    assert "forward/backward batch probe" in markdown
