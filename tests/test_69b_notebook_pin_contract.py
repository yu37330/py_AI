"""69b Notebookが修正済みrunnerを固定参照することを検証する。"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab/69b_openvla_selected_rlds_smoke.ipynb"
FIXED_PIN = "b1e6b70d9ee2fcc2941adbcb7a3771edf9f0fda0"
OLD_PINS = {
    "73bab2a1d87dd1b2ab80b2e053210a57f731d6f6",
    "5f7cbb4ed4b054b381a3a0074ab54b419b09a5de",
}


def test_69b_notebook_uses_uv_compatible_dependency_fix():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )

    assert FIXED_PIN in source
    for old_pin in OLD_PINS:
        assert old_pin not in source
    assert "tools/colab/rlds_smoke_recovery.py" in source
    assert "subprocess.run(cmd, check=True)" in source


def test_69b_notebook_documents_promise_exception():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
        for line in cell.get("source", [])
    )

    assert "promise==2.3" in markdown
    assert "binary wheel" in markdown
    assert "--no-binary promise" in markdown
