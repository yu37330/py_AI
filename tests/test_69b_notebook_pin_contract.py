"""69b Notebookが修正済みrunnerを固定参照し、失敗時に診断を表示することを検証する。"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab/69b_openvla_selected_rlds_smoke.ipynb"
FIXED_PIN = "d340168e3458107e7025b7dfb57a09f7b05ac4d9"
OLD_PINS = {
    "b1e6b70d9ee2fcc2941adbcb7a3771edf9f0fda0",
    "73bab2a1d87dd1b2ab80b2e053210a57f731d6f6",
    "5f7cbb4ed4b054b381a3a0074ab54b419b09a5de",
}


def code_source():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )


def test_69b_notebook_uses_compatibility_fixed_runner():
    source = code_source()
    assert FIXED_PIN in source
    for old_pin in OLD_PINS:
        assert old_pin not in source
    assert "tools/colab/rlds_smoke_recovery.py" in source
    assert "subprocess.run(cmd, check=True)" in source


def test_69b_notebook_surfaces_child_failure_diagnostics():
    source = code_source()
    assert "except subprocess.CalledProcessError" in source
    assert "bridge_smoke_status.json" in source
    assert "Full log: " in source
    assert "=== 69b DIAGNOSTICS ===" in source
    assert "=== 69b ERROR LOG ===" in source
    assert "lines[-200:]" in source


def test_69b_notebook_documents_dependency_compatibility():
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
    assert "tensorflow-metadata==1.16.1" in markdown
    assert "googleapis-common-protos==1.65.0" in markdown
    assert "protobuf==3.20.3" in markdown
    assert "bridge_smoke_status.json" in markdown
