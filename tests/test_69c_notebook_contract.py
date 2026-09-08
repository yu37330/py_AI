"""69c one-cell Notebook contract and safety gates."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab/69c_openvla_streaming_bridge_validation.ipynb"
FIXED_PIN = "734f8c4077deb4ca4b09c30b34303445b5cc1edb"


def _sources():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )
    markdown = "\n".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
        for line in cell.get("source", [])
    )
    return code, markdown


def test_69c_notebook_pins_validated_streaming_code():
    code, _ = _sources()
    assert FIXED_PIN in code
    assert "validate_lerobot_openvla_streaming_bridge.py" in code
    assert "bridge_smoke_report.json" in code
    assert "bridge_capacity_decision.json" in code
    assert "streaming_bridge_contract.json" in code
    assert "=== 69c COMPLETE ===" in code


def test_69c_notebook_does_not_materialize_full_rlds_or_train():
    code, markdown = _sources()
    assert "conversion_contract.json" not in code
    assert "finetune.py" not in code
    assert "full RLDS / 56.6 GiB" in markdown
    assert "D10再選択・manifest再生成はしません" in markdown
    assert "M3 trainingはまだ自動開始しません" in markdown
