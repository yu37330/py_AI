"""Notebook 72a-d must stay pinned to the complete probe runtime and preserve stage boundaries."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_PIN = "2b961ef03619fea21f36dc902fb8b7b02f793120"
OLD_PIN = "5baa5e3297ae476fe6f462ebcaac6d988f1a8e43"

NOTEBOOKS = {
    "72a_m3_pi05_batch_probe.ipynb": "run_m3_pi05_batch_probe.py",
    "72b_m3_smolvla_batch_probe.ipynb": "run_m3_smolvla_batch_probe.py",
    "72c_m3_openvla_oft_batch_probe.ipynb": "run_m3_openvla_batch_probe.py",
    "72d_m3_batch_probe_controller.ipynb": "validate_m3_batch_probes.py",
}


def _read_notebook(name: str) -> tuple[dict, str, str]:
    data = json.loads((ROOT / "colab" / name).read_text(encoding="utf-8"))
    code_cells = [cell for cell in data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) == 1
    source = "".join(code_cells[0].get("source", []))
    ast.parse(source)
    markdown = "".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
        for line in cell.get("source", [])
    )
    return data, source, markdown


def test_72_notebooks_pin_complete_probe_runtime():
    for name, entry in NOTEBOOKS.items():
        _, source, _ = _read_notebook(name)
        assert FIXED_PIN in source, name
        assert OLD_PIN not in source, name
        assert entry in source, name
        assert "checkout', '--detach', '--force', PIN" in source, name
        assert "rev-parse', 'HEAD" in source, name


def test_72a_to_72c_are_probe_only_and_not_benchmark_runs():
    for name in (
        "72a_m3_pi05_batch_probe.ipynb",
        "72b_m3_smolvla_batch_probe.ipynb",
        "72c_m3_openvla_oft_batch_probe.ipynb",
    ):
        _, source, markdown = _read_notebook(name)
        assert "Probe only. M3 benchmark training has NOT started." in source
        assert "probe-only" in markdown
        assert "4800" not in source
        assert "1800" not in source


def test_72d_only_validates_probe_results_and_stops_before_m3_training():
    _, source, markdown = _read_notebook("72d_m3_batch_probe_controller.ipynb")
    assert "validate_m3_batch_probes.py" in source
    assert "m3_batch_probe_summary.json" in source
    assert "Benchmark training has NOT started." in source
    assert "READY_FOR_M3_RUNNER_IMPLEMENTATION" in markdown
    for forbidden in ("lerobot-train", "finetune.py", "torchrun", "accelerate launch"):
        assert forbidden not in source
