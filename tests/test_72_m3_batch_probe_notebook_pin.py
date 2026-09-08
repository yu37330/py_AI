"""Notebook 72a-d must stay pinned to complete probe runtimes and preserve stage boundaries."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PROBE_PIN = "2b961ef03619fea21f36dc902fb8b7b02f793120"
OPENVLA_COMPAT_PIN = "d680d5acae4beed905fdeb7541b33d590d8f314f"
OLD_PIN = "5baa5e3297ae476fe6f462ebcaac6d988f1a8e43"
OLD_OPENVLA_DIAG_PIN = "ac9a5c403784bbc7624c9055ed6282b27f16b3a3"

NOTEBOOKS = {
    "72a_m3_pi05_batch_probe.ipynb": ("run_m3_pi05_batch_probe.py", BASE_PROBE_PIN),
    "72b_m3_smolvla_batch_probe.ipynb": ("run_m3_smolvla_batch_probe.py", BASE_PROBE_PIN),
    "72c_m3_openvla_oft_batch_probe.ipynb": ("run_m3_openvla_batch_probe.py", OPENVLA_COMPAT_PIN),
    "72d_m3_batch_probe_controller.ipynb": ("validate_m3_batch_probes.py", BASE_PROBE_PIN),
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
    for name, (entry, expected_pin) in NOTEBOOKS.items():
        _, source, _ = _read_notebook(name)
        assert expected_pin in source, name
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


def test_72c_surfaces_persisted_setup_and_probe_diagnostics():
    _, source, markdown = _read_notebook("72c_m3_openvla_oft_batch_probe.ipynb")
    assert OPENVLA_COMPAT_PIN in source
    assert OLD_OPENVLA_DIAG_PIN not in source
    assert "openvla_oft_status.json" in source
    assert "logs/openvla_oft/setup.log" in source
    assert "72c NOTEBOOK DIAGNOSTICS" in source
    assert "72c SETUP LOG TAIL" in source
    assert "72c PROBE LOG TAIL" in source
    assert "失敗時" in markdown
    assert "TensorFlow Metadata / protobuf互換pin" in markdown

    runner = (ROOT / "tools/colab/run_m3_openvla_batch_probe.py").read_text(encoding="utf-8")
    for token in (
        "install_python_310",
        "install_openvla_editable",
        "install_flash_attn",
        "pin_tf_metadata_protobuf",
        'TF_METADATA_VERSION = "1.16.1"',
        'PROTOBUF_VERSION = "3.20.3"',
        "import tensorflow_datasets as tfds",
        "import tensorflow_metadata",
        "import dlimp",
        "tfds_dlimp_ok",
        "verify_openvla_env",
        "72c DIAGNOSTICS",
        "72c SETUP LOG TAIL",
        "openvla_oft_status.json",
    ):
        assert token in runner


def test_72d_only_validates_probe_results_and_stops_before_m3_training():
    _, source, markdown = _read_notebook("72d_m3_batch_probe_controller.ipynb")
    assert "validate_m3_batch_probes.py" in source
    assert "m3_batch_probe_summary.json" in source
    assert "Benchmark training has NOT started." in source
    assert "READY_FOR_M3_RUNNER_IMPLEMENTATION" in markdown
    for forbidden in ("lerobot-train", "finetune.py", "torchrun", "accelerate launch"):
        assert forbidden not in source
