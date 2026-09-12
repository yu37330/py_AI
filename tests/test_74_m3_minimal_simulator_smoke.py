import ast
import json
from pathlib import Path


NOTEBOOK_PIN = "a2f517ed85f4a685e591f1be3b77d59f4a7fdaca"


def _notebook_code(root: Path) -> str:
    nb = json.loads((root / "colab/74_m3_minimal_simulator_smoke.ipynb").read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook74_is_pinned_and_runs_runtime_setup_before_smoke():
    root = Path(__file__).resolve().parents[1]
    code = _notebook_code(root)
    assert NOTEBOOK_PIN in code
    assert "prepare_m3_simulator_runtimes.py" in code
    assert "run_m3_minimal_simulator_smoke.py" in code
    assert code.index("prepare_m3_simulator_runtimes.py") < code.index("run_m3_minimal_simulator_smoke.py")
    assert "PARC_M3_EXECUTE" in code
    assert "HF_TOKEN" in code
    assert "74 FAILURE DIAGNOSTICS" in code
    assert "m3_minimal_simulator_smoke_summary.json" in code


def test_simulator_runtime_setup_is_syntax_valid_and_does_not_rerun_probes_or_benchmark():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_simulator_runtimes.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "prepare_m3_training_runtimes.py" in text
    assert "hf-libero==" in text
    assert "8f1084e3132a39270c3a13ebe37270a43ece2a01" in text
    assert '"probes_rerun": False' in text
    assert '"benchmark_training_started": False' in text
    assert '"simulator_started": False' in text
    assert '"final_800_episode_evaluation_started": False' in text
    assert "72a_m3" not in text
    assert "72b_m3" not in text
    assert "72c_m3" not in text
    assert "1800" not in text


def test_minimal_smoke_runner_is_frozen_to_60_nonpromotion_episodes():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/run_m3_minimal_simulator_smoke.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'SUITE = "libero_spatial"' in text
    assert "EXPECTED_PER_MODEL = 20" in text
    assert "EXPECTED_TOTAL = 60" in text
    assert '"promotion_evidence": False' in text
    assert '"benchmark_training_started": False' in text
    assert '"final_800_episode_evaluation_started": False' in text
