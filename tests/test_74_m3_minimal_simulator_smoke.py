import ast
import json
from pathlib import Path


NOTEBOOK_PIN = "1b983910509020dc2a01f1474140d72bf5664e8e"


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
    assert "ATTEMPT = '8'" in code
    assert "prepare_m3_disk_headroom.py" in code
    assert "'pre-setup'" in code
    assert "'pre-eval'" in code
    assert "prepare_m3_libero_noninteractive_config.py" in code
    assert "prepare_m3_simulator_runtimes.py" in code
    assert "prepare_m3_mujoco_compat.py" in code
    assert "run_m3_minimal_simulator_smoke.py" in code
    assert code.index("'pre-setup'") < code.index("prepare_m3_libero_noninteractive_config.py")
    assert code.index("prepare_m3_libero_noninteractive_config.py") < code.index("prepare_m3_simulator_runtimes.py")
    assert code.index("prepare_m3_simulator_runtimes.py") < code.index("prepare_m3_mujoco_compat.py")
    assert code.index("prepare_m3_mujoco_compat.py") < code.index("'pre-eval'")
    assert code.index("'pre-eval'") < code.index("run_m3_minimal_simulator_smoke.py")
    assert "LIBERO_CONFIG_PATH" in code
    assert "m3-libero-config/shared" in code
    assert "MPLBACKEND" in code
    assert "'Agg'" in code
    assert "MUJOCO_GL" in code
    assert "PYOPENGL_PLATFORM" in code
    assert "'egl'" in code
    assert "PARC_M3_EXECUTE" in code
    assert "PARC_M3_SIM_SMOKE_ATTEMPT" in code
    assert "HF_TOKEN" in code
    assert "74 FAILURE DIAGNOSTICS" in code
    assert "def safe_tail" in code
    assert "diagnostic read failed" in code
    assert "df', '-h'" in code
    assert "df', '-i'" in code
    assert "episode_records.partial.json" in code
    assert "m3_minimal_simulator_smoke_summary.json" in code
    assert "latest eval log" in code
    assert "attempt-{ATTEMPT}" in code


def test_noninteractive_libero_config_helper_is_syntax_valid_and_prompt_free():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_libero_noninteractive_config.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "LIBERO_CONFIG_PATH" in text
    assert "vendor/libero-openvla-m3/libero/libero" in text
    assert '"interactive_prompt_allowed": False' in text
    assert "input(" not in text


def test_simulator_runtime_setup_is_syntax_valid_and_does_not_rerun_probes_or_benchmark():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_simulator_runtimes.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "prepare_m3_training_runtimes.py" in text
    assert "hf-libero==" in text
    assert "8f1084e3132a39270c3a13ebe37270a43ece2a01" in text
    assert "parc_m3_pinned_libero.pth" in text
    assert '"probes_rerun": False' in text
    assert '"benchmark_training_started": False' in text
    assert '"simulator_started": False' in text
    assert '"final_800_episode_evaluation_started": False' in text
    assert "72a_m3" not in text
    assert "72b_m3" not in text
    assert "72c_m3" not in text
    assert "1800" not in text


def test_mujoco_compat_helper_pins_all_three_eval_runtimes():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_mujoco_compat.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'MUJOCO_VERSION = "3.3.1"' in text
    assert '"pi05"' in text
    assert '"smolvla"' in text
    assert '"openvla_oft"' in text
    assert "mujoco_robosuite_import_ok" in text
    assert '"benchmark_training_started": False' in text
    assert '"simulator_episode_started": False' in text


def test_minimal_smoke_runner_is_frozen_to_60_nonpromotion_episodes_and_attempt_scoped():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/run_m3_minimal_simulator_smoke.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'SUITE = "libero_spatial"' in text
    assert "EXPECTED_PER_MODEL = 20" in text
    assert "EXPECTED_TOTAL = 60" in text
    assert "PARC_M3_SIM_SMOKE_ATTEMPT" in text
    assert 'f"model-benchmark-v1/m3-simulator-minimal-smoke-v1/attempt-{attempt}"' in text
    assert "Use a new PARC_M3_SIM_SMOKE_ATTEMPT instead of deleting evidence" in text
    assert '"promotion_evidence": False' in text
    assert '"benchmark_training_started": False' in text
    assert '"final_800_episode_evaluation_started": False' in text
