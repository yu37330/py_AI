import ast
import json
from pathlib import Path


PIN = "27b1735d882dc0bdd8ae4d2ca84d93df27b6dc73"


def _code(root: Path, name: str) -> str:
    nb = json.loads((root / "colab" / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_75a_75b_share_exact_runtime_pin_and_attempt():
    root = Path(__file__).resolve().parents[1]
    forward = _code(root, "75a_m3_forward_training_benchmark.ipynb")
    reverse = _code(root, "75b_m3_reverse_training_benchmark.ipynb")
    for code in (forward, reverse):
        assert PIN in code
        assert "run_m3_benchmark_order.py" in code
        assert "PARC_M3_EXECUTE" in code
        assert "--attempt', '1" in code or '"--attempt", "1"' in code
    assert "--order', 'forward" in forward or '"--order", "forward"' in forward
    assert "--order', 'reverse" in reverse or '"--order", "reverse"' in reverse


def test_benchmark_order_runner_is_syntax_valid_and_same_schedule():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/run_m3_benchmark_order.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "TRAINING_SCHEDULE_SEED = 20260906" in text
    assert '"forward": ("pi05", "smolvla", "openvla_oft")' in text
    assert '"reverse": ("openvla_oft", "smolvla", "pi05")' in text
    assert "EQUAL_DATA_SAMPLES = 4800" in text
    assert "EQUAL_DATA_UPDATES = 150" in text
    assert "EQUAL_WALL_SEC = 1800.0" in text
    assert "_validate_smoke_gate" in text


def test_benchmark_order_runner_preserves_partial_evidence_and_skips_openvla_merge():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/run_m3_benchmark_order.py").read_text(encoding="utf-8")
    assert "preserved for diagnosis" in text
    assert "Use a new --attempt instead of deleting evidence" in text
    assert "shutil.rmtree" not in text
    assert "post_training_command" in text
    assert "ignore spec['post_training_command']" in text
    assert '"openvla_merged_checkpoint_materialized": False' in text
    assert '"openvla_merge_deferred_to_evaluation": True' in text


def test_forward_reverse_notebooks_do_not_start_evaluation_or_promotion():
    root = Path(__file__).resolve().parents[1]
    for name in ("75a_m3_forward_training_benchmark.ipynb", "75b_m3_reverse_training_benchmark.ipynb"):
        code = _code(root, name)
        assert "run_m3_libero" not in code
        assert "aggregate_m3_results" not in code
        assert "promotion" not in code.lower()
