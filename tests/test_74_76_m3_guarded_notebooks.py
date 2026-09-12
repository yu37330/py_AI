import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    notebook = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_notebook_74_is_explicit_guarded_smoke_only():
    src = _source("colab/74_m3_guarded_smoke_a100.ipynb")
    assert "51b1f1e71a53c559f46a688862f74c60c64048ff" in src
    assert "EXECUTE_FORWARD = False" in src
    assert "EXECUTE_REVERSE = False" in src
    assert "env['PARC_M3_EXECUTE'] = '1'" in src
    assert "'--mode', 'smoke'" in src
    assert "'--order', 'forward'" in src
    assert "'--order', 'reverse'" in src
    assert "'--track', 'equal_data'" in src
    assert "64" in src
    assert "optimizer_updates_per_model_per_order': 2" in src
    assert "READY_FOR_M3_BENCHMARK" in src
    assert "benchmark_training_started': False" in src
    assert "python', '-u'" not in src
    assert "'-m', 'tools.benchmark.run_m3_guarded'" in src
    assert "Do not regenerate or reselect" in src
    assert "full_rlds_materialized" in src


def test_notebook_76_never_auto_starts_benchmark_or_evaluation():
    src = _source("colab/76_m3_guarded_benchmark_a100.ipynb")
    assert "447d7f40de86c2a4b6f0bef5b8e7891dd6f66f17" in src
    assert "EXECUTE_BENCHMARK = False" in src
    assert "env['PARC_M3_EXECUTE'] = '1'" in src
    assert "'--mode', 'benchmark'" in src
    assert "ORDER = 'forward'" in src
    assert "TRACK = 'equal_data'" in src
    assert "4800" in src
    assert "150" in src
    assert "1800" in src
    assert "READY_FOR_M3_SIMULATOR_EVALUATION" in src
    assert "simulator_evaluation_completed': False" in src
    assert "promotion_completed': False" in src
    assert "No model has been promoted" in src
    assert "'-m', 'tools.benchmark.run_m3_guarded'" in src


def test_notebook_76_requires_smoke_gate_before_benchmark():
    src = _source("colab/76_m3_guarded_benchmark_a100.ipynb")
    assert "m3_smoke_summary.json" in src
    assert "smoke.get('status') != 'READY_FOR_M3_BENCHMARK'" in src
    assert "Smoke summary unexpectedly claims benchmark training started" in src
