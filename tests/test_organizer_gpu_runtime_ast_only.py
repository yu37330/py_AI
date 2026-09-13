import ast
from pathlib import Path


def test_organizer_runtime_python_files_parse():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "tools/benchmark/run_m3_organizer_simulator_smoke.py",
        "tools/colab/prepare_m3_disk_headroom.py",
        "tools/colab/run_m3_minimal_simulator_smoke.py",
    ):
        path = root / rel
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
