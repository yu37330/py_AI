"""Regression guards for Notebook 73 child import visibility."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_logged_wrapper_injects_repo_root_into_pythonpath():
    source = (ROOT / "tools/colab/run_m3_training_smoke_logged.py").read_text(encoding="utf-8")
    assert "def _prepend_pythonpath" in source
    assert 'env["PYTHONPATH"] = os.pathsep.join(parts)' in source
    assert "_prepend_pythonpath(env, repo)" in source
    assert '"repo_pythonpath_injected": True' in source


def test_smoke_orchestrator_children_inherit_environment():
    source = (ROOT / "tools/colab/run_m3_training_smoke.py").read_text(encoding="utf-8")
    assert "subprocess.run(cmd, cwd=str(repo), check=True, env=os.environ.copy())" in source
