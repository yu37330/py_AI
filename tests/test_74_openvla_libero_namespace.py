import ast
from pathlib import Path


def test_openvla_libero_setup_uses_pinned_checkout_via_venv_pth():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_simulator_runtimes.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))

    assert "def _install_source_pth" in text
    assert "parc_m3_pinned_libero.pth" in text
    assert "site.getsitepackages()[0]" in text
    assert "_install_source_pth(python_bin, libero_source, log)" in text
    assert "libero/libero/__init__.py" in text
    assert "LIBERO import path mismatch" in text
    assert "openvla_libero_import_ok" in text

    # The pinned upstream revision uses a namespace-style source layout that is
    # not made importable by its legacy setup.py/find_packages() editable install.
    # Do not reintroduce the misleading metadata-only `uv pip install -e` path.
    assert '"-e", str(libero_source)' not in text


def test_pinned_libero_checkout_is_not_modified_to_fix_packaging():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_simulator_runtimes.py"
    text = path.read_text(encoding="utf-8")

    assert 'LIBERO_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"' in text
    assert "find_namespace_packages" not in text
    assert "_install_source_pth(python_bin, libero_source, log)" in text
    assert "target.write_text(str(root)" in text
    assert "libero_source / 'setup.py'" not in text
    assert 'libero_source / "setup.py"' not in text
