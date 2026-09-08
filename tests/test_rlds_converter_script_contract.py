"""Regression contract for the 69b RLDS converter's direct-script TFDS builder."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "tools/data/convert_lerobot_manifest_to_openvla_rlds.py"


def test_direct_script_builder_sets_explicit_tfds_pkg_dir_path():
    source = CONVERTER.read_text(encoding="utf-8")
    assert "class ParcLiberoSelected(tfds.core.GeneratorBasedBuilder):" in source
    assert "pkg_dir_path = Path(__file__).resolve().parent" in source
    assert "if __name__ == \"__main__\":" in source


def test_converter_still_uses_directory_readback_gate():
    source = CONVERTER.read_text(encoding="utf-8")
    assert "tfds.builder_from_directory(str(prepared_dir))" in source
    assert '"tfds_builder_from_directory": "PASS"' in source
