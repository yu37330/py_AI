"""Static contract for 69c validator provenance and storage policy."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/data/validate_lerobot_openvla_streaming_bridge.py"


def test_69c_validator_requires_69b_and_fixed_openvla_revision():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert 'report.get("status") != "SMOKE_PASS"' in source
    assert '"STREAMING_BRIDGE_RECOMMENDED"' in source
    assert "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239" in source
    assert "e4287e94541f459edc4feabc4e181f537cd569a8" in source
    assert '"tfds_builder_from_directory"' in source


def test_69c_contract_forbids_full_rlds_materialization():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert '"bridge_type": "lerobot_streaming"' in source
    assert '"full_rlds_materialized": False' in source
    assert '"dataset_statistics": statistics' in source
    assert '"streaming_window_count": sample_window_count' in source
