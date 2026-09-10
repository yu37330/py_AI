from pathlib import Path


def test_scheduled_openvla_adapter_symbol_is_defined_and_shared():
    root = Path(__file__).resolve().parents[1]
    adapter = (root / "tools/data/openvla_streaming_training_adapter.py").read_text()
    scheduled = (root / "tools/data/openvla_scheduled_training.py").read_text()
    stream = (root / "tools/data/openvla_scheduled_stream.py").read_text()

    assert "def to_openvla_rlds_batch(" in adapter
    assert "def to_upstream_window(" in adapter
    assert "return to_openvla_rlds_batch(window)" in adapter
    assert "from openvla_streaming_training_adapter import to_upstream_window" in scheduled
    assert "from openvla_streaming_training_adapter import to_upstream_window" in stream
    assert "batch_transform(to_upstream_window(" in scheduled
    assert "batch_transform(to_upstream_window(" in stream
