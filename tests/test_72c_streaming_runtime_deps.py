"""Regression guard for OpenVLA-OFT streaming runtime dependencies."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_72c_prepare_pins_streaming_data_runtime():
    source = (ROOT / "tools/colab/prepare_m3_openvla_wandb_compat.py").read_text(encoding="utf-8")
    for token in (
        'PANDAS_VERSION = "2.2.3"',
        'PYARROW_VERSION = "17.0.0"',
        'AV_VERSION = "12.3.0"',
        '"pin_streaming_data_runtime"',
        'f"pandas=={PANDAS_VERSION}"',
        'f"pyarrow=={PYARROW_VERSION}"',
        'f"av=={AV_VERSION}"',
        '"verify_wandb_streaming_compat"',
        "import wandb, pandas, pyarrow, av",
        "wandb_streaming_import_ok",
    ):
        assert token in source


def test_streaming_adapter_requires_those_data_packages():
    source = (ROOT / "tools/data/openvla_lerobot_streaming.py").read_text(encoding="utf-8")
    assert "import pandas as pd" in source
    assert "import pyarrow.dataset as pads" in source
    assert "import pyarrow.parquet as pq" in source
    assert "import av" in source
