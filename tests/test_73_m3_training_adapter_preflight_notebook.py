import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PIN = "d0745cd15ac7a6bd87a480e98ab21d95d3a9ca9d"
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"


def test_73_is_preflight_only_and_pins_runtime():
    nb = json.loads((ROOT / "colab/73_m3_training_adapter_preflight.ipynb").read_text(encoding="utf-8"))
    code = "".join(
        line
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )
    ast.parse(code)
    assert RUNTIME_PIN in code
    assert D10_HASH in code
    assert "m3_batch_probe_summary.json" in code
    assert "m3_sampling_schedule.py" in code
    assert "m3_training_adapters.py" in code
    assert "m3_equal_data_seed_20260906.json" in code
    assert "READY_FOR_M3_RUNNER_IMPLEMENTATION" in code
    assert "READY_FOR_GUARDED_SMOKE_IMPLEMENTATION" in code
    assert "run_count" in code
    assert "training_started" in code
    assert "74_m3_guarded_smoke_a100.ipynb" in code
    assert "PARC_M3_EXECUTE=1" not in code
    for forbidden in ("lerobot-train", "finetune.py", "torchrun", "accelerate launch"):
        assert forbidden not in code


def test_73_refuses_to_regenerate_or_reselect_d10():
    text = (ROOT / "colab/73_m3_training_adapter_preflight.ipynb").read_text(encoding="utf-8")
    assert "Exact D10 group-aware manifest not found; do not regenerate/reselect it" in text
    assert "V2_SQRT_BALANCED_RAW.json" in text
