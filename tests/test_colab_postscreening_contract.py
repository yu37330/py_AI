import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    nb = json.loads((ROOT / path).read_text())
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


def test_top2_tiebreak_contract():
    text = _text("colab/65_pi05_top2_tiebreak_l4.ipynb")
    assert "V1_MULTI" in text and "V2_SQRT" in text
    assert "20260906" in text
    assert "STILL_TIED" in text
    assert "provisional_best_dataset_recipe.json" in text


def test_model_benchmark_smoke_is_lightweight_and_non_training():
    text = _text("colab/69_model_benchmark_bringup_smoke.ipynb")
    assert "V2_SQRT_BALANCED_RAW" in text
    assert "model-benchmark-smoke-v1" in text
    assert "openvla_selected_subset_rlds_missing" in text
    assert "No training or model-weight loading was performed" in text
    assert "lerobot-smolvla" in text and "openvla-oft" in text


def test_model_benchmark_three_candidates_and_two_protocols():
    text = _text("colab/70_model_benchmark_a100.ipynb")
    for model in ("pi05", "smolvla", "openvla_oft"):
        assert model in text
    assert "equal_data_exposure" in text
    assert "equal_wall_time" in text
    assert "openvla_selected_subset_rlds_missing" in text


def test_generalization_requires_m3_and_track1_regression_gate():
    text = _text("colab/75_generalization_screening_a100.ipynb")
    assert "model_shortlist.json" in text
    assert "track2_success_rate" in text
    assert "track1_success_rate" in text
    for aug in ("brightness", "contrast", "mild_color_jitter", "mild_crop_resize"):
        assert aug in text
    assert "targeted_public_supplemental" in text


def test_track3_forbids_naive_reversal_and_preregisters_ratios():
    text = _text("colab/80_track3_inverse_factory.ipynb")
    assert "forbid_naive_action_reversal" in text
    assert "time_reverse" in text
    assert "action_reverse" in text
    for ratio in ("0.00", "0.05", "0.10", "0.20"):
        assert ratio in text
    for gate in ("simulator_success", "replay_success", "converted_lerobot_20hz"):
        assert gate in text


def test_run_a_freeze_requires_all_pretraining_gates():
    text = _text("colab/90_run_a_freeze.ipynb")
    for stage in ("D10", "M3", "G1", "T3", "ORGANIZER"):
        assert stage in text
    assert "libero_combined_20hz" in text
    assert "READY" in text
    assert "20000" in text
