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


def test_openvla_rlds_bridge_smoke_is_exact_and_capacity_gated():
    text = _text("colab/69b_openvla_selected_rlds_smoke.ipynb")
    assert "V2_SQRT_BALANCED_RAW" in text
    assert "rlds_smoke_recovery.py" in text
    assert "bridge_smoke_status.json" in text
    assert "conversion_contract.json" in text
    assert "full RLDS" in text
    assert "--max-episodes 8" in (ROOT / "colab/POST_SCREENING.md").read_text()

    recovery = (ROOT / "tools/colab/rlds_smoke_recovery.py").read_text()
    for token in ("--max-episodes", "8", "av==12.3.0", "--only-binary", "SMOKE_PASS", "STREAMING_BRIDGE_RECOMMENDED", "bridge_capacity_decision.json", "bridge_smoke_status.json"):
        assert token in recovery
    assert "av>=12,<15" not in recovery

    bridge = (ROOT / "tools/data/convert_lerobot_manifest_to_openvla_rlds.py").read_text()
    for token in (
        "observation.images.front",
        "observation.images.wrist",
        "episode_ids_sha256",
        "builder_from_directory",
        "projected_full_gib",
        "raw_action_preserved",
        "no_noop_filter_applied",
    ):
        assert token in bridge
    assert "conversion_contract.json intentionally NOT written" in bridge


def test_model_benchmark_budget_is_frozen_screening_contract():
    budget = json.loads((ROOT / "experiments/plans/model_benchmark_budget_v1.json").read_text())
    assert budget["status"] == "FROZEN"
    assert budget["selected_dataset_variant"] == "V2_SQRT_BALANCED_RAW"
    assert budget["equal_data_exposure"]["sample_budget"] == 4800
    assert budget["equal_wall_time"]["train_loop_sec"] == 1800
    assert budget["promotion"]["max_models"] == 2
    assert budget["promotion"]["training_loss_alone_is_not_sufficient"] is True


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
