from pathlib import Path


def test_training_smoke_reuses_only_validated_completed_models():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/run_m3_training_smoke.py").read_text()

    for token in (
        "completed_results",
        "reuse validated completed smoke",
        "completed_models=set(completed_results)",
        "_validate_result(",
        "validated_partial_resume_supported",
    ):
        assert token in text

    assert "partial smoke output exists for incomplete model" in text
    assert "PARC_M3_SMOKE_RESET=1" in text


def test_completed_model_skip_does_not_enable_benchmark_or_delete_evidence():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/run_m3_training_smoke.py").read_text()

    assert 'plan.get("mode") != "smoke"' in text
    assert '"benchmark_training_started": False' in text
    assert '"full_1800_second_run_started": False' in text
    assert "if run_root.exists() and reset:" in text
    assert "if model in completed_results:" in text
