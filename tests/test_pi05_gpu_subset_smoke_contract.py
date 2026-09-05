from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "tools/colab/run_pi05_subset_gpu_smoke.py"


def test_gpu_subset_smoke_is_one_effective_step_with_ga8():
    text = SMOKE.read_text()
    assert '"ABLATION_BS": "1"' in text
    assert '"ABLATION_GA": "8"' in text
    assert '"ABLATION_STEPS": "1"' in text
    assert '"PI05_VIDEO_BACKEND": "pyav"' in text
    assert '"PI05 GPU SUBSET SMOKE: PASS"' in text


def test_gpu_subset_smoke_uses_non_contiguous_episode_subset():
    text = SMOKE.read_text()
    assert "min(100, last)" in text
    assert "max(0, last - 1)" in text
    assert "dataset-quality experiment" in text


def test_gpu_subset_smoke_does_not_call_full_ablation_runner():
    text = SMOKE.read_text()
    assert "run_pi05_group_aware_ablation.py" not in text
    assert "cheap_ablation_pi05.sh" in text


def test_gpu_subset_smoke_ensures_quantiles_before_training():
    text = SMOKE.read_text()
    quantile_pos = text.index("ensure_pi05_quantile_stats.py")
    training_pos = text.index("cheap_ablation_pi05.sh")
    assert quantile_pos < training_pos
    assert '"q01"' in text
    assert '"q99"' in text
    assert "PI05 Quantile Stats Gate: PASS" in text
