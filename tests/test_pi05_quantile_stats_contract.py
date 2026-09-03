from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quantile_stats_utility_contract():
    text = (ROOT / "tools/data/ensure_pi05_quantile_stats.py").read_text()
    assert '"observation.state"' in text
    assert '"action"' in text
    assert '"q01"' in text
    assert '"q99"' in text
    assert "np.quantile" in text
    assert "stats.json" in text
    assert "PI05 Quantile Stats Gate: PASS" in text


def test_colab_runner_requires_quantile_gate_before_ga_probe():
    text = (ROOT / "tools/colab/run_pi05_group_aware_ablation.py").read_text()
    quantile_pos = text.index("ensure_pi05_quantile_stats.py")
    ga_pos = text.index("Mandatory GA=8 runtime probe") if "Mandatory GA=8 runtime probe" in text else text.index("SMOKE_GA")
    assert quantile_pos < ga_pos
    assert "PI05 Quantile Stats Gate: PASS" in text
    assert 'PYTHONUNBUFFERED' in text
