from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"


def test_openvla_driver_reads_the_actual_69c_source_hash_field():
    source = (ROOT / "tools/benchmark/m3_openvla_adapter_driver.py").read_text(encoding="utf-8")
    assert 'contract.get("source_episode_ids_sha256") != D10_HASH' in source
    assert 'contract.get("selected_episode_ids_sha256") != D10_HASH' not in source


def test_openvla_train_entry_reads_the_actual_69c_source_hash_field():
    source = (ROOT / "tools/benchmark/m3_openvla_train_entry.py").read_text(encoding="utf-8")
    assert 'contract.get("source_episode_ids_sha256") != D10_HASH' in source
    assert 'contract.get("selected_episode_ids_sha256") != D10_HASH' not in source


def test_training_smoke_and_benchmark_runner_read_actual_69c_source_hash_field():
    for relative in (
        "tools/colab/run_m3_training_smoke.py",
        "tools/colab/run_m3_benchmark_order.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'contract.get("source_episode_ids_sha256") != D10_HASH' in source
        assert 'contract.get("selected_episode_ids_sha256") != D10_HASH' not in source


def test_smoke_results_do_not_claim_benchmark_training_started():
    for relative in (
        "tools/benchmark/m3_openvla_train_entry.py",
        "tools/benchmark/m3_lerobot_train_entry_v2.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '"benchmark_training_started": args.mode == "benchmark"' in source
        assert '"benchmark_training_started": True' not in source

    smoke_runner = (ROOT / "tools/colab/run_m3_training_smoke.py").read_text(encoding="utf-8")
    assert '"benchmark_training_started": False' in smoke_runner
