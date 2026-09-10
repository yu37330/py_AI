import json
from pathlib import Path


PIN = "a7ad0ac5c4ff5c7b21e904befebc0a9d4610ee0f"


def test_notebook_73_is_pinned_smoke_only_and_bootstraps_fresh_runtime():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "colab/73_m3_a100_training_smoke.ipynb").read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )
    assert PIN in code
    assert "prepare_m3_training_runtimes.py" in code
    assert "run_m3_training_smoke_logged.py" in code
    assert "venv-openvla-oft-m3/bin/python" in code
    assert "PARC_M3_EXECUTE" in code
    assert "RUNTIME SETUP LOG TAIL" in code
    assert "ORCHESTRATOR STATUS" in code
    assert "ORCHESTRATOR LOG TAIL" in code
    assert "LATEST SMOKE PLAN" in code
    assert "LATEST TRAIN RESULT" in code
    assert "--mode', 'benchmark" not in code
    assert '"--mode", "benchmark"' not in code
    assert "mode='benchmark'" not in code


def test_runtime_setup_rebuilds_ephemeral_colab_dependencies_without_probes():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/prepare_m3_training_runtimes.py").read_text()
    for token in (
        "setup_train.sh",
        'SMOL_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"',
        'OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"',
        "prepare_m3_openvla_wandb_compat.py",
        '"probes_rerun": False',
        '"benchmark_training_started": False',
        "runtime_preflight",
    ):
        assert token in text
    assert "run_m3_pi05_batch_probe.py" not in text
    assert "run_m3_smolvla_batch_probe.py" not in text
    assert "run_m3_openvla_batch_probe.py" not in text


def test_logged_smoke_wrapper_persists_child_stdout_stderr_without_enabling_benchmark():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/run_m3_training_smoke_logged.py").read_text()
    for token in (
        "run_m3_training_smoke.py",
        "stderr=subprocess.STDOUT",
        "orchestrator.log",
        "orchestrator_status.json",
        '"benchmark_training_started": False',
        '"full_1800_second_run_started": False',
        "sys.executable",
    ):
        assert token in text
    assert "run_m3_benchmark_order.py" not in text


def test_training_smoke_reuses_one_schedule_for_forward_reverse():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/run_m3_training_smoke.py").read_text()
    assert "TRAINING_SCHEDULE_SEED = 20260906" in text
    assert 'ORDERS = ("forward", "reverse")' in text
    assert "forward_reverse_reuse_same_schedule" in text
    assert "SEED_ORDER" not in text
    assert "20260907" not in text


def test_training_smoke_never_materializes_openvla_merged_checkpoint():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/run_m3_training_smoke.py").read_text()
    assert "post_training_command" in text
    assert "do NOT execute post_training_command" in text
    assert '"openvla_smoke_merged_checkpoint_materialized": False' in text
    assert '"openvla_merge_deferred_to_evaluation": True' in text
    assert '"full_1800_second_run_started": False' in text


def test_contract_separates_training_schedule_seed_from_eval_seed_set():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "experiments/plans/m3_model_adapter_contract_v1.json").read_text())
    assert contract["training_schedule_seed"] == 20260906
    assert contract["evaluation_seed_set"] == [20260906, 20260907]
    assert contract["sampling"]["forward_reverse_reuse_same_training_schedule_seed"] is True
    assert contract["smoke"]["openvla_merged_checkpoint_materialized"] is False
    assert contract["smoke"]["full_1800_second_run_started"] is False
