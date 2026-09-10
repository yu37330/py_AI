import json
from pathlib import Path


PIN = "6e907ef5663f37b85010d0d1620ebc45d539105e"


def test_notebook_73_is_pinned_smoke_only():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "colab/73_m3_a100_training_smoke.ipynb").read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )
    assert PIN in code
    assert "run_m3_training_smoke.py" in code
    assert "PARC_M3_EXECUTE" in code
    assert "benchmark" not in code.lower().replace("has not started", "")


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
