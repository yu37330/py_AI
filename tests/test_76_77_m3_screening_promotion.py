import ast
import json
from pathlib import Path


SCREENING_PIN = "659cbc51f84a466986dd7698b8d362358f3a24e0"
PROMOTION_PIN = "d21e4c231fb82a6aa90cfe556402372ffc473793"


def _notebook_code(root: Path, name: str) -> str:
    nb = json.loads((root / "colab" / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_76_forward_reverse_notebooks_share_screening_pin_and_attempt():
    root = Path(__file__).resolve().parents[1]
    forward = _notebook_code(root, "76a_m3_forward_screening_evaluation.ipynb")
    reverse = _notebook_code(root, "76b_m3_reverse_screening_evaluation.ipynb")
    for code in (forward, reverse):
        assert SCREENING_PIN in code
        assert "run_m3_screening_evaluation_order.py" in code
        assert "PARC_M3_EXECUTE" in code
        assert "--attempt', '1" in code or '"--attempt", "1"' in code
        assert "benchmark" not in code.lower()
    assert "--order', 'forward" in forward or '"--order", "forward"' in forward
    assert "--order', 'reverse" in reverse or '"--order", "reverse"' in reverse


def test_screening_runner_is_syntax_valid_and_frozen_at_80_per_checkpoint():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/run_m3_screening_evaluation_order.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "SCREENING_EPISODES = 80" in text
    assert '"--mode",\n                    "smoke"' in text
    assert "m3_promotion_record" in text
    assert "promotion_record.json" in text
    assert "final_800_episode_evaluation_started" in text
    assert "Use a new --attempt instead of deleting evidence" in text


def test_promotion_record_binds_training_sampling_and_evaluation_metrics():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/benchmark/m3_promotion_record.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "SCREENING_EPISODES = 80" in text
    assert 'training.get("sampling_schedule_sha256")' in text
    assert 'training.get("samples_consumed")' in text
    assert 'evaluation.get("metrics")' in text
    assert "evaluation does not point to exact training result" in text
    assert '"training_loss_used_for_promotion": False' in text


def test_77_controller_requires_complete_twelve_record_matrix():
    root = Path(__file__).resolve().parents[1]
    runner = root / "tools/colab/run_m3_promotion_controller.py"
    text = runner.read_text(encoding="utf-8")
    ast.parse(text, filename=str(runner))
    assert "EXPECTED_RECORDS = 12" in text
    assert "SCREENING_EPISODES_PER_CHECKPOINT = 80" in text
    assert "aggregate(records)" in text
    assert 'summary.get("primary_metric") != "simulator_success_rate"' in text
    assert 'summary.get("training_loss_used_for_promotion") is not False' in text
    assert '"automatic_final_training": False' in text


def test_77_notebook_is_cpu_controller_and_pinned():
    root = Path(__file__).resolve().parents[1]
    code = _notebook_code(root, "77_m3_forward_reverse_promotion_controller.ipynb")
    assert PROMOTION_PIN in code
    assert "run_m3_promotion_controller.py" in code
    assert "PARC_M3_EXECUTE" not in code
    assert "HF_TOKEN" not in code


def test_contract_splits_screening_80_from_final_800():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "experiments/plans/m3_libero_executor_v1.json").read_text(encoding="utf-8")
    )
    screening = contract["evaluation_protocol"]["m3_screening"]
    final = contract["evaluation_protocol"]["final_selected_candidate"]
    assert screening["episode_records_per_checkpoint"] == 80
    assert screening["checkpoint_count_complete_forward_reverse_matrix"] == 12
    assert screening["total_episode_records_complete_matrix"] == 960
    assert final["episode_records"] == 800
    assert final["candidate_count"] == 1
    assert contract["promotion_handoff"]["normal_complete_matrix_record_count"] == 12
    assert contract["promotion_handoff"]["training_loss_used"] is False
