import ast
import json
from pathlib import Path


def test_final_candidate_runner_is_syntax_valid_and_double_guarded():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/submission/run_track3_final_candidate.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "PARC_M3_EXECUTE" in text
    assert 'p.add_argument("--execute", action="store_true")' in text
    assert 'mode="benchmark"' in text
    assert "runtime_preflight" in text
    assert "gpu_info" in text
    assert "MIN_A100_VRAM_MIB = 38_000" in text
    assert "link_result" in text
    assert '"automatic_evaluation": False' in text
    assert '"automatic_upload": False' in text


def test_final_runner_uses_one_candidate_and_frozen_training_seed():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/submission/run_track3_final_candidate.py").read_text(encoding="utf-8")
    assert "TRAINING_SCHEDULE_SEED = 20260906" in text
    assert "batch_summary[\"models\"][candidate]" in text
    assert "runtime = runtimes[candidate]" in text
    assert 'order="forward"' in text
    assert '"order_is_not_a_final_comparison_variable": True' in text
    assert "final-run schedule must use frozen training schedule seed 20260906" in text


def test_final_runner_never_deletes_existing_final_output():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/submission/run_track3_final_candidate.py").read_text(encoding="utf-8")
    assert "final_output_not_empty" in text
    assert "shutil.rmtree" not in text
    assert ".unlink()" not in text


def test_track3_contract_requires_linked_final_run():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "experiments/plans/track3_final_candidate_freeze_v1.json").read_text(encoding="utf-8")
    )
    assert contract["schema_version"] >= 3
    assert contract["final_run"]["training_schedule_seed"] == 20260906
    assert contract["final_run"]["equal_data_sample_budget"] == 4800
    assert contract["final_run"]["equal_wall_train_loop_sec"] == 1800
    assert contract["final_run"]["raw_smoke_result_allowed"] is False
    assert contract["final_training"]["final_run_config_link_required"] is True
    assert contract["final_evaluation"]["episode_count"] == 800
