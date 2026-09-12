import ast
import json
from pathlib import Path


def test_openvla_evaluation_wrapper_is_syntax_valid():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/benchmark/run_m3_libero_evaluation.py"
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_intermediate_openvla_eval_always_has_cleanup_path():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/benchmark/run_m3_libero_evaluation.py").read_text(encoding="utf-8")
    assert "cleanup_command" in text
    assert 'model == "openvla_oft"' in text
    assert 'os.environ.get(KEEP_MERGED_ENV) != "1"' in text
    assert "materialized_here" not in text.split("finally:", 1)[1].split("status =", 1)[0]
    assert "m3_openvla_dematerialize_checkpoint.py" in text


def test_cleanup_is_not_limited_to_wrapper_materialized_weights():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/benchmark/run_m3_libero_evaluation.py").read_text(encoding="utf-8")
    assert "openvla_merged_present_before_evaluation" in text
    assert "openvla_dematerialized_after_evaluation" in text
    assert "merged_present_before" in text


def test_contract_requires_keep_only_for_final_selected_openvla():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "experiments/plans/m3_libero_executor_v1.json").read_text(encoding="utf-8")
    )
    openvla = contract["models"]["openvla_oft"]
    lifecycle = contract["openvla_storage_lifecycle"]
    assert openvla["default_dematerialize_merged_weights_after_evaluation"] is True
    assert openvla["default_cleanup_applies_even_if_weights_were_merged_before_wrapper"] is True
    assert openvla["final_selected_candidate_must_keep_merged_until_artifact_freeze"] is True
    assert lifecycle["intermediate_evaluation_always_dematerializes_without_keep_opt_in"] is True
    assert lifecycle["cleanup_does_not_depend_on_which_stage_created_merged_weights"] is True
    assert "PARC_M3_KEEP_MERGED_OPENVLA=1" in lifecycle["final_candidate_keep_rule"]
