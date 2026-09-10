from pathlib import Path


def test_openvla_checkpoint_finalizer_is_post_training_and_eval_ready():
    root = Path(__file__).resolve().parents[1]
    finalizer = (root / "tools/benchmark/m3_openvla_finalize_checkpoint.py").read_text()
    orchestrator = (root / "tools/benchmark/m3_model_adapters.py").read_text()
    for token in (
        "PeftModel.from_pretrained",
        "merge_and_unload",
        "merged.save_pretrained",
        "dataset_statistics.json",
        "action_head--*checkpoint.pt",
        "proprio_projector--*checkpoint.pt",
        'result["checkpoint_eval_ready"] = True',
        'result["merge_excluded_from_train_wall_time"] = True',
    ):
        assert token in finalizer
    assert "build_post_training_command" in orchestrator
    assert "m3_openvla_finalize_checkpoint.py" in orchestrator
    assert "post_training_work_excluded_from_train_wall_time" in orchestrator
    assert "OpenVLA checkpoint finalizer did not produce an eval-ready checkpoint" in orchestrator


def test_openvla_finalizer_keeps_merge_outside_training_timer():
    root = Path(__file__).resolve().parents[1]
    train_entry = (root / "tools/benchmark/m3_openvla_train_entry.py").read_text()
    finalizer = (root / "tools/benchmark/m3_openvla_finalize_checkpoint.py").read_text()
    assert "stopped = time.perf_counter()" in train_entry
    assert "save_training_checkpoint" in train_entry
    assert "merge_and_unload" not in train_entry
    assert "merge_and_unload" in finalizer
