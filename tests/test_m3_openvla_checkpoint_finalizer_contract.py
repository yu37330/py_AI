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


def test_openvla_dematerializer_removes_only_ephemeral_merged_weights():
    root = Path(__file__).resolve().parents[1]
    cleanup = (root / "tools/benchmark/m3_openvla_dematerialize_checkpoint.py").read_text()
    for token in (
        'checkpoint / "lora_adapter"',
        'checkpoint / "dataset_statistics.json"',
        'checkpoint.glob("action_head--*checkpoint.pt")',
        'checkpoint.glob("proprio_projector--*checkpoint.pt")',
        'checkpoint.glob("model*.safetensors")',
        'checkpoint / "model.safetensors.index.json"',
        'result["checkpoint_eval_ready"] = False',
        'result["merged_checkpoint_dematerialized"] = True',
        '"lora_components_and_d10_stats"',
    ):
        assert token in cleanup
    # Cleanup is deliberately narrow: it may unlink only the collected merged
    # weight/index targets. Persistent LoRA/components/stats are pre/post guards.
    assert "shutil.rmtree" not in cleanup
    assert "rmtree" not in cleanup
    assert "unlink()" in cleanup


def test_training_smoke_skips_post_training_openvla_merge():
    root = Path(__file__).resolve().parents[1]
    smoke = (root / "tools/colab/run_m3_training_smoke.py").read_text()
    assert "do NOT execute post_training_command" in smoke
    assert '"openvla_smoke_merged_checkpoint_materialized": False' in smoke
    assert '"openvla_merge_deferred_to_evaluation": True' in smoke
