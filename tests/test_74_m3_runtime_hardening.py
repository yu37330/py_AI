import ast
from pathlib import Path


def test_simulator_setup_forces_headless_graphics_contract():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_simulator_runtimes.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'child["MPLBACKEND"] = "Agg"' in text
    assert 'child["MUJOCO_GL"] = "egl"' in text
    assert 'child["PYOPENGL_PLATFORM"] = "egl"' in text
    assert "LIBERO_CONFIG_PATH" in text


def test_common_evaluator_forces_same_headless_graphics_contract():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/benchmark/m3_libero_executor.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'env["MPLBACKEND"] = "Agg"' in text
    assert 'env["MUJOCO_GL"] = "egl"' in text
    assert 'env["PYOPENGL_PLATFORM"] = "egl"' in text


def test_minimal_smoke_cleans_only_openvla_merge_created_by_this_attempt():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/run_m3_minimal_simulator_smoke.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'env["MPLBACKEND"] = "Agg"' in text
    assert 'env["MUJOCO_GL"] = "egl"' in text
    assert 'env["PYOPENGL_PLATFORM"] = "egl"' in text
    assert "merged_present_before" in text
    assert "finalization_attempted" in text
    assert "cleanup_needed" in text
    assert "and not merged_present_before" in text
    assert "m3_openvla_dematerialize_checkpoint.py" in text
    assert "active_exception = sys.exc_info()[0] is not None" in text
    assert "check=False" in text


def test_openvla_dematerializer_preserves_persistent_training_artifacts():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/benchmark/m3_openvla_dematerialize_checkpoint.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'checkpoint.glob("model*.safetensors")' in text
    assert 'checkpoint / "lora_adapter"' in text
    assert 'checkpoint / "dataset_statistics.json"' in text
    assert "action_head--*checkpoint.pt" in text
    assert "proprio_projector--*checkpoint.pt" in text
    assert 'result["checkpoint_eval_ready"] = False' in text


def test_disk_headroom_cleanup_is_safe_and_fail_fast():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/prepare_m3_disk_headroom.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert 'LOCAL_MIN_FREE_GIB = 24.0' in text
    assert 'DRIVE_MIN_FREE_GIB = 20.0' in text
    assert '["uv", "cache", "clean"]' in text
    assert '"pip", "cache", "purge"' in text
    assert '["apt-get", "clean"]' in text
    assert '.cache/huggingface' in text
    assert "Do not touch ~/.cache/huggingface" in text
    assert "D10 data/manifests" in text
    assert "insufficient local /content headroom" in text
    assert "insufficient Google Drive headroom" in text


def test_evaluators_persist_attempt_scoped_partial_episode_evidence():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "tools/benchmark/m3_lerobot_eval_entry.py",
        "tools/benchmark/m3_openvla_eval_entry.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        ast.parse(text, filename=rel)
        assert "def _write_progress" in text
        assert 'status="RUNNING"' in text
        assert 'status="PASS"' in text
        assert '"episode_count": len(records)' in text
        assert 'partial.json' in text
