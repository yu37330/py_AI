import ast
import importlib.util
import json
from pathlib import Path


D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_payload(checkpoint_ref: str) -> dict:
    return {
        "status": "PASS",
        "model": "pi05",
        "mode": "smoke",
        "order": "forward",
        "track": "equal_data",
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": 20260906,
        "effective_batch_size": 32,
        "samples_consumed": 64,
        "optimizer_updates": 2,
        "checkpoint_ref": checkpoint_ref,
    }


def test_checkpoint_rebase_uses_preserved_model_benchmark_relative_path(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/run_m3_minimal_simulator_smoke.py"
    module = _load_module(path, "m3_minimal_simulator_smoke_rebase")

    persist = tmp_path / "persist"
    relocated = persist / "model-benchmark-v1/m3-training-smoke-v1/run/pi05/checkpoint"
    relocated.mkdir(parents=True)
    original = Path(
        "/content/drive/MyDrive/parc2026-cache/"
        "model-benchmark-v1/m3-training-smoke-v1/run/pi05/checkpoint"
    )
    monkeypatch.setenv("PARC_M3_ALLOW_ARTIFACT_REBASE", "1")

    resolved, rebased = module._resolve_checkpoint_ref(
        {"checkpoint_ref": str(original)}, persist_root=persist
    )
    assert rebased is True
    assert resolved == relocated.resolve()


def test_training_view_does_not_mutate_original_evidence(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/colab/run_m3_minimal_simulator_smoke.py"
    module = _load_module(path, "m3_minimal_simulator_smoke_view")

    persist = tmp_path / "persist"
    relocated = persist / "model-benchmark-v1/m3-training-smoke-v1/run/pi05/checkpoint"
    relocated.mkdir(parents=True)
    original_checkpoint = (
        "/content/drive/MyDrive/parc2026-cache/"
        "model-benchmark-v1/m3-training-smoke-v1/run/pi05/checkpoint"
    )
    source = tmp_path / "source/train_result.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_training_payload(original_checkpoint)) + "\n", encoding="utf-8")
    before = source.read_bytes()
    monkeypatch.setenv("PARC_M3_ALLOW_ARTIFACT_REBASE", "1")

    data, view_path, rebased = module._prepare_training_view(
        source,
        model="pi05",
        persist_root=persist,
        output_root=tmp_path / "attempt",
    )

    assert rebased is True
    assert source.read_bytes() == before
    assert view_path != source
    assert view_path.is_file()
    assert data["checkpoint_ref"] == str(relocated.resolve())
    assert data["artifact_relocation"]["source_train_result_mutated"] is False


def test_organizer_launcher_is_generic_guarded_and_uses_same_74_helpers():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/benchmark/run_m3_organizer_simulator_smoke.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))

    assert "google.colab" not in text
    assert "PARC_LOCAL_SCRATCH_ROOT" in text
    assert "PARC_PERSIST_ROOT" in text
    assert 'env["PARC_ROOT"] = str(local_root)' in text
    assert 'env["PARC_DRIVE_ROOT"] = str(persist_root)' in text
    assert 'env["PARC_M3_ALLOW_ARTIFACT_REBASE"] = "1"' in text
    assert "PARC_EXPECTED_SOURCE_SHA" in text
    assert "nvidia-smi" in text
    assert "environment_inventory.json" in text
    assert "handoff_validation.json" in text
    assert "prepare_m3_disk_headroom.py" in text
    assert "prepare_m3_libero_noninteractive_config.py" in text
    assert "prepare_m3_simulator_runtimes.py" in text
    assert "prepare_m3_mujoco_compat.py" in text
    assert "run_m3_minimal_simulator_smoke.py" in text
    assert '"probes_rerun": False' in text
    assert '"benchmark_training_started": False' in text
    assert '"final_800_episode_evaluation_started": False' in text
    assert "1800" in text  # docstring states that this launcher never starts it
    assert "=== ORGANIZER GPU 74 EQUIVALENT: PASS ===" in text
