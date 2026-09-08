"""Regression contract for M3 runner provenance and no-training preflight."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
SMOL_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"


def test_runner_plan_freezes_budget_sources_and_execution_guard():
    plan = json.loads((ROOT / "experiments/plans/model_benchmark_runner_v1.json").read_text(encoding="utf-8"))
    assert plan["status"] == "FROZEN_PREFLIGHT"
    assert plan["selected_dataset"]["variant"] == "V2_SQRT_BALANCED_RAW"
    assert plan["selected_dataset"]["episode_count"] == 10758
    assert plan["selected_dataset"]["frame_count"] == 1620614
    assert plan["selected_dataset"]["episode_ids_sha256"] == EXPECTED_HASH
    assert plan["tracks"]["equal_data_exposure"]["sample_budget"] == 4800
    assert plan["tracks"]["equal_wall_time"]["train_loop_sec"] == 1800
    assert plan["models"]["smolvla"]["source_ref"] == SMOL_REF
    assert plan["models"]["openvla_oft"]["source_ref"] == OPENVLA_REF
    assert plan["execution_guard"]["default_mode"] == "preflight"
    assert plan["execution_guard"]["preflight_starts_training"] is False
    assert plan["execution_guard"]["execute_environment_variable"] == "PARC_M3_EXECUTE"


def test_candidate_yamls_match_registry_and_have_no_todos():
    smol = (ROOT / "experiments/configs/smolvla.yaml").read_text(encoding="utf-8")
    openvla = (ROOT / "experiments/configs/openvla_oft.yaml").read_text(encoding="utf-8")
    assert SMOL_REF in smol
    assert "optimizer_lr: 1.0e-4" in smol
    assert "PROBE_ON_A100" in smol
    assert "TODO_" not in smol
    assert OPENVLA_REF in openvla
    assert "https://github.com/small-zeng/openvla-oft.git" in openvla
    assert "learning_rate: 5.0e-4" in openvla
    assert "lora_r: 32" in openvla
    assert "dataset_bridge: lerobot_streaming" in openvla
    assert "TODO_" not in openvla


def test_preflight_validator_cannot_start_training():
    source = (ROOT / "tools/benchmark/validate_m3_runner_preflight.py").read_text(encoding="utf-8")
    assert "READY_FOR_BATCH_PROBE" in source
    assert '"training_started": False' in source
    assert "subprocess.Popen" not in source
    assert "lerobot-train" not in source
    assert "vla-scripts/finetune.py" not in source


def test_source_checkout_preflight_only_fetches_and_validates_entries():
    source = (ROOT / "tools/colab/run_m3_runner_preflight.py").read_text(encoding="utf-8")
    assert SMOL_REF in source and OPENVLA_REF in source
    assert "src/lerobot/scripts/lerobot_train.py" in source
    assert "vla-scripts/finetune.py" in source
    assert "READY_FOR_BATCH_PROBE" in source
    assert "Training started: False" in source
    assert "subprocess.Popen" not in source
    assert "lerobot-train" not in source
    assert "accelerate launch" not in source


def test_openvla_training_adapter_adds_only_expected_window_dimension():
    sys.path.insert(0, str(ROOT / "tools/data"))
    try:
        spec = importlib.util.spec_from_file_location(
            "openvla_streaming_training_adapter",
            ROOT / "tools/data/openvla_streaming_training_adapter.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    window = {
        "dataset_name": "parc_libero_selected_streaming",
        "episode_id": 12,
        "timestep": 7,
        "observation": {
            "image_primary": np.zeros((256, 256, 3), dtype=np.uint8),
            "image_wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            "proprio": np.arange(8, dtype=np.float32),
        },
        "task": {"language_instruction": b"pick up the object"},
        "action": np.zeros((8, 7), dtype=np.float32),
    }
    got = module.to_openvla_rlds_batch(window)
    assert got["observation"]["image_primary"].shape == (1, 256, 256, 3)
    assert got["observation"]["image_wrist"].shape == (1, 256, 256, 3)
    assert got["observation"]["proprio"].shape == (1, 8)
    assert got["action"].shape == (8, 7)
    assert got["task"]["language_instruction"] == b"pick up the object"
    assert got["episode_id"] == 12 and got["timestep"] == 7
