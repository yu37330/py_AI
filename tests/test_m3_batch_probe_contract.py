"""Regression contracts for the M3 A100 one-step batch probes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
SMOL_REF = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"


def test_probe_plan_freezes_effective_batch_and_candidate_search():
    plan = json.loads((ROOT / "experiments/plans/m3_batch_probe_v1.json").read_text(encoding="utf-8"))
    assert plan["status"] == "FROZEN"
    assert plan["probe_only"] is True
    assert plan["benchmark_training_started"] is False
    assert plan["selected_episode_ids_sha256"] == EXPECTED_HASH
    assert plan["hardware"]["required_gpu_family"] == "NVIDIA A100"
    assert plan["optimization"]["target_effective_batch_size"] == 32
    assert plan["optimization"]["one_optimizer_step_per_trial"] is True
    assert plan["optimization"]["candidate_micro_batches"]["pi05"] == [16, 8, 4, 2, 1]
    assert plan["optimization"]["candidate_micro_batches"]["smolvla"] == [32, 16, 8, 4, 2, 1]
    assert plan["optimization"]["candidate_micro_batches"]["openvla_oft"] == [8, 4, 2, 1]
    assert plan["promotion_gate"]["next_status"] == "READY_FOR_M3_RUNNER_IMPLEMENTATION"
    assert plan["promotion_gate"]["does_not_start_training"] is True


def test_openvla_candidate_freezes_pinned_libero_recipe():
    text = (ROOT / "experiments/configs/openvla_oft.yaml").read_text(encoding="utf-8")
    for token in (
        OPENVLA_REF,
        "lora_r: 32",
        "learning_rate: 5.0e-4",
        "use_l1_regression: true",
        "use_diffusion: false",
        "use_film: false",
        "num_images_in_input: 2",
        "use_proprio: true",
        "image_aug: true",
        "target_effective_batch_size: 32",
        "dataset_bridge: lerobot_streaming",
    ):
        assert token in text


def test_common_probe_gate_requires_71_hash_and_a100():
    source = (ROOT / "tools/benchmark/m3_batch_probe_common.py").read_text(encoding="utf-8")
    assert EXPECTED_HASH in source
    assert 'pf.get("status") != "READY_FOR_BATCH_PROBE"' in source
    assert 'pf.get("training_started") is not False' in source
    assert '"A100" not in name' in source
    assert "TARGET_EFFECTIVE_BATCH = 32" in source
    assert "while not stop.wait(0.5)" in source


def test_pi05_probe_is_one_step_exact_manifest_and_no_promotion():
    source = (ROOT / "tools/colab/run_m3_pi05_batch_probe.py").read_text(encoding="utf-8")
    assert '"ABLATION_MANIFEST": str(manifest)' in source
    assert '"ABLATION_STEPS": "1"' in source
    assert "grad_accum_for(micro_batch)" in source
    assert "optimizer_steps" in source
    assert "TARGET_EFFECTIVE_BATCH" in source
    assert "shutil.rmtree(out_root / run_name" in source
    assert "4800" not in source
    assert "1800" not in source


def test_smolvla_probe_uses_pinned_python_and_microstep_semantics():
    source = (ROOT / "tools/colab/run_m3_smolvla_batch_probe.py").read_text(encoding="utf-8")
    assert SMOL_REF in source
    assert '["uv", "python", "install", "3.12"]' in source
    assert 'f"--accelerator.gradient_accumulation.steps={ga}"' in source
    assert 'f"--steps={ga}"' in source
    assert '"--save_checkpoint=false"' in source
    assert '"--policy.path=lerobot/smolvla_base"' in source
    assert "4800" not in source
    assert "1800" not in source


def test_openvla_probe_uses_pinned_recipe_streaming_and_no_checkpoint():
    outer = (ROOT / "tools/colab/run_m3_openvla_batch_probe.py").read_text(encoding="utf-8")
    inner = (ROOT / "tools/benchmark/openvla_oft_one_step_probe.py").read_text(encoding="utf-8")
    assert OPENVLA_REF in outer
    assert '["uv", "python", "install", "3.10"]' in outer
    assert '"flash-attn==2.5.5"' in outer
    assert '"--nproc-per-node=1"' in outer
    assert "make_torch_iterable_dataset" in inner
    assert "max_episodes=8" in inner
    assert "vla.vision_backbone.set_num_images_in_input(2)" in inner
    assert "r=32" in inner
    assert "use_l1_regression=True" in inner
    assert "use_proprio=True" in inner
    assert "optimizer.step()" in inner
    assert '"checkpoint_saved": False' in inner
    assert '"benchmark_training_started": False' in inner
    assert "save_training_checkpoint" not in inner
    assert "4800" not in inner
    assert "1800" not in inner


def test_controller_requires_all_three_one_step_passes_before_runner_implementation():
    source = (ROOT / "tools/benchmark/validate_m3_batch_probes.py").read_text(encoding="utf-8")
    assert EXPECTED_HASH in source
    assert SMOL_REF in source and OPENVLA_REF in source
    assert 'result.get("status") != "PASS"' in source
    assert 'int(result.get("optimizer_steps", -1)) != 1' in source
    assert "bs * ga != 32" in source
    assert 'result.get("loss") is None' in source
    assert '"status": "READY_FOR_M3_RUNNER_IMPLEMENTATION"' in source
    assert '"sample_budget": 4800' in source
    assert '"optimizer_updates": 150' in source
    assert '"benchmark_training_started": False' in source
