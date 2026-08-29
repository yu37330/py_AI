from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_cheap_ablation_runner_uses_native_dataset_episode_subset():
    path = ROOT / "examples" / "pi05_libero_finetune" / "scripts" / "cheap_ablation_pi05.sh"
    text = path.read_text(encoding="utf-8")
    assert "--dataset.episodes=$EPISODES_JSON" in text
    assert "--dataset.root=$PI05_DATASET_ROOT" in text
    assert "--dataset.revision" not in text
    assert "LEROBOT_GRAD_ACCUM" in text
    assert "--peft.method_type=LORA" in text
    assert "--peft.r=16" in text
    assert "screening_only" in text
    assert "Do not promote a dataset variant from training loss alone" in text
