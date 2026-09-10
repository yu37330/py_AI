from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/data"))

from tools.data.openvla_scheduled_training_source import (  # noqa: E402
    IMAGE_AUGMENT_KWARGS,
    augmentation_seed_pair,
    transform_scheduled_batch,
)


class _FakeSource:
    def load_rlds_batch(self, refs):
        return [
            {
                "episode_id": int(ref["episode_id"]),
                "timestep": int(ref["timestep"]),
                "observation": {},
            }
            for ref in refs
        ]


def test_openvla_image_aug_recipe_matches_pinned_oft_contract():
    assert IMAGE_AUGMENT_KWARGS == {
        "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
        "random_brightness": [0.2],
        "random_contrast": [0.8, 1.2],
        "random_saturation": [0.8, 1.2],
        "random_hue": [0.05],
        "augment_order": [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    }


def test_augmentation_seed_is_deterministic_and_sample_specific():
    a = augmentation_seed_pair(20260906, 123)
    b = augmentation_seed_pair(20260906, 123)
    c = augmentation_seed_pair(20260906, 124)
    d = augmentation_seed_pair(20260907, 123)
    assert a == b
    assert a != c
    assert a != d
    assert all(0 <= x < 2**31 - 1 for x in a)


def test_transform_scheduled_batch_preserves_requested_identity_without_aug():
    refs = [
        {"sample_index": 0, "episode_id": 9, "timestep": 4},
        {"sample_index": 1, "episode_id": 2, "timestep": 8},
    ]
    transformed, identities = transform_scheduled_batch(
        _FakeSource(),
        refs,
        lambda x: {"seen": (x["episode_id"], x["timestep"])},
        run_seed=20260906,
        image_aug=False,
    )
    assert identities == [(9, 4), (2, 8)]
    assert transformed == [{"seen": (9, 4)}, {"seen": (2, 8)}]


def test_openvla_worker_contract_is_guarded_and_does_not_materialize_rlds():
    outer = (ROOT / "tools/benchmark/workers/run_m3_openvla_oft.py").read_text(encoding="utf-8")
    inner = (ROOT / "tools/benchmark/workers/openvla_oft_train_inner.py").read_text(encoding="utf-8")
    source = (ROOT / "tools/data/openvla_scheduled_training_source.py").read_text(encoding="utf-8")

    assert "build_runtime_spec(" in outer
    assert 'run["micro_batch"]' in outer
    assert 'run["gradient_accumulation"]' in outer
    assert "e4287e94541f459edc4feabc4e181f537cd569a8" in outer
    assert "wandb=={WANDB_VERSION}" in outer
    assert "protobuf=={PROTOBUF_VERSION}" in outer
    assert '"image_augmentation_applied": True' in outer
    assert '"full_rlds_materialized": False' in outer

    assert 'if int(runtime["sample_target"]) != 4800' in inner
    assert 'float(runtime.get("train_loop_sec") or 0.0) != 1800.0' in inner
    assert "ScheduledOpenVLASource(" in inner
    assert "transform_scheduled_batch(" in inner
    assert "image_aug=True" in inner
    assert "merge_lora_during_training=False" in inner
    assert '"full_rlds_materialized": False' in inner
    assert "save_dataset_statistics(statistics, checkpoint_dir)" in inner

    assert "iter_selected_trajectories" not in source
    assert "TFRecord" not in source
    assert "tfds.builder" not in source
