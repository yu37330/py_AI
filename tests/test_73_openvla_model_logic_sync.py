from pathlib import Path


def test_openvla_training_syncs_hf_model_logic_from_pinned_source_checkout():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/benchmark/m3_openvla_train_entry.py").read_text()

    for token in (
        "def _sync_pinned_model_logic(",
        'source_hf = source_root / "prismatic/extern/hf"',
        'required = ("modeling_prismatic.py", "configuration_prismatic.py")',
        "previous_cwd = Path.cwd()",
        "os.chdir(source_root)",
        "check_model_logic_mismatch(vla_path)",
        "os.chdir(previous_cwd)",
        "missing_synced",
        "_sync_pinned_model_logic(",
    ):
        assert token in text


def test_openvla_training_fails_fast_if_multi_image_methods_are_stale():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/benchmark/m3_openvla_train_entry.py").read_text()

    for token in (
        '"set_num_images_in_input"',
        '"get_num_images_in_input"',
        '"get_num_patches"',
        "missing_vision_methods",
        "OpenVLA loaded stale/incompatible vision-backbone logic after sync",
        "vla.vision_backbone.set_num_images_in_input(2)",
        "vla.vision_backbone.get_num_images_in_input()",
    ):
        assert token in text
