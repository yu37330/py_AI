from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "examples/pi05_libero_finetune/patches/subset-episode-sampler-relative-index.patch"
SETUP = ROOT / "examples/pi05_libero_finetune/scripts/setup_train.sh"


def test_subset_sampler_patch_maps_absolute_to_filtered_indices():
    text = PATCH.read_text()
    assert "index_mapping: dict[int, int] | None = None" in text
    assert "indices.extend(index_mapping[idx] for idx in frame_indices)" in text
    assert 'index_mapping=getattr(dataset, "_absolute_to_relative_idx", None)' in text


def test_training_setup_applies_all_patch_files():
    text = SETUP.read_text()
    assert 'for patch in "$HERE"/patches/*.patch; do' in text


def test_relative_mapping_keeps_subset_sampler_in_bounds():
    # Reproduce the shape of the failure: metadata frame indices stay absolute,
    # while a filtered HF Dataset is compacted to relative row indices.
    absolute_frames = range(100, 105)
    mapping = {absolute: relative for relative, absolute in enumerate(absolute_frames)}
    sampled = [mapping[idx] for idx in absolute_frames]
    assert sampled == [0, 1, 2, 3, 4]
    assert max(sampled) < len(mapping)
