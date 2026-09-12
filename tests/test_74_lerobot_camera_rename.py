import ast
from pathlib import Path


def test_lerobot_eval_resolves_camera_names_from_checkpoint_config():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/benchmark/m3_lerobot_eval_entry.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))

    assert "def _camera_rename_map" in text
    assert "config.json" in text
    assert "observation.images.image" in text
    assert "observation.images.image2" in text
    assert "observation.images.front" in text
    assert "observation.images.wrist" in text
    assert "--rename_map=" in text
    assert "camera_rename_map" in text
    assert "unsupported checkpoint visual feature layout" in text


def test_camera_rename_is_fail_closed_instead_of_guessing_unknown_layouts():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/benchmark/m3_lerobot_eval_entry.py").read_text(encoding="utf-8")

    assert "expected == source" in text
    assert "expected == canonical" in text
    assert "raise RuntimeError" in text


def test_lerobot_upstream_output_directory_exists_before_upstream_main_runs():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/benchmark/m3_lerobot_eval_entry.py").read_text(encoding="utf-8")

    mkdir = 'upstream_dir.mkdir(parents=True, exist_ok=True)'
    invoke = 'module.main()'
    assert 'upstream_dir = args.output_dir / "upstream"' in text
    assert mkdir in text
    assert 'f"--output_dir={upstream_dir}"' in text
    assert text.index(mkdir) < text.index(invoke)
