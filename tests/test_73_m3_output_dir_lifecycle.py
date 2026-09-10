from pathlib import Path


def test_lerobot_entry_does_not_precreate_trainer_output_dir():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/benchmark/m3_lerobot_train_entry_v2.py").read_text(encoding="utf-8")
    assert 'args.output_dir.parent.mkdir(parents=True, exist_ok=True)' in text
    assert 'args.output_dir.mkdir(parents=True, exist_ok=True)' not in text


def test_smoke_only_prunes_empty_partial_scaffolding():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/colab/run_m3_training_smoke.py").read_text(encoding="utf-8")
    assert 'def _has_partial_payload(root: Path) -> bool:' in text
    assert 'if not _has_partial_payload(run_root):' in text
    assert 'prune empty partial smoke scaffolding' in text
    assert 'elif not reset:' in text
    assert 'PARC_M3_SMOKE_RESET=1' in text
