import ast
from pathlib import Path


def _text(rel: str) -> str:
    root = Path(__file__).resolve().parents[1]
    path = root / rel
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    return text


def test_hardware_guard_has_explicit_a100_and_organizer_blackwell_profiles():
    text = _text("tools/benchmark/m3_hardware_guard.py")
    assert '"colab_a100"' in text
    assert '"organizer_rtx_pro_6000_blackwell"' in text
    assert 'required_name_tokens=("RTX PRO 6000", "Blackwell")' in text
    assert "min_vram_mib=90_000" in text
    assert "PARC_M3_HARDWARE_PROFILE" in text


def test_openvla_blackwell_runtime_overrides_only_hardware_runtime():
    text = _text("tools/colab/prepare_m3_openvla_blackwell_runtime.py")
    assert 'OPENVLA_REF = "e4287e94541f459edc4feabc4e181f537cd569a8"' in text
    assert 'TORCH_VERSION = "2.7.1"' in text
    assert 'TORCH_INDEX = "https://download.pytorch.org/whl/cu128"' in text
    assert '"sm_120"' in text
    assert "flash-attn" in text
    assert "historical_flash_attn_installed" in text
    assert '"benchmark_training_started": False' in text


def test_base_runtime_setup_routes_organizer_profile_before_a100_path():
    text = _text("tools/colab/prepare_m3_training_runtimes.py")
    assert 'ORGANIZER_PROFILE = "organizer_rtx_pro_6000_blackwell"' in text
    assert "prepare_m3_organizer_training_runtimes.py" in text
    assert "PARC_LOCAL_SCRATCH_ROOT" in text
    assert "PARC_PERSIST_ROOT" in text


def test_handoff_builder_is_checksum_scoped_and_excludes_dataset():
    text = _text("tools/benchmark/build_m3_organizer_handoff.py")
    assert "D10_MANIFEST_SHA256" in text
    assert "m3_training_smoke_summary.json" in text
    assert "m3_batch_probe_summary.json" in text
    assert "streaming_bridge_contract.json" in text
    assert "m3_equal_data_seed-" in text
    assert '"dataset_included": False' in text
    assert "refusing to overwrite existing handoff archive" in text


def test_handoff_restore_rejects_links_traversal_and_overwrite():
    text = _text("tools/benchmark/restore_m3_organizer_handoff.py")
    assert '".." in rel.parts' in text
    assert "member.issym() or member.islnk()" in text
    assert "refusing to overwrite existing restored artifact" in text
    assert "verify_m3_organizer_handoff.py" in text


def test_handoff_verifier_checks_every_manifest_file_sha():
    text = _text("tools/benchmark/verify_m3_organizer_handoff.py")
    assert "handoff SHA mismatch" in text
    assert "handoff size mismatch" in text
    assert "verified_file_count" in text
    assert '"source_training_results_mutated": False' in text


def test_dataset_materializer_uses_exact_revision_on_scratch_only():
    text = _text("tools/benchmark/prepare_m3_organizer_dataset.py")
    assert 'DATASET_REPO = "lerobot/libero_plus"' in text
    assert 'DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"' in text
    assert "PARC_LOCAL_SCRATCH_ROOT" in text
    assert "PARC_DATASET_ROOT" in text
    assert '"persistent_dataset_copy_created": False' in text
    assert '"benchmark_training_started": False' in text


def test_organizer_launchers_keep_frozen_gates_and_persistent_boundary():
    simulator = _text("tools/benchmark/run_m3_organizer_simulator_smoke.py")
    compat = _text("tools/benchmark/run_m3_organizer_training_compat.py")
    benchmark = _text("tools/benchmark/run_m3_organizer_benchmark_order.py")
    screening = _text("tools/benchmark/run_m3_organizer_screening_order.py")

    assert "verify_m3_organizer_handoff.py" in simulator
    assert 'ORGANIZER_HARDWARE_PROFILE = "organizer_rtx_pro_6000_blackwell"' in simulator
    assert 'summary.get("episode_count") != 60' in simulator
    assert 'summary.get("promotion_evidence") is not False' in simulator
    assert 'summary.get("benchmark_training_started") is not False' in simulator

    assert "samples_per_model" in compat
    assert "optimizer_updates_per_model" in compat
    assert '"ready_for_75": True' in compat
    assert '"compatibility_checkpoint_storage": "ephemeral_local_scratch"' in compat

    assert "_require_74" in benchmark
    assert "_require_training_compat" in benchmark
    assert "PARC_DRIVE_DATASET" in benchmark
    assert "run_m3_benchmark_order.py" in benchmark

    assert "run_m3_screening_evaluation_order.py" in screening
    assert "checkpoint_count" in screening
    assert "total_episode_records" in screening
    assert "ready_for_forward_reverse_promotion" in screening


def test_execution_runbook_uses_official_storage_roles_and_session_boundaries():
    text = _text("docs/PARC2026_ORGANIZER_GPU_EXECUTION_RUNBOOK_20260913.md")
    assert "NVIDIA RTX PRO 6000 Blackwell" in text
    assert "/opt/dlami/nvme" in text
    assert "~/data" in text
    assert "parc-home-sync data-push" in text
    assert "parc-home-sync data-pull" in text
    assert "最大12時間" in text
    assert "run_m3_organizer_simulator_smoke.py" in text
    assert "run_m3_organizer_training_compat.py" in text
    assert "run_m3_organizer_benchmark_order.py" in text
    assert "run_m3_organizer_screening_order.py" in text
