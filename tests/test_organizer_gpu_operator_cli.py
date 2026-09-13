from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_operator_shell_scripts_are_bash_syntax_valid():
    for rel in (
        "tools/benchmark/organizer_gpu_env.sh",
        "tools/benchmark/organizer_gpu_cli.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / rel)], check=True)


def test_environment_does_not_touch_persistent_root_before_pull():
    text = _text("tools/benchmark/organizer_gpu_env.sh")
    assert 'export PARC_PERSIST_ROOT="${PARC_PERSIST_ROOT:-$HOME/data/parc2026-cache}"' in text
    assert "mkdir -p" in text
    assert '"$PARC_LOCAL_SCRATCH_ROOT"' in text
    mkdir_block = text.split("mkdir -p", 1)[1].split("if git -C", 1)[0]
    assert '"$PARC_PERSIST_ROOT"' not in mkdir_block
    assert "must not write into ~/data before" in text


def test_operator_cli_has_complete_guarded_stage_commands():
    text = _text("tools/benchmark/organizer_gpu_cli.sh")
    for token in (
        "status)",
        "pull)",
        "preflight)",
        "restore)",
        "74)",
        "dataset)",
        "compat)",
        "75-forward)",
        "75-reverse)",
        "76-unit)",
        "76-finalize)",
        "77)",
        "push)",
        "stop-ready)",
    ):
        assert token in text
    assert "hardware_guard" in text
    assert "check_repo_identity" in text
    assert "require_hf_token" in text
    assert "operator-logs" in text
    assert "PIPESTATUS[0]" in text


def test_data_pull_logs_outside_data_until_restore_finishes():
    text = _text("tools/benchmark/organizer_gpu_cli.sh")
    pull = text.split("pull_data()", 1)[1].split("preflight()", 1)[0]
    assert 'bootstrap_log="$HOME/parc2026_data_pull_' in pull
    assert "parc-home-sync data-pull" in pull
    assert 'mkdir -p "$LOG_ROOT"' in pull
    assert pull.index("parc-home-sync data-pull") < pull.index('mkdir -p "$LOG_ROOT"')


def test_quickstart_explicitly_uses_jupyterlab_terminal_not_notebook_cells():
    text = _text("docs/PARC2026_ORGANIZER_GPU_OPERATOR_QUICKSTART_20260913.md")
    assert "Notebookセルではなく JupyterLab Terminal" in text
    assert "Launcher -> Terminal" in text
    assert "File -> New -> Terminal" in text
    assert "git pullしない" in text
    assert "organizer_gpu_cli.sh preflight" in text
    assert "organizer_gpu_cli.sh 74" in text
    assert "organizer_gpu_cli.sh 75-forward" in text
    assert "organizer_gpu_cli.sh 75-reverse" in text
    assert "organizer_gpu_cli.sh 76-unit" in text
    assert "organizer_gpu_cli.sh 77" in text
    assert "organizer_gpu_cli.sh stop-ready" in text
    assert "File -> Hub Control Panel -> Stop Server" in text
