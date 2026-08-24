from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
BDDL_ROOT = ASSETS_ROOT / "bddl_files"
INIT_FILES_ROOT = ASSETS_ROOT / "init_files"

SUITE_NAME = "libero_t3"
T3_BDDL_DIR = BDDL_ROOT / SUITE_NAME
T3_INIT_DIR = INIT_FILES_ROOT / SUITE_NAME


def t3_task_stems():
    return sorted(p.stem for p in T3_BDDL_DIR.glob("*.bddl"))
