#!/usr/bin/env python3
"""Create a non-interactive shared LIBERO config for Notebook 74.

hf-libero prompts on import when its config file does not exist. Colab smoke
runs are non-interactive, so create the config before importing any LIBERO
module. All three M3 evaluators use the same pinned LIBERO checkout target for
task definitions/assets; the simulator runtime setup materializes that checkout
before any episode is executed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026")).resolve()
    config_root = Path(
        os.environ.get("LIBERO_CONFIG_PATH", root / "m3-libero-config/shared")
    ).resolve()
    package_root = root / "vendor/libero-openvla-m3/libero/libero"
    config_root.mkdir(parents=True, exist_ok=True)
    config_file = config_root / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {package_root}",
                f"bddl_files: {package_root / 'bddl_files'}",
                f"init_states: {package_root / 'init_files'}",
                f"datasets: {package_root / 'datasets'}",
                f"assets: {package_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "stage": "M3_LIBERO_noninteractive_config",
        "status": "PASS",
        "config_root": str(config_root),
        "config_file": str(config_file),
        "pinned_libero_package_root": str(package_root),
        "interactive_prompt_allowed": False,
        "benchmark_training_started": False,
        "simulator_started": False,
    }
    print(json.dumps(evidence, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
