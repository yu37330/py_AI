#!/usr/bin/env python3
"""Run Notebook 73's M3 smoke orchestrator with persistent diagnostics.

This wrapper is intentionally invoked from the already validated OpenVLA
Python 3.10 environment. That environment contains numpy/pyarrow/pandas/PyAV,
which are also needed by the model-neutral D10 schedule metadata loader.
The wrapper only tees the real smoke orchestrator output to Drive; it does not
change budgets, rerun probes, or enable the 1800-second benchmark.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _prepend_pythonpath(env: dict[str, str], repo: Path) -> None:
    """Make repo-local ``tools.*`` imports resolvable in all child processes.

    Some M3 entrypoints are intentionally launched by absolute file path. In
    that mode Python puts the script directory, not the repository root, at
    sys.path[0]. Keep the repository root in PYTHONPATH so those entrypoints
    can still import ``tools.benchmark`` without depending on the caller's cwd.
    """
    root = str(repo)
    current = env.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if root not in parts:
        parts.insert(0, root)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def main() -> int:
    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).resolve()
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache")).resolve()
    diag_root = drive / "model-benchmark-v1/m3-training-smoke-v1/orchestrator"
    log_path = diag_root / "orchestrator.log"
    status_path = diag_root / "orchestrator_status.json"
    target = repo / "tools/colab/run_m3_training_smoke.py"
    if not target.is_file():
        raise FileNotFoundError(target)

    status = {
        "schema_version": 2,
        "stage": "M3_A100_training_smoke_orchestrator",
        "status": "RUNNING",
        "python": sys.executable,
        "target": str(target),
        "repo_pythonpath_injected": True,
        "benchmark_training_started": False,
        "full_1800_second_run_started": False,
    }
    _write_json(status_path, status)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    _prepend_pythonpath(env, repo)
    log_path.write_text(
        "=== M3 training smoke orchestrator ===\n"
        f"python={sys.executable}\n"
        f"target={target}\n"
        f"PYTHONPATH={env.get('PYTHONPATH', '')}\n",
        encoding="utf-8",
    )

    cmd = [sys.executable, "-u", str(target)]
    print(">>>", " ".join(cmd), flush=True)
    lines: list[str] = []
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            if len(lines) > 300:
                lines.pop(0)
            fh.write(line)
            fh.flush()
            print(line, end="", flush=True)
        rc = proc.wait()

    status.update(
        {
            "status": "PASS" if rc == 0 else "FAILED",
            "return_code": rc,
            "log_path": str(log_path),
            "output_tail": lines[-120:],
        }
    )
    _write_json(status_path, status)
    if rc != 0:
        print("=== 73 ORCHESTRATOR FAILURE TAIL ===", flush=True)
        print("\n".join(lines[-120:]), flush=True)
        raise RuntimeError(f"M3 training smoke orchestrator failed rc={rc}; log={log_path}")

    print("=== 73 ORCHESTRATOR: PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
