#!/usr/bin/env python3
"""Probe pi0.5 micro-batch size on A100 with exactly one effective optimizer step."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT_FALLBACK = Path("/content/parc2026")


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", str(ROOT_FALLBACK)))
    repo = Path(os.environ.get("PY_AI_REPO", str(root / "py_AI_m3_probe")))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    sys.path.insert(0, str(repo / "tools/benchmark"))
    from m3_batch_probe_common import (  # noqa: PLC0415
        EXPECTED_HASH,
        TARGET_EFFECTIVE_BATCH,
        candidate_result,
        gpu_info,
        grad_accum_for,
        load_gate_paths,
        run_logged,
        write_result,
    )

    gpu_name, gpu_vram = gpu_info()
    _, manifest, _ = load_gate_paths(drive)
    dataset_root = Path(
        os.environ.get(
            "PARC_DRIVE_DATASET",
            str(drive / "datasets/lerobot_libero_plus_v3_train"),
        )
    )
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(dataset_root / "meta/info.json")
    stats_path = dataset_root / "meta/stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for feature in ("observation.state", "action"):
        if "q01" not in stats.get(feature, {}) or "q99" not in stats.get(feature, {}):
            raise RuntimeError(f"pi0.5 M3 source dataset is missing frozen q01/q99 stats: {feature}")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for pi0.5 probe")

    pi05_dir = repo / "examples/pi05_libero_finetune"
    if not pi05_dir.is_dir():
        raise FileNotFoundError(pi05_dir)
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(["uv", "python", "install", "3.10"], check=True)
    py310 = subprocess.check_output(["uv", "python", "find", "3.10"], text=True).strip()

    data_root = root / "cache/m3-pi05-batch-probe"
    lerobot_root = root / "vendor/lerobot-pi05-m3-probe"
    out_root = data_root / "outputs"
    log_root = drive / "model-benchmark-v1/batch-probes/logs/pi05"
    setup_env = os.environ.copy()
    setup_env.update(
        {
            "PYTHON": py310,
            "DATA_ROOT": str(data_root),
            "LEROBOT_ROOT": str(lerobot_root),
            "INSTALL_FFMPEG": "0",
        }
    )
    subprocess.run(["bash", "scripts/setup_train.sh"], cwd=pi05_dir, env=setup_env, check=True)
    pi05_venv = lerobot_root / ".venv"
    if not (pi05_venv / "bin/activate").is_file():
        raise FileNotFoundError(pi05_venv / "bin/activate")

    candidates = [16, 8, 4, 2, 1]
    trials = []
    selected = None
    launcher = pi05_dir / "scripts/cheap_ablation_pi05.sh"
    for micro_batch in candidates:
        ga = grad_accum_for(micro_batch)
        run_name = f"m3_pi05_probe_bs{micro_batch}_ga{ga}"
        env = {
            "DATA_ROOT": str(data_root),
            "LEROBOT_ROOT": str(lerobot_root),
            "PI05_VENV": str(pi05_venv),
            "OUT_ROOT": str(out_root),
            "LOG_ROOT": str(log_root),
            "ABLATION_MANIFEST": str(manifest),
            "PI05_DATASET_ROOT": str(dataset_root),
            "PI05_DATASET_REPO_ID": "lerobot/libero_plus",
            "PI05_VIDEO_BACKEND": "pyav",
            "ABLATION_BS": str(micro_batch),
            "ABLATION_GA": str(ga),
            "ABLATION_STEPS": "1",
            "ABLATION_SEED": "1000",
            "RUN_NAME": run_name,
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "WANDB_DISABLED": "true",
        }
        outer_log = log_root / f"{run_name}.outer.log"
        rc, text, elapsed, sampled_peak = run_logged(
            ["bash", "-lc", f"source env_train.sh && bash {launcher}"],
            cwd=pi05_dir,
            env=env,
            log_path=outer_log,
        )
        summary_path = out_root / run_name / "cheap_ablation_summary.json"
        peak = sampled_peak
        loss = None
        if rc == 0 and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("episode_ids_sha256") != EXPECTED_HASH:
                raise RuntimeError("pi05 probe summary manifest hash mismatch")
            if int(summary.get("optimizer_steps", -1)) != 1:
                raise RuntimeError(f"pi05 probe optimizer step mismatch: {summary}")
            if int(summary.get("effective_batch", -1)) != TARGET_EFFECTIVE_BATCH:
                raise RuntimeError(f"pi05 probe effective batch mismatch: {summary}")
            peak = max(peak, int(summary.get("peak_vram_mib", 0) or 0))
            loss = summary.get("final_logged_loss_best_effort")
        trial = candidate_result(
            micro_batch=micro_batch,
            rc=rc,
            text=text,
            elapsed_sec=elapsed,
            peak_vram=peak,
        )
        if loss is not None:
            trial["loss"] = loss
        trials.append(trial)
        shutil.rmtree(out_root / run_name, ignore_errors=True)
        if rc == 0:
            selected = trial
            break
        if trial["status"] != "OOM":
            raise RuntimeError(f"pi05 probe failed for non-OOM reason: {trial}; log={outer_log}")

    result_path = drive / "model-benchmark-v1/batch-probes/pi05.json"
    if selected is None:
        write_result(
            result_path,
            {
                "status": "FAILED_NO_FIT",
                "model": "pi05",
                "source_ref": "v0.4.4",
                "gpu_name": gpu_name,
                "gpu_vram_mib": gpu_vram,
                "trials": trials,
            },
        )
        raise RuntimeError("pi05 did not fit even at micro_batch=1")
    result = {
        "status": "PASS",
        "model": "pi05",
        "source_ref": "v0.4.4",
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram,
        "selected_micro_batch": selected["micro_batch"],
        "gradient_accumulation": selected["gradient_accumulation"],
        "effective_batch_size": TARGET_EFFECTIVE_BATCH,
        "peak_vram_mib": selected["peak_vram_mib"],
        "loss": selected["loss"],
        "optimizer_steps": 1,
        "trials": trials,
    }
    write_result(result_path, result)
    print("=== 72a PI05 BATCH PROBE: PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
