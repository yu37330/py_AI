#!/usr/bin/env python3
"""Fail-fast, resumable 69b smoke runner. No model loading or training.

Use only the D10-selected episode pool. Preserve source and completed artifacts;
never silently reselect data or treat an old report as a new success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

VARIANT = "V2_SQRT_BALANCED_RAW"
REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
DATASET_ID = "lerobot/libero_plus"
DEPS = ["numpy<2", "pandas>=2,<3", "pyarrow>=16", "tensorflow-cpu==2.17.1", "tensorflow-datasets==4.9.6", "av==12.3.0"]
CHECK = "import sys, numpy, pandas, pyarrow, tensorflow as tf, tensorflow_datasets as tfds, av; assert sys.version_info[:2] == (3,10); assert tf.__version__ == '2.17.1'; assert tfds.__version__ == '4.9.6'; assert av.__version__ == '12.3.0'; print('conversion dependencies verified', flush=True)"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_manifest(path):
    m = read_json(path); ids = m.get("episode_ids")
    if m.get("schema_version") != 2 or m.get("group_aware") is not True or m.get("kind") != "training_episode_manifest" or m.get("variant") != VARIANT:
        raise ValueError(f"Not the selected group-aware training manifest: {path}")
    if m.get("dataset_id") != DATASET_ID or m.get("dataset_revision") != REVISION:
        raise ValueError("Dataset provenance mismatch")
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids) or len(ids) != 10758 or len(set(ids)) != len(ids):
        raise ValueError("Selected episode IDs are invalid")
    if m.get("episode_ids_sha256") != ids_hash(ids) or m.get("summary", {}).get("episode_count") != len(ids):
        raise ValueError("Selected manifest count/hash mismatch")
    if int(m.get("summary", {}).get("frame_count", 0)) <= 0:
        raise ValueError("Selected manifest frame count missing")
    return m


def find_manifest(drive, local):
    name = VARIANT + ".json"
    candidates = [local / "outputs/dataset_ablation_manifests_v2_group_aware" / name,
                  local / "outputs/rlds_bridge_manifest_rebuild/v2" / name,
                  drive / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware" / name,
                  drive / "pi05-ablation-group-aware-v2/outputs/dataset_ablation_manifests_v2_group_aware" / name,
                  drive / "dataset_ablation_manifests_v2_group_aware" / name]
    valid = [(p, validate_manifest(p)) for p in candidates if p.is_file()]
    if not valid:
        raise FileNotFoundError("Exact V2 manifest not found. Restore the original group-aware manifest from the completed dataset-screening run. Automatic reselection is disabled to avoid changing the D10 input.")
    if len({m["episode_ids_sha256"] for _, m in valid}) != 1:
        raise ValueError("Conflicting selected manifests; refusing to choose one silently")
    return valid[0]


def validate_report(report, manifest, expected_ids, prepared_required=True):
    if report.get("status") != "SMOKE_PASS" or report.get("selected_dataset_variant") != VARIANT:
        raise ValueError("Smoke report status/variant mismatch")
    if report.get("source_dataset_id") != DATASET_ID or report.get("source_dataset_revision") != REVISION:
        raise ValueError("Smoke report source mismatch")
    if report.get("source_episode_ids_sha256") != manifest["episode_ids_sha256"]:
        raise ValueError("Smoke report manifest hash mismatch")
    if report.get("converted_episode_ids") != expected_ids or report.get("converted_episode_count") != len(expected_ids):
        raise ValueError("Smoke report episode IDs/count mismatch")
    if report.get("tfds_builder_from_directory") != "PASS" or int(report.get("converted_frames", 0)) <= 0:
        raise ValueError("Smoke report TFDS/frame gate failed")
    if int(report["source_frame_count"]) != int(manifest["summary"]["frame_count"]):
        raise ValueError("Smoke report source frame count mismatch")
    for key in ("raw_action_preserved", "raw_state_preserved", "no_noop_filter_applied"):
        if report.get(key) is not True:
            raise ValueError(f"Smoke report {key} missing")
    if prepared_required:
        prepared = Path(report.get("prepared_dir", ""))
        if not prepared.is_dir() or not (prepared / "dataset_info.json").is_file():
            raise FileNotFoundError(f"Prepared TFDS missing: {prepared}")
    return report


def capacity_decision(report, manifest, threshold=35.0):
    projected = float(report["projected_full_gib"])
    if not math.isfinite(projected) or projected <= 0:
        raise ValueError("Invalid full-capacity projection")
    decision = "MATERIALIZE_CANDIDATE" if projected <= threshold else "STREAMING_BRIDGE_RECOMMENDED"
    return {"schema_version": 1, "status": "PASS", "selected_dataset_variant": VARIANT,
            "source_episode_ids_sha256": manifest["episode_ids_sha256"],
            "smoke_episode_count": report["converted_episode_count"], "smoke_frames": report["converted_frames"],
            "bytes_per_frame": report["bytes_per_frame"], "projected_full_gib": projected,
            "materialize_threshold_gib": threshold, "decision": decision,
            "next_step": "Full exact-manifest conversion may be considered; not started automatically" if decision == "MATERIALIZE_CANDIDATE" else "Implement a validated streaming LeRobot-to-OpenVLA adapter; do not duplicate the full dataset",
            "m3_budget_file": "experiments/plans/model_benchmark_budget_v1.json"}


def run_logged(cmd, label, log_dir, *, env=None, cwd=None, interval=30):
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / (label + ".log"); start = time.monotonic()
    print(f"[start] {label}; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        while True:
            try:
                rc = p.wait(timeout=interval); break
            except subprocess.TimeoutExpired:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
                try:
                    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
                except (OSError, subprocess.SubprocessError):
                    gpu = "CPU/no nvidia-smi"
                free = shutil.disk_usage(str(log_dir)).free / 1024**3
                print(f"[heartbeat] {label} elapsed={(time.monotonic()-start)/60:.1f}m gpu={gpu} free={free:.1f}GiB {' | '.join(tail)[-500:]}", flush=True)
        text = log.read_text(encoding="utf-8", errors="replace")
        if rc:
            print(text[-16000:], flush=True)
            raise RuntimeError(f"{label} failed (rc={rc}). Full log: {log}")
    print(f"[done] {label} elapsed={(time.monotonic()-start)/60:.1f}m rc=0", flush=True)
    return text


def ensure_env(root, log_dir):
    venv = root / "venv-openvla-rlds"; py = venv / "bin/python"
    if py.exists() and subprocess.run([str(py), "-c", CHECK], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        print("[reuse] verified conversion environment", flush=True); return py
    uv = shutil.which("uv")
    if not uv:
        run_logged([sys.executable, "-m", "pip", "install", "uv"], "install-uv", log_dir)
        uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    run_logged([uv, "python", "install", "3.10"], "python310", log_dir)
    if not py.exists():
        run_logged([uv, "venv", "--python", "3.10", str(venv)], "create-venv", log_dir)
    run_logged([uv, "pip", "install", "--python", str(py), "--only-binary", ":all:", *DEPS], "rlds-conversion-deps", log_dir)
    run_logged([str(py), "-c", CHECK], "verify-conversion-env", log_dir)
    return py


def run(args, *, exec_command=run_logged):
    root = Path(args.root); repo = Path(args.repo); drive = Path(args.drive)
    out = drive / "openvla-rlds-selected-v1"; out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    status_path = out / "bridge_smoke_status.json"
    report_path = out / "bridge_smoke_report.json"
    capacity_path = out / "bridge_capacity_decision.json"
    attempt = uuid.uuid4().hex[:12]
    status = {"schema_version": 1, "stage": "69b", "attempt_id": attempt, "status": "RUNNING", "last_completed_stage": None, "error": None}
    # Keep an old successful decision as history, never present it as this run's result.
    if capacity_path.exists():
        history = out / "history"; history.mkdir(exist_ok=True)
        shutil.copy2(capacity_path, history / f"bridge_capacity_decision_{attempt}.json")
    write_json(capacity_path, {"schema_version": 1, "status": "BLOCKED", "attempt_id": attempt, "reason": "69b verification in progress"})
    write_json(status_path, status)
    def mark(stage, state="RUNNING", error=None):
        status.update(status=state, last_completed_stage=stage, error=error)
        write_json(status_path, status)
    try:
        decision = read_json(drive / "pi05-top2-tiebreak-v1/provisional_best_dataset_recipe.json")
        if decision.get("status") != "DECIDED" or decision.get("selected_variant") != VARIANT:
            raise ValueError("D10 decision is not the selected V2_SQRT recipe")
        source = drive / "datasets/lerobot_libero_plus_v3_train"
        if not (source / "meta/info.json").is_file() or not (source / ".parc_prefetch_complete.json").is_file():
            raise FileNotFoundError("Prefetched source dataset is incomplete")
        manifest_path, manifest = find_manifest(drive, root)
        expected_ids = manifest["episode_ids"][:8]
        mark("preflight"); print("[1/3] D10 and exact manifest PASS", flush=True)
        py = ensure_env(root, log_dir)
        mark("dependencies"); print("[2/3] Conversion environment PASS", flush=True)
        report = None
        if report_path.exists() and not args.force:
            try:
                report = validate_report(read_json(report_path), manifest, expected_ids)
                # Revalidate TFDS readability in the isolated conversion environment.
                check = "import sys, tensorflow_datasets as tfds; b=tfds.builder_from_directory(sys.argv[1]); x=next(iter(tfds.as_numpy(b.as_dataset(split='train').take(1)))); assert 'steps' in x and 'episode_metadata' in x; print('prepared TFDS reuse gate PASS')"
                exec_command([str(py), "-c", check, report["prepared_dir"]], "verify-existing-tfds", log_dir)
                print("[reuse] existing provenance-matched smoke report", flush=True)
            except (ValueError, KeyError, TypeError, FileNotFoundError, json.JSONDecodeError, RuntimeError) as exc:
                print(f"[rebuild] existing report is not reusable: {exc}", flush=True); report = None
        if report is None:
            # An isolated attempt prevents stale reports/partial TFDS from being reused.
            smoke_root = root / "openvla-rlds-smoke-v1" / attempt
            staged_report = smoke_root / "bridge_smoke_report.json"
            env = os.environ.copy(); env.update(TF_CPP_MIN_LOG_LEVEL="2", TF_NUM_INTEROP_THREADS="2", TF_NUM_INTRAOP_THREADS="2", PYTHONUNBUFFERED="1")
            exec_command([str(py), "-u", str(repo / "tools/data/convert_lerobot_manifest_to_openvla_rlds.py"), "--lerobot-root", str(source), "--manifest", str(manifest_path), "--out-root", str(smoke_root), "--report", str(staged_report), "--dataset-repo-id", DATASET_ID, "--dataset-revision", REVISION, "--max-episodes", "8"], "rlds-8ep-conversion", log_dir, env=env)
            if not staged_report.is_file():
                raise RuntimeError(f"Converter returned success but did not create {staged_report}; inspect {log_dir / 'rlds-8ep-conversion.log'}")
            report = validate_report(read_json(staged_report), manifest, expected_ids)
            write_json(report_path, report)
        mark("conversion")
        capacity = capacity_decision(report, manifest)
        capacity["attempt_id"] = attempt
        write_json(capacity_path, capacity)
        mark("capacity", "PASS")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(json.dumps(capacity, ensure_ascii=False, indent=2), flush=True)
        print("=== 69b COMPLETE ===", flush=True)
        print("M3 is still blocked until the full selected-pool bridge and model runners are validated.", flush=True)
        return capacity
    except Exception as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(status_path, status)
        write_json(capacity_path, {"schema_version": 1, "status": "BLOCKED", "attempt_id": attempt, "reason": status["error"]})
        print(f"=== 69b FAILED at {status['last_completed_stage']} ===", flush=True)
        print(status["error"], flush=True)
        print(f"Logs: {log_dir}", flush=True)
        raise


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/content/parc2026"))
    p.add_argument("--repo", type=Path, default=Path("/content/parc2026/py_AI"))
    p.add_argument("--drive", type=Path, default=Path("/content/drive/MyDrive/parc2026-cache"))
    p.add_argument("--force", action="store_true", help="Rebuild only a new isolated local 8-episode smoke attempt")
    args = p.parse_args(argv); run(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
