#!/usr/bin/env python3
"""Run the guarded M3 A100 training smoke matrix.

Smoke only. It validates D10/72d/69c, materializes the two deterministic 4,800
reference schedules, then executes both training tracks with seed/order pairing:
20260906 -> forward, 20260907 -> reverse. This exercises all three model
runtimes, both canonical sampling paths, both seed streams and both orderings
without starting the 1,800-second benchmark.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"
SEED_ORDER = ((20260906, "forward"), (20260907, "reverse"))
TRACKS = ("equal_data", "equal_wall")
MODELS = ("pi05", "smolvla", "openvla_oft")
SMOKE_EQUAL_DATA_SAMPLES = 64
SMOKE_WALL_SEC = 5.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _validate_result(path: Path, *, seed: int, order: str, track: str, model: str) -> dict[str, Any]:
    result = _load_json(path)
    if result.get("status") != "PASS":
        raise RuntimeError(f"smoke result not PASS: {path}")
    expected = {
        "model": model,
        "mode": "smoke",
        "order": order,
        "track": track,
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": seed,
        "effective_batch_size": 32,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"smoke result mismatch {path}: {key}={result.get(key)!r} != {value!r}")
    micro = int(result.get("micro_batch", 0))
    ga = int(result.get("gradient_accumulation", 0))
    if micro <= 0 or ga <= 0 or micro * ga != 32:
        raise RuntimeError(f"invalid smoke batch accounting: {path}")
    samples = int(result.get("samples_consumed", -1))
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"smoke metrics missing: {path}")
    wall = float(metrics.get("train_wall_time", -1))
    peak = float(metrics.get("peak_train_vram", -1))
    if wall < 0 or peak < 0:
        raise RuntimeError(f"invalid smoke train metrics: {path}")
    if track == "equal_data":
        if samples != SMOKE_EQUAL_DATA_SAMPLES:
            raise RuntimeError(f"equal-data smoke sample mismatch: {samples} != {SMOKE_EQUAL_DATA_SAMPLES}")
        if int(result.get("optimizer_updates", -1)) != SMOKE_EQUAL_DATA_SAMPLES // 32:
            raise RuntimeError("equal-data smoke optimizer-update mismatch")
    else:
        if wall + 1e-9 < SMOKE_WALL_SEC:
            raise RuntimeError(f"equal-wall smoke stopped early: {wall}")
        if samples <= 0 or samples % 32 != 0:
            raise RuntimeError(f"equal-wall smoke sample accounting invalid: {samples}")
    checkpoint = Path(str(result.get("checkpoint_ref") or ""))
    if not checkpoint.exists():
        raise FileNotFoundError(f"smoke checkpoint missing: {checkpoint}")
    if model == "openvla_oft":
        if result.get("checkpoint_eval_ready") is not True:
            raise RuntimeError("OpenVLA smoke checkpoint is not evaluator-ready")
        if result.get("lora_merged_for_evaluation") is not True:
            raise RuntimeError("OpenVLA smoke checkpoint LoRA was not merged")
    return result


def main() -> int:
    if os.environ.get("PARC_M3_EXECUTE") != "1":
        raise RuntimeError("M3 smoke requires explicit PARC_M3_EXECUTE=1")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).resolve()
    parc_root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    dataset_root = Path(os.environ.get("PARC_DRIVE_DATASET", drive / "datasets/lerobot_libero_plus_v3_train"))
    manifest = Path(
        os.environ.get(
            "PARC_D10_MANIFEST",
            drive
            / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json",
        )
    )
    batch_summary = drive / "model-benchmark-v1/m3_batch_probe_summary.json"
    streaming_contract = drive / "openvla-streaming-selected-v1/streaming_bridge_contract.json"
    schedule_root = drive / "model-benchmark-v1/m3-schedules-v1"
    output_root = drive / "model-benchmark-v1/m3-training-smoke-v1"
    plan_root = output_root / "plans"
    summary_out = output_root / "m3_training_smoke_summary.json"

    sys.path.insert(0, str(repo))
    from tools.benchmark.m3_batch_probe_common import gpu_info  # noqa: PLC0415
    from tools.benchmark.m3_runner_core import validate_batch_probe_summary  # noqa: PLC0415
    from tools.benchmark.m3_sampling_schedule import (  # noqa: PLC0415
        build_equal_data_schedule,
        load_d10_episode_lengths,
        write_schedule,
    )
    from tools.benchmark.m3_scheduled_data import load_schedule  # noqa: PLC0415
    from tools.data.openvla_lerobot_streaming import load_manifest  # noqa: PLC0415

    gpu_name, gpu_vram_mib = gpu_info()
    _require_file(batch_summary)
    _require_file(streaming_contract)
    _require_file(manifest)
    _require_file(dataset_root / "meta/info.json")
    if _sha256_file(manifest) != D10_MANIFEST_SHA256:
        raise RuntimeError("D10 manifest file SHA mismatch")
    manifest_data = load_manifest(manifest)
    if manifest_data.get("episode_ids_sha256") != D10_HASH:
        raise RuntimeError("D10 manifest episode hash mismatch")
    validate_batch_probe_summary(_load_json(batch_summary))
    contract = _load_json(streaming_contract)
    if contract.get("status") != "PASS" or contract.get("bridge_type") != "lerobot_streaming":
        raise RuntimeError("69c streaming contract is not PASS lerobot_streaming")
    if contract.get("selected_episode_ids_sha256") != D10_HASH:
        raise RuntimeError("69c streaming D10 mismatch")

    episode_ids, episode_lengths = load_d10_episode_lengths(
        manifest_path=manifest,
        dataset_root=dataset_root,
    )
    schedule_root.mkdir(parents=True, exist_ok=True)
    schedules: dict[int, Path] = {}
    schedule_hashes: dict[str, str] = {}
    for seed, _ in SEED_ORDER:
        payload = build_equal_data_schedule(
            episode_ids=episode_ids,
            episode_lengths=episode_lengths,
            seed=seed,
        )
        path = schedule_root / f"m3_equal_data_seed-{seed}.json"
        if path.exists():
            existing = load_schedule(path, require_equal_data=True)
            if existing.get("schedule_sha256") != payload.get("schedule_sha256"):
                raise RuntimeError(f"existing M3 schedule differs from deterministic schedule: {path}")
        else:
            write_schedule(path, payload)
        schedules[seed] = path
        schedule_hashes[str(seed)] = str(payload["schedule_sha256"])

    results: list[dict[str, Any]] = []
    runner = repo / "tools/benchmark/m3_model_adapters.py"
    _require_file(runner)
    reset = os.environ.get("PARC_M3_SMOKE_RESET") == "1"
    for seed, order in SEED_ORDER:
        schedule = schedules[seed]
        for track in TRACKS:
            plan = plan_root / f"seed-{seed}_{order}_{track}.json"
            expected_paths = {
                model: output_root / order / track / f"seed-{seed}" / model / "train_result.json"
                for model in MODELS
            }
            if all(path.is_file() for path in expected_paths.values()):
                for model, path in expected_paths.items():
                    results.append(_validate_result(path, seed=seed, order=order, track=track, model=model))
                print(f"skip completed smoke: seed={seed} order={order} track={track}", flush=True)
                continue
            run_root = output_root / order / track / f"seed-{seed}"
            if run_root.exists():
                if not reset:
                    raise RuntimeError(
                        f"partial smoke output exists: {run_root}; set PARC_M3_SMOKE_RESET=1 to reset only this smoke run"
                    )
                shutil.rmtree(run_root)

            cmd = [
                sys.executable,
                "-u",
                str(runner),
                "--batch-summary",
                str(batch_summary),
                "--schedule",
                str(schedule),
                "--dataset-root",
                str(dataset_root),
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--track",
                track,
                "--order",
                order,
                "--mode",
                "smoke",
                "--repo-root",
                str(repo),
                "--parc-root",
                str(parc_root),
                "--streaming-contract",
                str(streaming_contract),
                "--plan-out",
                str(plan),
                "--execute",
            ]
            subprocess.run(cmd, cwd=str(repo), check=True, env=os.environ.copy())
            for model, path in expected_paths.items():
                results.append(_validate_result(path, seed=seed, order=order, track=track, model=model))

    expected_count = len(SEED_ORDER) * len(TRACKS) * len(MODELS)
    if len(results) != expected_count:
        raise RuntimeError(f"smoke result count mismatch: {len(results)} != {expected_count}")
    identities = {(r["sampling_seed"], r["order"], r["track"], r["model"]) for r in results}
    if len(identities) != expected_count:
        raise RuntimeError("duplicate/missing M3 smoke result identities")

    summary = {
        "schema_version": 1,
        "stage": "M3_A100_training_smoke",
        "status": "PASS",
        "selected_episode_ids_sha256": D10_HASH,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "seed_order": [{"seed": seed, "order": order} for seed, order in SEED_ORDER],
        "tracks": list(TRACKS),
        "models": list(MODELS),
        "schedule_sha256": schedule_hashes,
        "result_count": expected_count,
        "benchmark_training_started": False,
        "full_1800_second_run_started": False,
        "results": [
            {
                "seed": r["sampling_seed"],
                "order": r["order"],
                "track": r["track"],
                "model": r["model"],
                "samples_consumed": r["samples_consumed"],
                "optimizer_updates": r["optimizer_updates"],
                "train_wall_time": r["metrics"]["train_wall_time"],
                "peak_train_vram": r["metrics"]["peak_train_vram"],
                "checkpoint_ref": r["checkpoint_ref"],
            }
            for r in results
        ],
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== 73 M3 A100 TRAINING SMOKE: PASS ===", flush=True)
    print(json.dumps({"status": "PASS", "result_count": expected_count, "summary": str(summary_out)}, indent=2), flush=True)
    print("Benchmark training has NOT started. Full 1800-second runs remain disabled.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
