#!/usr/bin/env python3
"""Run/resume long PARC2026 pi0.5 LoRA training from a selected group-aware manifest.

Expected Colab inputs:
- A100/H100-class GPU (>=38 GiB VRAM)
- Google Drive mounted at /content/drive
- HF_TOKEN in the environment
- Drive-prefetched lerobot/libero_plus v3 dataset
- fixed simulator result and a decisive top-2 tie-break result on Drive

The runner stages the dataset locally, rebuilds group-aware manifests/leakage gates,
launches manifest-aware 20k LoRA training, and mirrors stable checkpoints/last to
Drive so a fresh Colab runtime can resume.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time


def out(*args) -> None:
    print(*args, flush=True)


def run(args, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    out(">>>", " ".join(str(x) for x in args))
    subprocess.run([str(x) for x in args], cwd=cwd, env=env, check=True)


def gpu_info() -> tuple[str, int]:
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    mem = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    return name, mem


def gpu_status() -> tuple[str, str, str]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        a, b, c = [x.strip() for x in raw.split(",")]
        return a, b, c
    except Exception:
        return "?", "?", "?"


def free_gib() -> float:
    return shutil.disk_usage("/content").free / 1024**3


def tree_latest_mtime(root: Path) -> float:
    latest = 0.0
    for path in root.rglob("*"):
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            pass
    return latest


def select_variant(fixed_path: Path, tiebreak_path: Path) -> tuple[str, str, dict]:
    fixed = json.loads(fixed_path.read_text())
    fixed_scores = {r["variant"]: r.get("overall_score") for r in fixed.get("results", [])}
    override = os.environ.get("SELECTED_VARIANT_OVERRIDE", "").strip()
    if override:
        if override not in {"V1_MULTI", "V2_SQRT"}:
            raise RuntimeError("SELECTED_VARIANT_OVERRIDE must be V1_MULTI or V2_SQRT")
        return override, "manual_override", fixed_scores

    if not tiebreak_path.exists():
        raise RuntimeError(
            f"top-2 tie-break result missing: {tiebreak_path}. "
            "Run the top-2 tie-break evaluation before full training."
        )
    tb = json.loads(tiebreak_path.read_text())
    scores = {
        r["variant"]: r.get("overall_score")
        for r in tb.get("results", [])
        if r.get("variant") in {"V1_MULTI", "V2_SQRT"}
    }
    if set(scores) != {"V1_MULTI", "V2_SQRT"} or any(v is None for v in scores.values()):
        raise RuntimeError(f"tie-break summary does not contain both scores: {scores}")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked[0][1] == ranked[1][1]:
        raise RuntimeError(
            f"tie-break is still tied: {scores}. Review the evidence before setting "
            "SELECTED_VARIANT_OVERRIDE."
        )
    return ranked[0][0], "top2_tiebreak", fixed_scores


def stage_dataset(src: Path, dst: Path) -> None:
    if dst.exists() and (dst / "meta/info.json").exists() and (dst / "videos").exists():
        out("dataset already staged:", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    out("=== staging dataset Drive -> local ===")
    p = subprocess.Popen(
        ["rsync", "-a", "--info=progress2", str(src) + "/", str(dst) + "/"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in iter(p.stdout.readline, ""):
        if line.strip():
            out("[copy]", line.rstrip())
    rc = p.wait()
    if rc:
        raise RuntimeError(f"dataset rsync failed rc={rc}")


def restore_checkpoint(drive_run: Path, out_dir: Path) -> None:
    meta_path = drive_run / "checkpoint_meta.json"
    drive_last = drive_run / "checkpoint_last"
    if not meta_path.exists() or not drive_last.exists():
        out("[resume] no Drive checkpoint; starting fresh")
        return
    meta = json.loads(meta_path.read_text())
    step = str(meta["step"])
    local_step = out_dir / "checkpoints" / step
    if not local_step.exists():
        local_step.parent.mkdir(parents=True, exist_ok=True)
        out(f"[resume] restoring Drive checkpoint step={step}")
        shutil.copytree(drive_last, local_step)
    last_link = out_dir / "checkpoints" / "last"
    if last_link.exists() or last_link.is_symlink():
        if last_link.is_symlink() or last_link.is_file():
            last_link.unlink()
        else:
            shutil.rmtree(last_link)
    last_link.symlink_to(step, target_is_directory=True)
    out("[resume] checkpoints/last restored")


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", root / "py_AI"))
    drive = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/parc2026-cache"))
    drive_dataset = Path(
        os.environ.get(
            "PARC_DRIVE_DATASET",
            str(drive / "datasets/lerobot_libero_plus_v3_train"),
        )
    )
    fixed_eval = Path(
        os.environ.get(
            "PI05_FIXED_EVAL_SUMMARY",
            str(drive / "pi05-fixed-sim-eval-v2/screening_summary.json"),
        )
    )
    tiebreak_eval = Path(
        os.environ.get(
            "PI05_TIEBREAK_SUMMARY",
            str(drive / "pi05-top2-tiebreak-v1/screening_summary.json"),
        )
    )
    drive_train_root = Path(
        os.environ.get("PI05_DRIVE_TRAIN_ROOT", str(drive / "pi05-full-train-v1"))
    )
    local_dataset = root / "datasets/public_libero_plus_v3_train"

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    if not (repo / ".git").exists():
        raise FileNotFoundError(repo)
    for p in [drive_dataset / "meta/info.json", drive_dataset / "meta/stats.json", fixed_eval]:
        if not p.exists():
            raise FileNotFoundError(p)

    name, vram = gpu_info()
    out("GPU:", name, vram, "MiB")
    if vram < 38000:
        raise RuntimeError(
            "long 20k training requires >=38 GiB VRAM here; use A100 40GB/80GB or H100. "
            "L4 is intended for simulator evaluation."
        )
    if free_gib() < 45 and not local_dataset.exists():
        raise RuntimeError(f"fresh local disk >=45 GiB required before dataset stage; free={free_gib():.1f}")

    selected, decision_source, fixed_scores = select_variant(fixed_eval, tiebreak_eval)
    out("SELECTED_VARIANT:", selected)
    out("decision source :", decision_source)
    drive_train_root.mkdir(parents=True, exist_ok=True)
    (drive_train_root / "selection.json").write_text(
        json.dumps(
            {
                "selected_variant": selected,
                "decision_source": decision_source,
                "fixed_eval_scores": fixed_scores,
                "tiebreak_summary": str(tiebreak_eval) if tiebreak_eval.exists() else None,
            },
            indent=2,
        )
        + "\n"
    )

    stage_dataset(drive_dataset, local_dataset)
    stats = json.loads((local_dataset / "meta/stats.json").read_text())
    for feature in ("observation.state", "action"):
        assert "q01" in stats[feature] and "q99" in stats[feature]
    out("q01/q99: PASS")
    out(f"local free after stage: {free_gib():.1f} GiB")

    # Rebuild and verify the exact schema-v2 manifests and leakage contract. This also
    # sets up the patched LeRobot v0.4.4 training venv and executes the strong GA=8 trace.
    gate_env = os.environ.copy()
    gate_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PARC_ROOT": str(root),
            "PY_AI_REPO": str(repo),
            "PI05_DATASET_ROOT": str(local_dataset),
            "PI05_DATASET_REPO_ID": "lerobot/libero_plus",
            "PI05_DATASET_REVISION": "f3f49f426d75030177b18778374005bc12ccd588",
            "RUN_ABLATIONS": "false",
        }
    )
    run([sys.executable, "-u", repo / "tools/colab/run_pi05_group_aware_ablation.py"], cwd=repo, env=gate_env)

    manifest_root = root / "outputs/dataset_ablation_manifests_v2_group_aware"
    manifest_map = {
        "V1_MULTI": manifest_root / "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL.json",
        "V2_SQRT": manifest_root / "V2_SQRT_BALANCED_RAW.json",
    }
    manifest = manifest_map[selected]
    m = json.loads(manifest.read_text())
    expected_counts = {"V1_MULTI": 13579, "V2_SQRT": 10758}
    assert m["schema_version"] == 2 and m["group_aware"] is True
    assert m["summary"]["episode_count"] == expected_counts[selected]
    out("selected manifest:", manifest)
    out("episodes:", m["summary"]["episode_count"])
    out("episode_ids_sha256:", m.get("episode_ids_sha256"))

    lerobot_root = root / "vendor/lerobot-pi05-ablation-v2"
    pi05_venv = lerobot_root / ".venv"
    if not (pi05_venv / "bin/activate").exists():
        raise FileNotFoundError(pi05_venv)

    if vram >= 60000:
        bs, ga = 16, 8
    else:
        bs, ga = 8, 16
    steps = int(os.environ.get("PI05_STEPS", "20000"))
    save_steps = int(os.environ.get("PI05_SAVE_STEPS", "500"))
    lora_r = int(os.environ.get("PI05_LORA_R", "16"))
    lr = os.environ.get("PI05_LR", "5e-5")
    short = "v1_multi" if selected == "V1_MULTI" else "v2_sqrt"
    run_name = os.environ.get(
        "RUN_NAME", f"pi05_full_{short}_r{lora_r}_bs{bs}x{ga}_{steps}"
    )
    out_root = root / "cache/pi05-full-train/outputs"
    log_root = root / "cache/pi05-full-train/logs"
    out_dir = out_root / run_name
    drive_run = drive_train_root / run_name
    drive_run.mkdir(parents=True, exist_ok=True)

    out("RUN_NAME       :", run_name)
    out("batch x GA     :", bs, "x", ga, "=", bs * ga)
    out("steps/save     :", steps, "/", save_steps)
    out("Drive run      :", drive_run)
    restore_checkpoint(drive_run, out_dir)

    persisted_step: str | None = None
    sync_lock = threading.Lock()

    def sync_latest(force: bool = False) -> None:
        nonlocal persisted_step
        last = out_dir / "checkpoints/last"
        if not last.exists():
            return
        try:
            src = last.resolve()
        except OSError:
            return
        if not (src / "pretrained_model/train_config.json").exists():
            return
        step = src.name
        if not force and time.time() - tree_latest_mtime(src) < 30:
            return
        if not force and persisted_step == step:
            return
        with sync_lock:
            tmp = drive_run / f"checkpoint_tmp_{step}"
            final = drive_run / "checkpoint_last"
            shutil.rmtree(tmp, ignore_errors=True)
            out(f"[drive] syncing stable checkpoint step={step} ...")
            shutil.copytree(src, tmp)
            shutil.rmtree(final, ignore_errors=True)
            tmp.rename(final)
            (drive_run / "checkpoint_meta.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "synced_at_unix": time.time(),
                        "variant": selected,
                        "run_name": run_name,
                        "repo_sha": subprocess.check_output(
                            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                        ).strip(),
                        "manifest_sha256": m.get("episode_ids_sha256"),
                    },
                    indent=2,
                )
                + "\n"
            )
            for src_file, name2 in (
                (log_root / f"{run_name}.log", "train.log"),
                (log_root / f"{run_name}.vram.csv", "vram.csv"),
            ):
                if src_file.exists():
                    shutil.copy2(src_file, drive_run / name2)
            persisted_step = step
            out(f"[drive] checkpoint step={step}: PASS")

    launcher = repo / "examples/pi05_libero_finetune/scripts/full_pi05_lora_manifest.sh"
    if not launcher.exists():
        raise FileNotFoundError(launcher)
    train_env = os.environ.copy()
    train_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "FULL_TRAIN_MANIFEST": str(manifest),
            "PI05_DATASET_ROOT": str(local_dataset),
            "PI05_DATASET_REPO_ID": "lerobot/libero_plus",
            "PI05_VIDEO_BACKEND": "pyav",
            "PI05_BS": str(bs),
            "PI05_GA": str(ga),
            "PI05_STEPS": str(steps),
            "PI05_SAVE_STEPS": str(save_steps),
            "PI05_LORA_R": str(lora_r),
            "PI05_LR": lr,
            "PI05_SEED": "1000",
            "DATALOADER_NUM_WORKERS": "4",
            "RUN_NAME": run_name,
            "LEROBOT_ROOT": str(lerobot_root),
            "PI05_VENV": str(pi05_venv),
            "OUT_ROOT": str(out_root),
            "LOG_ROOT": str(log_root),
        }
    )
    p = subprocess.Popen(
        ["bash", str(launcher)],
        cwd=launcher.parent.parent,
        env=train_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert p.stdout is not None
    tail: deque[str] = deque(maxlen=150)

    def reader() -> None:
        for line in iter(p.stdout.readline, ""):
            line = line.rstrip()
            tail.append(line)
            out("[train]", line)

    threading.Thread(target=reader, daemon=True).start()
    start = time.time()
    last_sync_try = 0.0
    try:
        while p.poll() is None:
            now = time.time()
            if now - last_sync_try >= 60:
                sync_latest(False)
                last_sync_try = now
            util, used, total = gpu_status()
            elapsed = int(now - start)
            out(
                f"[heartbeat] elapsed={elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
                f" | GPU={util}% | VRAM={used}/{total} MiB | free={free_gib():.1f} GiB"
                f" | Drive_step={persisted_step or 'none'} | running"
            )
            time.sleep(20)
    except KeyboardInterrupt:
        out("KeyboardInterrupt: terminating training process after trying to persist the latest stable checkpoint")
        try:
            sync_latest(True)
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        raise
    finally:
        if p.poll() is not None:
            sync_latest(True)

    rc = p.wait()
    out("training return code:", rc)
    if rc:
        out("=== LAST 150 TRAIN LINES ===")
        for line in tail:
            out(line)
        raise RuntimeError(
            "training process stopped. If this was a runtime interruption, Run all on a fresh "
            "A100 runtime to restore the latest Drive checkpoint."
        )

    sync_latest(True)
    summary = out_dir / "full_train_summary.json"
    if not summary.exists():
        raise FileNotFoundError(summary)
    shutil.copy2(summary, drive_run / "full_train_summary.json")
    final_adapter = out_dir / "checkpoints/last/pretrained_model"
    drive_adapter = drive_run / "final_pretrained_model"
    shutil.rmtree(drive_adapter, ignore_errors=True)
    shutil.copytree(final_adapter, drive_adapter)
    result = json.loads(summary.read_text())
    result.update(
        {
            "selected_variant_short": selected,
            "repo_sha": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "drive_run": str(drive_run),
        }
    )
    (drive_run / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
    out("=== FULL TRAIN: PASS ===")
    out(json.dumps(result, indent=2))
    out("Drive:", drive_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
