#!/usr/bin/env python3
"""Run PARC2026 pi0.5 group-aware cheap dataset ablation from Colab.

Safety contract:
- build/verify group-aware schema-v2 manifests
- require exact/action trajectory leakage == 0
- ensure π0.5 STATE/ACTION q01/q99 normalization stats exist
- setup LeRobot v0.4.4 training env
- verify GA=8 by runtime trace
- only then run the four cheap-screening variants when RUN_ABLATIONS=true
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def cmd(args, *, cwd=None, env=None):
    print(">>>", shlex.join([str(x) for x in args]), flush=True)
    child_env = (env or os.environ).copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(args, cwd=cwd, env=child_env, check=True)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    root = Path(os.environ.get("PARC_ROOT", "/content/parc2026"))
    repo = Path(os.environ.get("PY_AI_REPO", root / "py_AI"))
    pi05_dir = repo / "examples/pi05_libero_finetune"
    dataset_root = Path(os.environ["PI05_DATASET_ROOT"])
    dataset_id = os.environ.get("PI05_DATASET_REPO_ID", "lerobot/libero_plus")
    dataset_revision = os.environ.get("PI05_DATASET_REVISION") or None
    run_ablations = env_bool("RUN_ABLATIONS", False)

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for pi0.5 / PaliGemma access")
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(dataset_root / "meta" / "info.json")
    if not any(dataset_root.glob("data/**/*.parquet")):
        raise FileNotFoundError(f"no parquet under {dataset_root / 'data'}")
    if not (dataset_root / "videos").exists():
        raise FileNotFoundError(dataset_root / "videos")

    for p in [root / "cache", root / "outputs", root / "vendor"]:
        p.mkdir(parents=True, exist_ok=True)

    git_sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    print("dataset:", dataset_id, "@", dataset_revision, flush=True)
    print("dataset root:", dataset_root, flush=True)
    print("git:", git_sha, flush=True)

    print("\n=== Gate 1/5: Python/data tooling ===", flush=True)
    if shutil.which("uv") is None:
        cmd([sys.executable, "-m", "pip", "install", "-q", "uv"])
    cmd(["uv", "python", "install", "3.10"])
    py310 = subprocess.check_output(["uv", "python", "find", "3.10"], text=True).strip()
    print("python3.10:", py310, flush=True)
    cmd([sys.executable, "-m", "pip", "install", "-q", "pyarrow>=16", "pandas>=2"])
    import pandas as pd

    print("\n=== Gate 2/5: π0.5 quantile stats ===", flush=True)
    quantile_report = root / "outputs" / "pi05_quantile_stats_gate.json"
    cmd([
        sys.executable,
        str(repo / "tools/data/ensure_pi05_quantile_stats.py"),
        "--root", str(dataset_root),
        "--report", str(quantile_report),
    ])
    qreport = json.loads(quantile_report.read_text())
    assert set(qreport["features"]) == {"observation.state", "action"}
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    for feature in ("observation.state", "action"):
        assert "q01" in stats[feature] and "q99" in stats[feature]
    print("PI05 Quantile Stats Gate: PASS", flush=True)

    print("\n=== Gate 3/5: group-aware manifests ===", flush=True)
    static_out = root / "outputs" / "static_quality_v1"
    metrics = static_out / "episode_quality_metrics.csv"
    if not metrics.exists():
        cmd([
            sys.executable,
            str(repo / "tools/data/static_quality_analyzer.py"),
            "--root", str(dataset_root),
            "--out", str(static_out),
            "--smooth-window", "5",
            "--robust-z-threshold", "5.0",
        ])

    legacy = root / "outputs" / "dataset_ablation_manifests_v1"
    if not (legacy / "run_matrix.json").exists():
        args = [
            sys.executable,
            str(repo / "tools/data/build_dataset_ablation_manifests.py"),
            "--metrics-csv", str(metrics),
            "--out", str(legacy),
            "--dataset-id", dataset_id,
            "--seed", "20260830",
            "--eval-per-task", "2",
        ]
        if dataset_revision:
            args += ["--dataset-revision", dataset_revision]
        cmd(args)

    leak_v1 = root / "outputs" / "trajectory_group_leakage_v1"
    group_members = leak_v1 / "trajectory_group_members.csv"
    if not group_members.exists():
        cmd([
            sys.executable,
            str(repo / "tools/data/check_trajectory_group_leakage.py"),
            "--root", str(dataset_root),
            "--manifests-dir", str(legacy),
            "--out", str(leak_v1),
            "--metrics-csv", str(metrics),
            "--round-decimals", "6",
        ])

    manifests = root / "outputs" / "dataset_ablation_manifests_v2_group_aware"
    args = [
        sys.executable,
        str(repo / "tools/data/build_group_aware_ablation_manifests.py"),
        "--metrics-csv", str(metrics),
        "--trajectory-groups-csv", str(group_members),
        "--out", str(manifests),
        "--dataset-id", dataset_id,
        "--seed", "20260830",
        "--eval-per-task", "2",
    ]
    if dataset_revision:
        args += ["--dataset-revision", dataset_revision]
    cmd(args)

    matrix = json.loads((manifests / "run_matrix.json").read_text())
    eval_manifest = json.loads((manifests / "FIXED_EVAL_HOLDOUT.json").read_text())
    print("group-aware:", matrix.get("group_aware"), flush=True)
    print("fixed eval:", matrix["fixed_eval"]["episode_count"], "episodes /", matrix["fixed_eval_group_count"], "groups", flush=True)
    print("protected:", matrix["protected_episode_count"], f"({matrix['protected_fraction']:.2%})", flush=True)
    for name, row in matrix["variants"].items():
        print("variant:", name, "episodes=", row["episode_count"], "frames=", row["frame_count"], flush=True)

    print("\n=== Gate 4/5: trajectory leakage ===", flush=True)
    leak_v2 = root / "outputs" / "trajectory_group_leakage_v2_group_aware"
    cmd([
        sys.executable,
        str(repo / "tools/data/check_trajectory_group_leakage.py"),
        "--root", str(dataset_root),
        "--manifests-dir", str(manifests),
        "--out", str(leak_v2),
        "--metrics-csv", str(metrics),
        "--round-decimals", "6",
        "--fail-on-exact-leakage",
    ])
    leak_report = json.loads((leak_v2 / "trajectory_group_leakage_summary.json").read_text())
    leak_df = pd.read_csv(leak_v2 / "manifest_leakage_report.csv")

    assert matrix.get("schema_version") == 2
    assert matrix.get("group_aware") is True
    assert eval_manifest.get("schema_version") == 2
    assert eval_manifest.get("group_aware") is True
    assert matrix.get("dataset_id") == dataset_id
    assert eval_manifest.get("dataset_id") == dataset_id
    if dataset_revision:
        assert matrix.get("dataset_revision") == dataset_revision
        assert eval_manifest.get("dataset_revision") == dataset_revision
    assert leak_report["trajectory_leakage_gate"] == "PASS"
    assert int(leak_df["exact_leakage_group_count"].sum()) == 0
    assert int(leak_df["action_leakage_group_count"].sum()) == 0

    if dataset_id == "lerobot/libero_plus" and matrix["source_summary"]["episode_count"] == 14347:
        assert matrix["protected_episode_count"] == 674
        expected = {
            "V0_RAW": 13673,
            "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL": 13579,
            "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL": 13301,
            "V2_SQRT_BALANCED_RAW": 10758,
        }
        for name, n in expected.items():
            assert matrix["variants"][name]["episode_count"] == n

    print("Group-aware Manifest Gate: PASS", flush=True)
    print("Trajectory Leakage Gate: PASS", flush=True)
    print("TRAINING MANIFEST SAFETY GATE: PASS", flush=True)

    print("\n=== Gate 5/5: training env + GA=8 runtime ===", flush=True)
    train_data_root = root / "cache" / "pi05-ablation-group-aware-v2"
    lerobot_root = root / "vendor" / "lerobot-pi05-ablation-v2"
    train_data_root.mkdir(parents=True, exist_ok=True)
    setup_env = os.environ.copy()
    setup_env.update({
        "PYTHON": py310,
        "DATA_ROOT": str(train_data_root),
        "LEROBOT_ROOT": str(lerobot_root),
    })
    cmd(["bash", "scripts/setup_train.sh"], cwd=pi05_dir, env=setup_env)
    print("training env: ready", flush=True)

    trace = root / "outputs" / "pi05_ga8_probe_trace.jsonl"
    trace_summary = root / "outputs" / "pi05_ga8_probe_summary.json"
    trace.unlink(missing_ok=True)
    trace_summary.unlink(missing_ok=True)
    probe_env = os.environ.copy()
    probe_env.update({
        "PARC_GA_TRACE_FILE": str(trace),
        "PYTHONPATH": str(repo / "tools/pi05/ga_instrument") + (":" + probe_env["PYTHONPATH"] if probe_env.get("PYTHONPATH") else ""),
        "PI05_DATASET_ROOT": str(dataset_root),
        "PI05_DATASET_REPO_ID": dataset_id,
        "PI05_VIDEO_BACKEND": "pyav",
        "SMOKE_BS": "1",
        "SMOKE_GA": "8",
        "SMOKE_STEPS": "3",
        "SMOKE_SKIP_MERGE": "1",
        "RUN_NAME": "colab_ga8_probe",
        "DATA_ROOT": str(train_data_root),
        "LEROBOT_ROOT": str(lerobot_root),
        "HF_TOKEN": os.environ["HF_TOKEN"],
    })
    cmd(
        ["bash", "-lc", "source env_train.sh && bash scripts/smoke_pi05.sh"],
        cwd=pi05_dir,
        env=probe_env,
    )
    cmd([
        sys.executable,
        str(repo / "tools/pi05/summarize_ga_trace.py"),
        str(trace),
        "--expected-ga", "8",
        "--expected-steps", "3",
        "--json-out", str(trace_summary),
    ])
    print(trace_summary.read_text(), flush=True)
    print("GA Gate: PASS", flush=True)

    print("effective batch: 32", flush=True)
    print("variants:", matrix["cheap_ablation_order"], flush=True)
    print("manifest schema:", matrix["schema_version"], "group-aware:", matrix["group_aware"], flush=True)
    print("RUN_ABLATIONS:", run_ablations, flush=True)

    results_dir = root / "outputs" / "pi05_dataset_ablation_v2_group_aware"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []

    if not run_ablations:
        print("skip: Gate確認後、実行する時だけ RUN_ABLATIONS=True に変更", flush=True)
        if not (results_dir / "comparison.json").exists():
            print("no comparison.json yet", flush=True)
        return 0

    for variant in matrix["cheap_ablation_order"]:
        manifest_path = manifests / f"{variant}.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest.get("schema_version") == 2
        assert manifest.get("group_aware") is True
        assert manifest.get("dataset_id") == dataset_id
        assert manifest.get("protected_episode_count") == matrix["protected_episode_count"]
        if dataset_revision:
            assert manifest.get("dataset_revision") == dataset_revision

        run_name = "colab_data_" + variant.lower()
        run_env = os.environ.copy()
        run_env.update({
            "DATA_ROOT": str(train_data_root),
            "LEROBOT_ROOT": str(lerobot_root),
            "ABLATION_MANIFEST": str(manifest_path),
            "PI05_DATASET_ROOT": str(dataset_root),
            "PI05_DATASET_REPO_ID": dataset_id,
            "PI05_VIDEO_BACKEND": "pyav",
            "ABLATION_BS": "4",
            "ABLATION_GA": "8",
            "ABLATION_STEPS": "150",
            "ABLATION_SEED": "1000",
            "RUN_NAME": run_name,
            "HF_TOKEN": os.environ["HF_TOKEN"],
        })
        print("\n=== RUN", variant, "===", flush=True)
        cmd(
            ["bash", "-lc", "source env_train.sh && bash scripts/cheap_ablation_pi05.sh"],
            cwd=pi05_dir,
            env=run_env,
        )
        summary_path = train_data_root / "pi05-ft-outputs" / run_name / "cheap_ablation_summary.json"
        row = json.loads(summary_path.read_text())
        row.update({
            "dataset_revision": dataset_revision,
            "git_sha": git_sha,
            "protected_episode_count": matrix["protected_episode_count"],
            "manifest_schema_version": manifest["schema_version"],
            "group_aware": manifest["group_aware"],
        })
        result_rows.append(row)
        (results_dir / f"{variant}.json").write_text(json.dumps(row, indent=2) + "\n")

    (results_dir / "comparison.json").write_text(json.dumps(result_rows, indent=2) + "\n")
    print("SCREENING COMPLETE — do not promote from loss alone", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
