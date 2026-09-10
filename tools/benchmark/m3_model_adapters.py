#!/usr/bin/env python3
"""Guarded M3 model-adapter orchestration for π0.5, SmolVLA and OpenVLA-OFT.

Preflight validates model/runtime/data contracts. Smoke and benchmark modes
launch the real scheduled training entries. Final micro-batch and gradient
accumulation are always loaded from the 72d summary at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tools.benchmark.m3_runner_core import (
    D10_HASH,
    EQUAL_DATA_BUDGET,
    EQUAL_WALL_SEC,
    EXECUTE_ENV,
    EXECUTE_VALUE,
    ORDERS,
    require_execution_guard,
    validate_batch_probe_summary,
)
from tools.benchmark.m3_scheduled_data import load_schedule

ROOT_DEFAULT = Path("/content/parc2026")
DRIVE_DEFAULT = Path("/content/drive/MyDrive/parc2026-cache")
SEEDS = (20260906, 20260907)

SOURCE_REFS = {
    "pi05": "v0.4.4",
    "smolvla": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e",
    "openvla_oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}


@dataclass(frozen=True)
class AdapterRuntime:
    model: str
    python: str
    source_root: str
    driver: str
    source_ref: str
    checkpoint: str


def default_runtimes(root: Path, repo: Path) -> dict[str, AdapterRuntime]:
    root = Path(root)
    repo = Path(repo)
    return {
        "pi05": AdapterRuntime(
            model="pi05",
            python=str(root / "vendor/lerobot-pi05-m3-probe/.venv/bin/python"),
            source_root=str(root / "vendor/lerobot-pi05-m3-probe"),
            driver=str(repo / "tools/benchmark/m3_lerobot_adapter_driver.py"),
            source_ref=SOURCE_REFS["pi05"],
            checkpoint="lerobot/pi05_libero_base",
        ),
        "smolvla": AdapterRuntime(
            model="smolvla",
            python=str(root / "venv-smolvla-m3/bin/python"),
            source_root=str(root / "vendor/lerobot-smolvla-m3"),
            driver=str(repo / "tools/benchmark/m3_lerobot_adapter_driver.py"),
            source_ref=SOURCE_REFS["smolvla"],
            checkpoint="lerobot/smolvla_base",
        ),
        "openvla_oft": AdapterRuntime(
            model="openvla_oft",
            python=str(root / "venv-openvla-oft-m3/bin/python"),
            source_root=str(root / "vendor/openvla-oft-m3"),
            driver=str(repo / "tools/benchmark/m3_openvla_adapter_driver.py"),
            source_ref=SOURCE_REFS["openvla_oft"],
            checkpoint="openvla/openvla-7b",
        ),
    }


def load_batch_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_batch_probe_summary(payload)
    return payload


def runtime_preflight(
    runtime: AdapterRuntime,
    *,
    repo: Path | None = None,
    strict_files: bool = True,
) -> list[str]:
    blockers: list[str] = []
    items = [("python", runtime.python), ("source_root", runtime.source_root), ("driver", runtime.driver)]
    if repo is not None:
        entry = (
            Path(repo) / "tools/benchmark/m3_openvla_train_entry.py"
            if runtime.model == "openvla_oft"
            else Path(repo) / "tools/benchmark/m3_lerobot_train_entry_v2.py"
        )
        items.append(("train_entry", str(entry)))
        if runtime.model == "openvla_oft":
            items.append(("checkpoint_finalizer", str(Path(repo) / "tools/benchmark/m3_openvla_finalize_checkpoint.py")))
    for label, raw in items:
        p = Path(raw)
        if strict_files and not p.exists():
            blockers.append(f"{runtime.model}:{label}_missing:{p}")
    return blockers


def build_adapter_command(
    *,
    runtime: AdapterRuntime,
    batch_cfg: dict[str, Any],
    schedule_path: Path,
    batch_summary_path: Path,
    dataset_root: Path,
    manifest_path: Path,
    track: str,
    order: str,
    mode: str,
    out: Path,
    streaming_contract: Path | None = None,
) -> list[str]:
    """Build the contract-only preflight command."""
    if track not in {"equal_data", "equal_wall"}:
        raise ValueError(f"unsupported track: {track}")
    if order not in ORDERS:
        raise ValueError(f"unsupported order: {order}")
    if mode not in {"preflight", "smoke", "benchmark"}:
        raise ValueError(f"unsupported mode: {mode}")
    cmd = [
        runtime.python,
        "-u",
        runtime.driver,
        "--model",
        runtime.model,
        "--source-root",
        runtime.source_root,
        "--dataset-root",
        str(dataset_root),
        "--manifest",
        str(manifest_path),
        "--schedule",
        str(schedule_path),
        "--batch-summary",
        str(batch_summary_path),
        "--track",
        track,
        "--order",
        order,
        "--mode",
        mode,
        "--micro-batch",
        str(int(batch_cfg["micro_batch"])),
        "--grad-accum",
        str(int(batch_cfg["gradient_accumulation"])),
        "--out",
        str(out),
    ]
    if runtime.model == "openvla_oft":
        if streaming_contract is None:
            raise ValueError("OpenVLA adapter requires streaming contract")
        cmd.extend(["--streaming-contract", str(streaming_contract)])
    return cmd


def build_training_command(
    *,
    runtime: AdapterRuntime,
    repo: Path,
    batch_cfg: dict[str, Any],
    schedule_path: Path,
    dataset_root: Path,
    manifest_path: Path,
    track: str,
    order: str,
    mode: str,
    output_dir: Path,
    result_out: Path,
    streaming_contract: Path | None = None,
) -> list[str]:
    if mode not in {"smoke", "benchmark"}:
        raise ValueError("live M3 training command is only valid for smoke/benchmark")
    require_execution_guard(mode)
    common = [
        "--repo-root",
        str(repo),
        "--source-root",
        runtime.source_root,
        "--dataset-root",
        str(dataset_root),
        "--manifest",
        str(manifest_path),
        "--schedule",
        str(schedule_path),
        "--micro-batch",
        str(int(batch_cfg["micro_batch"])),
        "--grad-accum",
        str(int(batch_cfg["gradient_accumulation"])),
        "--track",
        track,
        "--mode",
        mode,
        "--order",
        order,
        "--output-dir",
        str(output_dir),
        "--result-out",
        str(result_out),
    ]
    if runtime.model == "openvla_oft":
        if streaming_contract is None:
            raise ValueError("OpenVLA live training requires validated 69c streaming contract")
        entry = Path(repo) / "tools/benchmark/m3_openvla_train_entry.py"
        return [
            runtime.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            str(entry),
            *common,
            "--streaming-contract",
            str(streaming_contract),
        ]
    entry = Path(repo) / "tools/benchmark/m3_lerobot_train_entry_v2.py"
    return [runtime.python, "-u", str(entry), "--model", runtime.model, *common]


def build_post_training_command(
    *, runtime: AdapterRuntime, repo: Path, output_dir: Path, result_out: Path
) -> list[str] | None:
    if runtime.model != "openvla_oft":
        return None
    return [
        runtime.python,
        "-u",
        str(Path(repo) / "tools/benchmark/m3_openvla_finalize_checkpoint.py"),
        "--source-root",
        runtime.source_root,
        "--checkpoint-root",
        str(output_dir),
        "--result",
        str(result_out),
    ]


def build_run_specs(
    *,
    batch_summary: dict[str, Any],
    schedule_path: Path,
    batch_summary_path: Path,
    dataset_root: Path,
    manifest_path: Path,
    output_root: Path,
    root: Path,
    repo: Path,
    track: str,
    order: str,
    mode: str,
    streaming_contract: Path | None,
) -> list[dict[str, Any]]:
    require_execution_guard(mode)
    schedule = load_schedule(schedule_path, require_equal_data=True)
    if schedule.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("schedule D10 mismatch")
    seed = int(schedule.get("seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"unexpected M3 schedule seed: {seed}")
    batches = validate_batch_probe_summary(batch_summary)
    runtimes = default_runtimes(root, repo)
    specs: list[dict[str, Any]] = []
    for sequence_index, model in enumerate(ORDERS[order]):
        runtime = runtimes[model]
        run_root = Path(output_root) / order / track / f"seed-{seed}" / model
        result_out = run_root / "train_result.json"
        output_dir = run_root / "checkpoint"
        if mode == "preflight":
            cmd = build_adapter_command(
                runtime=runtime,
                batch_cfg=batches[model],
                schedule_path=schedule_path,
                batch_summary_path=batch_summary_path,
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                track=track,
                order=order,
                mode=mode,
                out=result_out,
                streaming_contract=streaming_contract,
            )
            post_cmd = None
        else:
            cmd = build_training_command(
                runtime=runtime,
                repo=repo,
                batch_cfg=batches[model],
                schedule_path=schedule_path,
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                track=track,
                order=order,
                mode=mode,
                output_dir=output_dir,
                result_out=result_out,
                streaming_contract=streaming_contract,
            )
            post_cmd = build_post_training_command(
                runtime=runtime,
                repo=repo,
                output_dir=output_dir,
                result_out=result_out,
            )
        specs.append(
            {
                "sequence_index": sequence_index,
                "model": model,
                "track": track,
                "order": order,
                "mode": mode,
                "source_ref": runtime.source_ref,
                "runtime": asdict(runtime),
                "sampling_seed": seed,
                "micro_batch": int(batches[model]["micro_batch"]),
                "gradient_accumulation": int(batches[model]["gradient_accumulation"]),
                "effective_batch_size": 32,
                "sample_budget": EQUAL_DATA_BUDGET if track == "equal_data" else None,
                "train_loop_sec": EQUAL_WALL_SEC if track == "equal_wall" else None,
                "sampling_schedule_sha256": schedule.get("schedule_sha256"),
                "equal_wall_stream_continues_after_materialized_prefix": track == "equal_wall",
                "command": cmd,
                "post_training_command": post_cmd,
                "post_training_work_excluded_from_train_wall_time": post_cmd is not None,
                "output_dir": str(output_dir),
                "out": str(result_out),
            }
        )
    return specs


def execute_specs(specs: list[dict[str, Any]], *, mode: str) -> None:
    if mode == "preflight":
        raise RuntimeError("preflight mode never executes model training")
    require_execution_guard(mode)
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")
    for spec in specs:
        subprocess.run(spec["command"], check=True, env=env)
        post_command = spec.get("post_training_command")
        if post_command:
            subprocess.run(post_command, check=True, env=env)
        result_path = Path(spec["out"])
        if not result_path.is_file():
            raise RuntimeError(f"M3 training result missing: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            raise RuntimeError(f"M3 training result is not PASS: {result_path}")
        if spec["model"] == "openvla_oft" and result.get("checkpoint_eval_ready") is not True:
            raise RuntimeError("OpenVLA checkpoint finalizer did not produce an eval-ready checkpoint")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-summary", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--track", choices=("equal_data", "equal_wall"), required=True)
    p.add_argument("--order", choices=("forward", "reverse"), required=True)
    p.add_argument("--mode", choices=("preflight", "smoke", "benchmark"), default="preflight")
    p.add_argument("--repo-root", type=Path, default=Path(os.environ.get("PY_AI_REPO", Path.cwd())))
    p.add_argument("--parc-root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--streaming-contract", type=Path)
    p.add_argument("--plan-out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    summary = load_batch_summary(args.batch_summary)
    specs = build_run_specs(
        batch_summary=summary,
        schedule_path=args.schedule,
        batch_summary_path=args.batch_summary,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        output_root=args.output_root,
        root=args.parc_root,
        repo=args.repo_root,
        track=args.track,
        order=args.order,
        mode=args.mode,
        streaming_contract=args.streaming_contract,
    )
    blockers: list[str] = []
    for runtime in default_runtimes(args.parc_root, args.repo_root).values():
        blockers.extend(
            runtime_preflight(
                runtime,
                repo=args.repo_root,
                strict_files=args.mode != "preflight",
            )
        )
    plan = {
        "schema_version": 3,
        "stage": "M3_model_adapter_plan",
        "status": "READY_FOR_EXECUTION" if not blockers else "BLOCKED",
        "mode": args.mode,
        "track": args.track,
        "order": args.order,
        "selected_episode_ids_sha256": D10_HASH,
        "sampling_seed": specs[0]["sampling_seed"] if specs else None,
        "execute_guard": {"environment_variable": EXECUTE_ENV, "required_value": EXECUTE_VALUE},
        "blockers": blockers,
        "runs": specs,
        "benchmark_training_started": False,
        "automatic_full_run": False,
    }
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "runs": len(specs), "plan": str(args.plan_out)}, indent=2))
    if args.execute:
        execute_specs(specs, mode=args.mode)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
