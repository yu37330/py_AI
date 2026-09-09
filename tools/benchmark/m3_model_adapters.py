#!/usr/bin/env python3
"""Guarded M3 model-adapter orchestration for π0.5, SmolVLA and OpenVLA-OFT.

This layer connects the already-frozen runner/sampling contracts to concrete
model runtime environments.  It deliberately keeps the final micro-batch and
gradient-accumulation values late-bound from 72d.

It does not silently start a benchmark: smoke/benchmark modes require the same
PARC_M3_EXECUTE=1 opt-in as m3_runner_core.
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
    MODELS,
    ORDERS,
    require_execution_guard,
    validate_batch_probe_summary,
)
from tools.benchmark.m3_scheduled_data import load_schedule

ROOT_DEFAULT = Path("/content/parc2026")
DRIVE_DEFAULT = Path("/content/drive/MyDrive/parc2026-cache")

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


def runtime_preflight(runtime: AdapterRuntime, *, strict_files: bool = True) -> list[str]:
    blockers: list[str] = []
    for label, raw in (("python", runtime.python), ("source_root", runtime.source_root), ("driver", runtime.driver)):
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
    schedule = load_schedule(schedule_path, require_equal_data=(track == "equal_data"))
    if schedule.get("selected_episode_ids_sha256") != D10_HASH:
        raise ValueError("schedule D10 mismatch")
    batches = validate_batch_probe_summary(batch_summary)
    runtimes = default_runtimes(root, repo)
    specs: list[dict[str, Any]] = []
    for sequence_index, model in enumerate(ORDERS[order]):
        runtime = runtimes[model]
        out = Path(output_root) / order / track / model / "train_result.json"
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
            out=out,
            streaming_contract=streaming_contract,
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
                "micro_batch": int(batches[model]["micro_batch"]),
                "gradient_accumulation": int(batches[model]["gradient_accumulation"]),
                "effective_batch_size": 32,
                "sample_budget": EQUAL_DATA_BUDGET if track == "equal_data" else None,
                "train_loop_sec": EQUAL_WALL_SEC if track == "equal_wall" else None,
                "sampling_schedule_sha256": schedule.get("schedule_sha256"),
                "command": cmd,
                "out": str(out),
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
        blockers.extend(runtime_preflight(runtime, strict_files=args.mode != "preflight"))
    plan = {
        "schema_version": 1,
        "stage": "M3_model_adapter_plan",
        "status": "READY_FOR_EXECUTION" if not blockers else "BLOCKED",
        "mode": args.mode,
        "track": args.track,
        "order": args.order,
        "selected_episode_ids_sha256": D10_HASH,
        "execute_guard": {"environment_variable": EXECUTE_ENV, "required_value": EXECUTE_VALUE},
        "blockers": blockers,
        "runs": specs,
        "benchmark_training_started": False,
    }
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "runs": len(specs), "plan": str(args.plan_out)}, indent=2))
    if args.execute:
        execute_specs(specs, mode=args.mode)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
