#!/usr/bin/env python3
"""Apply the guarded M3 schedule/time hook to exact LeRobot trainer revisions.

This is a narrow source transformation, not a floating monkey patch.  It checks
revision-specific anchors and refuses to edit an unexpected trainer.  π0.5's
v0.4.4 source must already contain this repository's gradient-accumulation
patch; SmolVLA uses its frozen 2026 commit natively.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# PARC2026_M3_GUARDED_RUNTIME_V1"


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count}")
    return text.replace(old, new, 1)


def patch_pi05_v044(text: str) -> str:
    if MARKER in text:
        return text
    if "gradient_accumulation_steps=_grad_accum" not in text or "while step < cfg.steps:" not in text:
        raise RuntimeError(
            "π0.5 LeRobot v0.4.4 must have the proven grad-accum patch before the M3 runtime patch"
        )
    if "import os" not in text:
        raise RuntimeError("π0.5 patched trainer unexpectedly lacks import os")

    text = _replace_once(
        text,
        "    dataloader = torch.utils.data.DataLoader(\n",
        (
            f"    {MARKER}\n"
            "    _parc_m3 = None\n"
            "    if os.environ.get(\"PARC_M3_RUNTIME_SPEC\"):\n"
            "        from tools.benchmark.m3_lerobot_train_hook import setup_runtime\n"
            "        _parc_m3 = setup_runtime(dataset)\n"
            "        shuffle = False\n"
            "        sampler = _parc_m3.sampler\n\n"
            "    dataloader = torch.utils.data.DataLoader(\n"
        ),
        label="pi05 dataloader hook",
    )
    text = _replace_once(
        text,
        "        start_time = time.perf_counter()\n        batch = next(dl_iter)\n        batch = preprocessor(batch)\n",
        (
            "        start_time = time.perf_counter()\n"
            "        if _parc_m3 is not None:\n"
            "            _parc_m3.before_batch_fetch()\n"
            "        batch = next(dl_iter)\n"
            "        if _parc_m3 is not None:\n"
            "            _parc_m3.observe_raw_batch(batch)\n"
            "        batch = preprocessor(batch)\n"
        ),
        label="pi05 raw-batch evidence hook",
    )
    text = _replace_once(
        text,
        "        if not accelerator.sync_gradients:\n            continue\n\n        step += 1\n",
        (
            "        if not accelerator.sync_gradients:\n"
            "            if _parc_m3 is not None:\n"
            "                _parc_m3.optimizer_boundary(sync_gradients=False)\n"
            "            continue\n\n"
            "        _parc_m3_stop = (\n"
            "            _parc_m3.optimizer_boundary(sync_gradients=True)\n"
            "            if _parc_m3 is not None else False\n"
            "        )\n"
            "        step += 1\n"
        ),
        label="pi05 optimizer-boundary hook",
    )
    text = _replace_once(
        text,
        "        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps\n",
        "        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps or _parc_m3_stop\n",
        label="pi05 forced checkpoint hook",
    )
    text = _replace_once(
        text,
        "\n    if is_main_process:\n        progbar.close()\n",
        (
            "\n        if _parc_m3_stop:\n"
            "            break\n\n"
            "    if _parc_m3 is not None:\n"
            "        _parc_m3.finalize()\n\n"
            "    if is_main_process:\n"
            "        progbar.close()\n"
        ),
        label="pi05 final stop/evidence hook",
    )
    return text


def patch_smolvla_pinned(text: str) -> str:
    if MARKER in text:
        return text
    required = (
        "def make_dataloaders(",
        "cfg.accelerator.gradient_accumulation.steps",
        "for _ in range(step, cfg.steps):",
        "should_save_checkpoint(step, cfg.save_freq, cfg.steps)",
    )
    if any(token not in text for token in required):
        raise RuntimeError("SmolVLA pinned trainer anchors do not match expected revision")

    text = _replace_once(
        text,
        "import logging\nimport sys\n",
        "import logging\nimport os\nimport sys\n",
        label="smolvla import os",
    )
    text = _replace_once(
        text,
        "    device_type = parallel_dims.device_type\n",
        (
            f"    {MARKER}\n"
            "    if os.environ.get(\"PARC_M3_RUNTIME_SPEC\"):\n"
            "        from tools.benchmark.m3_lerobot_train_hook import setup_runtime\n"
            "        _parc_m3_runtime = setup_runtime(dataset)\n"
            "        shuffle = False\n"
            "        sampler = _parc_m3_runtime.sampler\n\n"
            "    device_type = parallel_dims.device_type\n"
        ),
        label="smolvla dataloader hook",
    )
    text = _replace_once(
        text,
        "    dl_iter = cycle(dataloader)\n    policy.train()\n",
        (
            "    dl_iter = cycle(dataloader)\n"
            "    from tools.benchmark.m3_lerobot_train_hook import get_active_runtime\n"
            "    _parc_m3 = get_active_runtime()\n"
            "    policy.train()\n"
        ),
        label="smolvla runtime handle",
    )
    text = _replace_once(
        text,
        "        step_start = time.perf_counter()\n        batch = next(dl_iter)\n        preprocessing_start = time.perf_counter()\n",
        (
            "        step_start = time.perf_counter()\n"
            "        if _parc_m3 is not None:\n"
            "            _parc_m3.before_batch_fetch()\n"
            "        batch = next(dl_iter)\n"
            "        if _parc_m3 is not None:\n"
            "            _parc_m3.observe_raw_batch(batch)\n"
            "        preprocessing_start = time.perf_counter()\n"
        ),
        label="smolvla raw-batch evidence hook",
    )
    text = _replace_once(
        text,
        "        step += 1\n        if is_main_process():\n            progbar.update(1)\n",
        (
            "        _parc_m3_stop = (\n"
            "            _parc_m3.optimizer_boundary(sync_gradients=accelerator.sync_gradients)\n"
            "            if _parc_m3 is not None else False\n"
            "        )\n"
            "        step += 1\n"
            "        if is_main_process():\n"
            "            progbar.update(1)\n"
        ),
        label="smolvla optimizer-boundary hook",
    )
    text = _replace_once(
        text,
        "        is_saving_step = should_save_checkpoint(step, cfg.save_freq, cfg.steps)\n",
        "        is_saving_step = should_save_checkpoint(step, cfg.save_freq, cfg.steps) or _parc_m3_stop\n",
        label="smolvla forced checkpoint hook",
    )
    text = _replace_once(
        text,
        "\n    if is_main_process():\n        progbar.close()\n        logging.info(\"End of training\")\n",
        (
            "\n        if _parc_m3_stop:\n"
            "            break\n\n"
            "    if _parc_m3 is not None:\n"
            "        _parc_m3.finalize()\n\n"
            "    if is_main_process():\n"
            "        progbar.close()\n"
            "        logging.info(\"End of training\")\n"
        ),
        label="smolvla final stop/evidence hook",
    )
    return text


def patch_file(path: Path, kind: str) -> bool:
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    if kind == "pi05_v044":
        updated = patch_pi05_v044(original)
    elif kind == "smolvla_pinned":
        updated = patch_smolvla_pinned(original)
    else:
        raise ValueError(kind)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--kind", choices=("pi05_v044", "smolvla_pinned"), required=True)
    args = p.parse_args()
    changed = patch_file(args.trainer, args.kind)
    print(f"M3 LeRobot trainer patch: {'APPLIED' if changed else 'ALREADY_APPLIED'} :: {args.trainer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
