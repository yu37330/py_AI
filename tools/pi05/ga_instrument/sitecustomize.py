"""Runtime instrumentation injected into lerobot-train via PYTHONPATH.

Enable only when PARC_GA_TRACE_FILE is set. The module is intentionally named
``sitecustomize`` because Python imports it automatically at interpreter startup.
It records gradient-accumulation behavior without modifying upstream LeRobot.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

TRACE_FILE = os.environ.get("PARC_GA_TRACE_FILE")

if TRACE_FILE:
    _trace_path = Path(TRACE_FILE)
    _trace_path.parent.mkdir(parents=True, exist_ok=True)
    _lock = threading.Lock()
    _seq = 0

    def _emit(event: str, **payload: object) -> None:
        global _seq
        with _lock:
            _seq += 1
            row = {
                "seq": _seq,
                "time_ns": time.time_ns(),
                "pid": os.getpid(),
                "event": event,
                **payload,
            }
            with _trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _emit("sitecustomize_loaded")

    try:
        from accelerate import Accelerator

        _orig_backward = Accelerator.backward

        def _traced_backward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            _emit(
                "backward",
                sync_gradients=bool(getattr(self, "sync_gradients", False)),
                gradient_accumulation_steps=int(
                    getattr(self, "gradient_accumulation_steps", 1)
                ),
            )
            return _orig_backward(self, *args, **kwargs)

        Accelerator.backward = _traced_backward  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover
        _emit("instrumentation_error", target="Accelerator.backward", error=repr(exc))

    try:
        from accelerate.optimizer import AcceleratedOptimizer

        _orig_acc_step = AcceleratedOptimizer.step

        def _traced_acc_step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            gradient_state = getattr(self, "gradient_state", None)
            sync = bool(getattr(gradient_state, "sync_gradients", False))
            _emit("accelerated_optimizer_step_call", sync_gradients=sync)
            return _orig_acc_step(self, *args, **kwargs)

        AcceleratedOptimizer.step = _traced_acc_step  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover
        _emit("instrumentation_error", target="AcceleratedOptimizer.step", error=repr(exc))

    try:
        import torch.optim as optim

        for _name in ("AdamW", "Adam", "SGD", "Adagrad", "RMSprop"):
            _cls = getattr(optim, _name, None)
            if _cls is None or not hasattr(_cls, "step"):
                continue
            _orig = _cls.step
            if getattr(_orig, "_parc_ga_wrapped", False):
                continue

            def _make_step_wrapper(orig, name):  # type: ignore[no-untyped-def]
                def _wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                    _emit("underlying_optimizer_step", optimizer=name)
                    return orig(self, *args, **kwargs)

                _wrapped._parc_ga_wrapped = True  # type: ignore[attr-defined]
                return _wrapped

            _cls.step = _make_step_wrapper(_orig, _name)  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover
        _emit("instrumentation_error", target="torch.optim", error=repr(exc))

    try:
        import torch.optim.lr_scheduler as lrs

        _scheduler_base = getattr(lrs, "LRScheduler", None) or getattr(
            lrs, "_LRScheduler", None
        )
        if _scheduler_base is not None:
            _orig_sched_step = _scheduler_base.step

            def _traced_sched_step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                _emit("scheduler_step", scheduler=type(self).__name__)
                return _orig_sched_step(self, *args, **kwargs)

            _scheduler_base.step = _traced_sched_step  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover
        _emit("instrumentation_error", target="lr_scheduler", error=repr(exc))
