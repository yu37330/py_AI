from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import torch

_SUITE = "libero_t2"
_PKG_DIR = Path(__file__).resolve().parent
_CSV = _PKG_DIR / "T2_TASKS.csv"


_SUFFIX_RE = re.compile(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")

_REGISTERED = False


def _load_rows() -> list[dict]:
    with open(_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{_CSV} is empty; cannot register {_SUITE}.")
    return rows


def _init_states_for(task, get_libero_path):
    fn = task.init_states_file
    root = get_libero_path("init_states")
    if "_add_" in fn or "_level" in fn:
        p = os.path.join(root, "libero_newobj", task.problem_folder, fn)
        return torch.load(p, weights_only=False).reshape(1, -1)
    stem, ext = os.path.splitext(fn)
    stripped = _SUFFIX_RE.sub("", stem) + ext
    p = os.path.join(root, task.problem_folder, stripped)
    return torch.load(p, weights_only=False)


def register_t2():
    global _REGISTERED
    from libero.libero import benchmark as _b
    from libero.libero import get_libero_path

    rows = _load_rows()
    _b.task_maps[_SUITE] = {
        r["task_id"]: _b.Task(
            name=r["task_id"],
            language=r["instruction"],
            problem="Libero",
            problem_folder=r["suite"],
            bddl_file=f"{r['task_id']}.bddl",
            init_states_file=f"{r['task_id']}.pruned_init",
        )
        for r in rows
    }
    if _SUITE not in _b.libero_suites:
        _b.libero_suites.append(_SUITE)

    if _REGISTERED:
        return _b.get_benchmark(_SUITE)

    @_b.register_benchmark
    class LIBERO_T2(_b.Benchmark):
        def __init__(self, task_order_index: int = 0):
            assert task_order_index == 0, (
                f"{_SUITE} has a variable task count; only task_order_index=0 supported."
            )
            super().__init__(task_order_index=task_order_index)
            self.name = _SUITE
            self._make_benchmark()

        def _make_benchmark(self):
            self.tasks = list(_b.task_maps[self.name].values())
            self.n_tasks = len(self.tasks)

        def get_task_init_states(self, i):
            return _init_states_for(self.tasks[i], get_libero_path)

    _REGISTERED = True

    return _b.get_benchmark(_SUITE)
