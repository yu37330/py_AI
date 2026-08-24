from __future__ import annotations

import csv
import re
from pathlib import Path

from compe.t3.paths import SUITE_NAME, T3_BDDL_DIR, t3_task_stems

_REGISTERED = False


_CONTACT_GUARD_INSTALLED = False

_LANGUAGE_RE = re.compile(r"\(:language\s+([^()]+?)\s*\)")


def _resting_on_static_scene(env, obj_name) -> bool:
    sim = env.sim
    skip = getattr(env, "_t3_nonsupport_geom_ids", None)
    if skip is None:
        skip = set()
        for gid in range(sim.model.ngeom):
            nm = sim.model.geom_id2name(gid) or ""
            if nm.startswith(("gripper", "robot", "mount")):
                skip.add(gid)
        for _, obj in getattr(env, "objects_dict", {}).items():
            for g in getattr(obj, "contact_geoms", []):
                try:
                    skip.add(sim.model.geom_name2id(g))
                except Exception:
                    pass
        env._t3_nonsupport_geom_ids = skip
    obj_gids = set()
    for g in getattr(env.get_object(obj_name), "contact_geoms", []):
        try:
            obj_gids.add(sim.model.geom_name2id(g))
        except Exception:
            pass
    if not obj_gids:
        return True
    for c in range(sim.data.ncon):
        con = sim.data.contact[c]
        g1, g2 = con.geom1, con.geom2
        if g1 in obj_gids and g2 not in skip and g2 not in obj_gids:
            return True
        if g2 in obj_gids and g1 not in skip and g1 not in obj_gids:
            return True
    return False


def _install_table_contact_guard():
    global _CONTACT_GUARD_INSTALLED
    if _CONTACT_GUARD_INSTALLED:
        return

    from libero.libero.envs.object_states.base_object_states import SiteObjectState

    if getattr(SiteObjectState, "_t3_table_contact_guard", False):
        _CONTACT_GUARD_INSTALLED = True
        return

    _orig_check_ontop = SiteObjectState.check_ontop

    def check_ontop(self, other):
        base = _orig_check_ontop(self, other)
        if not base or self.env.get_object(self.parent_name) is not None:
            return base
        try:
            return _resting_on_static_scene(self.env, other.object_name)
        except Exception:
            return base

    SiteObjectState.check_ontop = check_ontop
    SiteObjectState._t3_table_contact_guard = True
    _CONTACT_GUARD_INSTALLED = True


def _bddl_language(stem: str, fallback) -> str:
    try:
        text = (T3_BDDL_DIR / f"{stem}.bddl").read_text()
    except (OSError, UnicodeDecodeError):
        return fallback(SUITE_NAME, stem + ".bddl")
    m = _LANGUAGE_RE.search(text)
    if not m:
        return fallback(SUITE_NAME, stem + ".bddl")
    return " ".join(m.group(1).split())


def register_t3():
    global _REGISTERED

    _install_table_contact_guard()

    from libero.libero import benchmark as _b
    from libero.libero.benchmark import libero_suite_task_map as _m

    stems = t3_task_stems()

    _m.libero_task_map[SUITE_NAME] = stems

    if SUITE_NAME not in _b.libero_suites:
        _b.libero_suites.append(SUITE_NAME)

    _b.task_maps[SUITE_NAME] = {
        task: _b.Task(
            name=task,
            language=_bddl_language(task, _b.grab_language_from_filename),
            problem="Libero",
            problem_folder=SUITE_NAME,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )
        for task in stems
    }

    if _REGISTERED:
        return _b.get_benchmark(SUITE_NAME)

    @_b.register_benchmark
    class LIBERO_T3(_b.Benchmark):
        def __init__(self, task_order_index: int = 0):
            assert task_order_index == 0, (
                "LIBERO_T3 has a variable task count; only task_order_index=0 is supported."
            )
            super().__init__(task_order_index=task_order_index)
            self.name = SUITE_NAME
            self._make_benchmark()

        def _make_benchmark(self):
            tasks = list(_b.task_maps[self.name].values())
            self.tasks = tasks
            self.n_tasks = len(tasks)

        def get_task_init_states(self, i):
            import os
            import torch
            from libero.libero import get_libero_path

            task = self.tasks[i]
            path = os.path.join(
                get_libero_path("init_states"),
                task.problem_folder,
                task.init_states_file,
            )
            return torch.load(path, weights_only=False)

    _REGISTERED = True
    return _b.get_benchmark(SUITE_NAME)
