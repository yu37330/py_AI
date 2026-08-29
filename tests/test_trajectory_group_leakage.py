from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "trajectory_group_leakage",
    ROOT / "tools" / "data" / "check_trajectory_group_leakage.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def _fingerprint(state, action, task=1, decimals=6):
    return mod.fingerprint_arrays(
        task_index=task,
        state=np.asarray(state, dtype=np.float64),
        action=np.asarray(action, dtype=np.float64),
        decimals=decimals,
    )


def test_fingerprint_exact_and_action_groups():
    state = [[0, 0], [1, 1], [2, 2]]
    action = [[0.1], [0.2], [0.3]]
    exact1, action1, state1 = _fingerprint(state, action)

    state_tiny = [[0, 0], [1 + 1e-8, 1], [2, 2]]
    exact2, action2, state2 = _fingerprint(state_tiny, action)
    assert exact1 == exact2
    assert action1 == action2
    assert state1 == state2

    state_changed = [[0, 0], [1.01, 1], [2, 2]]
    exact3, action3, state3 = _fingerprint(state_changed, action)
    assert exact3 != exact1
    assert state3 != state1
    assert action3 == action1


def test_manifest_group_overlap_is_detected():
    members = pd.DataFrame(
        [
            {
                "episode_index": 1,
                "task_index": 0,
                "frames": 3,
                "exact_group_hash": "A",
                "action_group_hash": "a",
            },
            {
                "episode_index": 2,
                "task_index": 0,
                "frames": 3,
                "exact_group_hash": "A",
                "action_group_hash": "a",
            },
            {
                "episode_index": 3,
                "task_index": 0,
                "frames": 3,
                "exact_group_hash": "B",
                "action_group_hash": "b",
            },
            {
                "episode_index": 4,
                "task_index": 0,
                "frames": 3,
                "exact_group_hash": "C",
                "action_group_hash": "a",
            },
        ]
    )
    eval_manifest = {"episode_ids": [1]}
    training = {
        "LEAK_EXACT": {"episode_ids": [2, 3]},
        "NO_EXACT_BUT_ACTION": {"episode_ids": [3, 4]},
    }
    summary, detail = mod.evaluate_manifests(members, eval_manifest, training)

    by_variant = summary.set_index("variant")
    assert by_variant.loc["LEAK_EXACT", "exact_leakage_group_count"] == 1
    assert not bool(by_variant.loc["LEAK_EXACT", "exact_gate_pass"])
    assert by_variant.loc["NO_EXACT_BUT_ACTION", "exact_leakage_group_count"] == 0
    assert bool(by_variant.loc["NO_EXACT_BUT_ACTION", "exact_gate_pass"])
    assert by_variant.loc["NO_EXACT_BUT_ACTION", "action_leakage_group_count"] == 1
    assert set(detail["leakage_kind"]) == {"exact", "action"}


def test_read_manifests(tmp_path: Path):
    (tmp_path / "FIXED_EVAL_HOLDOUT.json").write_text(
        json.dumps({"kind": "fixed_eval_holdout", "episode_ids": [1]}),
        encoding="utf-8",
    )
    (tmp_path / "V0_RAW.json").write_text(
        json.dumps(
            {
                "kind": "training_episode_manifest",
                "variant": "V0_RAW",
                "episode_ids": [2],
            }
        ),
        encoding="utf-8",
    )
    evaluation, training = mod.read_manifest_files(tmp_path)
    assert evaluation["episode_ids"] == [1]
    assert set(training) == {"V0_RAW"}
