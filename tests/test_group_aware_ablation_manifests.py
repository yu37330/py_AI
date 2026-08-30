from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "group_aware",
    ROOT / "tools" / "data" / "build_group_aware_ablation_manifests.py",
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    metrics = []
    groups = []
    episode = 0
    for task in range(2):
        for group in range(4):
            group_hash = f"task{task}-group{group}"
            for sibling in range(3):
                metrics.append(
                    {
                        "episode_index": episode,
                        "task_index": task,
                        "task_name": f"task-{task}",
                        "frames": 10 + sibling,
                        "flag_integrity": False,
                        "quality_flag_count": 0,
                        "quality_review_candidate": False,
                    }
                )
                groups.append(
                    {
                        "episode_index": episode,
                        "task_index": task,
                        "exact_group_hash": group_hash,
                        "exact_group_id": group_hash,
                    }
                )
                episode += 1
    metrics_path = tmp_path / "metrics.csv"
    groups_path = tmp_path / "groups.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(groups).to_csv(groups_path, index=False)
    return metrics_path, groups_path


def test_group_aware_holdout_excludes_all_siblings(tmp_path: Path):
    metrics_path, groups_path = _fixtures(tmp_path)
    out = tmp_path / "out"
    matrix = MOD.build(
        metrics_csv=metrics_path,
        groups_csv=groups_path,
        out_dir=out,
        dataset_id="test/dataset",
        dataset_revision="rev",
        seed=20260830,
        eval_per_task=2,
    )

    fixed = json.loads((out / "FIXED_EVAL_HOLDOUT.json").read_text())
    v0 = json.loads((out / "V0_RAW.json").read_text())
    groups = pd.read_csv(groups_path)

    assert fixed["group_aware"] is True
    assert len(fixed["episode_ids"]) == 4
    assert len(fixed["exact_group_hashes"]) == 4
    assert len(set(fixed["exact_group_hashes"])) == 4
    assert fixed["protected_episode_count"] == 12
    assert matrix["protected_episode_count"] == 12

    train_groups = set(
        groups.loc[
            groups["episode_index"].isin(v0["episode_ids"]), "exact_group_hash"
        ].astype(str)
    )
    assert train_groups.isdisjoint(set(fixed["exact_group_hashes"]))
    assert set(v0["episode_ids"]).isdisjoint(set(fixed["protected_episode_ids"]))


def test_group_aware_build_is_deterministic(tmp_path: Path):
    metrics_path, groups_path = _fixtures(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    kwargs = dict(
        metrics_csv=metrics_path,
        groups_csv=groups_path,
        dataset_id="test/dataset",
        dataset_revision="rev",
        seed=7,
        eval_per_task=2,
    )
    MOD.build(out_dir=out_a, **kwargs)
    MOD.build(out_dir=out_b, **kwargs)

    a = json.loads((out_a / "FIXED_EVAL_HOLDOUT.json").read_text())
    b = json.loads((out_b / "FIXED_EVAL_HOLDOUT.json").read_text())
    assert a["episode_ids"] == b["episode_ids"]
    assert a["exact_group_hashes"] == b["exact_group_hashes"]
    assert a["protected_episode_ids"] == b["protected_episode_ids"]
    assert a["episode_ids_sha256"] == b["episode_ids_sha256"]


def test_quality_filtering_stays_nested_after_group_exclusion(tmp_path: Path):
    metrics_path, groups_path = _fixtures(tmp_path)
    metrics = pd.read_csv(metrics_path)
    metrics.loc[metrics["episode_index"] == 23, "quality_flag_count"] = 2
    metrics.loc[metrics["episode_index"] == 23, "quality_review_candidate"] = True
    metrics.to_csv(metrics_path, index=False)

    out = tmp_path / "out"
    MOD.build(
        metrics_csv=metrics_path,
        groups_csv=groups_path,
        out_dir=out,
        dataset_id="test/dataset",
        dataset_revision=None,
        seed=1,
        eval_per_task=1,
    )
    v0 = set(json.loads((out / "V0_RAW.json").read_text())["episode_ids"])
    multi = set(
        json.loads((out / "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL.json").read_text())["episode_ids"]
    )
    all_review = set(
        json.loads((out / "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL.json").read_text())["episode_ids"]
    )
    assert all_review.issubset(multi)
    assert multi.issubset(v0)
