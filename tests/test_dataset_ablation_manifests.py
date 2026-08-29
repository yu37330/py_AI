from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "data"))

from build_dataset_ablation_manifests import build_manifests  # noqa: E402


def _synthetic_metrics() -> pd.DataFrame:
    rows = []
    ep = 0
    for task_index, count in [(0, 8), (1, 12), (2, 18)]:
        for i in range(count):
            review = i == count - 1
            two_flags = i == count - 2
            rows.append(
                {
                    "episode_index": ep,
                    "task_index": task_index,
                    "task_name": f"task-{task_index}",
                    "frames": 100 + task_index * 20 + i,
                    "flag_integrity": False,
                    "quality_flag_count": 2 if two_flags else (1 if review else 0),
                    "quality_review_candidate": review or two_flags,
                }
            )
            ep += 1
    return pd.DataFrame(rows)


def test_build_manifests_keeps_fixed_eval_disjoint_and_deterministic(tmp_path: Path):
    metrics = tmp_path / "episode_quality_metrics.csv"
    _synthetic_metrics().to_csv(metrics, index=False)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    m1 = build_manifests(
        metrics,
        out1,
        dataset_id="test/libero",
        dataset_revision="abc123",
        seed=7,
        eval_per_task=2,
    )
    m2 = build_manifests(
        metrics,
        out2,
        dataset_id="test/libero",
        dataset_revision="abc123",
        seed=7,
        eval_per_task=2,
    )

    eval1 = json.loads((out1 / "FIXED_EVAL_HOLDOUT.json").read_text())
    eval2 = json.loads((out2 / "FIXED_EVAL_HOLDOUT.json").read_text())
    assert eval1["episode_ids"] == eval2["episode_ids"]
    assert len(eval1["episode_ids"]) == 6

    eval_ids = set(eval1["episode_ids"])
    for name in m1["variants"]:
        manifest1 = json.loads((out1 / f"{name}.json").read_text())
        manifest2 = json.loads((out2 / f"{name}.json").read_text())
        assert manifest1["episode_ids"] == manifest2["episode_ids"]
        assert not eval_ids.intersection(manifest1["episode_ids"])


def test_filter_variants_are_nested_and_sqrt_balance_reduces_dominant_tasks(tmp_path: Path):
    metrics = tmp_path / "episode_quality_metrics.csv"
    _synthetic_metrics().to_csv(metrics, index=False)
    out = tmp_path / "out"
    build_manifests(
        metrics,
        out,
        dataset_id="test/libero",
        dataset_revision=None,
        seed=11,
        eval_per_task=1,
    )

    raw = json.loads((out / "V0_RAW.json").read_text())
    integrity = json.loads((out / "V1_INTEGRITY_ONLY.json").read_text())
    multi = json.loads((out / "V1_MULTI_FLAG_PRUNED_EXPERIMENTAL.json").read_text())
    all_review = json.loads((out / "V1_ALL_REVIEW_PRUNED_EXPERIMENTAL.json").read_text())
    sqrt = json.loads((out / "V2_SQRT_BALANCED_RAW.json").read_text())

    assert raw["episode_ids"] == integrity["episode_ids"]
    assert set(all_review["episode_ids"]).issubset(set(multi["episode_ids"]))
    assert set(multi["episode_ids"]).issubset(set(raw["episode_ids"]))
    assert set(sqrt["episode_ids"]).issubset(set(raw["episode_ids"]))
    assert sqrt["balancing"]["basis"] == "task frame exposure"
    assert sqrt["summary"]["frame_count"] < raw["summary"]["frame_count"]


def test_run_matrix_skips_integrity_duplicate_when_no_hard_failures(tmp_path: Path):
    metrics = tmp_path / "episode_quality_metrics.csv"
    _synthetic_metrics().to_csv(metrics, index=False)
    out = tmp_path / "out"
    matrix = build_manifests(
        metrics,
        out,
        dataset_id="test/libero",
        dataset_revision="r1",
        seed=1,
        eval_per_task=1,
    )
    assert "V1_INTEGRITY_ONLY" not in matrix["cheap_ablation_order"]
    assert matrix["skip_reason"]["V1_INTEGRITY_ONLY"]
    assert matrix["promotion_rule"].startswith("screening only")


def test_run_matrix_includes_integrity_variant_when_it_differs_from_raw(tmp_path: Path):
    df = _synthetic_metrics()
    # Mark one non-review episode as a hard integrity failure so it cannot be
    # selected into the OK-only eval holdout and remains a real training diff.
    idx = df.index[(~df["quality_review_candidate"])][0]
    df.loc[idx, "flag_integrity"] = True
    df.loc[idx, "quality_review_candidate"] = True
    df.loc[idx, "quality_flag_count"] = 1
    metrics = tmp_path / "episode_quality_metrics.csv"
    df.to_csv(metrics, index=False)
    out = tmp_path / "out"
    matrix = build_manifests(
        metrics,
        out,
        dataset_id="test/libero",
        dataset_revision="r1",
        seed=3,
        eval_per_task=1,
    )
    assert "V1_INTEGRITY_ONLY" in matrix["cheap_ablation_order"]
    assert "V1_INTEGRITY_ONLY" not in matrix["skip_reason"]
    assert (
        matrix["variants"]["V1_INTEGRITY_ONLY"]["episode_count"]
        == matrix["variants"]["V0_RAW"]["episode_count"] - 1
    )
