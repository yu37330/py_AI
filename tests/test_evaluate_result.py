"""build_omnicampus_result のテスト（インフラ障害と参加者 0 点の切り分け）

    python -m pytest tests/test_evaluate_result.py -v
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import evaluate as ev  # noqa: E402


def _write_result(dirpath: Path, submission_id: str, tracks: list[dict]) -> None:
    (dirpath / f"{submission_id}.json").write_text(
        json.dumps({"submission_id": submission_id, "tracks": tracks}),
        encoding="utf-8",
    )


def _track(name: str, score: float, **metrics) -> dict:
    return {
        "track": name,
        "benchmark": "libero_t1",
        "overall_score": score,
        "overall_metrics": {"mean_success_rate": score, **metrics},
        "tasks": [],
    }


# ------------------------------------------------------------------
# 正常系
# ------------------------------------------------------------------
def test_scores_are_averaged_across_tracks(tmp_path):
    _write_result(tmp_path, "server_8000", [_track("track1", 1.0), _track("track2", 0.5)])

    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert r["public_score"] == pytest.approx(0.75)
    assert r["base_passed"] is True
    assert "Score: 0.75" in r["course_top_message"]


def test_genuine_zero_score_is_not_treated_as_infra_failure(tmp_path):
    """参加者のポリシーが本当に 0% のケースは、これまで通り 0 点として採点する。

    インフラ障害の検出を入れたことで正常な 0 点まで例外にしてしまうと、
    全滅した提出が採点不能になる。error フラグの有無だけで区別する。
    """
    _write_result(tmp_path, "server_8000", [_track("track1", 0.0), _track("track2", 0.0)])

    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert r["public_score"] == 0.0
    assert r["base_passed"] is False
    assert "Success rate: 0%" in r["course_top_message"]


# ------------------------------------------------------------------
# インフラ障害: スコアとして記録してはいけない
# ------------------------------------------------------------------
def test_errored_track_raises_instead_of_reporting_zero(tmp_path):
    """摂動アセット欠損などで pipeline がトラックを 0 点(error=1.0) にした場合。

    これを平均に混ぜると参加者が本当に 0% だったケースと区別できなくなる。
    """
    _write_result(tmp_path, "server_8000", [_track("track2", 0.0, error=1.0)])

    with pytest.raises(ev.EvaluationInfraError) as exc:
        ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert "track2" in str(exc.value)


def test_partial_failure_raises_even_if_other_tracks_succeeded(tmp_path):
    """一部トラックだけ失敗した場合も、平均を薄めた偽スコアを出さない。"""
    _write_result(
        tmp_path,
        "server_8000",
        [_track("track1", 1.0), _track("track3", 0.0, error=1.0)],
    )

    with pytest.raises(ev.EvaluationInfraError) as exc:
        ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert "track3" in str(exc.value)
    assert "track1" not in str(exc.value)


def test_all_errored_tracks_are_listed(tmp_path):
    _write_result(
        tmp_path,
        "server_8000",
        [_track("track1", 0.0, error=1.0), _track("track2", 0.0, error=1.0)],
    )

    with pytest.raises(ev.EvaluationInfraError) as exc:
        ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert "track1" in str(exc.value) and "track2" in str(exc.value)


# ------------------------------------------------------------------
# 結果ファイルが読めない系（従来通り error_result を返す）
# ------------------------------------------------------------------
def test_missing_result_file_returns_error_result(tmp_path):
    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert r["public_score"] == 0.0
    assert "結果ファイルが見つかりません" in r["course_top_message"]


def test_empty_tracks_returns_error_result(tmp_path):
    _write_result(tmp_path, "server_8000", [])

    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert "トラック結果がありません" in r["course_top_message"]
