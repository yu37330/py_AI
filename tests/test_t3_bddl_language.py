"""Track 3 のタスク説明文の取得（`_bddl_language`）のテスト

    python -m pytest tests/test_t3_bddl_language.py -v

`.bddl` が読めない場合と `(:language ...)` が無い場合は、LIBERO 側の
`grab_language_from_filename` へ退避する。**この退避経路は引数の数を間違えても
正常系では一切現れない。** 正常系だけ通していると、退避が必要になった本番で
初めて `TypeError` として出る。

LIBERO-plus の実シグネチャは 2 引数である。

    def grab_language_from_filename(suite_name, x):   # libero/libero/benchmark/__init__.py

呼び出し側がこれとずれていないことを、4 つの経路すべてで固定する。
"""

import inspect
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from compe.t3.paths import SUITE_NAME  # noqa: E402
from compe.t3.register import _bddl_language  # noqa: E402


def _fallback(suite_name, x):
    """LIBERO-plus の `grab_language_from_filename` と同じシグネチャのスタブ。

    **引数を減らさないこと。** ここを 1 引数にすると本体のバグを検出できなくなる。
    """
    return f"fallback:{suite_name}:{x}"


@pytest.fixture
def bddl_dir(tmp_path, monkeypatch):
    """`_bddl_language` が読む先を一時ディレクトリに差し替える。"""
    monkeypatch.setattr("compe.t3.register.T3_BDDL_DIR", tmp_path)
    return tmp_path


def test_language_を_bddl_から読む(bddl_dir):
    (bddl_dir / "pick_up_the_mug.bddl").write_text(
        "(define (problem X)\n  (:language pick up the mug)\n)\n"
    )
    assert _bddl_language("pick_up_the_mug", _fallback) == "pick up the mug"


def test_language_の空白は畳まれる(bddl_dir):
    (bddl_dir / "t.bddl").write_text("(:language   pick   up\n   the mug  )\n")
    assert _bddl_language("t", _fallback) == "pick up the mug"


def test_ファイルが無ければ退避する(bddl_dir):
    # OSError 経路
    assert _bddl_language("missing", _fallback) == f"fallback:{SUITE_NAME}:missing.bddl"


def test_utf8_で読めなければ退避する(bddl_dir):
    # UnicodeDecodeError 経路
    (bddl_dir / "broken.bddl").write_bytes(b"\x00\x05\x16\x07" + b"\xa3" * 60)
    assert _bddl_language("broken", _fallback) == f"fallback:{SUITE_NAME}:broken.bddl"


def test_language_が無ければ退避する(bddl_dir):
    # 正規表現が当たらない経路
    (bddl_dir / "nolang.bddl").write_text("(define (problem X)\n  (:domain robosuite)\n)\n")
    assert _bddl_language("nolang", _fallback) == f"fallback:{SUITE_NAME}:nolang.bddl"


# ---------------------------------------------------------------------------
# **ここが本体。** 上のテストは `_fallback` スタブに対して引数の数を固定している
# だけで、**LIBERO-plus 側がシグネチャを変えたら素通りする。** スタブを実物に
# 突き合わせておかないと、守れているのは「呼び出し側を書き換えた」場合だけになる。
#
# libero を入れていない環境（CI の軽いジョブなど）では skip する。
# setup.sh を通した環境と採点イメージでは必ず走る。
# ---------------------------------------------------------------------------


def _real_grab_language_from_filename():
    try:
        from libero.libero.benchmark import grab_language_from_filename
    except Exception:
        pytest.skip("libero が入っていないため実物との突き合わせを飛ばす")
    return grab_language_from_filename


def test_スタブが実物と同じシグネチャである():
    real = _real_grab_language_from_filename()
    assert inspect.signature(real) == inspect.signature(_fallback), (
        "LIBERO-plus の grab_language_from_filename のシグネチャが変わっている。"
        "_fallback と _bddl_language の呼び出しを合わせること"
    )


def test_実物に対して呼び出しが成立する():
    real = _real_grab_language_from_filename()
    # _bddl_language が退避時に行う呼び出しがそのまま通ることを確かめる。
    inspect.signature(real).bind(SUITE_NAME, "some_task.bddl")
