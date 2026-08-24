#!/usr/bin/env python3
"""配布物に内部情報が混入していないか検査する。

公開してよいのは「配布タスクそのもの」と「参加者向けの手順」だけである。
非公開タスクの識別子、タスクの生成方法、運用フェーズの区分、内部の人名や
リポジトリ名が配布物に残っていると、参加者が非公開タスクを推測できてしまう。

使い方:

    python scripts/check_no_leaks.py           # リポジトリ全体を検査
    python scripts/check_no_leaks.py <path>    # 特定のディレクトリだけ検査

問題が 1 件でもあれば終了コード 1 を返す。CI や公開前の手動確認で使う。

設計上の前提: このリポジトリ自体が将来公開される可能性があるため、
**非公開タスクの一覧をこのスクリプトに書かない**。代わりに、配布タスク CSV と
BDDL から「公開してよい識別子」を組み立て、それ以外のタスク風識別子を
検出する許可リスト方式を採る。ハッシュを埋め込む方式は、候補集合が公開されている
以上（LIBERO-plus のタスク名は公開）総当たりで逆引きできるため採らない。
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# 検査から除く。.git は履歴、examples の notebook は参加者向けの学習例。
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
SKIP_FILES = {"check_no_leaks.py"}

# --- 禁止パターン -----------------------------------------------------------
# (正規表現, 説明) の組。バイナリを含む全ファイルに対して照合する。
FORBIDDEN: list[tuple[str, str]] = [
    # タスクの生成方法（どの既存タスクをどう改変したか）
    (r"source_forward_bddl", "タスク生成の元 BDDL を示すキー"),
    (r"\b(A_direct_flip|C_pure_primitive|D_role_swap|E_cross_fixture)\b", "タスク生成機構の分類名"),
    (r'"mechanism"\s*:', "タスク生成機構を示すキー"),
    (r'"family"\s*:', "タスクファミリー分類を示すキー"),
    (r"\bT3[ab]\b", "Track3 のサブトラック区分"),
    (r"(?<![A-Za-z0-9])(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+", "元タスク（順方向 BDDL）の識別子"),
    # 逆転タスクの命名体系（分類記号・連番から母集団の構造が読める）
    (r"(?<![A-Za-z0-9])rev_[a-z0-9_]+", "逆転タスクの内部命名 (rev_*)"),
    (r"(?<![A-Za-z0-9])REV_[A-Z0-9_]+", "逆転タスクの内部命名 (REV_*)"),
    (r"(?<![A-Za-z0-9])[fF][1-6]_\d{2}(?![A-Za-z0-9])", "ファミリー記号つき連番"),
    # 運用フェーズ（どのタスクがどの評価に使われるか）
    (r"1_配布|2_予選|3_Omni|4_最終", "評価フェーズ区分"),
    (r"TASK_USAGE|task_usage|USAGE_EPISODES\s*[:=]\s*\{\s*[^}\s]", "運用フェーズの切り替え機構"),
    # 内部の体制・インフラ
    (r"PAI_compe|matsuolab/PAI|vla-competition|eval-pipeline", "内部リポジトリ名"),
    (r"村上|西浦|酒井|小野|MURAKAMI|NISHIURA|SAKAI|Shinnosuke", "内部の人名"),
    (r"p-shared|192\.168\.\d+\.\d+|/storage/home/", "内部ホスト・内部パス"),
    (r"libero_t12", "内部の suite パッケージ名"),
]

# --- 許可リスト方式でのタスク識別子検査 -------------------------------------
# LIBERO-plus 由来のタスク名（摂動サフィックスつき）と、Track3 の BDDL stem を
# 「公開してよい識別子」とする。これに一致しないタスク風の識別子を報告する。
TASKISH = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[a-z][a-z0-9_]{15,}"
    r"(?:_view_\d[\w]*|_light_\d+|_table_\d+|_noise_\d+|_add_\d+|_initstate_\d+|level\d_sample\d+)"
    r"(?![A-Za-z0-9_])"
)


def allowed_task_ids() -> set[str]:
    allowed: set[str] = set()
    for csv_path in REPO.glob("compe/t*/T*_TASKS.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if tid := (row.get("task_id") or "").strip():
                    allowed.add(tid)
    bddl_dir = REPO / "compe" / "t3" / "assets" / "bddl_files" / "libero_t3"
    allowed.update(p.stem for p in bddl_dir.glob("*.bddl"))
    return allowed


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.name in SKIP_FILES:
            continue
        yield path


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO
    allowed = allowed_task_ids()
    patterns = [(re.compile(rx), why) for rx, why in FORBIDDEN]
    findings: list[str] = []
    n_files = 0

    for path in iter_files(root):
        n_files += 1
        try:
            text = path.read_text(errors="ignore")
        except OSError as e:
            findings.append(f"{path}: 読めない ({e})")
            continue
        rel = path.relative_to(root)
        for rx, why in patterns:
            for m in rx.finditer(text):
                findings.append(f"{rel}: {why}: {m.group(0)[:80]!r}")
                break  # 同じ理由は1ファイル1件だけ報告する
        for m in TASKISH.finditer(text):
            if m.group(0) not in allowed:
                findings.append(f"{rel}: 配布対象でないタスク識別子: {m.group(0)!r}")

    print(f"検査ファイル数: {n_files} / 配布タスク識別子: {len(allowed)} 件")
    if findings:
        print(f"\n混入の疑い {len(findings)} 件:\n")
        for line in findings:
            print(f"  - {line}")
        return 1
    print("混入なし")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
