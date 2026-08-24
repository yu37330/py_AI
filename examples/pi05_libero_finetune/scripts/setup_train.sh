#!/usr/bin/env bash
# pi0.5 学習環境の構築。
#
#   bash scripts/setup_train.sh     # このディレクトリ（examples/pi05_libero_finetune）で実行
#   source env_train.sh             # 以後、学習を回すシェルで毎回 source する
#
# ここで作るのは **学習専用の venv** であり、配布環境の評価用 venv（リポジトリ
# ルートの setup.sh が作るもの）とは別に作る。
# 評価・提出前チェックは必ず評価用の setup.sh + env.sh 側で行うこと。
#
# やること:
#   1. Python の確認
#   2. FFmpeg の導入（torchcodec 用。入れられなければ pyav にフォールバック）
#   3. lerobot を固定タグで clone
#   4. patches/ を適用
#   5. venv 作成 + 依存インストール、env_train.sh の生成

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HERE="$PWD"

PYTHON="${PYTHON:-python3.10}"
# 秘密情報（HF_TOKEN / WANDB_API_KEY）は .env から読む。git 管理下に置かないこと。
ENV_FILE="${ENV_FILE:-${HOME}/.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
# venv・重み・データセット・チェックポイントの置き場。ホームは狭いので、
# 既定を大容量ディスク（${HOME}/data）に置く。
DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${DATA_ROOT}/pi05-ft/lerobot}"
# 採点環境が Python 3.10 のため、推論側で入る最新の lerobot は 0.4.4 である
# （0.5.0 以降は Python 3.12 以上を要求する）。学習と推論で版を揃えるため、
# 学習側も同じタグに固定する。ここを動かすと patches/ が当たらなくなる。
LEROBOT_REF="v0.4.4"

echo "[setup-train] 1/5 Python の確認"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "$PYTHON が見つかりません。PYTHON=/path/to/python3.10 で指定してください。" >&2
    exit 1
fi
echo "[setup-train]   $("$PYTHON" --version)"

# torchcodec は FFmpeg の共有ライブラリ（libavutil.so.56 等）を dlopen するため、
# 入っていないと学習の DataLoader が起動直後に落ちる。Ubuntu 22.04 の ffmpeg 4.4
# は torchcodec の対応範囲（FFmpeg 4〜7）に収まる。入れられない環境でも pyav に
# フォールバックできる（scripts/full_pi05_lora.sh の PI05_VIDEO_BACKEND 参照）
# ので、ここで失敗しても止めない。INSTALL_FFMPEG=0 でスキップできる。
echo "[setup-train] 2/5 FFmpeg の導入（torchcodec 用）"
if ldconfig -p 2>/dev/null | grep -q 'libavutil\.so\.'; then
    echo "[setup-train]   導入済み: $(ffmpeg -version 2>/dev/null | head -1 || echo 'libavutil あり')"
elif [ "${INSTALL_FFMPEG:-1}" != "1" ]; then
    echo "[setup-train]   INSTALL_FFMPEG=${INSTALL_FFMPEG} のためスキップ（pyav で回すこと）"
elif command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
    echo "[setup-train]   apt-get install ffmpeg"
    if $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq \
        && $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; then
        echo "[setup-train]   $(ffmpeg -version 2>/dev/null | head -1)"
    else
        echo "[setup-train]   ffmpeg の導入に失敗した。PI05_VIDEO_BACKEND=pyav で回すこと" >&2
    fi
else
    echo "[setup-train]   apt-get が無い。ffmpeg を手動で入れるか、pyav で回すこと" >&2
fi

echo "[setup-train] 3/5 lerobot の取得（$LEROBOT_REF）"
if [ ! -d "$LEROBOT_ROOT/.git" ]; then
    mkdir -p "$(dirname "$LEROBOT_ROOT")"
    git clone --filter=blob:none https://github.com/huggingface/lerobot.git "$LEROBOT_ROOT"
fi
git -C "$LEROBOT_ROOT" fetch --quiet --tags origin
git -C "$LEROBOT_ROOT" checkout --quiet "$LEROBOT_REF"
echo "[setup-train]   $LEROBOT_ROOT @ $(git -C "$LEROBOT_ROOT" describe --tags)"

echo "[setup-train] 4/5 パッチ適用"
for patch in "$HERE"/patches/*.patch; do
    name=$(basename "$patch")
    if git -C "$LEROBOT_ROOT" apply --reverse --check "$patch" 2>/dev/null; then
        echo "[setup-train]   $name（適用済みのためスキップ）"
    elif git -C "$LEROBOT_ROOT" apply --check "$patch" 2>/dev/null; then
        git -C "$LEROBOT_ROOT" apply "$patch"
        echo "[setup-train]   $name を適用"
    else
        echo "[setup-train]   $name を適用できません（$LEROBOT_REF からずれています）" >&2
        exit 1
    fi
done

echo "[setup-train] 5/5 venv 作成 + 依存インストール"
VENV="$LEROBOT_ROOT/.venv"
if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel -q
# pi0.5 は lerobot の pi extra を使う。これは transformers の fork
# （fix/lerobot_openpi ブランチ）を要求する。素の transformers では
# lerobot が「An incorrect transformer version is used」で失敗する。
# peft extra は transformers-dep を引いて pi と衝突するので、単体で入れる。
pip install -q -e "${LEROBOT_ROOT}[pi]"
pip install -q "peft>=0.18.0,<1.0.0"
# 実際に入った版を記録しておく（再現用。バージョン確定はこのファイルを見ること）
pip freeze --exclude-editable > "$HERE/requirements-train.lock.txt"
echo "[setup-train]   依存を requirements-train.lock.txt に記録"

cat > "$HERE/env_train.sh" <<EOF
# 学習用シェルで source する（評価用の env.sh とは併用しないこと）
# HF_TOKEN / WANDB_API_KEY は .env から読む（このファイルには書かない）。
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
source "$VENV/bin/activate"
export LEROBOT_ROOT="$LEROBOT_ROOT"
export PI05_VENV="$VENV"
# 大きいものはすべて $DATA_ROOT に置く。ホームには入らない。
export HF_HOME="$DATA_ROOT/hf"                  # 重みのキャッシュ + データセット（\$HF_HOME/lerobot/<name>）
export OUT_ROOT="$DATA_ROOT/pi05-ft-outputs"    # チェックポイント
export LOG_ROOT="$DATA_ROOT/pi05-ft-logs"       # 学習ログ + VRAM CSV
# W&B。save_freq ごとに LoRA アダプタ（未マージ）が artifact として上がる
# （lerobot の WandBLogger.log_policy、wandb.disable_artifact=false が既定）。
export WANDB_PROJECT="\${WANDB_PROJECT:-lerobot-libero-ft}"
export WANDB_DIR="$DATA_ROOT/wandb"             # run ディレクトリ
export WANDB_CACHE_DIR="$DATA_ROOT/wandb-cache" # artifact のステージング
EOF

echo "[setup-train] 動作確認"
python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), \
    'transformers が fork 版になっていません（pi extra を確認）'
print('  transformers: fork 版 ok')
"
python -c "
import lerobot, torch
from lerobot.policies.pi05.configuration_pi05 import PI05Config
cfg = PI05Config()
assert hasattr(cfg, 'drop_n_last_frames'), 'pi05-config-defaults.patch が当たっていません'
print(f'  lerobot={lerobot.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')
print(f'  drop_n_last_frames={cfg.drop_n_last_frames}')
"
# torchcodec は import 時に FFmpeg の共有ライブラリを dlopen する。ここで通れば
# PI05_VIDEO_BACKEND=torchcodec で回せる（各学習スクリプトの既定は pyav）。
python -c "
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    print('  torchcodec: ok（PI05_VIDEO_BACKEND=torchcodec が使える）')
except Exception as e:
    print(f'  torchcodec: 使えない（{type(e).__name__}）。PI05_VIDEO_BACKEND=pyav で回すこと')
"
grep -q "LEROBOT_GRAD_ACCUM" "$LEROBOT_ROOT/src/lerobot/scripts/lerobot_train.py" \
    || { echo "grad-accum-env-var.patch が当たっていません" >&2; exit 1; }
echo "[setup-train]   grad-accum-env-var.patch: ok"

cat <<EOF

[setup-train] 完了。次の手順:

  source env_train.sh
  BS=16 STEPS=30 bash scripts/probe_pi05_bs.sh    # VRAM の確認
  bash scripts/full_pi05_lora.sh                  # 学習

学習用 venv: $VENV
評価・提出前チェックはルートの setup.sh + env.sh 側で行うこと。
EOF
