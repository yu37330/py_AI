#!/usr/bin/env bash
# 配布環境の環境構築（本番採点イメージと同じ手順をローカルに再現する）
#
#   bash setup.sh          # このディレクトリ（配布環境のルート）で実行
#   source env.sh          # 以後、評価を回すシェルで毎回 source する
#
# 前提: NVIDIA GPU + ドライバ R580 系以降（本番採点環境と同一の CUDA 13 / GPU 前提。
#       CPU のみの環境はサポートしない）
#
# やること:
#   1. システム依存ライブラリの導入（ImageMagick / GL 系。pip では入らない）
#   2. venv 作成 + 依存インストール（本番と同じピン止めバージョン・CUDA 版 torch）
#   3. LIBERO-plus（評価環境）と LIBERO（base assets）の取得
#   4. LIBERO-plus への既知パッチ（__init__.py 追加 / torch.load weights_only）
#   5. タスクアセットのダウンロードと配線（HF assets.zip / Track 3 symlink）
#   6. ~/.libero/config.yaml の生成（既存があれば .bak に退避）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"

PY="${PYTHON:-python3.10}"

echo "[setup] 1/6 システム依存ライブラリ"
# LIBERO-plus の envs/env_wrapper.py が import する wand は ImageMagick の共有ライブラリ
# （libMagickWand）への ctypes バインディングでしかなく、pip install wand だけでは動かない。
# mujoco / robosuite の GL 系も同様に apt でしか入らないため、Dockerfile と同じ一式を導入する。
if ldconfig -p 2>/dev/null | grep -q libMagickWand; then
    echo "[setup]   導入済みのためスキップ"
elif [ "$(id -u)" = 0 ] || command -v sudo >/dev/null 2>&1; then
    SUDO=""
    [ "$(id -u)" = 0 ] || SUDO="sudo"
    $SUDO apt-get update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        libmagickwand-dev \
        python3.10-dev \
        libosmesa6 libosmesa6-dev \
        libgl1 libglfw3 libglew-dev \
        libegl1 \
        libsm6 libxext6 libxrender-dev \
        libglib2.0-0
else
    echo "[setup] ERROR: ImageMagick の共有ライブラリが無く、apt を実行する権限もありません。" >&2
    echo "[setup]        root で次を実行してから再試行してください:" >&2
    echo "[setup]          apt-get install -y libmagickwand-dev python3.10-dev libosmesa6 libosmesa6-dev \\" >&2
    echo "[setup]            libgl1 libglfw3 libglew-dev libegl1 libsm6 libxext6 libxrender-dev libglib2.0-0" >&2
    exit 1
fi

echo "[setup] 2/6 venv + 依存"
if [ ! -d venv ]; then
    "$PY" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
# 本番採点環境と同じ CUDA 13 版 torch（GPU 前提）
pip install -q --timeout 120 --retries 10 \
    "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple
pip install -q --timeout 120 \
    mujoco==3.7.0 robosuite==1.4.0 numpy==1.26.4 "gym==0.25.2" bddl==3.6.0 \
    cloudpickle==3.1.2 easydict==1.13 hydra-core==1.3.2 einops==0.8.2 \
    opencv-python-headless==4.11.0.86 \
    scipy pyyaml h5py Pillow termcolor tqdm matplotlib \
    requests msgpack fastapi uvicorn huggingface_hub wand scikit-image pytest

# 順序事故の検出。numpy が 1.26.4 でないと LIBERO / robosuite が想定外の
# dtype 昇格規則で動くことになるため、ここで止める。
python - <<'PY'
import numpy, sys, glob, os
import torch
# cu12 と cu13 の wheel が同じパス（nvidia/*/lib）へ上書きされると、torch が読む
# cuDNN が CUDA 12 用に化けて GPU 推論が壊れる。両系統の同居をここで検出する。
import importlib.metadata as _md
_names = {d.metadata["Name"].lower() for d in _md.distributions()}
_cu12 = {n for n in _names if n.startswith("nvidia-") and n.endswith("-cu12")}
_cu13 = {n for n in _names if n.startswith("nvidia-") and n.endswith("-cu13")}
if _cu12 and _cu13:
    sys.exit(f"[setup] ERROR: CUDA 12 系と 13 系の nvidia wheel が同居しています "
             f"（cu12: {sorted(_cu12)[:3]}... / cu13: {sorted(_cu13)[:3]}...）。"
             "cuDNN が上書きされ GPU 推論が壊れます")
if numpy.__version__ != "1.26.4":
    sys.exit(f"[setup] ERROR: numpy が {numpy.__version__} です（1.26.4 であるべき）。"
             "依存のインストール順序を確認してください")
print("[setup]   依存の版を確認: numpy", numpy.__version__,
      "| torch", torch.__version__, "| cudnn", torch.backends.cudnn.version())
PY

echo "[setup] 3/6 LIBERO-plus / LIBERO の取得"
# 本番採点イメージは submodule でこの 2 コミットを固定している。配布側もコミットを
# 指すことで、いつ・どこでセットアップしても採点環境と同じ版の LIBERO になる。
# （上流のブランチ先端を取ると、取得時期によって中身が変わりうる）
LIBERO_PLUS_REF="${LIBERO_PLUS_REF:-4976dc30028e805ff8094b55501d532c48fec182}"
LIBERO_REF="${LIBERO_REF:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"

# コミット指定のまま depth 1 で取得する（clone --depth 1 は ref しか受け付けない）
clone_pinned() {  # <dir> <url> <commit>
    [ -d "$1" ] && return 0
    git init -q "$1"
    git -C "$1" remote add origin "$2"
    git -C "$1" fetch -q --depth 1 origin "$3"
    git -C "$1" checkout -q FETCH_HEAD
}
clone_pinned LIBERO-plus https://github.com/sylvestf/LIBERO-plus "$LIBERO_PLUS_REF"
clone_pinned LIBERO https://github.com/Lifelong-Robot-Learning/LIBERO "$LIBERO_REF"

echo "[setup] 4/6 LIBERO-plus パッチ"
touch LIBERO-plus/libero/__init__.py LIBERO-plus/libero/libero/__init__.py
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' \
    LIBERO-plus/libero/libero/benchmark/__init__.py || true

echo "[setup] 5/6 アセット"
# アセットも版を固定する（データセットリポジトリの main が更新されうるため）
ASSETS_REV="${ASSETS_REV:-dd2bd61b7d9a6fef1abc52d606e983b41886a149}"
ASSETS="LIBERO-plus/libero/libero/assets"
if [ "$(ls "$ASSETS"/textures 2>/dev/null | wc -l)" -lt 100 ]; then
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Sylvest/LIBERO-plus','assets.zip',repo_type='dataset',revision='$ASSETS_REV',local_dir='.tmp_assets')"
    unzip -q .tmp_assets/assets.zip -d LIBERO-plus/libero/libero
    rm -rf .tmp_assets
    # assets.zip は作者環境の深いネストパスごと展開されることがあるため対応する
    if [ "$(ls "$ASSETS"/textures 2>/dev/null | wc -l)" -lt 100 ]; then
        rm -rf "$ASSETS"
        NESTED="$(find LIBERO-plus/libero/libero -type d -path '*/assets' -name assets | grep -v '^LIBERO-plus/libero/libero/assets$' | head -1)"
        test -n "$NESTED" && ln -sfn "$(realpath "$NESTED")" "$ASSETS"
    fi
fi
test -e "$ASSETS/scenes/libero_floor_base_style.xml"
echo "[setup]   textures=$(ls "$ASSETS"/textures | wc -l)"

# Track 3（reverse タスク）の bddl / init をベンチマークの参照先へ配線する
ln -sfn "$ROOT/compe/t3/assets/bddl_files/libero_t3" LIBERO-plus/libero/libero/bddl_files/libero_t3
ln -sfn "$ROOT/compe/t3/assets/init_files/libero_t3" LIBERO-plus/libero/libero/init_files/libero_t3

echo "[setup] 6/6 libero 設定"
mkdir -p "$HOME/.libero"
if [ -f "$HOME/.libero/config.yaml" ]; then
    cp "$HOME/.libero/config.yaml" "$HOME/.libero/config.yaml.bak"
    echo "[setup]   既存の ~/.libero/config.yaml を config.yaml.bak に退避しました"
fi
cat > "$HOME/.libero/config.yaml" <<EOF
benchmark_root: $ROOT/LIBERO-plus/libero/libero
bddl_files: $ROOT/LIBERO-plus/libero/libero/bddl_files
init_states: $ROOT/LIBERO-plus/libero/libero/init_files
datasets: $ROOT/LIBERO-plus/libero/libero/datasets
assets: $ROOT/LIBERO/libero/libero/assets
EOF

# env.sh には ~/.libero/config.yaml の生成も入れる。この設定はユーザーのホームに
# 置かれるため、共有環境（AMI へ焼き込む等、setup.sh を別ユーザーで実行する構成）では
# setup.sh 実行時のホームにしか作られない。source した時点で無ければ作る。
cat > env.sh <<EOF
source "$ROOT/venv/bin/activate"
export MUJOCO_GL="\${MUJOCO_GL:-egl}"
export PYTHONPATH="$ROOT/LIBERO-plus:$ROOT:$ROOT/compe"
export LIBERO_ROOT="$ROOT/LIBERO-plus"

if [ ! -f "\$HOME/.libero/config.yaml" ]; then
    mkdir -p "\$HOME/.libero"
    cat > "\$HOME/.libero/config.yaml" <<'LIBERO_CFG'
benchmark_root: $ROOT/LIBERO-plus/libero/libero
bddl_files: $ROOT/LIBERO-plus/libero/libero/bddl_files
init_states: $ROOT/LIBERO-plus/libero/libero/init_files
datasets: $ROOT/LIBERO-plus/libero/libero/datasets
assets: $ROOT/LIBERO/libero/libero/assets
LIBERO_CFG
    echo "[env] ~/.libero/config.yaml を生成しました"
fi
EOF

echo "[setup] 動作確認（suite 登録）"
PYTHONPATH="$ROOT/LIBERO-plus:$ROOT:$ROOT/compe" \
    python -c "import libero.libero.benchmark; from compe.t1 import register_t1; from compe.t2 import register_t2; from compe.t3 import register_t3; register_t1(); register_t2(); register_t3(); print('suite 登録 OK')"

echo
echo "セットアップ完了。評価を回すシェルで次を実行してください:"
echo "  source env.sh"
