# 配布環境用 Dockerfile（本番採点環境と同一の GPU ベース構成）
#
#   docker build -t parc2026 .
#
#   # 対話シェル（中で README の自己評価コマンドをそのまま使える）
#   docker run -it --rm --gpus all parc2026
#
#   # 提出 zip をエンドツーエンド評価
#   docker run --rm --gpus all -v $PWD/my_submission.zip:/sub.zip parc2026 \
#       python evaluate.py /sub.zip --n-episodes 2
#
# 実行要件: NVIDIA GPU（ドライバ R580 系以降）+ nvidia-container-toolkit。
# 環境構築は setup.sh（ローカル構築と同一手順）をビルド時に実行する。
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3.10-dev \
    libosmesa6 libosmesa6-dev \
    libgl1 libglfw3 libglew-dev \
    libegl1 \
    libsm6 libxext6 libxrender-dev \
    libglib2.0-0 \
    libmagickwand-dev \
    build-essential cmake git wget curl zip unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# venv 作成・依存・LIBERO-plus 取得とパッチ・アセット・config.yaml まで一式
RUN bash setup.sh

# lerobot をあらかじめ用意する。参加者が requirements.txt に lerobot を書いた場合、
# 提出物 venv には lerobot 0.4.4 の制約により CUDA 12 版の torch と numpy 2.x が入る。
# ベース環境（torch 2.11.0+cu130 / numpy 1.26.4）を壊さないよう、専用の venv に分けている。
# wheelhouse は提出物 venv の構築にも使われ（PIP_FIND_LINKS）、インストール時間と
# 採点中のネットワーク障害リスクを下げる。
RUN python3 -m pip wheel --no-cache-dir -w /opt/wheelhouse \
        lerobot==0.4.4 transformers==4.53.2 \
 && python3 -m venv /opt/lerobot-venv \
 && /opt/lerobot-venv/bin/pip install --no-cache-dir --no-index --find-links /opt/wheelhouse \
        lerobot==0.4.4 transformers==4.53.2
ENV PIP_FIND_LINKS=/opt/wheelhouse

ENV PATH="/workspace/venv/bin:${PATH}"
ENV PYTHONPATH="/workspace/LIBERO-plus:/workspace:/workspace/compe"
ENV LIBERO_ROOT="/workspace/LIBERO-plus"

CMD ["bash"]
