#!/usr/bin/env bash
# バッチサイズ探索用の VRAM プローブ。少数ステップだけ学習を回して
# ピーク VRAM とスループットを測る。手持ちの GPU に載る PI05_BS を
# 決めてから full_pi05_lora.sh を回すこと。
#
# full_pi05_lora.sh と同じ LoRA 条件で測る。ここで PEFT を付け忘れると
# 4B 全体を学習する full-FT になり、VRAM probe が本学習条件を反映しない。
#
# 使い方:
#   BS=16 STEPS=30 bash scripts/probe_pi05_bs.sh
#   BS=32 STEPS=30 bash scripts/probe_pi05_bs.sh
#
# LORA_R=16 で LoRA rank を変更できる。
# GC=1 を付けると gradient checkpointing を有効にする（VRAM は減るが遅くなる）。

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./_train_common.sh
source "$HERE/_train_common.sh"

BS=${BS:-16}
STEPS=${STEPS:-30}
LORA_R=${LORA_R:-16}
TAG=${TAG:-bs${BS}_r${LORA_R}}
RUN_NAME=probe_pi05_${TAG}

PI05_DATASET_REPO_ID="${PI05_DATASET_REPO_ID:-local/libero_combined_20hz}"
PI05_DATASET_ROOT="${PI05_DATASET_ROOT:-${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}/libero_combined_20hz}"

# 動画デコーダ。torchcodec は FFmpeg の共有ライブラリ（libavutil 等）を必要とし、
# 入っていないと DataLoader が落ちる。scripts/setup_train.sh がその ffmpeg を
# 入れるので、既定は速い torchcodec にしてある。入れられなかった環境（setup の
# 動作確認で「torchcodec: 使えない」と出る）では PI05_VIDEO_BACKEND=pyav に落と
# すこと。pyav は自前で FFmpeg を同梱するため追加の system 依存が要らない。
PI05_VIDEO_BACKEND="${PI05_VIDEO_BACKEND:-torchcodec}"

_resolve_paths
rm -rf "$OUT_DIR"
_require_hf_token
_activate_venv

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

_start_vram_sampler

echo "=== probe pi05 LoRA BS=$BS R=$LORA_R STEPS=$STEPS ==="
START=$(date +%s)

# ハイパーパラメータは full_pi05_lora.sh に合わせ、ステップ数だけ短くする。
lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_libero_base \
    --policy.dtype=bfloat16 \
    --policy.n_obs_steps=1 \
    --policy.n_action_steps=10 \
    --policy.optimizer_lr=5e-5 \
    --policy.scheduler_decay_lr=5e-6 \
    --policy.scheduler_decay_steps=30000 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.num_inference_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --peft.method_type=LORA \
    --peft.r=$LORA_R \
    ${GC:+--policy.gradient_checkpointing=true} \
    --dataset.repo_id="$PI05_DATASET_REPO_ID" \
    --dataset.root="$PI05_DATASET_ROOT" \
    --dataset.video_backend="$PI05_VIDEO_BACKEND" \
    --batch_size=$BS \
    --steps=$STEPS \
    --num_workers=8 \
    --save_freq=99999 \
    --log_freq=10 \
    --output_dir="$OUT_DIR" \
    --job_name="$RUN_NAME" \
    --seed=1000 \
    --wandb.enable=false \
    2>&1 | tee "$LOG_FILE" || true

_summarize_run "$START" "$(date +%s)"
RATE=$(grep -aoE "[0-9.]+(step/s|s/step|it/s|s/it)" "$LOG_FILE" | tail -1 || true)
echo "=== rate=$RATE ==="
