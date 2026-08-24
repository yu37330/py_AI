#!/usr/bin/env bash
# pi0.5 の LoRA 追加学習。gemma_expert の q/v + action projection（lerobot の
# PI05 既定の target_modules）に LoRA アダプタを学習し、それ以外は凍結する。
# 学習率は full-FT より高めに取る（LoRA では一般的）。
# 実効バッチサイズ = PI05_BS × PI05_GA。resume に対応する
# （$OUT_DIR/checkpoints/last があればそこから再開する）。
#
# 使い方:
#   bash scripts/full_pi05_lora.sh
#
# env で上書きできる主な変数:
#   PI05_BS / PI05_GA / PI05_STEPS / PI05_SAVE_STEPS / PI05_LORA_R / PI05_LR / RUN_NAME
#   PI05_DATASET_REPO_ID / PI05_DATASET_ROOT  学習データセット
#   PI05_VENV / OUT_ROOT / LOG_ROOT           環境と出力先（_train_common.sh 参照）
#   WANDB_API_KEY                             設定時のみ W&B に記録する
#   HF_TOKEN                                  gated な重みを使う場合のみ必要

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./_train_common.sh
source "$HERE/_train_common.sh"

PI05_BS="${PI05_BS:-16}"
PI05_GA="${PI05_GA:-8}"
PI05_STEPS="${PI05_STEPS:-20000}"
PI05_SAVE_STEPS="${PI05_SAVE_STEPS:-1000}"
PI05_LORA_R="${PI05_LORA_R:-16}"
PI05_LR="${PI05_LR:-5e-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
RUN_NAME="${RUN_NAME:-pi05_lora_r${PI05_LORA_R}_bs${PI05_BS}x${PI05_GA}_${PI05_STEPS}}"

# 学習データセット。既定は README の「2. データセット」で用意するもの。
PI05_DATASET_REPO_ID="${PI05_DATASET_REPO_ID:-local/libero_combined_20hz}"
PI05_DATASET_ROOT="${PI05_DATASET_ROOT:-${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}/libero_combined_20hz}"

# 動画デコーダ。torchcodec は FFmpeg の共有ライブラリ（libavutil 等）を必要とし、
# 入っていないと DataLoader が落ちる。scripts/setup_train.sh がその ffmpeg を
# 入れるので、既定は速い torchcodec にしてある。入れられなかった環境（setup の
# 動作確認で「torchcodec: 使えない」と出る）では PI05_VIDEO_BACKEND=pyav に落と
# すこと。pyav は自前で FFmpeg を同梱するため追加の system 依存が要らない。
PI05_VIDEO_BACKEND="${PI05_VIDEO_BACKEND:-torchcodec}"

_resolve_paths
_maybe_resume
_require_hf_token
_activate_venv
_wandb_args

export PYTHONUNBUFFERED=1
export LEROBOT_GRAD_ACCUM=$PI05_GA   # grad-accum-env-var.patch が読む

DECAY_LR=$(awk "BEGIN { printf \"%g\", $PI05_LR / 10 }")

_start_vram_sampler

echo "=== $RUN_NAME :: bs=$PI05_BS ga=$PI05_GA eff=$((PI05_BS * PI05_GA)) lora_r=$PI05_LORA_R lr=$PI05_LR steps=$PI05_STEPS ==="
echo "=== started at $(date -u +%FT%TZ) ==="
START=$(date +%s)

lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_libero_base \
    --policy.dtype=bfloat16 \
    --policy.n_obs_steps=1 \
    --policy.n_action_steps=10 \
    --policy.optimizer_lr=$PI05_LR \
    --policy.scheduler_decay_lr=$DECAY_LR \
    --policy.scheduler_decay_steps=$((PI05_STEPS * 3 / 2)) \
    --policy.scheduler_warmup_steps=1000 \
    --policy.num_inference_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --peft.method_type=LORA \
    --peft.r=$PI05_LORA_R \
    --dataset.repo_id="$PI05_DATASET_REPO_ID" \
    --dataset.root="$PI05_DATASET_ROOT" \
    --dataset.video_backend="$PI05_VIDEO_BACKEND" \
    --batch_size=$PI05_BS \
    --steps=$PI05_STEPS \
    --num_workers=$DATALOADER_NUM_WORKERS \
    --save_freq=$PI05_SAVE_STEPS \
    --log_freq=100 \
    --output_dir="$OUT_DIR" \
    --job_name="$RUN_NAME" \
    --seed=1000 \
    "${WANDB_ARGS[@]}" \
    $RESUME_ARGS \
    2>&1 | tee "$LOG_FILE"

_summarize_run "$START" "$(date +%s)"
