#!/usr/bin/env bash
# Long pi0.5 LoRA training on one group-aware dataset manifest.
#
# Required:
#   FULL_TRAIN_MANIFEST=/path/V1_MULTI_FLAG_PRUNED_EXPERIMENTAL.json
#   PI05_DATASET_ROOT=/path/to/local/LeRobot-dataset
#
# Optional:
#   PI05_BS=8 PI05_GA=16 PI05_STEPS=20000 PI05_SAVE_STEPS=500
#   PI05_DATASET_REPO_ID=lerobot/libero_plus PI05_VIDEO_BACKEND=pyav
#   PI05_LORA_R=16 PI05_LR=5e-5 RUN_NAME=...
#   OUT_ROOT / LOG_ROOT / PI05_VENV / WANDB_API_KEY / HF_TOKEN
#
# The manifest episode_ids are passed through LeRobot v0.4.4 DatasetConfig.episodes,
# preserving the exact group-aware training recipe selected by simulator evidence.
# Resume is supported through _train_common.sh when checkpoints/last exists.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./_train_common.sh
source "$HERE/_train_common.sh"

: "${FULL_TRAIN_MANIFEST:?set FULL_TRAIN_MANIFEST to a group-aware training manifest JSON}"
: "${PI05_DATASET_ROOT:?set PI05_DATASET_ROOT to the local LeRobot dataset}"

PI05_BS="${PI05_BS:-8}"
PI05_GA="${PI05_GA:-16}"
PI05_STEPS="${PI05_STEPS:-20000}"
PI05_SAVE_STEPS="${PI05_SAVE_STEPS:-500}"
PI05_LORA_R="${PI05_LORA_R:-16}"
PI05_LR="${PI05_LR:-5e-5}"
PI05_DATASET_REPO_ID="${PI05_DATASET_REPO_ID:-lerobot/libero_plus}"
PI05_VIDEO_BACKEND="${PI05_VIDEO_BACKEND:-pyav}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
SEED="${PI05_SEED:-1000}"

readarray -t META < <(python - "$FULL_TRAIN_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
ids = m.get("episode_ids")
if not isinstance(ids, list) or not ids:
    raise SystemExit("manifest must contain non-empty episode_ids")
if m.get("schema_version") != 2 or m.get("group_aware") is not True:
    raise SystemExit("manifest must be schema_version=2 and group_aware=true")
print(m.get("variant", "unknown"))
print(json.dumps([int(x) for x in ids], separators=(",", ":")))
print(m.get("episode_ids_sha256", ""))
print(m.get("dataset_id", ""))
print(m.get("summary", {}).get("episode_count", len(ids)))
print(m.get("summary", {}).get("frame_count", ""))
PY
)
VARIANT="${META[0]}"
EPISODES_JSON="${META[1]}"
EPISODES_SHA="${META[2]}"
MANIFEST_DATASET_ID="${META[3]}"
EPISODE_COUNT="${META[4]}"
FRAME_COUNT="${META[5]}"

if [[ -n "$MANIFEST_DATASET_ID" && "$MANIFEST_DATASET_ID" != "$PI05_DATASET_REPO_ID" ]]; then
    echo "manifest dataset_id=$MANIFEST_DATASET_ID does not match PI05_DATASET_REPO_ID=$PI05_DATASET_REPO_ID" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-pi05_full_${VARIANT,,}_r${PI05_LORA_R}_bs${PI05_BS}x${PI05_GA}_${PI05_STEPS}}"

_resolve_paths
_maybe_resume
_require_hf_token
_activate_venv
_wandb_args

export PYTHONUNBUFFERED=1
export LEROBOT_GRAD_ACCUM="$PI05_GA"

DECAY_LR=$(awk "BEGIN { printf \"%g\", $PI05_LR / 10 }")
DATASET_ARGS=(
    "--dataset.repo_id=$PI05_DATASET_REPO_ID"
    "--dataset.root=$PI05_DATASET_ROOT"
    "--dataset.video_backend=$PI05_VIDEO_BACKEND"
    "--dataset.episodes=$EPISODES_JSON"
)

_start_vram_sampler

echo "=== full manifest train :: variant=$VARIANT episodes=$EPISODE_COUNT frames=$FRAME_COUNT ==="
echo "=== bs=$PI05_BS ga=$PI05_GA eff=$((PI05_BS * PI05_GA)) lora_r=$PI05_LORA_R lr=$PI05_LR steps=$PI05_STEPS save=$PI05_SAVE_STEPS seed=$SEED ==="
echo "=== manifest=$FULL_TRAIN_MANIFEST episodes_sha256=$EPISODES_SHA ==="
echo "=== started at $(date -u +%FT%TZ) ==="
START=$(date +%s)

lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_libero_base \
    --policy.dtype=bfloat16 \
    --policy.n_obs_steps=1 \
    --policy.n_action_steps=10 \
    --policy.optimizer_lr="$PI05_LR" \
    --policy.scheduler_decay_lr="$DECAY_LR" \
    --policy.scheduler_decay_steps=$((PI05_STEPS * 3 / 2)) \
    --policy.scheduler_warmup_steps=1000 \
    --policy.num_inference_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --peft.method_type=LORA \
    --peft.r="$PI05_LORA_R" \
    "${DATASET_ARGS[@]}" \
    --batch_size="$PI05_BS" \
    --steps="$PI05_STEPS" \
    --num_workers="$DATALOADER_NUM_WORKERS" \
    --save_freq="$PI05_SAVE_STEPS" \
    --eval_freq=999999999 \
    --log_freq=100 \
    --output_dir="$OUT_DIR" \
    --job_name="$RUN_NAME" \
    --seed="$SEED" \
    "${WANDB_ARGS[@]}" \
    $RESUME_ARGS \
    2>&1 | tee "$LOG_FILE"

END=$(date +%s)
_summarize_run "$START" "$END"

CKPT="$OUT_DIR/checkpoints/last/pretrained_model"
[[ -d "$CKPT" ]] || { echo "NG: checkpoint missing: $CKPT" >&2; exit 1; }

SUMMARY="$OUT_DIR/full_train_summary.json"
python - "$SUMMARY" "$FULL_TRAIN_MANIFEST" "$START" "$END" "$PI05_BS" "$PI05_GA" "$PI05_STEPS" "$PI05_SAVE_STEPS" "$PI05_LORA_R" "$PI05_LR" "$SEED" "$VRAM_FILE" <<'PY'
import json, sys
from pathlib import Path
(
    out, manifest, start, end, bs, ga, steps, save_steps,
    lora_r, lr, seed, vram_path
) = sys.argv[1:]
m = json.loads(Path(manifest).read_text())
peak = 0
for line in Path(vram_path).read_text().splitlines()[1:]:
    try:
        peak = max(peak, int(float(line.split(',')[1])))
    except Exception:
        pass
result = {
    "variant": m.get("variant"),
    "manifest": str(Path(manifest).resolve()),
    "dataset_id": m.get("dataset_id"),
    "dataset_revision_provenance": m.get("dataset_revision"),
    "episode_ids_sha256": m.get("episode_ids_sha256"),
    "episode_count": m.get("summary", {}).get("episode_count"),
    "frame_count": m.get("summary", {}).get("frame_count"),
    "batch_size": int(bs),
    "grad_accum": int(ga),
    "effective_batch": int(bs) * int(ga),
    "optimizer_steps": int(steps),
    "save_steps": int(save_steps),
    "lora_r": int(lora_r),
    "learning_rate": float(lr),
    "seed": int(seed),
    "wall_sec": int(end) - int(start),
    "peak_vram_mib": peak,
    "screening_only": False,
}
Path(out).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "OK: $SUMMARY"
