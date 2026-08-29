#!/usr/bin/env bash
# Run one short pi0.5 LoRA dataset-ablation experiment.
#
# Required:
#   ABLATION_MANIFEST=/path/V0_RAW.json
#   PI05_DATASET_ROOT=/path/to/LeRobot-dataset
#
# Optional:
#   ABLATION_BS=4 ABLATION_GA=8 ABLATION_STEPS=150
#   PI05_DATASET_REPO_ID=lerobot/libero_plus
#   RUN_NAME=ablation_v0_raw
#
# The manifest's episode_ids are passed through LeRobot v0.4.4's native
# DatasetConfig.episodes field. No dataset copy and no trainer sampling patch is
# required. A fixed eval holdout is already excluded by the manifest builder.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./_train_common.sh
source "$HERE/_train_common.sh"

: "${ABLATION_MANIFEST:?set ABLATION_MANIFEST to a training manifest JSON}"
: "${PI05_DATASET_ROOT:?set PI05_DATASET_ROOT to the local LeRobot dataset}"

ABLATION_BS="${ABLATION_BS:-4}"
ABLATION_GA="${ABLATION_GA:-8}"
ABLATION_STEPS="${ABLATION_STEPS:-150}"
PI05_DATASET_REPO_ID="${PI05_DATASET_REPO_ID:-lerobot/libero_plus}"
PI05_VIDEO_BACKEND="${PI05_VIDEO_BACKEND:-pyav}"
SEED="${ABLATION_SEED:-1000}"

readarray -t META < <(python - "$ABLATION_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
ids = m.get("episode_ids")
if not isinstance(ids, list) or not ids:
    raise SystemExit("manifest must contain non-empty episode_ids")
print(m.get("variant", "unknown"))
print(json.dumps([int(x) for x in ids], separators=(",", ":")))
print(m.get("episode_ids_sha256", ""))
PY
)
VARIANT="${META[0]}"
EPISODES_JSON="${META[1]}"
EPISODES_SHA="${META[2]}"
RUN_NAME="${RUN_NAME:-cheap_${VARIANT,,}}"

_resolve_paths
rm -rf "$OUT_DIR"
_require_hf_token
_activate_venv
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export LEROBOT_GRAD_ACCUM="$ABLATION_GA"
_start_vram_sampler

# The source revision is recorded in the manifest for provenance, but the
# trainer consumes an already-materialized local root. Passing a Hub commit SHA
# as DatasetConfig.revision is unnecessary and can trigger remote resolution in
# some LeRobot paths, so it is intentionally omitted here.
DATASET_ARGS=(
    "--dataset.repo_id=$PI05_DATASET_REPO_ID"
    "--dataset.root=$PI05_DATASET_ROOT"
    "--dataset.video_backend=$PI05_VIDEO_BACKEND"
    "--dataset.episodes=$EPISODES_JSON"
)

echo "=== cheap ablation :: variant=$VARIANT bs=$ABLATION_BS ga=$ABLATION_GA eff=$((ABLATION_BS * ABLATION_GA)) steps=$ABLATION_STEPS ==="
echo "=== manifest=$ABLATION_MANIFEST episodes_sha256=$EPISODES_SHA ==="
START=$(date +%s)

lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_libero_base \
    --policy.dtype=bfloat16 \
    --policy.n_obs_steps=1 \
    --policy.n_action_steps=10 \
    --policy.optimizer_lr=5e-5 \
    --policy.num_inference_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --peft.method_type=LORA \
    --peft.r=16 \
    "${DATASET_ARGS[@]}" \
    --batch_size="$ABLATION_BS" \
    --steps="$ABLATION_STEPS" \
    --num_workers=2 \
    --save_freq="$ABLATION_STEPS" \
    --eval_freq=999999999 \
    --log_freq=1 \
    --output_dir="$OUT_DIR" \
    --job_name="$RUN_NAME" \
    --seed="$SEED" \
    --wandb.enable=false \
    2>&1 | tee "$LOG_FILE"

END=$(date +%s)
_summarize_run "$START" "$END"

CKPT="$OUT_DIR/checkpoints/last/pretrained_model"
[[ -d "$CKPT" ]] || { echo "NG: checkpoint missing: $CKPT" >&2; exit 1; }

SUMMARY="$OUT_DIR/cheap_ablation_summary.json"
python - "$LOG_FILE" "$VRAM_FILE" "$SUMMARY" "$VARIANT" "$ABLATION_MANIFEST" "$START" "$END" "$ABLATION_BS" "$ABLATION_GA" "$ABLATION_STEPS" "$SEED" <<'PY'
import json, re, sys
from pathlib import Path
log_path, vram_path, out, variant, manifest, start, end, bs, ga, steps, seed = sys.argv[1:]
text = Path(log_path).read_text(errors="replace")
# Best-effort only: log formats vary across LeRobot versions.
losses = []
for pat in (r"\bloss[=: ]+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", r"\btrain_loss[=: ]+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"):
    losses.extend(float(x) for x in re.findall(pat, text))
peak = 0
for line in Path(vram_path).read_text().splitlines()[1:]:
    try: peak = max(peak, int(float(line.split(',')[1])))
    except Exception: pass
m = json.loads(Path(manifest).read_text())
result = {
    "variant": variant,
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
    "seed": int(seed),
    "wall_sec": int(end) - int(start),
    "peak_vram_mib": peak,
    "final_logged_loss_best_effort": losses[-1] if losses else None,
    "last20_logged_loss_mean_best_effort": sum(losses[-20:]) / len(losses[-20:]) if losses else None,
    "screening_only": True,
    "promotion_note": "Do not promote a dataset variant from training loss alone; require fixed evaluation/simulator evidence.",
}
Path(out).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "OK: $SUMMARY"
