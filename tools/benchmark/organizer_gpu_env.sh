#!/usr/bin/env bash
# PARC2026 organizer GPU canonical environment.
#
# Usage:
#   source tools/benchmark/organizer_gpu_env.sh
#
# This file never stores secrets. Set HF_TOKEN in the current shell separately.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file instead of executing it:" >&2
  echo "  source tools/benchmark/organizer_gpu_env.sh" >&2
  exit 2
fi

export PY_AI_REPO="${PY_AI_REPO:-$HOME/src/py_AI}"
export PARC_LOCAL_SCRATCH_ROOT="${PARC_LOCAL_SCRATCH_ROOT:-/opt/dlami/nvme/parc2026}"
export PARC_PERSIST_ROOT="${PARC_PERSIST_ROOT:-$HOME/data/parc2026-cache}"
export PARC_DATASET_ROOT="${PARC_DATASET_ROOT:-$PARC_LOCAL_SCRATCH_ROOT/datasets/lerobot_libero_plus_v3_train}"

# Legacy aliases used by the canonical M3 code. On organizer hardware these
# deliberately point to scratch/persistent storage rather than Colab/Drive.
export PARC_ROOT="$PARC_LOCAL_SCRATCH_ROOT"
export PARC_DRIVE_ROOT="$PARC_PERSIST_ROOT"
export PARC_DRIVE_DATASET="$PARC_DATASET_ROOT"

export PARC_M3_EXECUTE="${PARC_M3_EXECUTE:-1}"
export PARC_M3_HARDWARE_PROFILE="${PARC_M3_HARDWARE_PROFILE:-organizer_rtx_pro_6000_blackwell}"
export PARC_M3_SIM_SMOKE_ATTEMPT="${PARC_M3_SIM_SMOKE_ATTEMPT:-organizer-1}"
export PARC_M3_TRAINING_COMPAT_ATTEMPT="${PARC_M3_TRAINING_COMPAT_ATTEMPT:-organizer-1}"
export PARC_M3_BENCHMARK_ATTEMPT="${PARC_M3_BENCHMARK_ATTEMPT:-organizer-1}"

# All regenerable caches stay on NVMe.
export HF_HOME="${HF_HOME:-$PARC_LOCAL_SCRATCH_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PARC_LOCAL_SCRATCH_ROOT/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PARC_LOCAL_SCRATCH_ROOT/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PARC_LOCAL_SCRATCH_ROOT/cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PARC_LOCAL_SCRATCH_ROOT/cache/pip}"
export TMPDIR="${TMPDIR:-$PARC_LOCAL_SCRATCH_ROOT/tmp}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

mkdir -p \
  "$PARC_LOCAL_SCRATCH_ROOT" \
  "$PARC_PERSIST_ROOT" \
  "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

if git -C "$PY_AI_REPO" rev-parse HEAD >/dev/null 2>&1; then
  export PARC_EXPECTED_SOURCE_SHA="${PARC_EXPECTED_SOURCE_SHA:-$(git -C "$PY_AI_REPO" rev-parse HEAD)}"
fi

printf '%s\n' \
  "PYSICALAI organizer environment loaded" \
  "  repo:       $PY_AI_REPO" \
  "  scratch:    $PARC_LOCAL_SCRATCH_ROOT" \
  "  persistent: $PARC_PERSIST_ROOT" \
  "  dataset:    $PARC_DATASET_ROOT" \
  "  profile:    $PARC_M3_HARDWARE_PROFILE" \
  "  attempt:    $PARC_M3_BENCHMARK_ATTEMPT"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Export it in this terminal before 74/dataset/runtime setup." >&2
else
  echo "  HF_TOKEN:   set (value hidden)"
fi
