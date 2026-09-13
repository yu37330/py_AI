#!/usr/bin/env bash
# Operator CLI for PARC2026 organizer GPU execution.
# Intended to be run from a JupyterLab Terminal, not from notebook cells.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/organizer_gpu_env.sh"

LOG_ROOT="$PARC_PERSIST_ROOT/operator-logs"
mkdir -p "$LOG_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash tools/benchmark/organizer_gpu_cli.sh status
  bash tools/benchmark/organizer_gpu_cli.sh pull
  bash tools/benchmark/organizer_gpu_cli.sh preflight
  bash tools/benchmark/organizer_gpu_cli.sh restore
  bash tools/benchmark/organizer_gpu_cli.sh 74
  bash tools/benchmark/organizer_gpu_cli.sh dataset
  bash tools/benchmark/organizer_gpu_cli.sh compat
  bash tools/benchmark/organizer_gpu_cli.sh 75-forward
  bash tools/benchmark/organizer_gpu_cli.sh 75-reverse
  bash tools/benchmark/organizer_gpu_cli.sh 76-unit <forward|reverse> <equal_data|equal_wall> <pi05|smolvla|openvla_oft> <unit-attempt>
  bash tools/benchmark/organizer_gpu_cli.sh 76-finalize <forward|reverse>
  bash tools/benchmark/organizer_gpu_cli.sh 77
  bash tools/benchmark/organizer_gpu_cli.sh push
  bash tools/benchmark/organizer_gpu_cli.sh stop-ready

Default handoff input:
  $HOME/data/incoming/m3-organizer-handoff-v1.tar
  $HOME/data/incoming/m3-organizer-handoff-v1.tar.sha256.json

Override with:
  PARC_M3_HANDOFF_ARCHIVE=/path/to/archive.tar
  PARC_M3_HANDOFF_SIDECAR=/path/to/archive.tar.sha256.json
EOF
}

stamp() { date '+%Y%m%d_%H%M%S'; }

run_logged() {
  local name="$1"; shift
  local log="$LOG_ROOT/$(stamp)_${name}.log"
  echo "=== $name ==="
  echo "log: $log"
  set +e
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAILED rc=$rc; log preserved: $log" >&2
    return "$rc"
  fi
  echo "PASS: $name"
}

require_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set in this terminal." >&2
    echo "Run: export HF_TOKEN='hf_...'; then retry." >&2
    return 2
  fi
}

check_repo_identity() {
  local got
  got="$(git -C "$PY_AI_REPO" rev-parse HEAD)"
  if [[ -n "${PARC_EXPECTED_SOURCE_SHA:-}" && "$got" != "$PARC_EXPECTED_SOURCE_SHA" ]]; then
    echo "ERROR: repo SHA drift: $got != $PARC_EXPECTED_SOURCE_SHA" >&2
    return 2
  fi
  echo "repo SHA: $got"
}

hardware_guard() {
  PYTHONPATH="$PY_AI_REPO${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from tools.benchmark.m3_batch_probe_common import gpu_info
from tools.benchmark.m3_hardware_guard import validate_hardware
name, vram = gpu_info()
profile = validate_hardware(name, vram)
print(f"hardware PASS: profile={profile.name} gpu={name} vram_mib={vram}")
PY
}

status() {
  echo "=== PARC2026 ORGANIZER GPU STATUS ==="
  date -Is || true
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    nvidia-smi || true
  else
    echo "nvidia-smi: NOT FOUND"
  fi
  echo
  df -h "$HOME" "$HOME/data" /opt/dlami/nvme 2>/dev/null || true
  df -i "$HOME" "$HOME/data" /opt/dlami/nvme 2>/dev/null || true
  echo
  printf 'python: '; python --version 2>&1 || true
  printf 'git: '; git --version 2>&1 || true
  printf 'uv: '; uv --version 2>&1 || echo 'not installed yet'
  printf 'parc-home-sync: '; command -v parc-home-sync || echo 'NOT FOUND'
  printf 'tmux: '; command -v tmux || echo 'not installed (optional)'
  printf 'HF_TOKEN: '; [[ -n "${HF_TOKEN:-}" ]] && echo 'set (hidden)' || echo 'NOT SET'
  printf 'repo: '; git -C "$PY_AI_REPO" rev-parse HEAD 2>/dev/null || echo 'not available'
  echo "expected source: ${PARC_EXPECTED_SOURCE_SHA:-not-set}"
  echo "scratch: $PARC_LOCAL_SCRATCH_ROOT"
  echo "persistent: $PARC_PERSIST_ROOT"
  echo "dataset: $PARC_DATASET_ROOT"
  echo "attempt: $PARC_M3_BENCHMARK_ATTEMPT"
}

pull_data() {
  command -v parc-home-sync >/dev/null 2>&1 || { echo "ERROR: parc-home-sync not found" >&2; return 2; }
  run_logged data_pull parc-home-sync data-pull
}

preflight() {
  require_hf_token
  check_repo_identity
  status
  hardware_guard
  mkdir -p "$HOME/data/incoming" "$PARC_PERSIST_ROOT"
  run_logged disk_headroom python -u "$PY_AI_REPO/tools/colab/prepare_m3_disk_headroom.py" --phase pre-setup
  echo "=== OPERATOR PREFLIGHT: PASS ==="
}

restore_handoff() {
  local archive="${PARC_M3_HANDOFF_ARCHIVE:-$HOME/data/incoming/m3-organizer-handoff-v1.tar}"
  local sidecar="${PARC_M3_HANDOFF_SIDECAR:-$HOME/data/incoming/m3-organizer-handoff-v1.tar.sha256.json}"
  [[ -f "$archive" ]] || { echo "ERROR: handoff archive missing: $archive" >&2; return 2; }
  [[ -f "$sidecar" ]] || { echo "ERROR: handoff sidecar missing: $sidecar" >&2; return 2; }
  run_logged handoff_restore python -u "$PY_AI_REPO/tools/benchmark/restore_m3_organizer_handoff.py" \
    --archive "$archive" --sidecar "$sidecar" --root "$PARC_PERSIST_ROOT"
}

run_74() {
  require_hf_token
  check_repo_identity
  hardware_guard
  run_logged m3_74 python -u "$PY_AI_REPO/tools/benchmark/run_m3_organizer_simulator_smoke.py"
}

run_dataset() {
  require_hf_token
  check_repo_identity
  hardware_guard
  run_logged m3_dataset python -u "$PY_AI_REPO/tools/benchmark/prepare_m3_organizer_dataset.py"
}

run_compat() {
  require_hf_token
  check_repo_identity
  hardware_guard
  run_logged m3_blackwell_compat python -u "$PY_AI_REPO/tools/benchmark/run_m3_organizer_training_compat.py"
}

run_75() {
  local order="$1"
  require_hf_token
  check_repo_identity
  hardware_guard
  run_logged "m3_75_${order}" python -u "$PY_AI_REPO/tools/benchmark/run_m3_organizer_benchmark_order.py" \
    --order "$order" --attempt "$PARC_M3_BENCHMARK_ATTEMPT"
}

run_76_unit() {
  local order="${1:-}" track="${2:-}" model="${3:-}" unit_attempt="${4:-}"
  [[ "$order" == "forward" || "$order" == "reverse" ]] || { usage; return 2; }
  [[ "$track" == "equal_data" || "$track" == "equal_wall" ]] || { usage; return 2; }
  [[ "$model" == "pi05" || "$model" == "smolvla" || "$model" == "openvla_oft" ]] || { usage; return 2; }
  [[ -n "$unit_attempt" ]] || { usage; return 2; }
  require_hf_token
  check_repo_identity
  hardware_guard
  run_logged "m3_76_${order}_${track}_${model}_${unit_attempt}" \
    python -u "$PY_AI_REPO/tools/benchmark/run_m3_organizer_screening_unit.py" \
      --order "$order" --track "$track" --model "$model" \
      --attempt "$PARC_M3_BENCHMARK_ATTEMPT" --unit-attempt "$unit_attempt"
}

finalize_76() {
  local order="${1:-}"
  [[ "$order" == "forward" || "$order" == "reverse" ]] || { usage; return 2; }
  check_repo_identity
  run_logged "m3_76_finalize_${order}" \
    python -u "$PY_AI_REPO/tools/benchmark/finalize_m3_organizer_screening_order.py" \
      --order "$order" --attempt "$PARC_M3_BENCHMARK_ATTEMPT"
}

run_77() {
  check_repo_identity
  run_logged m3_77 python -u "$PY_AI_REPO/tools/colab/run_m3_promotion_controller.py" \
    --attempt "$PARC_M3_BENCHMARK_ATTEMPT"
}

push_data() {
  command -v parc-home-sync >/dev/null 2>&1 || { echo "ERROR: parc-home-sync not found" >&2; return 2; }
  run_logged data_push parc-home-sync data-push
}

stop_ready() {
  push_data
  echo
  echo "PERSISTENCE PUSH COMPLETE. Now stop GPU from the JupyterHub UI:"
  echo "  File -> Hub Control Panel -> Stop Server"
  echo "Then confirm the page shows: Start My Server"
  echo "Do NOT only close the browser tab."
}

cmd="${1:-}"
case "$cmd" in
  status) status ;;
  pull) pull_data ;;
  preflight) preflight ;;
  restore) restore_handoff ;;
  74) run_74 ;;
  dataset) run_dataset ;;
  compat) run_compat ;;
  75-forward) run_75 forward ;;
  75-reverse) run_75 reverse ;;
  76-unit) shift; run_76_unit "$@" ;;
  76-finalize) shift; finalize_76 "$@" ;;
  77) run_77 ;;
  push) push_data ;;
  stop-ready) stop_ready ;;
  -h|--help|help|'') usage ;;
  *) echo "ERROR: unknown command: $cmd" >&2; usage; exit 2 ;;
esac
