# 学習ランチャーが source する共通ヘルパー。
#
# 提供する関数と、出力先のディレクトリ規約:
#   _resolve_paths       $RUN_NAME から OUT_DIR / LOG_FILE / VRAM_FILE を決める
#   _activate_venv       パッチ適用済み lerobot の venv を有効化する
#   _wandb_args          WANDB_API_KEY があれば W&B を有効にする引数を組み立てる
#   _maybe_resume        $OUT_DIR/checkpoints/last から resume、無ければ出力先を wipe
#   _start_vram_sampler  nvidia-smi のバックグラウンドサンプラ + EXIT トラップ
#   _summarize_run START END   ウォールタイムとピーク VRAM を出力する
#
# 呼び出し側は _resolve_paths の前に RUN_NAME を設定すること。
# OUT_ROOT / LOG_ROOT / PI05_VENV は env で上書きできる。

: "${OUT_ROOT:=${HOME}/pi05-ft-outputs}"
: "${LOG_ROOT:=${HOME}/pi05-ft-logs}"
# パッチ適用済み lerobot の clone 先（README の手順に対応）。
: "${LEROBOT_ROOT:=${HOME}/pi05-ft/lerobot}"
: "${PI05_VENV:=${LEROBOT_ROOT}/.venv}"

_resolve_paths() {
    OUT_DIR="$OUT_ROOT/$RUN_NAME"
    LOG_FILE="$LOG_ROOT/$RUN_NAME.log"
    VRAM_FILE="$LOG_ROOT/$RUN_NAME.vram.csv"
    mkdir -p "$LOG_ROOT" "$OUT_ROOT"
}

_activate_venv() {
    if [[ ! -f "$PI05_VENV/bin/activate" ]]; then
        echo "学習用 venv が見つからない: $PI05_VENV" >&2
        echo "scripts/setup_train.sh を実行するか、PI05_VENV を設定すること。" >&2
        echo "（評価用 venv とは別物である。ルートの env.sh を source した" >&2
        echo "  シェルでは学習を回せない。）" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$PI05_VENV/bin/activate"
}

# W&B は既定で無効。WANDB_API_KEY があるときだけ有効化する。
# プロジェクト名は WANDB_PROJECT で変更できる。
_wandb_args() {
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        WANDB_ARGS=(--wandb.enable=true --wandb.project="${WANDB_PROJECT:-pi05-libero-ft}")
    else
        WANDB_ARGS=(--wandb.enable=false)
    fi
}

# pi0.5 のトークナイザ/重みは gated repo（google/paligemma-3b-pt-224）を引く。
# 事前に HF でライセンスに同意し、その権限を持つトークンを渡すこと。
_require_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN が未設定。pi0.5 は gated repo の" >&2
        echo "google/paligemma-3b-pt-224 を取得するため必須である。" >&2
        echo "  https://huggingface.co/google/paligemma-3b-pt-224 で利用条件に同意し、" >&2
        echo "  export HF_TOKEN=hf_... を設定してから再実行すること。" >&2
        exit 1
    fi
    export HF_TOKEN
}

_maybe_resume() {
    RESUME_ARGS=""
    if [[ -d "$OUT_DIR/checkpoints/last/pretrained_model" ]]; then
        echo "==> $OUT_DIR/checkpoints/last から resume する"
        RESUME_ARGS="--resume=true --config_path=$OUT_DIR/checkpoints/last/pretrained_model/train_config.json"
    else
        echo "==> 新規実行のため、既存の出力ディレクトリを削除する"
        rm -rf "$OUT_DIR"
    fi
}

_SAMPLER_PID=""
_sampler_cleanup() {
    [[ -n "$_SAMPLER_PID" ]] && kill "$_SAMPLER_PID" 2>/dev/null || true
}

_start_vram_sampler() {
    trap _sampler_cleanup EXIT
    (
        echo "ts,used_MiB,util_%" > "$VRAM_FILE"
        while true; do
            read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
            echo "$(date +%s),${used},${util}" >> "$VRAM_FILE"
            sleep 5
        done
    ) &
    _SAMPLER_PID=$!
}

_summarize_run() {
    local start=$1 end=$2
    echo
    echo "=== $RUN_NAME :: total_wall_sec=$((end - start)) ==="
    local peak
    peak=$(awk -F',' 'NR>1 {if ($2+0 > max) max=$2+0} END {print max+0}' "$VRAM_FILE")
    echo "=== peak VRAM=$peak MiB ==="
}
