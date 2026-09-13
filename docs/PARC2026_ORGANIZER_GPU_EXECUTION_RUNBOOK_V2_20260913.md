# PARC2026 運営GPU移行・M3実行Runbook v2（2026-09-13）

## 1. 目的

Colab中心で成立させたD10 / M3実行基盤を、PARC2026公式の配布開発GPUへ安全に移し、74 → 75 → 76 → 77までを12時間session制約の中で再開可能に実行する。

本Runbookは旧 `PARC2026_ORGANIZER_GPU_EXECUTION_RUNBOOK_20260913.md` を置き換える。

---

## 2. 公式環境と重要な分離

PARC2026本選資料 v1.1で開発用GPUは以下。

```text
NVIDIA RTX PRO 6000 Blackwell
本選1 GPU時間: 60時間
1 session: 最大12時間
idle: 1時間で自動停止
```

storage:

```text
$HOME             40 GiB  auto persistent
~/data            100 GiB manual persistent
~/dataset          read-only organizer dataset/baseline
/opt/dlami/nvme   約1.7 TB ephemeral
```

manual persistence:

```bash
parc-home-sync data-push
parc-home-sync data-pull
```

本番採点環境は別物。

```text
single NVIDIA L4 24 GB
10 sec / inference
```

したがって:

```text
RTX PRO 6000 Blackwell = 開発 / M3比較
L4                     = 最終submission互換性
```

を混同しない。

---

## 3. canonical paths

```bash
export PY_AI_REPO=$HOME/src/py_AI
export PARC_LOCAL_SCRATCH_ROOT=/opt/dlami/nvme/parc2026
export PARC_PERSIST_ROOT=$HOME/data/parc2026-cache
export PARC_DATASET_ROOT=$PARC_LOCAL_SCRATCH_ROOT/datasets/lerobot_libero_plus_v3_train
```

### `$HOME/src/py_AI`

Git repositoryのみ。

### `/opt/dlami/nvme/parc2026`

再生成可能な大容量物のみ。

```text
venv
model runtime
HF cache
torch cache
uv/pip cache
exact training dataset
temporary OpenVLA merge
compatibility checkpoint
```

### `~/data/parc2026-cache`

失ってはいけないものだけ。

```text
D10 / manifest
69c streaming contract
72d batch summary
fixed schedule
73 evidence + exact smoke checkpoints
74 evidence
75 benchmark checkpoints/results
76 screening summaries / promotion records
77 promotion summary
migration integrity records
```

---

## 4. immutable M3 contract

```text
D10 = V2_SQRT_BALANCED_RAW
source dataset = lerobot/libero_plus
source revision = f3f49f426d75030177b18778374005bc12ccd588
selected episodes = 10,758
selected frames = 1,620,614
tasks = 40

episode_ids_sha256 =
73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239

manifest_file_sha256 =
cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395
```

M3:

```text
effective batch = 32
training seed = 20260906
eval seeds = 20260906 / 20260907

equal-data = 4,800 samples/model = 150 optimizer updates
equal-wall = 1,800 sec train-loop/model
```

禁止:

- D10再選定
- manifest置換
- 72/73 evidence上書き
- full RLDS materialization
- failed evidence削除による成功扱い
- lossだけでpromotion

---

## 5. Blackwell runtime contract

過去72/73はA100で成立したhistorical evidenceとして保持する。

運営GPUでは別profileを使う。

```text
PARC_M3_HARDWARE_PROFILE=organizer_rtx_pro_6000_blackwell
GPU name contains: RTX PRO 6000 + Blackwell
VRAM >= 90,000 MiB
```

小さいMIG/vGPU sliceはfail-closed。

OpenVLAのpinned sourceは変更しない。

```text
source commit:
e4287e94541f459edc4feabc4e181f537cd569a8
```

ただしhistorical upstream dependency `torch==2.2.0` / old FlashAttentionはBlackwell用runtimeとして使わない。

organizer OpenVLA runtime:

```text
PyTorch 2.7.1
CUDA 12.8 wheel
TorchVision 0.22.1
TorchAudio 2.7.1
historical flash-attn removed
standard PyTorch attention
```

実機で必須:

```text
compute capability = 12.0
sm_120 available
CUDA >= 12.8
BF16 CUDA matmul PASS
```

π0.5 / SmolVLAも同じ実GPU上でtorch runtimeを検証してから実行する。

---

## 6. GPU起動前: Colab handoff bundle

Colab / Drive側で実artifactからbundleを作る。

```bash
export PARC_DRIVE_ROOT=/content/drive/MyDrive/parc2026-cache

python -u tools/benchmark/build_m3_organizer_handoff.py \
  --source-root "$PARC_DRIVE_ROOT" \
  --archive /content/m3-organizer-handoff-v1.tar
```

生成物:

```text
m3-organizer-handoff-v1.tar
m3-organizer-handoff-v1.tar.sha256.json
```

bundle内容:

```text
73 PASS summary
73 forward/equal-data 3-model train_result
73 exact smoke checkpoints
72d batch summary
fixed M3 schedule
69c streaming contract
D10 manifest
```

含めない:

```text
full dataset
venv
HF cache
other regenerable cache
```

builderはD10 / manifest / checkpoint identityを検証し、全file SHA256を内部manifestへ記録する。

---

## 7. organizer起動直後

既存persistent dataがある場合:

```bash
parc-home-sync data-pull
```

確認:

```bash
nvidia-smi
df -h "$HOME" "$HOME/data" /opt/dlami/nvme
df -i "$HOME" "$HOME/data" /opt/dlami/nvme
```

異なるGPU / 小さいsliceならSTOP。code側の閾値をその場で下げない。

---

## 8. source checkout

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"

if [ ! -d py_AI/.git ]; then
  git clone https://github.com/yu37330/py_AI.git py_AI
fi

cd py_AI
git fetch origin feat/organizer-gpu-runtime
git checkout --detach origin/feat/organizer-gpu-runtime

git rev-parse HEAD
```

review済みheadを固定する。

```bash
export PARC_EXPECTED_SOURCE_SHA=$(git -C "$PY_AI_REPO" rev-parse HEAD)
```

---

## 9. environment

```bash
export PY_AI_REPO=$HOME/src/py_AI
export PARC_LOCAL_SCRATCH_ROOT=/opt/dlami/nvme/parc2026
export PARC_PERSIST_ROOT=$HOME/data/parc2026-cache
export PARC_DATASET_ROOT=$PARC_LOCAL_SCRATCH_ROOT/datasets/lerobot_libero_plus_v3_train

export PARC_M3_EXECUTE=1
export PARC_M3_HARDWARE_PROFILE=organizer_rtx_pro_6000_blackwell
export PARC_M3_SIM_SMOKE_ATTEMPT=organizer-1
export PARC_M3_TRAINING_COMPAT_ATTEMPT=organizer-1
export PARC_M3_BENCHMARK_ATTEMPT=organizer-1
export PARC_EXPECTED_SOURCE_SHA=$(git -C "$PY_AI_REPO" rev-parse HEAD)

export HF_TOKEN='...'
```

secretをrepo/logへ保存しない。

cacheはlauncherがNVMeへ向ける。

---

## 10. safe handoff restore

bundleを運営環境へtransfer後:

```bash
mkdir -p "$PARC_PERSIST_ROOT"

python -u "$PY_AI_REPO/tools/benchmark/restore_m3_organizer_handoff.py" \
  --archive "$HOME/data/incoming/m3-organizer-handoff-v1.tar" \
  --sidecar "$HOME/data/incoming/m3-organizer-handoff-v1.tar.sha256.json" \
  --root "$PARC_PERSIST_ROOT"
```

restoreは:

```text
archive SHA
-> path traversal拒否
-> symlink/hardlink拒否
-> overwrite拒否
-> extraction
-> internal per-file SHA verification
```

expected:

```text
=== M3 ORGANIZER HANDOFF INTEGRITY: PASS ===
=== M3 ORGANIZER HANDOFF RESTORE: PASS ===
```

---

## 11. 74 organizer smoke

74ではfull training datasetは不要。

```bash
cd "$PY_AI_REPO"
python -u tools/benchmark/run_m3_organizer_simulator_smoke.py
```

contract:

```text
libero_spatial
10 tasks x 2 seeds x 3 models = 60 episodes
promotion evidence = false
benchmark training = false
```

旧Drive absolute `checkpoint_ref` は元JSONを変更せず、attempt-local viewだけをpersistent rootへrebaseする。

expected:

```text
M3 ORGANIZER HANDOFF INTEGRITY: PASS
M3 DISK HEADROOM: PASS
M3 ORGANIZER TRAINING RUNTIMES: PASS
M3 OPENVLA BLACKWELL RUNTIME: PASS
M3 SIMULATOR RUNTIME SETUP: PASS
74 M3 MINIMAL SIMULATOR SMOKE: PASS
ORGANIZER GPU 74 EQUIVALENT: PASS
```

74 PASS前に75へ進まない。

---

## 12. exact dataset on NVMe

74 PASS後:

```bash
python -u tools/benchmark/prepare_m3_organizer_dataset.py
```

固定:

```text
lerobot/libero_plus
revision f3f49f426d75030177b18778374005bc12ccd588
```

保存先:

```text
$PARC_DATASET_ROOT
```

expected:

```text
=== M3 ORGANIZER DATASET: PASS ===
```

server停止でNVMeは消える。次sessionでは同revisionを再materializeする。

---

## 13. Blackwell training compatibility

72/73を上書きせず、新しいhardware migration evidenceを作る。

```bash
python -u tools/benchmark/run_m3_organizer_training_compat.py
```

scope:

```text
forward
only equal-data
3 models
64 samples/model
2 optimizer updates/model
effective batch 32
```

compatibility checkpointはNVMeのみ。result/log/summaryだけpersistent。

expected:

```text
=== M3 ORGANIZER TRAINING COMPATIBILITY: PASS ===
ready_for_75=true
```

OOM/runtime incompatibilityなら75へ進まない。

---

## 14. 75 forward

```bash
python -u tools/benchmark/run_m3_organizer_benchmark_order.py \
  --order forward \
  --attempt organizer-1
```

expected:

```text
=== M3 ORGANIZER FORWARD BENCHMARK: PASS ===
```

直後:

```bash
parc-home-sync data-push
```

必要ならStop Server。

---

## 15. 75 reverse

新sessionなら:

```text
data-pull
source SHA確認
NVMe再構築
dataset exact revision再materialize
runtime再構築
```

その後:

```bash
python -u tools/benchmark/run_m3_organizer_benchmark_order.py \
  --order reverse \
  --attempt organizer-1
```

expected:

```text
=== M3 ORGANIZER REVERSE BENCHMARK: PASS ===
```

直後:

```bash
parc-home-sync data-push
```

---

## 16. 76はcheckpoint単位で実行する

12時間session制約のため、**480 episodes/orderをmonolithicに開始しない。**

1 unit:

```text
1 order
x 1 track
x 1 model
= 80 episodes
```

unitはattempt-scoped stagingへ書き、80 episodes PASS後だけcanonical screening pathへpromoteする。

途中でserverが強制停止しても、completed canonical unitは壊れない。失敗したstaging evidenceはそのまま残し、新しい `--unit-attempt` で再試行する。

### forward units

```bash
python -u tools/benchmark/run_m3_organizer_screening_unit.py \
  --order forward --track equal_data --model pi05 \
  --attempt organizer-1 --unit-attempt unit-1

python -u tools/benchmark/run_m3_organizer_screening_unit.py \
  --order forward --track equal_data --model smolvla \
  --attempt organizer-1 --unit-attempt unit-1

python -u tools/benchmark/run_m3_organizer_screening_unit.py \
  --order forward --track equal_data --model openvla_oft \
  --attempt organizer-1 --unit-attempt unit-1

python -u tools/benchmark/run_m3_organizer_screening_unit.py \
  --order forward --track equal_wall --model pi05 \
  --attempt organizer-1 --unit-attempt unit-1

python -u tools/benchmark/run_m3_organizer_screening_unit.py \
  --order forward --track equal_wall --model smolvla \
  --attempt organizer-1 --unit-attempt unit-1

python -u tools/benchmark/run_m3_organizer_screening_unit.py \
  --order forward --track equal_wall --model openvla_oft \
  --attempt organizer-1 --unit-attempt unit-1
```

各unit完了後、session残時間が少なければ:

```bash
parc-home-sync data-push
```

してStop Serverする。

6 unit揃ったらCPU-only finalizer:

```bash
python -u tools/benchmark/finalize_m3_organizer_screening_order.py \
  --order forward --attempt organizer-1
```

expected:

```text
=== M3 ORGANIZER FORWARD SCREENING FINALIZED: PASS ===
```

### reverse units

model orderに関係なくunitは個別指定できる。

```bash
python -u tools/benchmark/run_m3_organizer_screening_unit.py \
  --order reverse --track equal_data --model openvla_oft \
  --attempt organizer-1 --unit-attempt unit-1
```

以下同様に:

```text
reverse/equal_data/smolvla
reverse/equal_data/pi05
reverse/equal_wall/openvla_oft
reverse/equal_wall/smolvla
reverse/equal_wall/pi05
```

6 unit揃ったら:

```bash
python -u tools/benchmark/finalize_m3_organizer_screening_order.py \
  --order reverse --attempt organizer-1
```

expected:

```text
=== M3 ORGANIZER REVERSE SCREENING FINALIZED: PASS ===
```

---

## 17. screening unit failure

canonical pathへpromoteされるのはPASS後だけ。

途中失敗はstagingに残る。

```text
unit-1 failed
-> evidence preserved
-> retry with unit-2
```

canonical summaryが既にPASSなら同unitはreuseする。

canonical pathにpartialがある場合は自動削除しない。診断して明示対応する。

---

## 18. 77 promotion

forward/reverse finalizer PASS後:

```bash
export PARC_DRIVE_ROOT=$PARC_PERSIST_ROOT
python -u tools/colab/run_m3_promotion_controller.py --attempt organizer-1
```

required:

```text
12 promotion records
simulator_success_rate primary
loss not used
promoted models = 1..2
```

expected:

```text
=== 77 M3 FORWARD/REVERSE PROMOTION: READY ===
```

final candidate / 800 episode / artifact freezeはPR #68へhandoffする。

---

## 19. session plan

75は約数時間レンジを想定し、order単位で分割する。

76は時間予測に依存せずcheckpoint unitで分割する。

推奨:

```text
Session A
restore
74
exact dataset
Blackwell training compatibility
75 forward
data-push
Stop Server

Session B
restore/data-pull
NVMe rebuild
75 reverse
data-push
Stop Server

Session C+
76 screening unitを残時間に応じて1個ずつ
各unit後に残時間判断
必要ならdata-push -> Stop Server

Final CPU step
forward finalize
reverse finalize
77 promotion
data-push
```

「次の80 episode unitが12時間内に終わる余裕があるか不明」なら新unitを開始しない。

---

## 20. stop procedure

必ず:

```bash
parc-home-sync data-push
```

その後:

```text
File -> Hub Control Panel
Stop Server
Start My Server表示を確認
```

ブラウザtabを閉じるだけでは停止しない。

---

## 21. final submission separation

M3 development PASS後も、最終candidateはL4互換性を別Gateで確認する。

```text
selected candidate
-> submission/model_weights
-> zip <= 20GB
-> validate_submission.py
-> organizer pre-check Docker
-> single L4 24GB
-> <=10 sec/inference
-> end-to-end round-trip
-> submission
```

評価中外部network accessは禁止されるため、必要weight/tokenizer/codeはzip内へ同梱する。

---

## 22. GPU起動前 readiness

コード側で完了:

```text
[x] official Blackwell hardware profile
[x] generic scratch/persistent path
[x] cache -> NVMe
[x] checksum handoff builder
[x] safe archive restore
[x] per-file integrity verifier
[x] old Drive checkpoint_ref relocation without source mutation
[x] Blackwell OpenVLA torch/cu128 runtime
[x] all 3 runtime sm_120 verification
[x] organizer 74 launcher
[x] exact dataset materializer
[x] Blackwell training compatibility gate
[x] organizer 75 forward/reverse launcher
[x] checkpoint-scoped 76 unit runner
[x] CPU-only 76 order finalizer
[x] 77 canonical handoff
[x] session persistence strategy
```

物理環境でしかできない残作業:

```text
[ ] actual Colab/Drive artifactからhandoff bundle生成
[ ] bundle transfer
[ ] live nvidia-smi
[ ] 74 live PASS
```

ここまではGPU起動前準備として完了状態とする。
