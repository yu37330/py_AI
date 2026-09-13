# PARC2026 運営GPU移行・M3実行Runbook（2026-09-13）

## 1. 目的

PR #79 / #80 / #81で固定したM3方針を、PARC2026公式の配布GPU環境へ安全に移行する。

本Runbookは、GPUを起動してからコード修正やpath調査を始めないことを目的とし、以下を事前に固定する。

- 公式hardware / storage / session制約
- Colab / Google Drive成果物のhandoff
- source checkout
- scratch / persistent / dataset / cache path
- Notebook 74相当の60 episode smoke
- Blackwell training compatibility gate
- 75 forward / reverse benchmark
- 76 forward / reverse screening
- 77 promotionへのhandoff

D10、manifest、72/73 evidenceは再生成・上書きしない。

---

## 2. 公式配布GPU環境

PARC2026本選資料 v1.1（2026-08-26更新）で、開発用の配布GPUは次のように明記されている。

```text
GPU:
NVIDIA RTX PRO 6000 Blackwell

本選1 GPU時間:
60時間

1セッション:
最大12時間

完全idle:
1時間で自動停止
```

RTX PRO 6000 Blackwellは96GBクラスのGPUである。M3では小さいvGPU slice等を誤って受け入れないため、organizer hardware profileは以下でfail-closedとする。

```text
GPU name contains:
RTX PRO 6000
Blackwell

VRAM:
>= 90,000 MiB
```

これは過去のA100 evidenceを置き換えるものではない。

```text
72 / 73:
historical A100 evidence

organizer migration:
RTX PRO 6000 Blackwell compatibility evidence
```

なお、本番採点環境は別であり、単一NVIDIA L4 24GB。最終submissionはL4用の運営Dockerで別途round-tripする。

---

## 3. 公式storage制約とpath設計

公式資料:

```text
$HOME
40 GiB
自動永続化

~/data
100 GiB
手動永続化
save:    parc-home-sync data-push
restore: parc-home-sync data-pull

~/dataset
read-only
配布dataset / baseline

/opt/dlami/nvme
約1.7 TB
非永続 instance storage
```

したがってM3では次を正本mappingとする。

```bash
export PY_AI_REPO=$HOME/src/py_AI
export PARC_LOCAL_SCRATCH_ROOT=/opt/dlami/nvme/parc2026
export PARC_PERSIST_ROOT=$HOME/data/parc2026-cache
export PARC_DATASET_ROOT=$PARC_LOCAL_SCRATCH_ROOT/datasets/lerobot_libero_plus_v3_train
```

### `$HOME/src/py_AI`

- Git repositoryだけ
- 小さいため40GiBの自動永続領域に置く
- venv / model cache / datasetを置かない

### `/opt/dlami/nvme/parc2026`

再生成可能な大容量物だけ。

- 3 model runtimes / venv
- Hugging Face cache
- torch cache
- uv / pip cache
- temporary files
- exact `lerobot/libero_plus` dataset
- compatibility smoke checkpoints
- OpenVLA JIT materialization

### `~/data/parc2026-cache`

失ってはいけないものだけ。

- D10 / manifest
- 69c streaming contract
- 72d batch summary
- fixed schedule
- Notebook 73 evidence / exact smoke checkpoints
- 74 evidence
- 75 benchmark checkpoints / results
- 76 screening evidence
- 77 promotion evidence
- migration inventory / integrity records

**venv、HF cache、dataset本体は `~/data` に置かない。**

---

## 4. immutable contract

```text
D10 variant:
V2_SQRT_BALANCED_RAW

dataset:
lerobot/libero_plus

revision:
f3f49f426d75030177b18778374005bc12ccd588

episodes:
10,758

frames:
1,620,614

tasks:
40

episode_ids_sha256:
73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239

manifest_file_sha256:
cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395
```

M3:

```text
effective batch size = 32
training schedule seed = 20260906
evaluation seeds = 20260906 / 20260907

equal-data:
4,800 samples / model
150 optimizer updates

equal-wall:
1,800 sec train-loop / model
```

禁止:

- D10再選定
- manifest再生成で置換
- 72/73 PASS evidence上書き
- full RLDS materialization
- failed attempt削除による成功扱い
- training lossだけでpromotion

---

## 5. GPUを起動する前: Colab側handoff bundle作成

運営GPUへ移す最小artifactをchecksum付きbundleにする。

bundleに含めるもの:

```text
Notebook73 summary
Notebook73 forward/equal-data 3-model train_result
その3つのexact smoke checkpoints
72d batch summary
fixed M3 schedule
69c streaming contract
D10 manifest
```

含めないもの:

```text
full dataset
venv
HF cache
過去の巨大な再生成可能cache
```

ColabでPR #82の実行対象SHAをcheckoutした後:

```bash
export PARC_DRIVE_ROOT=/content/drive/MyDrive/parc2026-cache

python -u tools/benchmark/build_m3_organizer_handoff.py \
  --source-root "$PARC_DRIVE_ROOT" \
  --archive /content/m3-organizer-handoff-v1.tar
```

生成物:

```text
/content/m3-organizer-handoff-v1.tar
/content/m3-organizer-handoff-v1.tar.sha256.json
```

builderは:

- Notebook73 PASSを検証
- D10 hashを検証
- manifest file SHAを検証
- checkpointがpersistent root外を指していないか検証
- symlinkを拒否
- 全ファイルSHA256をmanifestへ保存

archiveを上書きしない。再生成時は古いarchiveを明示的に退避する。

### transport

bundleのtransport方法自体はbenchmark contractではない。

安全な方法で以下2ファイルを運営JupyterHubへ持ち込む。

```text
m3-organizer-handoff-v1.tar
m3-organizer-handoff-v1.tar.sha256.json
```

transfer後もsidecar SHAと内部manifest SHAの二段検証を必ず行う。

---

## 6. 運営GPUを起動した直後

### 6.1 `~/data` restore

過去に `data-push` 済みの場合:

```bash
parc-home-sync data-pull
```

その後:

```bash
df -h "$HOME" "$HOME/data" /opt/dlami/nvme
df -i "$HOME" "$HOME/data" /opt/dlami/nvme
nvidia-smi
```

期待:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell
VRAM: >= 90,000 MiB
NVMe: 十分な空き容量
```

異なるGPU / 小さいsliceならそこでSTOP。閾値をその場で下げない。

---

## 7. source checkout

repoは小さいため `$HOME` に置く。

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

実行時にはPR #82のreview済みhead SHAを `PARC_EXPECTED_SOURCE_SHA` として固定する。

---

## 8. organizer環境変数

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

secretをrepo / logsへ保存しない。

launcherがcacheをNVMeへ向ける。

```text
HF_HOME
TORCH_HOME
XDG_CACHE_HOME
UV_CACHE_DIR
PIP_CACHE_DIR
TMPDIR
```

---

## 9. handoff restore

例:

```bash
mkdir -p "$PARC_PERSIST_ROOT"

python -u "$PY_AI_REPO/tools/benchmark/restore_m3_organizer_handoff.py" \
  --archive "$HOME/data/incoming/m3-organizer-handoff-v1.tar" \
  --sidecar "$HOME/data/incoming/m3-organizer-handoff-v1.tar.sha256.json" \
  --root "$PARC_PERSIST_ROOT"
```

restoreは:

```text
archive SHA検証
-> path traversal拒否
-> symlink/hardlink拒否
-> overwrite拒否
-> extract
-> internal manifest全ファイルSHA検証
```

expected marker:

```text
=== M3 ORGANIZER HANDOFF INTEGRITY: PASS ===
=== M3 ORGANIZER HANDOFF RESTORE: PASS ===
```

以降の各sessionでも必要に応じてintegrity verifierを再実行する。

---

## 10. Notebook 74相当

74ではdataset本体は不要。

```bash
cd "$PY_AI_REPO"
python -u tools/benchmark/run_m3_organizer_simulator_smoke.py
```

内部:

```text
source SHA
-> RTX PRO 6000 Blackwell profile
-> persistent handoff全SHA
-> Notebook73 exact checkpoint identity
-> disk headroom
-> non-interactive LIBERO config
-> 3 model runtime再構築
-> mujoco 3.3.1
-> 60 episode simulator smoke
```

expected:

```text
=== M3 ORGANIZER HANDOFF INTEGRITY: PASS ===
=== M3 DISK HEADROOM: PASS ===
=== M3 SIMULATOR RUNTIME SETUP: PASS ===
=== 74 M3 MINIMAL SIMULATOR SMOKE: PASS ===
=== ORGANIZER GPU 74 EQUIVALENT: PASS ===
```

74 contract:

```text
suite = libero_spatial
10 tasks x 2 seeds x 3 models = 60 episodes
promotion_evidence = false
benchmark_training_started = false
```

旧Drive absolute `checkpoint_ref` は元JSONを変更せず、attempt-local execution viewだけを新しいpersistent rootへrebaseする。

---

## 11. datasetをNVMeへ再構築

74 PASS後、本番training前にexact dataset revisionをNVMeへmaterializeする。

```bash
python -u tools/benchmark/prepare_m3_organizer_dataset.py
```

固定:

```text
repo = lerobot/libero_plus
revision = f3f49f426d75030177b18778374005bc12ccd588
root = $PARC_DATASET_ROOT
```

expected:

```text
=== M3 ORGANIZER DATASET: PASS ===
```

`~/data` へdatasetを複製しない。

server停止後はNVMeが消えるため、次sessionでは同じexact revisionから再構築する。

---

## 12. Blackwell training compatibility Gate

過去72/73はA100 evidenceなので上書きしない。

RTX PRO 6000 Blackwell上で、新しい独立evidenceとして:

```text
3 models
forward only
equal-data only
64 samples/model
2 optimizer updates/model
effective batch = 32
```

だけを実行する。

```bash
python -u tools/benchmark/run_m3_organizer_training_compat.py
```

compatibility checkpointはNVMeのみ。永続化するのはresult/log/summaryだけ。

expected:

```text
=== M3 ORGANIZER TRAINING COMPATIBILITY: PASS ===
ready_for_75 = true
```

ここでOOM / runtime incompatibilityが出た場合は75へ進まない。

micro-batch / GAを変更する必要が出た場合は、effective batch 32を維持し、hardware migration evidenceとして別途reviewする。72dを上書きしない。

---

## 13. 75a forward benchmark

古いColab Notebook 75aは使用しない。

```bash
python -u tools/benchmark/run_m3_organizer_benchmark_order.py \
  --order forward \
  --attempt organizer-1
```

wrapperは以下をmandatory gateとする。

```text
handoff integrity PASS
74 PASS
organizer dataset PASS
Blackwell training compatibility PASS
hardware profile PASS
```

その後canonical `run_m3_benchmark_order.py`を使用する。

```text
equal-data 4,800 samples / model
equal-wall 1,800 sec / model
sequence: pi05 -> smolvla -> openvla_oft
```

expected:

```text
=== M3 ORGANIZER FORWARD BENCHMARK: PASS ===
```

完了後すぐ:

```bash
parc-home-sync data-push
```

必要ならここでServerを停止して次sessionへ分割する。

---

## 14. 75b reverse benchmark

新sessionの場合:

```text
parc-home-sync data-pull
source checkout確認
NVMe scratch再作成
dataset exact revision再materialize
training runtime再構築
```

その後:

```bash
python -u tools/benchmark/run_m3_organizer_benchmark_order.py \
  --order reverse \
  --attempt organizer-1
```

sequence:

```text
openvla_oft -> smolvla -> pi05
```

expected:

```text
=== M3 ORGANIZER REVERSE BENCHMARK: PASS ===
```

完了後:

```bash
parc-home-sync data-push
```

---

## 15. 76 screening

screeningは1 orderあたり:

```text
2 tracks x 3 models = 6 checkpoints
80 episodes / checkpoint
480 episode records / order
```

forward:

```bash
python -u tools/benchmark/run_m3_organizer_screening_order.py \
  --order forward \
  --attempt organizer-1
```

reverse:

```bash
python -u tools/benchmark/run_m3_organizer_screening_order.py \
  --order reverse \
  --attempt organizer-1
```

各order後に `data-push` する。

expected:

```text
=== M3 ORGANIZER FORWARD SCREENING: PASS ===
=== M3 ORGANIZER REVERSE SCREENING: PASS ===
```

古い76 Notebookを直接実行しない。

---

## 16. 77 promotion

77はCPU-only controller。

forward / reverse screeningが揃った後:

```bash
export PARC_DRIVE_ROOT=$PARC_PERSIST_ROOT
python -u tools/colab/run_m3_promotion_controller.py --attempt organizer-1
```

必須:

```text
12 promotion records
simulator_success_rate = primary metric
training loss = promotionに使用しない
promoted_models = 1〜2
```

expected:

```text
=== 77 M3 FORWARD/REVERSE PROMOTION: READY ===
```

ここから先のfinal candidate / 800 episode / artifact freezeはPR #68の実evidence gateへhandoffする。

---

## 17. 12時間session境界

1sessionに全工程を詰め込まない。

推奨:

```text
Session A:
handoff restore
-> 74
-> dataset
-> Blackwell training compat
-> 75 forward
-> data-push
-> Stop Server

Session B:
data-pull
-> dataset rebuild
-> 75 reverse
-> data-push
-> Stop Server

Session C:
data-pull
-> simulator runtime rebuild
-> 76 forward
-> data-push
-> Stop Server

Session D:
data-pull
-> simulator runtime rebuild
-> 76 reverse
-> 77 promotion
-> data-push
-> Stop Server
```

実測時間によってはさらに細分化する。

---

## 18. failure handling

### partial attempt

削除して上書きしない。

```text
organizer-1 failed
-> organizer-2
```

### hardware mismatch

STOP。profile閾値を下げない。

### storage不足

D10 / checkpoint / evidenceは削除しない。

削除対象は再生成可能なNVMe cacheのみ。

### session残時間不足

新しい高コストstageを開始しない。まず:

```bash
parc-home-sync data-push
```

その後Stop Server。

---

## 19. server停止

ブラウザtabを閉じるだけでは停止しない。

必ず:

```text
parc-home-sync data-push
File -> Hub Control Panel
Stop Server
Start My Server表示を確認
```

GPU時間はStop Server完了まで消費される。

---

## 20. 最終submissionとの分離

開発GPUがRTX PRO 6000 Blackwellでも、本番採点は単一L4 24GB。

最終候補決定後は必ず:

```text
selected candidate
-> submission/model_weights
-> zip <= 20GB
-> validate_submission.py
-> organizer pre-check Docker
-> L4 memory / 10 sec inference compatibility
-> end-to-end round-trip
-> submission
```

M3比較用GPUで動くことと、提出物がL4で動くことを同一視しない。

---

## 21. GPU起動前 readiness checklist

コード側で完了していること:

```text
[ ] PR #82 review対象SHA固定
[x] generic scratch / persistent path
[x] RTX PRO 6000 Blackwell hardware profile
[x] reduced VRAM slice fail-closed
[x] handoff bundle builder
[x] archive SHA sidecar
[x] safe restore
[x] per-file checksum verifier
[x] Notebook73 checkpoint relocation view
[x] 74 organizer launcher
[x] exact dataset revision materializer
[x] Blackwell training compatibility launcher
[x] 75 organizer forward/reverse launcher
[x] 76 organizer forward/reverse launcher
[x] 77 CPU controller handoff
[x] NVMe cache routing
[x] session/persistence Runbook
```

物理環境でしかできない残作業:

```text
[ ] Colab上で実artifactからhandoff bundleを生成
[ ] bundleを運営環境へtransfer
[ ] nvidia-smi実測
[ ] 74 live PASS
```

この4項目以外は、GPU起動前にコード上の準備を終えておく。
