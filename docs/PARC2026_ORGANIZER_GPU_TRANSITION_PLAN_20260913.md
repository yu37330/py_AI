# PARC2026 運営GPU移行前レビュー・実行方針（2026-09-13）

## 1. この文書の目的

本書は、PR #79 / #80 で整理した現状を前提に、Colab中心の実行から運営側GPU環境へ切り替える前に、事実関係・未解決リスク・実行順序を再確認し、期限までのクリティカルパスを定義するための運用方針である。

参照元:

- PR #79: `docs/PARC2026_CURRENT_STATUS_20260913.md`
- PR #80: `docs/PARC2026_PROJECT_OVERVIEW_20260913.md`
- `docs/PARC2026_M3_OPERATIONAL_CAUTIONS_20260912.md`
- `docs/PARC2026_M3_PREEXECUTION_AUDIT_ADDENDUM_20260912.md`
- `docs/PARC2026_GPU_VALIDATION_AND_NEXT_STEPS_20260829.md`
- repository root `README.md`
- open PR #67: M3 LIBERO evaluator / Notebook 74系
- open PR #68: final candidate / artifact freeze系
- open PR #70: alternate training implementation（canonical pathとしては採用しない）

本書は過去のsnapshotを書き換えるものではない。PR #79 / #80 のGPU移行記述を、過去の運営GPU実績と現行コード監査を踏まえて具体化する追加方針として扱う。

---

## 2. 結論

2026-09-13 時点では、**運営側GPUへ切り替える判断は妥当**である。

理由は、現在の主blockerがモデル設計やD10ではなく、Notebook 74で発生しているColab storage exhaustionだからである。

ただし、移行後すぐに75a / 75bの本番M3 trainingへ進んではいけない。

正しい順序は以下。

```text
運営GPU移行準備
  ↓
環境inventory / artifact identity確認
  ↓
Notebook 74相当のminimal simulator smokeを運営GPUで完走
  ↓
60 episode records + PASS
  ↓
PR #67 final review / main統合判断
  ↓
75a / 75bを最新runtime + 運営GPUprofileへrefresh
  ↓
Full M3 benchmark
  ↓
76 screening
  ↓
77 promotion
  ↓
final candidate
  ↓
800-episode final evaluation
  ↓
artifact freeze
  ↓
運営Dockerでsubmission round-trip
  ↓
submission
```

**74 PASSまでは75を開始しない。**

---

## 3. PR #79 / #80 の評価

PR #79 / #80 の現状整理は、Dataset / D10 / 69b / 69c / 71 / 72 / 73 / 74 / 75以降の依存関係を正しく整理できている。

特に以下はそのまま維持する。

- D10は固定済み
- full RLDS materializationはしない
- OpenVLAは69c streaming bridgeを使う
- 72a〜72dはPASS evidenceとして保存する
- Notebook 73 training smokeはPASS済み
- Notebook 74は未PASS
- 75 / 76 / 77 / finalは未開始
- training lossだけでpromotionしない
- Colabのattempt番号だけ増やして74を再試行しない

一方、GPU移行部分については補足が必要である。

PR #79 / #80 では「運営側GPU仕様がrepo上で未確定」と整理されているが、repoには過去の運営環境についてかなりの実績が残っている。

したがって、今後は「何も分かっていないGPU」ではなく、**既知の運用情報がある60時間GPU環境を、現行M3 contractへ再接続する作業**として扱う。

---

## 4. 「運営GPU」は2種類に分けて考える

ここを混同しないことが重要。

### 4.1 60時間の運営配布GPU / JupyterHub環境

2026-08-29の記録では、P&R2026本選側の60時間GPU枠で実際に以下を行っている。

- π0.5学習環境構築
- LIBERO dataset読込
- LoRA smoke training
- LoRA merge
- VRAM probe
- checkpoint / log永続化
- `parc-home-sync data-push`
- Hub Control PanelからStop Server

確認済みpath / 運用:

```text
read-only dataset area:
~/dataset

一時高速領域:
/opt/dlami/nvme/...

永続化:
~/data

停止前:
parc-home-sync data-push
File -> Hub Control Panel -> Stop Server
```

さらに、BS=32 LoRA probeでは `80,435 MiB` のpeak VRAMが記録され、OOMなしで完走している。

したがって、この60時間環境は少なくとも過去実行時点では大容量GPUを使える環境だったと判断できる。ただし、**GPU model / exact VRAMを文書から断定せず、再開時に `nvidia-smi` で再取得する。**

### 4.2 運営Docker / 本番評価互換環境

root `README.md` では、本番採点GPUは NVIDIA L4 と明記されている。

この環境の主目的は:

- submission zip validation
- policy server起動
- Track1〜3 example task評価
- 本番依存構成に近いend-to-end確認

である。

NVIDIA公式仕様ではL4は24GB GPU memoryである。

よって、**60時間のtraining GPUと、L4の本番評価Dockerは同一用途として扱わない。**

現在のM3 training / multi-model simulator smokeはまず60時間GPU側で行い、最終submission互換性はL4運営Dockerで確認する。

---

## 5. 現在の固定contract

### Dataset / D10

```text
variant: V2_SQRT_BALANCED_RAW
dataset: lerobot/libero_plus
revision: f3f49f426d75030177b18778374005bc12ccd588
episodes: 10,758
frames: 1,620,614
tasks: 40

episode_ids_sha256:
73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239

manifest_file_sha256:
cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395
```

### M3 fairness

```text
effective batch size: 32
training schedule seed: 20260906
eval seeds: 20260906, 20260907

equal-data:
4,800 actual samples / model
150 optimizer updates

equal-wall:
1,800 sec train-loop only / model
```

### OpenVLA

```text
full RLDS materialization: 禁止
streaming bridge: 使用
action_dim: 7
proprio_dim: 8
action_chunk: 8
normalization: bounds_q99
```

これらは環境移行を理由に変更しない。

---

## 6. 絶対に壊さないartifact / evidence

```text
D10 selected episode IDs
D10 SHA256
V2 group-aware manifest
V2 manifest SHA256
69b PASS evidence
69c PASS evidence
72a / 72b / 72c / 72d PASS evidence
Notebook 73 PASS summary
Notebook 73 per-model train_result.json
Notebook 73 exact smoke checkpoints
Notebook 74 attempt 1〜8 evidence
```

禁止:

- D10再選定
- manifest再生成での置換
- 73 checkpointを別checkpointへ黙って置換
- failed attemptの上書き
- evidence directoryの一括削除
- full RLDS materialization

---

## 7. 運営GPU切替前に、GPU時間を使わず済ませる作業

運営GPUを起動してからコード修正を始めない。

以下は事前にGitHub側で完了させる。

### 7.1 Notebook 74 / PR #67のfilesystem generic化

PR #67の現在の74実装にはまだColab専用の前提が残る。

特に:

```text
PARC_ROOT default      = /content/parc2026
PARC_DRIVE_ROOT default = /content/drive/MyDrive/parc2026-cache
local disk check       = /content
```

`tools/colab/prepare_m3_disk_headroom.py` は `/content` の存在を必須とし、Google Drive前提のメッセージを持つ。

運営GPU向けには少なくとも以下へ分離する。

```text
PARC_LOCAL_SCRATCH_ROOT
PARC_PERSIST_ROOT
PY_AI_REPO
HF_HOME
TORCH_HOME
XDG_CACHE_HOME
```

推奨mapping:

```text
PARC_LOCAL_SCRATCH_ROOT=/opt/dlami/nvme/parc2026
PARC_PERSIST_ROOT=$HOME/data/parc2026

venv / dataset cache / HF cache / temp merge:
/opt/dlami/nvme/parc2026/...

logs / result JSON / immutable evidence / required checkpoints:
$HOME/data/parc2026/...
```

### 7.2 74のA100-name hard gateを見直す

PR #67の `prepare_m3_simulator_runtimes.py` は現在:

```text
GPU name contains A100
AND
VRAM >= 38000 MiB
```

を要求している。

運営60時間GPUのexact modelが文書上固定されていないため、GPU名文字列ではなく、まず実機inventoryを取得する。

その後、74のacceptanceは以下のどちらかにする。

1. exact A100環境なら既存contractを維持
2. A100以外の大容量GPUなら、VRAM / runtime capabilityを検証した新しいorganizer profileとして明示的に許可

**24GB L4に合わせるために閾値を雑に下げることは禁止する。**

L4は最終評価互換性確認のターゲットであり、現在の3-model training/smokeを無検証で移す対象ではない。

### 7.3 73 artifact transfer manifestを作る

74はNotebook 73のforward/equal-data smoke checkpointを正確に要求する。

運営GPUへ持っていく前に:

- `m3_training_smoke_summary.json`
- 各modelの `train_result.json`
- 各 `checkpoint_ref`
- source ref
- D10 hash
- checkpoint directory hash / file inventory

を一覧化する。

運営GPU側で同じidentityをread-backしてから74を開始する。

**転送できないことを理由に73を自動再学習しない。**

### 7.4 75 / 76の修正は先に準備する

74 PASS後にGPU上でコードを書き始めると60時間枠を消費する。

そのため、実行はしなくても以下のpatchは事前準備しておく。

#### 75

- Notebook 73で確定したfresh-runtime fixesを反映
- 74 PASSをrunner-level mandatory gateへ追加
- generic filesystem rootへ対応
- D10 hash fail-closed
- benchmark default OFFを維持

#### 76

Notebook 74で確定した以下を伝播する。

- non-interactive LIBERO config
- `MPLBACKEND=Agg`
- `MUJOCO_GL=egl`
- `PYOPENGL_PLATFORM=egl`
- pinned LIBERO namespace
- `mujoco==3.3.1`
- camera feature rename
- output directory lifecycle
- OpenVLA JIT merge / cleanup
- disk headroom
- partial evidence persistence
- repo-root importability / `PYTHONPATH` または module invocation

Screeningはgeneric evaluatorの `mode=smoke` を使う既存設計を維持し、誤って `benchmark` に変えない。

---

## 8. 運営GPUを起動した直後のinventory

最初にtrainingを実行しない。

以下のようなinventoryを取り、結果をpersistent storageへ保存する。

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi

python3 --version
git --version
docker --version || true
uv --version || true

pwd
whoami

# storage
(df -h / || true)
(df -h /opt/dlami/nvme || true)
(df -h "$HOME/data" || true)
(df -h "$HOME/dataset" || true)

(df -i / || true)
(df -i /opt/dlami/nvme || true)
(df -i "$HOME/data" || true)

command -v parc-home-sync || true
parc-home-sync --help || true
```

確認項目:

```text
GPU model
GPU total VRAM
NVIDIA driver
CUDA visible version
local NVMe total/free
persistent storage total/free
inode headroom
Python
Docker availability
uv availability
GitHub access
Hugging Face access
session/job timeout
parc-home-sync behavior
```

ここで取得した結果を以降のenvironment provenanceにする。

---

## 9. GPU inventory後のGo / No-Go

### Case A: 旧60時間環境と同等の大容量GPU

```text
VRAM >= 38 GiB
NVMeに十分なscratchあり
persistent storage利用可
必要なnetwork / package accessあり
```

なら、74 migrationへGO。

### Case B: GPUはA100ではないがVRAM >= 38 GiB

GPU名だけで拒否せず、organizer runtime profileを追加する。

ただし:

- model load
- simulator runtime
- EGL
- checkpoint lifecycle
- inference peak VRAM

を74の前段preflightで確認する。

### Case C: 24GB級GPU / L4のみ

現在のA100ベースM3 pathをそのまま実行しない。

ここでA100 guardを外して強行するのではなく、以下を分離する。

```text
M3 training / 3-model smoke:
別の大容量GPU環境

submission compatibility / organizer Docker round-trip:
L4
```

期限優先のため、新しいlow-memory training architectureの研究には戻らない。

---

## 10. storage運用

8/29の実績から、`~/data` にvenv / HF cacheまで置くと `data-push` が約93.8GiBになり、停止前同期コストが増えた。

今後は明確に分離する。

### persistent: `~/data`

保存する:

- exact git SHA / environment snapshot
- D10 manifest / provenance
- Notebook 73 checkpoint evidence
- M3 training checkpoints
- train_result.json
- simulator episode records
- summaries
- promotion evidence
- final artifact
- submission zip
- compact logs

### scratch: `/opt/dlami/nvme`

置く:

- dataset checkout / cache
- venv
- pip / uv cache
- HF cache
- model download cache
- temporary evaluator files
- temporary OpenVLA merged artifact
- reproducible intermediate files

### OpenVLA JIT merge

74 / 76のintermediate evaluationでは、merged 7B artifactをpersistent storageに常駐させない。

可能ならlocal NVMeにmaterializeし、evaluation後にdematerializeする。

最終候補がOpenVLAの場合だけ、final 800 evaluationからartifact freezeまで必要なmerged evaluator-ready artifactを保持する。

---

## 11. Artifact restore gate

運営GPU上では、以下を通すまで74を開始しない。

### D10

```text
episode_ids_sha256 ==
73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239

manifest_file_sha256 ==
cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395
```

### Notebook 73

```text
summary status: PASS
stage: M3_A100_training_smoke
benchmark_training_started: false
selected_episode_ids_sha256: exact D10 hash
```

各model:

```text
mode: smoke
order: forward
track: equal_data
samples_consumed: 64
optimizer_updates: 2
effective_batch_size: 32
checkpoint_ref: exists
source_ref: exact expected ref
```

artifact identityが一致しない場合はfail closed。

---

## 12. 運営GPUで最初に実行するもの

**最初の本番実行は74 minimal simulator smoke。**

75ではない。

### 74 acceptance contract

```text
suite: libero_spatial
models: pi05 / smolvla / openvla_oft
seeds: 20260906 / 20260907
10 tasks x 2 seeds x 3 models
= 60 episodes
max steps: 300
promotion evidence: false
benchmark training started: false
```

PASS条件:

```text
=== 74 M3 MINIMAL SIMULATOR SMOKE: PASS ===
=== 74 COMPLETE ===

60 episode records
D10 exact match
Notebook 73 exact checkpoint linkage
3 model summaries
```

### 74前の低コストpreflight

環境移行直後なので、full 60 episodesの前にsetup-onlyで以下を確認する。

- all source pins
- exact checkpoint read
- runtime imports
- EGL
- MuJoCo / robosuite
- model load
- OpenVLA JIT finalize / cleanup path
- local/persistent storage headroom

ここで失敗したら60 episode smokeを開始しない。

---

## 13. 74 PASS後のPR #67扱い

PR #67はNotebook 74 / evaluator修正を蓄積しているが、現時点ではopen / unmergedである。

74が運営GPUでPASSしたら:

1. exact PASS runtime SHAを固定
2. 60 episode evidenceを確認
3. current mainとの差分を再確認
4. conflict / duplicate fixを整理
5. testsを再実行
6. #67 final review
7. merge判断

とする。

**74 PASS前に「ほぼ動いたから」でmergeしない。**

---

## 14. 75 benchmarkへ進む前の追加Gate

### 14.1 75 runtime refresh

現在の75 notebookはそのまま実行しない。

必須:

- 73 fresh-runtime fixes
- OpenVLA model-logic sync
- scheduled adapter compatibility
- output-dir lifecycle
- organizer filesystem profile
- 74 PASS gate
- persistent logging

### 14.2 hardware-specific batch feasibility

72a〜72dはA100で得たPASS evidenceであり、比較contractの根拠として保存する。

ただし、運営GPUが72実行時と別hardwareなら、A100のmicro-batch / gradient-accumulation値を無条件で再利用しない。

その場合は72を上書きせず、例えば:

```text
organizer_gpu_batch_compat_v1
```

のような別artifactで、各modelのfeasible micro-batchをbounded smokeする。

守るもの:

```text
effective batch size = 32
D10 unchanged
sampling schedule unchanged
forward/reverseで同じper-model batch semantics
```

これはdataset再選定でも72の再実行でもなく、**hardware migration compatibility evidence**として扱う。

---

## 15. Full M3 benchmarkの推奨順序

長時間runの前に短い高情報量runを優先する。

推奨:

```text
1. forward / equal-data
2. reverse / equal-data
3. forward / equal-wall
4. reverse / equal-wall
```

理由:

- equal-dataでsample accounting / checkpoint / schedule linkageを先に検証できる
- forward/reverse双方の実行pathを早期に踏める
- equal-wall 1800秒へ入る前にidentity問題を発見できる

各run完了後:

- PASS JSON検証
- sample/update/wall accounting検証
- checkpoint存在確認
- source SHA記録
- `~/data` へ必要artifact保存
- 必要に応じて `parc-home-sync data-push`
- temporary merged model / disposable cache cleanup

を行う。

一括で最後まで走らせてから確認しない。

---

## 16. 76 screening

75がcompleteした後だけ実行する。

76は74のvalidated evaluator stackへrefreshする。

重要:

```text
screening = 4 suites x 10 tasks x 2 seeds x 1 trial
          = 80 episodes / checkpoint
```

generic evaluatorの `mode=smoke` を意図的に使う。

`mode=benchmark` にすると10 trials/taskとなり、意図せず800 episode規模になるため変更しない。

---

## 17. 77 promotion

Promotion primary metric:

```text
simulator success rate
```

補助:

```text
steps-to-success
inference latency
peak inference VRAM
train wall time
peak train VRAM
```

最大2モデルをpromotion。

training lossはranking keyに使わない。

forward / reverse × equal-data / equal-wallの必要evidenceがそろっていることを確認する。

---

## 18. final candidate / 800 episode evaluation

PR #68は実装済みだが、現時点ではreal M3 evidence待ち。

Promotion後にcurrent mainへrefreshし、selected candidateだけをfinalへ進める。

Final evaluation:

```text
2 seeds
400 episodes / seed
800 episodes total
```

ここで初めてfinal-scale evaluationを行う。

非選択modelで800 episodeを回さない。

---

## 19. submission前はL4運営Dockerへ戻す

最終checkpointができたら、training GPU上で成功しただけでは提出可としない。

root READMEの本番互換Docker経路で:

```text
final model artifact
  ↓
submission/model_weights
  ↓
zip
  ↓
validate_submission.py
  ↓
organizer Docker
  ↓
policy server startup
  ↓
example task end-to-end
  ↓
submission artifact freeze
```

を確認する。

特にtraining environmentとevaluation DockerではPython / torch / CUDA / OS packageが異なり得るため、このround-tripを省略しない。

---

## 20. 60時間GPU枠の使い方

過去方針どおり、GPU時間は「コードを書く時間」ではなく「実model executionと運営環境固有の検証」に使う。

優先順位:

```text
P0: environment inventory / 74 completion
P0: 75 benchmark
P0: 76 screening
P0: final candidate evaluation
P0: organizer-specific compatibility check

P1: high-value retry
P1: bounded single issue diagnosis

P2: broad ablation
P2: new model family
P2: dataset reselection
P2: cosmetic refactor
```

途中でblockerが出た場合も、同じ失敗runを繰り返してGPU枠を消費しない。

---

## 21. Fail-closed / rollback条件

以下では即停止する。

### D10 mismatch

```text
STOP
manifestを作り直して続行しない
```

### Notebook 73 checkpoint identity mismatch

```text
STOP
黙って新しいsmoke checkpointへ置換しない
```

### organizer GPU VRAM不足

```text
STOP current M3 runtime
A100 guardを雑に外さない
```

### local / persistent disk不足

```text
STOP before execution
D10 / checkpoints / evidenceを削除しない
scratch配置・cache配置を見直す
```

### source ref mismatch

```text
STOP
current mainだから正しいとみなさない
```

### partial execution failure

```text
attempt evidenceを保存
新しいattempt IDを使う
既存failed attemptを上書きしない
```

---

## 22. 2026-09-13時点のGo / No-Go

| 項目 | 判定 | 条件 |
|---|---|---|
| 運営60時間GPUへの移行準備 | GO | filesystem / artifact transferを事前整備 |
| 運営GPU inventory | GO | 最初に実行、training禁止 |
| D10再選定 | NO-GO | frozen |
| manifest再生成 | NO-GO | fixed hashを維持 |
| 72a-d上書き再実行 | NO-GO | historical evidenceとして保持 |
| hardware migration batch compat | CONDITIONAL GO | 別artifact、D10/effective batch不変 |
| Notebook 73再学習 | 原則NO-GO | exact checkpoint transferを優先 |
| Notebook 74 minimal smoke | GO | environment/artifact gate通過後 |
| PR #67 merge | NO-GO | 74 60 episodes PASSまで |
| 75 full benchmark | NO-GO | 74 PASS + 75 refresh + hardware compatまで |
| 76 screening | NO-GO | 75 completeまで |
| 77 promotion | NO-GO | 76 completeまで |
| final 800 eval | NO-GO | selected final candidate決定まで |
| artifact freeze | NO-GO | final evidence completeまで |
| submission | NO-GO | organizer Docker round-tripまで |

---

## 23. 直近の実行順

GPU切替前:

```text
A. #67のfilesystem generic化
B. organizer environment profile追加
C. disk headroom checkのgeneric化
D. 73 artifact transfer manifest作成
E. 75に74 PASS gate追加準備
F. 76へ74 evaluator fixes伝播準備
G. generic evaluatorのrepo import risk修正
```

GPU起動後:

```text
1. environment inventory保存
2. local scratch / persistent root確認
3. exact source checkout
4. D10 / manifest hash確認
5. 73 checkpoint identity確認
6. setup-only runtime preflight
7. 74 minimal simulator smoke
8. 60 records + PASS確認
9. #67 final review
10. 75 refresh確定
11. 必要ならhardware migration batch compat
12. M3 benchmark
13. screening
14. promotion
15. final 800 eval
16. artifact freeze
17. L4 organizer Docker round-trip
18. submission
```

---

## 24. Source-of-truth metadata

本書作成時:

```text
main:
424765c2f022665453c349d8f1c2e4969662e98f

PR #79:
merged

PR #80:
merged

PR #67:
open
head observed: 4bc24f5e9b4c9bbb092bbf6ea303cca395260994

Notebook 74 attempt 8 runtime pin:
1b983910509020dc2a01f1474140d72bf5664e8e

PR #68:
open / IMPLEMENTED_PENDING_REAL_M3_EVIDENCE

PR #70:
open / alternate implementation / do not mix into canonical path
```

---

## 25. 最終判断

現在は、設計やdataset探索を続ける段階ではない。

最も価値が高いのは、Colab storage問題から離れ、過去に実績のある60時間の運営GPU環境を**再現性を保ったexecution platform**として使うことである。

ただし成功条件は「GPUを変えること」ではない。

成功条件は:

```text
同じD10
同じcheckpoint provenance
同じM3 fairness contract
同じevaluation contract
十分なscratch storage
persistent evidence
74 PASS
75/76/77/finalの順序維持
最後にL4運営Dockerで提出互換性確認
```

を崩さずに、期限までのクリティカルパスを完走することである。

したがって次の開発作業は、**運営GPUを起動する前にPR #67/75/76のenvironment依存を整理し、74を運営GPUで一度で検証できる状態へ持っていくこと**とする。
