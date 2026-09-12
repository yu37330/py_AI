# PARC2026 プロジェクト全体像（2026-09-13）

## 1. この文書の目的

本書は、2026-09-13 時点の `yu37330/py_AI` における PARC2026 / Track3 プロジェクト全体を、技術詳細だけでなく「どこまで終わっていて、何が未完了で、何が現在のblockerで、ここから何をするか」という観点で俯瞰するためのプロジェクト概要である。

詳細な障害履歴・再発防止・実行注意事項は以下を参照する。

- `docs/PARC2026_CURRENT_STATUS_20260913.md`
- `docs/PARC2026_M3_OPERATIONAL_CAUTIONS_20260912.md`
- `docs/PARC2026_M3_PREEXECUTION_AUDIT_ADDENDUM_20260912.md`
- `docs/PARC2026_CURRENT_STATUS_20260906.md`

本書は、上記の詳細文書より一段上の粒度で、プロジェクト全体の進行状況と提出までのクリティカルパスを説明する。

---

## 2. プロジェクトの最終目的

最終目的は、PARC2026 Track3向けに、固定されたdataset条件の下で複数モデルを公平に比較し、最良候補を選び、再現可能な最終artifactを生成し、期限内に提出することである。

比較対象モデルは以下の3つ。

1. π0.5
2. SmolVLA
3. OpenVLA-OFT

最終的な理想フローは次のとおり。

```text
Dataset Factory / D10 freeze
        ↓
M3 runtime readiness
        ↓
3-model training smoke
        ↓
3-model simulator smoke
        ↓
Full M3 benchmark training
        ↓
Simulator screening
        ↓
Promotion
        ↓
Final candidate
        ↓
800-episode final evaluation
        ↓
Artifact freeze
        ↓
Fresh-session reproduction
        ↓
Submission
```

2026-09-13 時点では、**3-model simulator smoke の途中**にいる。

---

## 3. 現在地を一枚で見る

| フェーズ | 状態 | 現在地 |
|---|---|---|
| Dataset inventory / quality | ✅ 完了 | public LIBERO-plusを分析済み |
| Leakage-safe split | ✅ 完了 | group-aware V2へ移行 |
| Dataset recipe screening | ✅ 完了 | fixed simulatorで比較 |
| D10 selection | ✅ 完了 | `V2_SQRT_BALANCED_RAW` 固定 |
| RLDS feasibility | ✅ 完了 | full materialization非推奨を確認 |
| Streaming bridge | ✅ 完了 | OpenVLA用streaming path成立 |
| M3 comparison contract | ✅ 完了 | equal-data / equal-wall / forward-reverse固定 |
| Batch probe | ✅ 完了 | 72a / 72b / 72c / 72d PASS |
| Training smoke | ✅ 完了 | Notebook 73 A100 PASS |
| Simulator smoke | 🟡 進行中 | Notebook 74、π0.5実機rollout確認済み |
| Full benchmark training | ⛔ 未開始 | 75a / 75b |
| Simulator screening | ⛔ 未開始 | 76a / 76b |
| Promotion | ⛔ 未開始 | 最大2モデル選定 |
| Final candidate | ⛔ 未開始 | 実M3 evidence待ち |
| 800-episode final eval | ⛔ 未開始 | 最終候補のみ |
| Artifact freeze | 🟡 実装準備済み | PR #68、実evidence待ち |
| Submission | ⛔ 未実施 | 最終段階 |

重要: **1800秒の本番M3 benchmark trainingはまだ開始していない。**

---

## 4. Dataset Factoryで達成したこと

### 4.1 Public LIBERO-plusをproxyとして利用

開発・screening用datasetとして以下を使用してきた。

```text
dataset: lerobot/libero_plus
revision: f3f49f426d75030177b18778374005bc12ccd588
```

Inventory、quality analysis、dataset ablation、fixed simulator screeningを通じて、単純なraw dataset利用ではなく、品質とtask exposureを考慮したdataset recipeを比較した。

### 4.2 trajectory leakageを発見し、legacy splitを廃止

初期のepisode-level splitでは、同一trajectory由来のsiblingsがtraining / evaluationへ跨る問題が発見された。

このため、legacy splitを廃止し、group-aware V2を正本とした。

以降のdataset選定では以下を必須条件にしている。

```text
exact_leakage_group_count = 0
action_leakage_group_count = 0
Trajectory Leakage Gate = PASS
```

### 4.3 Dataset recipe screening

主な候補は以下だった。

- V0_RAW
- V1_MULTI_FLAG_PRUNED_EXPERIMENTAL
- V1_ALL_REVIEW_PRUNED_EXPERIMENTAL
- V2_SQRT_BALANCED_RAW

π0.5の短時間trainingとfixed simulator評価を用い、training lossだけでなくSimulator successを重視して比較した。

V1_MULTIとV2_SQRTが同率首位となり、その後のtie-breakを経てV2をM3用D10として固定した。

---

## 5. 現在のD10は固定済み

M3比較に使用するdataset contractは以下で固定されている。

```text
variant: V2_SQRT_BALANCED_RAW
dataset: lerobot/libero_plus
revision: f3f49f426d75030177b18778374005bc12ccd588
selected episodes: 10,758
selected frames: 1,620,614
selected episode IDs SHA256:
73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239
```

V2 manifest file SHA256:

```text
cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395
```

以降の最重要ルール:

- D10を再選定しない
- manifestを再生成して別inputにしない
- 失敗を理由にdataset recipeを変更しない
- 72 / 73 / 74の既存evidenceを不用意に消さない

この固定により、今後のトラブル対応はdatasetを変えるのではなく、runtime / evaluator / storageを直す方針とする。

---

## 6. OpenVLA用データ経路も成立済み

### 6.1 69b RLDS smoke

69bではselected datasetのRLDS bridgeを検証した。

```text
smoke episodes: 8
smoke frames: 1,206
projected full size: 56.61 GiB
threshold: 35 GiB
recommendation: STREAMING_BRIDGE_RECOMMENDED
```

これにより、full RLDS materializationは採用しない方針となった。

### 6.2 69c streaming bridge

69cでOpenVLA-OFT向けstreaming bridgeを検証し、`READY_FOR_RUNNER` まで到達した。

現在のOpenVLA系の原則:

- full RLDSを作らない
- selected LeRobot dataからstreamingする
- D10 canonical orderを維持する
- forward / reverse間で同じsampling semanticsを使う

---

## 7. M3比較の公平性contract

3モデル比較では、単純な「同じstep数」ではなく、framework差を吸収するために以下の2軸を固定している。

### equal-data

```text
4,800 actual samples / model
effective batch size = 32
nominal optimizer updates = 150
```

### equal-wall

```text
1,800 sec / model
```

setup、download、conversion、evaluation、checkpoint serializationはequal-wallのtraining timerから除外する。

### 実行順バイアス対策

Forward:

```text
π0.5 → SmolVLA → OpenVLA-OFT
```

Reverse:

```text
OpenVLA-OFT → SmolVLA → π0.5
```

同じD10、同じcanonical sample stream、同じseed semantics、同じmetric定義を維持する。

PromotionはForward / Reverse双方のevidenceを使う。

---

## 8. 71 / 72でM3 readinessは成立済み

以下は完了している。

```text
71  preflight                      PASS
72a π0.5 A100 batch probe          PASS
72b SmolVLA A100 batch probe       PASS
72c OpenVLA-OFT A100 batch probe   PASS
72d controller                     PASS
```

このフェーズで、各frameworkのbatch semantics、VRAM、dependency、runtime互換性をかなり潰した。

特にOpenVLAでは以下のような互換性問題を修正してきた。

- TensorFlow Metadata / protobuf
- W&B / protobuf compatibility
- pandas / pyarrow / PyAV
- FlashAttention
- model logic mismatch
- streaming sample adapter

72のprobe evidenceはすでに得られているため、74以降のトラブル対応で不要に再実行しない。

---

## 9. Notebook 73: 3-model training smokeはPASS済み

Notebook 73ではA100上で、π0.5 / SmolVLA / OpenVLA-OFTすべてについて実training smokeを行った。

確認済み:

- fresh runtime reconstruction
- 実dataset読み込み
- 実model load
- 実optimizer update
- effective batch 32
- D10固定
- 実checkpoint生成
- OpenVLA streaming training path
- 3モデルのtrain-result evidence

この成功により、少なくとも短時間のtraining pathについては実機で成立している。

ただし明確に区別する。

```text
Notebook 73 = training smoke
Notebook 75 = full M3 benchmark training
```

Notebook 73 PASSは1800秒training済みという意味ではない。

---

## 10. Notebook 74: 現在の主戦場

### 10.1 74の目的

本番trainingを始める前に、Notebook 73で作成した各モデルのsmoke checkpointが実LIBERO simulatorで評価できることを確認する。

契約:

```text
suite: libero_spatial
models: π0.5 / SmolVLA / OpenVLA-OFT
seeds: 20260906 / 20260907
10 tasks × 2 seeds × 3 models
= 60 episodes
max steps: 300
promotion evidence: false
```

### 10.2 attempt 1〜8で直した主な問題

| Attempt | 主な問題 | 対応 |
|---:|---|---|
| 1 | hf-libero configが対話入力を要求 | non-interactive config |
| 2 | Colab Matplotlib / EGL問題 | `Agg` / `egl` / `PYOPENGL_PLATFORM=egl` |
| 3 | OpenVLA LIBERO namespace import不整合 | pinned source `.pth` exposure |
| 4 | MuJoCo 3.13 + robosuite 1.4 incompatibility | `mujoco==3.3.1` |
| 5 | camera feature名不一致 | `image/image2 -> front/wrist` rename |
| 6 | rollout後のupstream出力dir不足 | output directory事前生成 |
| 7 | failure diagnostics中にもdisk full | bounded diagnosticsへ変更 |
| 8 | 事前headroom PASS後に実行中disk full | 現在の主要blocker |

### 10.3 attempt 6で実機確認できたこと

π0.5は以下を実際に通過した。

```text
runtime setup
checkpoint load
camera rename
LIBERO environment creation
model inference
simulator stepping
libero_spatial 10 tasks completion
10 / 10 success
```

これは重要であり、現在の問題は「Simulatorが根本的に動かない」段階ではない。

### 10.4 attempt 8のblocker

現在確認されているエラー:

```text
OSError: [Errno 28] No space left on device
```

attempt 8では事前にlocal / Driveのheadroom確認と再生成可能cache cleanupを入れていたが、実行中に容量枯渇が発生した。

診断ログ自体も同じ`Errno 28`で読めなくなったため、元のchild process exceptionは確定していない。

したがって、現時点の74 statusは:

```text
π0.5 simulator path: 実機確認済み
SmolVLA: attempt 8で到達形跡あり
OpenVLA-OFT: 74全体として未完了
60 episodes total PASS: 未達
```

---

## 11. 現在の主要blockerはColab storage

2026-09-13時点の主blockerは、GPU capabilityそのものよりColab runtime / mounted filesystem容量である。

これまで対策済み:

- pre-setup disk headroom check
- pre-eval disk headroom check
- uv / pip / apt cache cleanup
- D10 / checkpoint / HF cache / past evidence保護
- bounded log tail
- `df -h` / `df -i` diagnostics
- partial episode evidence保存

それでも実行中に容量が尽きた。

そのため、同じColab構成でattempt番号だけ増やす再試行は避ける。

---

## 12. 運営側GPUへの移行案

Colab容量枯渇を受け、Notebook 74以降を運営側GPUへ移す案を検討中。

2026-09-13時点では、運営側GPUの詳細仕様はrepo上で未確定。

確認が必要な情報:

```text
GPU model
VRAM
OS
NVIDIA driver
CUDA
local SSD total/free
persistent storage path/capacity
GitHub access
Hugging Face access
HF_TOKEN handling
Python / uv / Docker可否
session/job timeout
```

M3 runtime guardとの整合上、少なくともA100-class相当またはVRAM >= 38 GiBを満たすことが望ましい。

ただし、今回の直接blockerはstorageのため、GPU性能だけでなくSSD容量・persistent storage設計が重要。

移行時に変えてはいけないもの:

- D10
- manifest
- canonical sample ordering
- 73 checkpoint evidence
- model source refs
- evaluation seeds
- max step contract
- equal-data / equal-wall contract

変えてよいもの:

- filesystem path
- cache path
- persistent storage root
- Colab固有mount path
- bootstrap方法

---

## 13. 75a / 75b: 本番M3 trainingはまだ未実行

Notebook 75が初めて本番比較trainingになる。

予定されている比較:

```text
3 models
× equal-data
× equal-wall
× forward
× reverse
```

重要: 75a / 75bは現在の古いruntime pinのまま実行しない。

74 PASS後に必ず以下を確認・反映する。

1. 73で確定したfresh-runtime修正
2. OpenVLA training logic fixes
3. 74 PASSを明示gate化
4. fresh machine bootstrap
5. D10 hash再検証
6. filesystem / storage preflight
7. benchmark開始前のfail-fast

ここを更新してから1800秒本番へ進む。

---

## 14. 76a / 76b: Simulator screening

75で生成した本番checkpointをLIBEROで評価するフェーズ。

74で見つかったevaluation側の修正は76へ必ず伝播する。

必要なもの:

- non-interactive LIBERO config
- headless Agg + EGL
- pinned LIBERO namespace
- `mujoco==3.3.1`
- camera rename map
- output directory precreation
- OpenVLA JIT merge / cleanup
- disk headroom guard
- partial evidence保存

76の古い実装をそのまま実行しない。

---

## 15. 77: Promotion

Promotionでは、training lossではなくSimulator success rateをprimary metricとする。

想定metrics:

- simulator success rate
- steps to success
- episode duration
- inference latency
- peak inference VRAM
- train wall time
- peak train VRAM

Forward / Reverse双方のevidenceを集約し、最大2モデルまで昇格する。

この時点で初めて「どのモデルが最も有望か」を確定できる。

2026-09-13時点では最良モデルは未決定。

---

## 16. Final candidate / 800-episode evaluation / artifact freeze

PR #68では、最終候補選定後の後段パイプラインがかなり実装されている。

想定フロー:

```text
promotion summary
        ↓
final candidate selection
        ↓
final-run config freeze
        ↓
linked final training result
        ↓
800-episode final evaluation
        ↓
artifact / provenance verification
        ↓
SUBMISSION_ARTIFACT_FROZEN
```

ただし現状は `IMPLEMENTED_PENDING_REAL_M3_EVIDENCE`。

必要な実evidence:

- 75 forward / reverse benchmark
- 76 simulator evaluation
- 77 promotion summary
- selected final candidate
- linked final run
- full 800-episode evaluation

後段の実装はあるが、前段の実測データがまだないためfinal artifactは作れない。

---

## 17. PR構造の現在地

### 主系統

#### PR #66

M3 training adapters / runnersの主系統。

- merged
- Notebook 73 training smokeに使用

#### PR #67

LIBERO simulator executor / screening / promotion handoffの主要PR。

- open
- unmerged
- Notebook 74の修正を蓄積
- 74 PASS後にfinal review / main統合判断

#### PR #68

Final candidate → artifact freeze / submission handoff。

- open
- real M3 evidence待ち

### 代替・注意

#### PR #70

alternate training implementation。

主系統として採用しない。既存主系統へ不用意に混ぜない。

### ドキュメント系

#### PR #77 / #78

M3 operational cautions / audit ledger。

- merged

#### PR #79

2026-09-13 current status snapshot。

- merged

---

## 18. 提出スケジュールと現在の遅れ

Issue #50で定義された公式deadline:

```text
2026-09-17 23:59 JST
```

内部方針は、9/15までにartifact freeze、9/16をdry-run submission day、9/17をbufferとするものだった。

当初計画:

```text
9/11 forward benchmark
9/12 reverse benchmark
9/13 simulator + promotion
9/14 final candidate
9/15 artifact freeze
9/16 dry-run submission
9/17 final upload
```

しかし2026-09-13時点で:

```text
74 simulator smoke: 未完
75 benchmark: 未開始
76 screening: 未開始
77 promotion: 未開始
```

そのため計画に対して遅延している。

今後は研究的な追加改善より、クリティカルパス完走を優先する。

---

## 19. ここからの最優先順位

### P0

1. Notebook 74を安定環境で完走
2. 60 episode records + PASSを確認
3. PR #67のfinal review
4. 75a / 75bを最新runtimeへrefresh
5. Full M3 benchmarkをforward / reverseで完走
6. 76a / 76b simulator screening
7. 77 promotion
8. Final candidate決定
9. 800-episode final eval
10. artifact freeze
11. submission verification

### P1

P0がgreenになった後のみ:

- extra seed
- extra eval episodes
- small high-value ablation
- richer reporting

### P2

期限が近いため原則やらない:

- 新モデルfamily追加
- broad hyperparameter search
- dataset recipe再選定
- full RLDS materialization
- cosmetic refactor

---

## 20. Go / No-Go

| 項目 | 判定 |
|---|---|
| D10再選定 | NO-GO |
| group-aware manifest再生成 | NO-GO |
| 72 probe再実行 | 原則NO-GO |
| 73 training smoke再実行 | 原則不要 |
| 74 minimal simulator smoke | GO（環境を安定化してから） |
| 75a / 75b full benchmark | NO-GO until 74 PASS + refresh |
| 76a / 76b screening | NO-GO until 75 complete |
| 77 promotion | NO-GO until screening complete |
| final 800 eval | NO-GO until final candidate selected |
| artifact freeze | NO-GO until final evidence complete |
| submission | NO-GO until freeze/integrity check complete |

---

## 21. 絶対に壊さないもの

以下は現在のプロジェクトのimmutable evidenceとして扱う。

```text
D10 selected episode IDs
D10 SHA256
V2 group-aware manifest
V2 manifest SHA256
69b PASS evidence
69c PASS evidence
72a / 72b / 72c / 72d PASS evidence
Notebook 73 PASS checkpoint evidence
Notebook 74 past attempt evidence
```

さらに:

- failed attemptを上書きして成功扱いにしない
- intermediate logs / partial evidenceを不用意に削除しない
- new runtimeへ移行してもprovenanceを変えない

---

## 22. 現在のリスク

### 最大リスク: 時間

公式deadlineまで残り日数が少なく、M3本番benchmarkがまだ未開始。

### 技術リスク: storage

Colabで74完走に必要なstorage headroomが不足。

### 統合リスク

74で見つかった修正を75 / 76へ漏れなく伝播しないと、同じ種類のruntime failureを再度踏む可能性がある。

### 判断リスク

deadlineが近い中で新しいdataset / architectureへ戻ると提出自体を失う可能性が高い。

---

## 23. 2026-09-13 時点のプロジェクト評価

プロジェクトは「初期研究」段階ではない。

すでに以下はかなり固まっている。

- dataset recipe
- leakage-safe holdout
- D10
- model candidate set
- sampling fairness
- training metrics
- evaluation metrics
- execution order control
- 3-model training smoke
- final artifact provenance design

一方、提出結果を決める最重要部分はまだこれから。

未完了の本質は:

```text
本番M3 comparison
Simulator screening
Promotion
Final candidate evidence
Final evaluation
Submission artifact
```

したがって現在は、

> 開発・設計フェーズはかなり終わっており、本番比較実験と提出証拠作成へ移る直前

と評価するのが最も正確。

---

## 24. 次の推奨アクション

最優先はNotebook 74を、Colab以外を含む十分なstorageを持つGPU環境で完走させること。

運営側GPUを使う場合は、まずGPU / VRAM / SSD / persistent storage / CUDA / timeout / networkを確認し、74のfilesystem依存だけをgeneric化する。

そのうえで:

```text
74 PASS
  ↓
#67 final review
  ↓
75 refresh
  ↓
75 forward / reverse benchmark
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
submission
```

へ一気に進む。

---

## 25. 一言でまとめる

2026-09-13時点のPARC2026プロジェクトは、

> **Dataset / D10 / 3-model execution基盤 / training smokeまでは成立済み。Simulator smokeの実機互換性もかなり確認できている。一方、Colab storage枯渇により74全体PASSが未達で、そのため本番M3 benchmark以降がまだ開始できていない。ここからは74を安定環境で完走し、75→76→77→final eval→artifact freeze→submissionを期限内に通すことが最優先。**

という状態である。
