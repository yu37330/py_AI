# PARC2026 Colab並列計画: Model Selection × Dataset Factory

更新日: 2026-08-29

## 1. Decision

Colab A100では、次の2本を**並列**で進める。

```text
Lane M: Model Selection
  π0.5 / SmolVLA / OpenVLA-OFT
  → 同一評価契約で比較
  → 本命モデル候補を1〜2個へ絞る

Lane D: Dataset Factory
  Organizer Raw
  → inventory
  → quality filtering
  → task balancing
  → safe augmentation
  → public supplemental
  → Track3 inverse generation
  → Dataset V0〜V5を作る

                 ↓
        Sync Point / Ablation
                 ↓
    Best Model × Best Dataset Recipe
                 ↓
          Organizer GPU Run A
```

片方だけを先に完成させてからもう片方へ進むのではなく、短い実験を交互に回す。

理由:

- モデル性能とデータ品質は相互依存する。
- π0.5で効くdataset recipeがSmolVLA/OpenVLA-OFTでも同程度に効くとは限らない。
- ただし全モデル×全datasetの総当たりはGPU時間を浪費するため、段階的に絞る。
- 運営GPU 60hは探索ではなく、最終候補のcheckpoint生成と評価へ集中させる。

---

## 2. Lane M: Model Selection

### M0. Preflight

完了済み:

- Colab A100 workspace作成
- `py_AI` clone/fetch
- experiment config / manifest基盤

次に確認:

- Python / CUDA / A100 VRAM
- upstream revision固定
- HF token / gated checkpointアクセス

### M1. π0.5 reproducibility gate

目的: 運営GPUで通ったπ0.5 recipeをColabでも再現し、GA semanticsを確定する。

成果物:

- 20-step LoRA smoke log
- peak VRAM
- wall time
- merged checkpoint
- GA=1/2/8 verification log
- optimizer step count
- scheduler step count
- `sync_gradients`周期

合格条件:

- GA=8で8 micro-stepごとにoptimizer/schedulerが1回進む
- `steps=20` が20 optimizer updatesを意味する
- mergeまで成功

### M2. Candidate bring-up

候補:

1. π0.5
2. SmolVLA
3. OpenVLA-OFT

各モデルで最低限そろえる契約:

- exact upstream repo + commit/tag
- exact base checkpoint revision
- same dataset manifest
- same local eval split
- same seed set
- success
- steps-to-success
- episode time
- trajectory metrics
- inference latency
- peak VRAM
- train wall time

### M3. Fair comparison protocol

2方式を分ける。

#### Protocol M-A: Equal Data Exposure

- 同じtrain episode pool
- 同じsampling policy
- 同じseed
- できる限り同程度のtraining samplesを見せる

目的: データを同程度見た時のsample efficiency比較。

#### Protocol M-B: Equal Wall Time

- Colab A100の学習時間を同じにする
- 例: 30分 / 1hなど短時間budget

目的: 実際のGPUコストに対する性能比較。

### M4. Promotion rule

最終候補へ残す条件:

- local successが明確に低くない
- L4 24GB推論に現実的に載る見込み
- startup / inference timeout条件に入る見込み
- submissionサイズ制約へ収まる
- Colabで再現可能
- organizer GPU 60hで本学習可能

本命を1モデルに固定できない場合でも、最大2モデルまでに絞る。

---

## 3. Lane D: Dataset Factory

### D0. Organizer Rawを正本として固定

正本:

- `libero_combined_20hz.tar`
- observed: 3,028,708 frames
- observed: 19,533 episodes
- 20Hz

公開データは補助候補であり、Organizer Rawの代替として無条件に混ぜない。

### D1. Dataset Inventory V1

最初に全episodeの構成を可視化する。

episode-level出力:

- episode_index
- task_index
- task_name / language instruction
- source suite（判別可能ならLIBERO / LIBERO-plus）
- frames
- duration_sec
- state shape
- action shape
- camera keys

summary出力:

- task数
- task別episode数
- task別frame数
- task別duration分布
- suite別episode数
- 上位/下位の不均衡task
- inverse pair候補

成果物:

- `episode_inventory.csv`
- `task_inventory.csv`
- dataset manifest更新

### D2. Static Quality Analyzer V1

まずLeRobotデータだけで計算できる安価な指標を全episodeへ付与する。

候補:

- duration_sec
- EEF / state movement
- path length
- net displacement
- path efficiency
- Cartesian jerk
- joint/state jerk（schema確認後）
- action RMS / max
- idle ratio
- gripper switch count
- timestamp/fps整合
- NaN / Inf

注意:

- この段階ではtask success / collision / true replayabilityを断定しない。
- MuJoCo raw stateが無い変換データだけでは完全replayできない可能性がある。

成果物:

- `episode_quality_static.csv`
- task別percentile report
- outlier candidate list

### D3. Replay Validator V2

raw simulator state / original HDF5 / BDDL等が取得できるtaskのみ実装する。

記録:

- replay_attempted
- replay_success
- task_success
- collision/contact pair
- harmful_collision_count
- max collision impulse/force（取得可能な場合）
- final-state drift
- trajectory drift

結果はStatic AnalyzerのCSVへJOINする。

### D4. Quality Tiering

最初から固定閾値を決めない。task別分布を見て設定する。

- Gold: replay/success確認済み、明確な異常なし、軌道品質上位
- Silver: 成功/利用可能だが効率・smoothness等に弱点
- Reject candidate: broken、NaN、replay不能、重大衝突、極端なoutlier等

重要:

- `high jerk = reject` のような単一指標ルールは使わない。
- task特性差を考慮する。

### D5. Dataset Versions

```text
V0 Organizer Raw
  19,533 episodesを基本そのまま

V1 Organizer Clean
  明確なbroken / invalidのみ除外

V2 Clean + Task Balanced
  task samplingを調整

V3 V2 + Safe Visual Augmentation
  Track2/generalization狙い

V4 V3 + Public Supplemental
  不足task/variationのみ補完

V5 V4 + Track3 Inverse
  simulator-valid inverse demosを追加
```

### D6. Task Balancing candidates

最低3つ比較する。

- raw proportional: episode数比例
- uniform-task: taskを均等sampling
- sqrt-balanced: `sqrt(task_episode_count)`相当の中間sampling

最初の第一候補はsqrt-balancedだが、実測で決める。

### D7. Augmentation candidates

安全側から開始:

- brightness
- contrast
- mild color jitter
- mild crop/resize
- camera-like small perturbation（action/stateとの整合を壊さない範囲）

初期段階では避ける:

- naive horizontal flip
- 大きな画像rotation
- state/action座標系を変えずに幾何変換だけかける処理

### D8. Public Supplemental

目的は「データ量を増やす」ことではなく、「不足を補う」こと。

追加前に確認:

- license
- source URL/revision
- task mapping
- episode duplication
- frame rate / schema compatibility
- Organizer Rawとの重複

追加対象例:

- Organizer Rawで極端に少ないtask
- Track2の視覚/domain variation不足
- inverse方向の操作不足

### D9. Track3 Inverse Factory

禁止する案:

- action sequenceを単純に時間反転して正解demoとする

生成手順:

```text
forward task
→ inverse task definition
→ inverse task用initial state
→ simulatorでexpert/script/policy実行
→ task success確認
→ replay validation
→ LeRobot 20Hz schemaへ変換
→ inverse dataset manifestへ登録
```

追加比率候補:

- 0%
- 5%
- 10%
- 20%

Track3 successを上げつつTrack1/2を壊さない点を探す。

---

## 4. 並列実行の同期ポイント

### Sync S0: 環境固定

Lane M:
- π0.5 smoke / GA verification

Lane D:
- Organizer dataset schema / task inventory

両方完了後、共通eval subsetとexperiment manifestを固定する。

### Sync S1: Cheap baseline

モデル側:
- π0.5を基準モデルとして固定

データ側:
- V0 / V1 / V2を作成

ここではまずπ0.5だけでdataset差を比較する。

理由: 3モデル×3datasetを最初から回さない。

### Sync S2: Model shortlist

V2程度の安定した共通datasetを使って、

- π0.5
- SmolVLA
- OpenVLA-OFT

を比較し、1〜2モデルへ絞る。

### Sync S3: Dataset ablation

shortlistされたモデルだけで、

- V2 balanced
- V3 augmentation
- V4 public supplemental
- V5 inverse

を順番に比較する。

### Sync S4: Final recipe

最終的に1つのRun A recipeを固定する。

- model
- base checkpoint revision
- dataset version
- sampling policy
- augmentation
- LoRA rank
- LR
- BS / GA
- seed
- steps

Track3 inverseが有効な場合のみRun B recipeも固定する。

---

## 5. 実験マトリクス

### Stage 1: Dataset cheap ablation

モデルはπ0.5固定。

| ID | Dataset | 目的 |
|---|---|---|
| D-A | V0 Raw | 基準 |
| D-B | V1 Clean | quality filter効果 |
| D-C | V2 sqrt-balanced | task balance効果 |
| D-D | V2 uniform | balance方式比較 |

### Stage 2: Model comparison

DatasetはStage 1のbestを固定。

| ID | Model | Protocol |
|---|---|---|
| M-A | π0.5 | equal-data + equal-wall-time |
| M-B | SmolVLA | equal-data + equal-wall-time |
| M-C | OpenVLA-OFT | equal-data + equal-wall-time |

### Stage 3: Generalization / Track3

shortlist modelのみ。

| ID | Dataset | 目的 |
|---|---|---|
| G-A | V2 | 基準 |
| G-B | V3 | augmentation / Track2 |
| G-C | V4 | public supplemental |
| G-D | V5 inverse 5% | Track3 |
| G-E | V5 inverse 10% | Track3 |
| G-F | V5 inverse 20% | Track3 |

---

## 6. 評価指標

training lossだけで昇格判定しない。

必須:

- task success rate
- track-like success rate
- steps-to-success
- episode duration
- cartesian path length
- joint/state path length（取得可能な場合）
- jerk / smoothness
- inference latency
- peak inference VRAM
- training wall time
- peak training VRAM

Dataset評価には追加:

- task coverage
- episode count per task
- Gold/Silver/Reject比率
- source distribution
- duplicate rate
- inverse pair coverage

---

## 7. Colab notebook / toolの実装順

既存:

- `colab/00_a100_preflight.ipynb`

次に追加する候補:

1. `colab/10_pi05_smoke_ga.ipynb`
   - π0.5 setup
   - 20-step smoke
   - GA instrumentation

2. `colab/20_dataset_inventory.ipynb`
   - dataset schema
   - task/episode inventory
   - inventory CSV

3. `colab/21_dataset_quality_static.ipynb`
   - movement/path/jerk/idle等
   - quality CSV

4. `colab/30_model_benchmark.ipynb`
   - 共通benchmark launcher
   - equal-data / equal-wall-time

5. `colab/40_dataset_ablation.ipynb`
   - V0〜V5切替
   - sampling / augmentation比較

6. `colab/50_track3_inverse_factory.ipynb`
   - inverse task definition
   - simulator trajectory生成
   - replay validation
   - LeRobot変換

再利用可能な処理はnotebookへ埋め込まず、`tools/`配下のPythonへ切り出す。

---

## 8. Git管理する成果物 / しない成果物

Git管理する:

- notebook
- Python tools
- dataset manifest
- task inventory summary
- experiment config
- eval split
- run manifest
- metrics JSON/CSVの小さいsummary
- decision log

Git管理しない:

- raw dataset
- videos
- HF cache
- venv
- full checkpoints
- model weights
- 大容量intermediate artifacts

---

## 9. Organizer GPUへ戻る条件

次が揃うまで、運営GPUで20k本学習を始めない。

- [ ] π0.5 GA semantics確認
- [ ] Dataset Inventory V1
- [ ] Static Quality Analyzer V1
- [ ] V0/V1/V2 short ablation結果
- [ ] model shortlist 1〜2個
- [ ] augmentation採否
- [ ] public supplemental採否
- [ ] Track3 inverse小規模Evidence
- [ ] Run A exact dataset manifest
- [ ] Run A exact model/config/command
- [ ] smoke submission round-trip
- [ ] 可能ならL4 24GB inference確認

---

## 10. 次の実装順

並列ではあるが、依存関係を守る。

```text
Now
├─ Lane M
│   └─ 10_pi05_smoke_ga
│
└─ Lane D
    └─ 20_dataset_inventory

        ↓ 両方完了

π0.5 × V0/V1/V2 cheap ablation
        ↓
model comparison
        ↓
augmentation / public / inverse ablation
        ↓
Run A固定
```

直近の2成果物は以下とする。

1. `10_pi05_smoke_ga`: GA=8を含むπ0.5 Colab再現
2. `20_dataset_inventory`: 19,533 episodeのtask構成をCSV化

この2本を並列で進めるのが次のフェーズ。
