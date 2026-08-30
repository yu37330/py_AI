# PARC2026 現状まとめ・判断ログ・次アクション（2026-08-30）

## 1. この文書の目的

本書は、2026-08-29〜2026-08-30に実施した運営GPUでのπ0.5検証、Colab A100でのDataset Factory検証、Static Quality Analyzer、Dataset Ablation、Trajectory-group Leakage検査、Group-aware holdout V2までを一つにまとめた現在地のSource of Truthである。

詳細は既存文書へ分割しているため、本書では「何が事実として確認できたか」「何を採用判断したか」「何がまだ未確認か」「次に何をするか」を優先して整理する。

関連文書:

- `docs/PARC2026_FINAL_ANALYSIS.md`
- `docs/PARC2026_GPU_VALIDATION_AND_NEXT_STEPS_20260829.md`
- `docs/PARC2026_PARALLEL_MODEL_DATA_PLAN_20260829.md`
- `docs/PARC2026_STATIC_QUALITY_RESULTS_20260830.md`
- `docs/PARC2026_DATASET_ABLATION_V1.md`
- `docs/PARC2026_TRAJECTORY_GROUP_LEAKAGE_V1.md`
- `docs/PARC2026_GROUP_AWARE_HOLDOUT_V2.md`

---

## 2. 現時点の結論

現時点では、**運営GPU 60時間枠で20k step本学習を開始しない**。

理由は、運営GPUで必要な最低限の成立性確認は終わっており、残る探索はColab A100等の外部GPUで安価に進められるためである。

現在の基本方針は以下。

```text
運営GPU
  → 最終候補のcheckpoint生成
  → merge / packaging
  → organizer Dockerでの最終確認
  → leaderboard / final submission

Colab A100
  → Dataset Factory
  → dataset ablation
  → model comparison
  → GA semantics確認
  → augmentation / inverse data検討
```

さらにDataset Factoryでは、従来のepisode単位eval splitを破棄し、**trajectory-group-aware split V2**を正とする。

---

## 3. 運営GPUで確認できたこと

### 3.1 公式π0.5 baselineのend-to-end評価は動作する

運営Docker相当環境で、配布済みπ0.5 submissionを使って以下を確認した。

- static validation: PASS
- policy server起動: 成功
- Track1 local evaluation: 完走
- fresh Docker再試行でserver ready: 約11秒
- local Track1 score: `0.500`

この`0.500`は開発用local評価値であり、非公開重みを含む本番Total Scoreや公式leaderboard参考値と同一ではない。

### 3.2 学習環境は構築済み

確認済み:

- Python 3.10.12
- LeRobot v0.4.4
- torch 2.10.0+cu128
- CUDA available
- FFmpeg / torchcodec利用可
- π0.5 config patch適用
- gradient accumulation patch適用
- PaliGemma gated checkpoint access確認

評価側はtorch 2.11.0+cu130等であり、学習環境と完全一致ではないため、提出前は必ず運営Dockerで再検証する。

### 3.3 運営配布学習データを読めた

`libero_combined_20hz.tar`を一時NVMeへ展開して利用できることを確認した。

実測:

- 3,028,708 frames
- 19,533 episodes
- `data/`
- `meta/`
- `videos/`

再生成可能なdataset展開・venv・HF cacheは`/opt/dlami/nvme`へ置き、`~/data`にはcheckpoint、log、submission、manifest等の保存必須物のみを置く。

### 3.4 LoRA smoke / mergeまで完走

π0.5 LoRA r16でsmoke trainingからmergeまで完走した。

確認済み:

```text
base load
→ dataset load
→ LoRA training
→ checkpoint save
→ LoRA merge
→ tokenizer同梱
→ merged checkpoint検証
```

### 3.5 VRAM probe

最初のBS16 probeはLoRA指定漏れによりfull fine-tuningになっていたため無効と判断した。

修正版LoRA r16では:

| 条件 | Learnable params | Peak VRAM | 結果 |
|---|---:|---:|---|
| BS16 | 約1.29M | 44,249 MiB | 完走 |
| BS32 | 約1.29M | 80,435 MiB | 完走 |

BS32はVRAM消費が大きい一方、今回の短時間probeでは明確な高速化を示さなかった。

現時点の第一候補:

```text
BS=16
GA=8
LoRA r=16
```

ただし、GA=8が「8 micro-stepごとに1 optimizer update」として正しく動いていることを外部GPUで確認するまで本学習には進まない。

### 3.6 GPU停止運用

停止前に`parc-home-sync data-push`を行い、Hub Control PanelからStop Serverを実行し、JupyterHubが`Start My Server`表示になることを確認した。

ブラウザを閉じるだけでは停止確認としない。

---

## 4. Colabでの開発方針

Colab A100では以下を並列で進める。

```text
Lane M: Model Selection
  π0.5
  SmolVLA
  OpenVLA-OFT
  → 同一評価契約で比較
  → 1〜2モデルへ絞る

Lane D: Dataset Factory
  inventory
  static quality
  filtering
  task balancing
  augmentation
  public supplemental
  Track3 inverse
  → dataset recipeを絞る
```

最初から3モデル×全datasetを総当たりせず、まずπ0.5を基準モデルとしてDataset Factoryを安価に比較する。

---

## 5. Public LIBERO-plus proxyで確認したDataset事実

Colabで使ったpublic proxy:

```text
lerobot/libero_plus
revision: f3f49f426d75030177b18778374005bc12ccd588
LeRobot: v3.0
fps: 20
```

実測:

- 14,347 episodes
- 2,238,036 frames
- 40 tasks
- missing episode: 0

これはDataset Factory開発用proxyであり、最終Run Aの正本は運営`libero_combined_20hz`である。

---

## 6. Static Quality Analyzer V1の結果

### 6.1 全episode解析に成功

公開proxy 14,347 / 14,347 episodesを解析した。

結果:

| Status | Episodes |
|---|---:|
| OK | 13,975 |
| REVIEW | 372 |
| Total | 14,347 |

REVIEW率:

```text
2.5929%
```

重要なのは、**hard integrity failureが0件**だったことである。

### 6.2 REVIEWはRejectではない

Static Analyzerで付けている主な指標:

- duration
- EEF path length
- displacement
- path efficiency
- jerk
- action RMS
- idle ratio
- gripper switches
- timestamp consistency
- frame gap
- invalid value

REVIEW 372件はtask内分布のoutlier候補であり、task success / collision / replayabilityを示していない。

したがって:

```text
REVIEW != Reject
```

を維持する。

### 6.3 Dataset ablation候補

Static Qualityだけを根拠に次を定義した。

- `V0_RAW`
- `V1_INTEGRITY_ONLY`
- `V1_MULTI_FLAG_PRUNED_EXPERIMENTAL`
- `V1_ALL_REVIEW_PRUNED_EXPERIMENTAL`
- `V2_SQRT_BALANCED_RAW`

`V1_INTEGRITY_ONLY`はpublic proxyではhard integrity failure=0のためV0と同一になる。

`V1_MULTI_FLAG` / `V1_ALL_REVIEW`はproduction clean確定版ではなく、controlled ablation用である。

---

## 7. 最初のDataset Manifest V1で見つかった重大な問題

### 7.1 旧split

当初はStatic Quality OKからtaskごとに2 episodesを選び、40 tasks × 2 = 80 episodesをfixed evalにした。

この時点ではepisode IDの重複やtrain/eval episode直接重複は0だった。

しかし、episode IDが別でも**同じstate+action trajectoryを共有する兄弟episode**が存在する可能性が見つかった。

### 7.2 Trajectory-group Leakage Gateを追加

各episodeについて以下をfingerprint化した。

#### exact group

```text
task index
+ sequence length
+ observation.state sequence
+ action sequence
```

#### action group

```text
task index
+ sequence length
+ action sequence
```

floatの微小差を吸収するため小数6桁へ量子化し、SHA256でgroup化した。

### 7.3 Public proxyの実測

14,347 episodesをhashしたところ:

- exact trajectory groups: 1,681
- duplicate exact groups: 1,681
- duplicate exact groupに属するepisodes: 14,347
- action groups: 1,681

つまりpublic proxyでは、全episodeが何らかのnon-visual trajectory sibling groupに属している。

ここから「なぜ重複しているか」は断定しない。visual/domain perturbation siblingsは有力仮説だが、Gateは観測されたstate+action一致だけを使う。

### 7.4 Legacy fixed evalはLeakage FAIL

旧80-episode fixed evalは79 exact groupsしか持たず、全80 eval episodesにtraining siblingが存在した。

Legacy manifestの実測:

| Variant | Train episodes | Leaked exact groups | Leaked train siblings |
|---|---:|---:|---:|
| V0_RAW | 14,267 | 79 | 642 |
| V1_INTEGRITY_ONLY | 14,267 | 79 | 642 |
| V1_MULTI_FLAG | 14,173 | 79 | 642 |
| V1_ALL_REVIEW | 13,895 | 79 | 642 |
| V2_SQRT_BALANCED | 11,242 | 79 | 489 |

結論:

```text
旧episode-level fixed evalはDataset Ablation比較に使用しない。
```

---

## 8. Group-aware Fixed Eval V2

### 8.1 split単位をepisodeからtrajectory groupへ変更

各taskで:

1. Static Quality OKのみをeval候補にする
2. `exact_group_hash`でdedupeする
3. seed固定で2 distinct groupsを選ぶ
4. 各groupから1 episodeだけeval代表にする
5. 選択groupに属する全siblingsをprotectedにする
6. protected episodesを全training variantから除外する
7. その後にfiltering / balancingを適用する

これにより、filtering/balancingの比較とleakage preventionを分離できる。

### 8.2 実測結果

public proxyで:

- fixed eval: 80 episodes
- fixed eval: 40 tasks
- protected episodes: 674
- protected fraction: 4.70%

Group-aware後のtraining variant:

| Variant | Episodes |
|---|---:|
| V0_RAW | 13,673 |
| V1_INTEGRITY_ONLY | 13,673 |
| V1_MULTI_FLAG_PRUNED_EXPERIMENTAL | 13,579 |
| V1_ALL_REVIEW_PRUNED_EXPERIMENTAL | 13,301 |
| V2_SQRT_BALANCED_RAW | 10,758 |

### 8.3 Leakage Gate PASS

Group-aware V2では全variantについて:

```text
exact_leakage_group_count = 0
action_leakage_group_count = 0
Trajectory Leakage Gate = PASS
```

したがって、public proxy上ではgroup-aware manifest V2をDataset Ablation用の正とする。

---

## 9. 現在の実装状態

### Colab

主要な実行順:

```text
00_a100_preflight.ipynb
10_pi05_smoke_ga.ipynb
20_dataset_inventory.ipynb
30_static_quality_analyzer.ipynb
40_dataset_ablation_manifests.ipynb       # legacy比較用
45_trajectory_group_leakage.ipynb         # legacy splitの問題を検出
47_group_aware_ablation_manifests.ipynb   # current split
50_pi05_dataset_ablation.ipynb             # current cheap ablation入口
```

`50_pi05_dataset_ablation.ipynb`はPR #20で以下に変更済み。

- group-aware schema v2のみ許可
- legacy manifest使用禁止
- training前にTrajectory Leakage Gateを再実行
- dataset ID / revisionを検証
- protected group / episode数を検証
- expensive run直前にも再検証
- 初期値`RUN_ABLATIONS=False`

### Training manifest safety gate

まず`RUN_ABLATIONS=False`のまま、以下3行が出ることを確認する。

```text
Group-aware Manifest Gate: PASS
Trajectory Leakage Gate: PASS
TRAINING MANIFEST SAFETY GATE: PASS
```

この確認前に4本学習を開始しない。

---

## 10. 次に実行するDataset Cheap Ablation

π0.5を固定し、以下を同条件で短く比較する。

```text
V0_RAW
V1_MULTI_FLAG_PRUNED_EXPERIMENTAL
V1_ALL_REVIEW_PRUNED_EXPERIMENTAL
V2_SQRT_BALANCED_RAW
```

固定条件:

- π0.5 base
- LoRA r16
- LR 5e-5
- BS 4
- GA 8
- effective batch 32
- 150 optimizer steps
- seed 1000

目的はwinner確定ではなく、明らかに弱いdataset recipeを落とすcheap screeningである。

training loss単独でdatasetを昇格させない。

比較時には少なくとも:

- loss
- wall time
- peak VRAM
- manifest hash
- dataset revision
- protected episode count
- fixed/simulator evaluation

を残す。

---

## 11. Model Selectionとの同期

Dataset cheap ablation後は、安定したdataset recipeを1つ固定し、以下を同一評価契約へ載せる。

- π0.5
- SmolVLA
- OpenVLA-OFT

比較軸:

### Equal Data Exposure

同程度のtraining samplesを見せる。

### Equal Wall Time

同じA100学習時間で比較する。

最終的には1〜2モデルへ絞り、運営GPUで本学習する候補を限定する。

---

## 12. Track別戦略への接続

Trackごとに別model提出も同一model提出も可能であるため、最終的に1 checkpointへ固定する必要はない。

現時点の方向:

### Track1

- success最優先
- organizer domainで安定動作
- clean/balanceがbase性能を壊さないことを確認

### Track2

- domain / viewpoint / visual perturbationへのgeneralization
- safe visual augmentation
- public supplementalの効果を比較

### Track3

Track3は既存LIBERO taskのinverse操作であることが公式開示されているため、inverse data生成を独立テーマとして扱う。

単純なaction sequence時間反転は採用しない。

候補フロー:

```text
forward task
→ inverse task definition
→ inverse initial state
→ simulatorでtrajectory生成
→ task success確認
→ replay/collision validation
→ LeRobot 20Hzへ変換
→ inverse manifestへ登録
```

---

## 13. Public proxyとOrganizer Rawを混同しない

ここまでのDataset Factoryの主要な定量値:

```text
14,347 episodes
1,681 trajectory groups
372 REVIEW
protected 674
V0 13,673
V2 10,758
```

は**public `lerobot/libero_plus` proxyでの実測**である。

運営`libero_combined_20hz`では:

```text
19,533 episodes
3,028,708 frames
```

を確認しているが、同じtrajectory group数、REVIEW率、protected率になるとは限らない。

Run A固定前にOrganizer Rawへ以下を再実行する。

```text
Inventory
→ Static Quality
→ Trajectory Grouping
→ Group-aware holdout
→ Leakage Gate
→ Dataset manifest生成
```

public proxyの件数をOrganizer Rawへハードコードしない。

---

## 14. まだ未確認の事項

以下は未完了または最終確認前。

- GA=8のoptimizer/scheduler step semantics実測
- group-aware V0/V1/V2 cheap ablation結果
- dataset recipeのsimulator success比較
- π0.5 vs SmolVLA vs OpenVLA-OFTの公平比較
- Organizer RawでのStatic Quality / trajectory grouping / group-aware manifest再生成
- submission round-tripを最新fine-tuned checkpointで完走
- Track2 augmentation効果
- Track3 inverse data factory
- L4 24GBでの最終推論VRAM / latency確認

---

## 15. 次アクションの優先順位

### Next 1: Notebook 50 Safety Gate

更新版`colab/50_pi05_dataset_ablation.ipynb`を先頭から実行し、`RUN_ABLATIONS=False`のまま3 Gate PASSを確認する。

### Next 2: Dataset cheap ablation

Gate確認後のみ`RUN_ABLATIONS=True`へ変更し、4 variantを実行する。

### Next 3: 固定/simulator評価

training lossで決めず、V0 + 有望candidateをlocal/simulator評価する。

### Next 4: Model comparison

best dataset recipeを固定してπ0.5 / SmolVLA / OpenVLA-OFTを比較する。

### Next 5: Organizer Rawへ移植

public proxyで確立したData Factoryを運営`libero_combined_20hz`へ再適用する。

### Next 6: Run A recipe固定

以下を固定してから運営GPUを再開する。

```text
model
base checkpoint revision
dataset manifest
sampling
augmentation
LoRA rank
LR
BS / GA
seed
steps
submission path
```

---

## 16. PR履歴と判断の流れ

| PR | 内容 | 主な判断 |
|---|---|---|
| #13 | Static AnalyzerのHF 429回避 | compact LeRobot v3をproxyに採用 |
| #14 | LeRobot v3 tasks.parquet対応 | v3 metadataを正式対応 |
| #15 | Static Quality結果固定 | REVIEW != Reject |
| #16 | Static Quality追加分析 | 94 multi-flagを優先確認対象へ |
| #17 | Dataset Ablation V1 | V0/V1/V2比較基盤を実装 |
| #18 | Trajectory Leakage Gate | episode-level splitの危険を検査 |
| #19 | Group-aware Fixed Eval V2 | trajectory group単位splitへ変更 |
| #20 | π0.5 ablation V2専用化 | legacy manifestでの学習を禁止 |

重要な流れ:

```text
Static Qualityだけでは削除根拠が弱い
→ controlled ablationへ
→ episode-level evalを作る
→ trajectory sibling leakageを発見
→ legacy splitをReject
→ group-aware holdoutへ移行
→ leakage 0を確認
→ cheap ablation開始可能な状態まで到達
```

---

## 17. 現在地

2026-08-30時点で、Dataset Factoryは「データを眺める段階」から「leakageを制御した再現可能ablationを実行できる段階」まで進んだ。

現在はまだ**本学習を始める段階ではなく、安価な実験でdataset/model候補を狭める段階**である。

運営GPU 60時間は探索用ではなく、evidenceが揃った最終候補へ集中投入する。
