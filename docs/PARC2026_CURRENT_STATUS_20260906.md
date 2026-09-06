# PARC2026 現状まとめ・判断ログ・次アクション（2026-09-06）

## 1. この文書の位置づけ

本書は `docs/PARC2026_CURRENT_STATUS_20260830.md`、
`docs/PARC2026_PARALLEL_MODEL_DATA_PLAN_20260829.md`、
`docs/PARC2026_TRACK_STRATEGY_AND_OFFLINE_PREFLIGHT_20260829.md` の後続となる現在地のSource of Truthである。

2026-09-05〜2026-09-06に実施した以下を反映する。

- group-aware V2 dataset cheap ablation
- π0.5 LoRA 150 optimizer-step screening
- L4 24GBでの固定Simulator評価
- dataset recipeの暫定順位
- model comparison / Track3 inverse対応を含む本学習前の残Gate

重要: 今回の結果は「π0.5の20k本学習開始」を意味しない。Dataset FactoryのScreening Gateを一段進めた結果であり、model selectionとTrack3 inverse検証がまだ残る。

---

## 2. 現時点の結論

現在地は次のとおり。

```text
Dataset Factory
  inventory / quality / leakage-safe split
  → 4 dataset recipe × π0.5 150-step screening
  → fixed simulator eval
  → V1_MULTI と V2_SQRT が同率首位
  → top-2 tie-break                    ★ 次

Model Selection
  π0.5 / SmolVLA / OpenVLA-OFT
  → best provisional datasetで公平比較
  → 最大2モデルへshortlist            ★ 未実施

Generalization / Track3
  safe augmentation / supplemental
  → inverse data 0/5/10/20%比較        ★ 未実施

Final Recipe
  Best Model × Best Dataset Recipe
  → Run A 20k LoRA
  → Track1/2/3評価
  → Track3だけ弱ければRun B continuation
```

したがって、**現時点では20k full trainingを開始しない**。

---

## 3. Group-aware Dataset Factoryの成立状況

Public LIBERO-plus proxy:

```text
dataset: lerobot/libero_plus
revision: f3f49f426d75030177b18778374005bc12ccd588
episodes: 14,347
tasks: 40
fps: 20
```

Static Quality Analyzerでは372 episodes（2.59%）をREVIEW候補として検出したが、hard integrity failureは0件だった。

Legacy episode-level splitはtrajectory sibling leakageを起こしたため廃止し、group-aware V2を正とした。

Group-aware V2ではfixed eval groupのsiblingsを全training variantからprotectし、全variantで以下を達成した。

```text
exact_leakage_group_count = 0
action_leakage_group_count = 0
Trajectory Leakage Gate = PASS
```

Training episode数:

| Variant | Episodes | 意図 |
|---|---:|---|
| V0_RAW | 13,673 | post-holdout raw baseline |
| V1_MULTI_FLAG_PRUNED_EXPERIMENTAL | 13,579 | quality flagが複数立つepisodeのみ除外 |
| V1_ALL_REVIEW_PRUNED_EXPERIMENTAL | 13,301 | REVIEW候補を全除外する攻めたablation |
| V2_SQRT_BALANCED_RAW | 10,758 | raw poolをtask frame exposureでsqrt balance |

このpublic datasetはDataset Factory開発・screening用proxyであり、**最終Run Aの正本は運営 `libero_combined_20hz`** とする。最終recipeは運営datasetへ再適用してmanifestを固定する。

---

## 4. π0.5 Dataset Cheap Ablation結果

固定条件:

```text
model: π0.5
base: lerobot/pi05_libero_base
adaptation: LoRA r16
optimizer steps: 150
batch size: 4
grad accumulation: 8
effective batch: 32
lr: 5e-5
seed: 1000
```

結果:

| Variant | Final logged loss* | Peak VRAM | Wall time |
|---|---:|---:|---:|
| V0_RAW | 0.458 | 16,930 MiB | 655s |
| V1_MULTI | **0.439** | 16,930 MiB | 604s |
| V1_ALL_REVIEW | 0.473 | 16,930 MiB | 598s |
| V2_SQRT | 0.458 | 16,930 MiB | **546s** |

`*` best-effort log値。promotionはtraining lossだけで決めない。

この時点ではV1_MULTIがloss上は最良だったが、正式判断をSimulator successへ委ねた。

---

## 5. L4 Fixed Simulator Evaluation結果

### 5.1 評価契約

新しいColab L4 24GB runtimeで、4 adapterを同一条件で逐次評価した。

```text
GPU: NVIDIA L4 24GB
track: Track1
models: 4
tasks/model: 4
episodes/task: 8
episodes/model: 32
total episodes: 128
seed: 20260905
max steps: 300
```

各adapterは1つずつbaseへmergeし、policy server起動、fixed simulator evaluation、Drive保存、merged checkpoint削除の順で処理した。

全run完走:

```text
UNATTENDED FIXED SIM EVAL: PASS
total elapsed: 03:43:10
```

### 5.2 結果

| Rank | Variant | Overall score | Success |
|---|---|---:|---:|
| 1 | **V1_MULTI** | **0.53125** | **17 / 32** |
| 1 | **V2_SQRT** | **0.53125** | **17 / 32** |
| 3 | V0_RAW | 0.43750 | 14 / 32 |
| 3 | V1_ALL_REVIEW | 0.43750 | 14 / 32 |

V1_MULTI / V2_SQRTはV0_RAW比で +0.09375、+3 successful episodes / 32。

V2_SQRT task-level:

| Task | Success |
|---|---:|
| black bowl in top drawer → plate | 37.5% |
| tomato sauce → basket | 0.0% |
| milk → basket | 87.5% |
| bowl → stove | 87.5% |

Machine-readable record:

- `experiments/results/pi05_dataset_screening_fixed_sim_20260906.json`

### 5.3 解釈

確定したこと:

- V1_MULTIとV2_SQRTは、今回のscreening契約ではRAWより有望。
- REVIEW候補をすべて削除するV1_ALL_REVIEWは改善を示さなかった。
- 「異常候補を多く削れば良い」とは言えない。
- training lossだけでなくSimulator successを必須にした判断は妥当だった。

まだ確定していないこと:

- V1_MULTIとV2_SQRTのどちらが本当に優位か。
- 4 tasks × 8 episodesのscreening結果が40 tasks全体へ一般化するか。
- このdataset recipeがSmolVLA / OpenVLA-OFTでも同様に有効か。
- public proxyでのrecipeがOrganizer Rawでも同程度に有効か。

---

## 6. Dataset selectionの次Gate

### Gate D10: Top-2 tie-break

対象:

```text
V1_MULTI_FLAG_PRUNED_EXPERIMENTAL
V2_SQRT_BALANCED_RAW
```

別seedで同一fixed simulator contractを再実行する。

判定:

1. success rateを最優先
2. 同率ならtask別の安定性を見る
3. なお同率ならtraining loss / dataset size / wall timeを二次指標にする

現時点の二次指標ではV1_MULTIがloss 0.439で優位、V2_SQRTはdatasetが小さくscreening wall timeが短い。ただしこれだけでは昇格を決めない。

Tie-break後に「Model Selection用のprovisional best dataset」を1つ固定する。

---

## 7. Model Selectionはまだ未実施

正式候補:

1. π0.5
2. SmolVLA
3. OpenVLA-OFT

X-VLA / MolmoAct2等は過去に候補検討したが、現在のformal comparison setには入れていない。GPU探索を広げすぎないため、まず上記3本でStage S2を行う。

### 共通比較contract

全モデルで最低限固定する。

- same provisional-best dataset manifest
- same eval tasks / episodes
- same seed set
- exact upstream revision
- exact base checkpoint revision
- success rate
- steps-to-success
- episode duration
- inference latency
- peak inference VRAM
- train wall time
- peak train VRAM

### 2つの公平比較

#### Equal Data Exposure

同程度のtraining samplesを見せ、sample efficiencyを比較する。

#### Equal Wall Time

同じA100時間budgetで学習し、GPUコスト当たり性能を比較する。

Promotionは最大2モデル。

**π0.5だけを今すぐ20kへ進めない。** Dataset screening後にmodel comparisonを挟む。

---

## 8. Track3 inverse / reversed task対応

Track3対策もまだ未実施。

禁止:

```text
forward action sequenceを単純に時間反転
→ inverseの正解demoとして使用
```

これはinitial state、contact、gripper状態、object placement等の整合を保証しないため採用しない。

採用する生成フロー:

```text
forward task
→ inverse task definition
→ inverse task用initial state
→ simulatorでexpert / script / policy実行
→ task success確認
→ replay validation
→ LeRobot 20Hz schemaへ変換
→ inverse dataset manifestへ登録
```

比較するinverse混合率:

```text
0%
5%
10%
20%
```

評価ではTrack3 successを主指標とし、Track1/2 regressionも必ず確認する。

---

## 9. Run A / Run B戦略

### Run A: 3 Track共通モデル

最終的に、

```text
Best Model
× Best normal Dataset Recipe
× fixed training recipe
```

を1本のRun Aとして20k本学習する。

Run AはTrack1専用ではなく、Track1/2/3共通の第一提出候補。

### Run B: Track3特化 continuation

Run A評価後、以下を満たす場合のみ発動する。

- Track1/2は十分だがTrack3だけ明確に弱い
- inverse local evalで弱点が再現する
- 5/10/20% inverse小規模実験で改善Evidenceがある

Evidenceが無ければRun Bは行わず、GPU時間をRun A continuation / checkpoint比較 /再評価へ残す。

---

## 10. 20k Full Trainingまでの正しい順序

```text
1. V1_MULTI vs V2_SQRT tie-break
2. provisional best datasetを固定
3. π0.5 / SmolVLA / OpenVLA-OFT model comparison
4. 最大2モデルへshortlist
5. shortlistでaugmentation / public supplementalを必要最小限比較
6. Track3 inverse factory: 0/5/10/20% screening
7. Organizer Rawへ最終dataset recipeを再適用
8. Run A manifestを固定
9. 20k LoRA full training
10. intermediate checkpointsをfixed simulatorで比較
11. best checkpointをmerge
12. L4 / organizer-Docker相当環境でTrack1/2/3評価
13. Track3だけ弱い場合のみRun B continuation
14. package / validate / submission
```

これが2026-09-06時点の正しい本学習前フローである。

---

## 11. 次アクション

### P0

- [ ] V1_MULTI vs V2_SQRT tie-break notebook / run
- [ ] tie-break resultをmachine-readable JSONへ保存
- [ ] provisional dataset winnerをdecision logへ固定

### P1

- [ ] common model benchmark notebookを実装
- [ ] π0.5 / SmolVLA / OpenVLA-OFT bring-up
- [ ] Equal Data Exposure比較
- [ ] Equal Wall Time比較
- [ ] model shortlist最大2本

### P2

- [ ] Track3 inverse factory notebook/tool
- [ ] simulator-valid inverse demos生成
- [ ] inverse 0/5/10/20% ablation
- [ ] Track1/2 regression gate

### P3

- [ ] Organizer `libero_combined_20hz`へfinal recipe再適用
- [ ] Run A exact command / manifest固定
- [ ] 20k full training

---

## 12. 関連ファイル

Existing strategy / status:

- `docs/PARC2026_CURRENT_STATUS_20260830.md`
- `docs/PARC2026_PARALLEL_MODEL_DATA_PLAN_20260829.md`
- `docs/PARC2026_TRACK_STRATEGY_AND_OFFLINE_PREFLIGHT_20260829.md`
- `docs/PARC2026_GROUP_AWARE_HOLDOUT_V2.md`

Current execution assets:

- `colab/50_pi05_dataset_ablation.ipynb`
- `colab/60_pi05_fixed_sim_eval_l4.ipynb`
- `examples/pi05_libero_finetune/scripts/full_pi05_lora.sh`
- `examples/pi05_libero_finetune/scripts/merge_lora.py`

Current result record:

- `experiments/results/pi05_dataset_screening_fixed_sim_20260906.json`

Updated executable plan:

- `experiments/plans/parallel_model_data_v2.yaml`
