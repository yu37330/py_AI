# PARC2026 Dataset Ablation V1

Date: 2026-08-30

## Purpose

Static Quality Analyzer V1で得たepisode-level evidenceを、π0.5の短い比較実験へ接続する。

この段階では「cleaning」と「task balancing」を混ぜない。モデル、seed、optimizer step、LoRA設定を固定し、dataset variantだけを変える。

また、training lossだけで最終datasetを決めない。Cheap ablationはscreeningであり、promotionには固定評価またはsimulator success evidenceを要求する。

## Source proxy

Colab開発用proxy:

- dataset: `lerobot/libero_plus`
- revision: Static Quality runでpinしたHub SHAをmanifestへ保存
- LeRobot codebase: v3.0
- fps: 20
- tasks: 40
- source episodes: 14,347
- source frames: 2,238,036
- Static Quality REVIEW: 372
- hard integrity failure: 0

Run A固定前には運営 `libero_combined_20hz` で同じpipelineを再実行する。

## Fixed eval holdout

全training variantから同じholdoutを除外する。

- seed: `20260830`
- 2 episodes / task
- 40 tasks -> 80 episodes
- eligibility: Static Quality `OK` only
- selection: deterministic task-stratified sampling

公開proxyの実測では、holdout後のV0 training poolは:

- 14,267 episodes
- 2,224,619 frames

holdoutはtraining lossの比較そのものでは使わず、後続の固定評価を同一episodeで行うための契約として先にfreezeする。

## Training variants

### V0_RAW

固定holdoutのみ除外。Filteringもbalancingもしない。

Public proxy expected:

- 14,267 episodes
- 2,224,619 frames

### V1_INTEGRITY_ONLY

NaN/Inf、timestamp非単調、frame gapなどhard integrity failureのみ除外。

今回のpublic proxyではhard integrity failure=0なのでV0と完全に同一。Cheap ablationでは重複runをskipする。

### V1_MULTI_FLAG_PRUNED_EXPERIMENTAL

Static Qualityの `quality_flag_count >= 2` のepisodeを除外する。

Public proxy expected:

- 14,173 episodes
- 2,206,235 frames

これは「壊れたデータ」と断定した除外ではなく、statistical pruningの実験候補。

### V1_ALL_REVIEW_PRUNED_EXPERIMENTAL

Static QualityのREVIEW 372件をすべて除外するaggressive variant。

Public proxy expected:

- 13,895 episodes
- 2,153,927 frames

長い/難しい成功demoまで削るリスクがあるため、効果が明確でない限り採用しない。

### V2_SQRT_BALANCED_RAW

Filteringはせず、固定holdout後のRaw poolからtask exposureだけをsqrt方向へ平坦化する。

LeRobot trainingは選択dataset内のframeをsampleするため、episode数ではなくtaskごとのframe数を基準にする。

式:

```text
target_frames(task) = sqrt(min_task_frames * raw_task_frames)
```

各task内のepisodeをseed固定でshuffleし、累積frame数がtargetに到達するまでepisodeを採用する。

Public proxyの現在のmanifest generatorでは概ね:

- 11,242 episodes
- 1,692,630 frames

となる。重要なのは件数そのものではなく、Rawよりtask frame shareの偏りが下がること。

## Why subset-based balancing

LeRobot v0.4.4にはnative `DatasetConfig.episodes` があるため、今回のablationはepisode subsetだけで実装する。

利点:

- dataset copy不要
- trainerへのsampling patch不要
- manifest hashで再現可能
- organizer datasetへ同じ方法を移植しやすい
- 重複episode oversamplingを導入しない

## Cheap screening contract

π0.5側のdefault:

- base: `lerobot/pi05_libero_base`
- LoRA: r=16
- LR: 5e-5
- batch size: 4
- gradient accumulation: 8
- effective batch: 32
- optimizer steps: 150
- seed: 1000
- n_action_steps: 10
- num_inference_steps: 10

Run order:

1. V0_RAW
2. V1_MULTI_FLAG_PRUNED_EXPERIMENTAL
3. V1_ALL_REVIEW_PRUNED_EXPERIMENTAL
4. V2_SQRT_BALANCED_RAW

V1_INTEGRITY_ONLYはV0と同一ならskipする。

各runで最低限保存するもの:

- manifest path/hash
- episode/frame count
- model/training config
- optimizer steps/effective batch
- wall time
- peak VRAM
- final/last-window training loss（best effort, screening参考値のみ）
- checkpoint path

## Promotion gate

Cheap screeningだけでwinnerを決めない。

次の固定評価へ最低限送る:

- V0_RAW
- cleaning系のbest candidate（差が無ければV0のみ）
- V2_SQRT_BALANCED_RAW（trainingが正常なら）

固定評価では、同じtask/episode splitと同じmodel settingsを使い、可能ならsimulator successを優先する。

## Colab entry points

- `colab/40_dataset_ablation_manifests.ipynb`
  - static metricsが無ければ自動生成
  - fixed eval + V0/V1/V2 manifests生成
  - manifest gate
- `colab/50_pi05_dataset_ablation.ipynb`
  - compact public v3 training dataset取得
  - π0.5 training env setup
  - explicit `RUN_ABLATIONS` safety gate
  - 4 variant cheap screening

## Organizer promotion

運営GPUへ持ち込む前に:

1. organizer `libero_combined_20hz` でInventory再実行
2. Static Quality Analyzer再実行
3. fixed eval + V0/V1/V2 manifest再生成
4. public proxyで有望だったdataset ruleをorganizer dataへ再適用
5. organizer short smokeでschema/episode selection確認
6. Run A本学習

Public proxyのepisode IDをorganizer datasetへそのまま流用してはいけない。
