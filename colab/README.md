# Colab A100 benchmark workspace

`colab/` is the entry point for experiments that should be completed before spending the PARC organizer GPU budget.

## Goal

Use a Colab A100 for model comparison, data ablation, Track3 inverse-data research, and submission preflight. Keep the organizer GPU for the final candidate training/evaluation runs.

## Repository roles

- `py_AI`: experiment definitions, manifests, benchmark outputs, PARC evaluation/submission code.
- `/content/vendor/*`: disposable upstream clones such as LeRobot/OpenVLA-OFT. Do not vendor them into this repository.
- `/content/cache/*`: Hugging Face/model caches and temporary datasets.
- `/content/parc2026/datasets/*`: disposable organizer/public datasets used in Colab.
- Google Drive: optional persistence for checkpoints and large generated artifacts only.

## Notebook order

`00_a100_preflight.ipynb` remains the recommended first notebook, but `10`, `20`, `30`, `40`, `45`, and `50` are **self-contained**. Each repeats the essential preflight steps at the beginning: runtime check, `/content/parc2026` workspace creation, and `py_AI` clone/update. Therefore a fresh Colab runtime can open them directly without failing because `00` was not executed in the same runtime.

1. `00_a100_preflight.ipynb`
   - GPU / Python / workspace確認
   - `/content/parc2026/py_AI` clone
2. `10_pi05_smoke_ga.ipynb`
   - self-contained preflight
   - Python 3.10学習環境をColab側へ構築
   - dataset priority: explicit `PI05_DATASET_ROOT` → organizer combined → public fallback
   - π0.5 LoRAの短いsmoke
   - runtime traceで `GA=8` が8 micro-stepごとに1 optimizer updateになることを検証
   - 20-step + mergeは明示的に有効化した場合だけ実行
3. `20_dataset_inventory.ipynb`
   - self-contained preflight
   - dataset priority: explicit `PARC_DATASET_ROOT` → organizer combined → public `Sylvest/libero_plus_lerobot`
   - public fallbackはInventoryに必要なmetadataだけを取得し、動画15GB級をdownloadしない
   - LeRobot metadataからtask / episode構成をCSV化
   - raw / uniform / sqrt-balancedのsampling候補を生成
   - success/collision/replayabilityはこの段階では推測しない
4. `30_static_quality_analyzer.ipynb`
   - self-contained preflight
   - dataset priority: explicit `PARC_DATASET_ROOT` → organizer combined → public `lerobot/libero_plus` v3
   - Static Analyzerのpublic fallbackだけは、14,347個の小parquetを持つv2.1ではなく、同じ14,347 episodes / 2,238,036 frames / 40 tasks / 20Hzを少数parquetへrepackしたv3を使う
   - Hub revisionを取得してpinし、parquetを1ファイルずつ逐次downloadするため、14k requestsによる429を避ける
   - videosはdownloadしない
   - analyzerはv2の1-episode-per-fileとv3のmulti-episode parquetの両方に対応
   - 全episodeについて EEF path/displacement、raw/smoothed jerk、action RMS、idle ratio、gripper switches、timestamp/frame integrityを計測
   - task-relative robust-zで `REVIEW` 候補を作るが、自動Rejectはしない
   - task success / collision / replayabilityはReplay Validatorまで保留する
5. `40_dataset_ablation_manifests.ipynb`
   - self-contained preflight
   - Static Analyzer出力がなければcompact public v3から自動生成
   - 2 episodes/taskのOK-only固定holdoutを作り、すべてのtraining variantから除外
   - `V0_RAW`, `V1_INTEGRITY_ONLY`, `V1_MULTI_FLAG_PRUNED_EXPERIMENTAL`, `V1_ALL_REVIEW_PRUNED_EXPERIMENTAL`, `V2_SQRT_BALANCED_RAW` をJSON manifest化
   - V2はLeRobotのframe-level samplingを考慮し、task frame exposureを `sqrt(min_frames * task_frames)` へdownsample
   - episode IDsとhash、task/frame distributionを固定して再現可能にする
6. `45_trajectory_group_leakage.ipynb`
   - self-contained preflight。GPU不要
   - compact public v3のmeta + trajectory parquetだけを使い、画像・動画はdownloadしない
   - task + state + action sequenceを量子化SHA256し、`exact_group` を作る
   - task + action sequenceだけの `action_group` もsoft signalとして作る
   - fixed evalと各training manifestのtrajectory-group overlapを検査
   - exact overlapが1件でもあれば `Trajectory Leakage Gate: FAIL`
   - FAIL時は50を回さず、group-aware fixed eval / train exclusionへmanifestを再設計する
7. `50_pi05_dataset_ablation.ipynb`
   - self-contained preflight
   - training用public fallbackはcompact `lerobot/libero_plus` v3をrevision pinして取得
   - static metrics/manifestsが無ければ自動生成
   - LeRobot v0.4.4 native `DatasetConfig.episodes` でdataset copy無しにvariantを切替
   - π0.5 LoRA / seed / optimizer steps / effective batchを固定してcheap screening
   - 初期値は `BS=4 / GA=8 / 150 optimizer steps`
   - `RUN_ABLATIONS=False` を安全gateとし、設定確認後に明示的にTrueへする
   - V1_INTEGRITY_ONLYがV0と同一なら重複runをskip
   - loss / wall time / peak VRAM / manifest hashを保存するが、training lossだけで最終選定しない

`10` はModel Selection側、`20`〜`50` はDataset Factory側です。モデル側のGateとDataset Factory側のinventory/quality/leakage/ablationを並列に進め、固定評価の同期点で合流します。

## Organizer vs public dataset

公開fallbackはColab開発を止めないためのものです。公開LIBERO-plusの結果を、運営 `libero_combined_20hz` の正本Inventoryや最終学習結果として扱ってはいけません。

`20`〜`50` は公開LIBERO-plusでpipelineを先に完成させてよいですが、Run A固定前には運営combined datasetへ同じInventory / Static Quality / Manifest / Trajectory-group Leakage pipelineを再適用します。`success / collision / replayability` はconverted LeRobot dataだけから推測せず、raw simulator stateを確保できた範囲でReplay Validatorを別途実行します。

## Comparison contract

Every comparable run must pin:

1. git SHA / upstream model revision,
2. dataset manifest,
3. eval split,
4. seed,
5. effective batch size and optimizer-step semantics,
6. wall time, peak VRAM, latency and task metrics.

Do not select a model or dataset using training loss alone. Prefer simulator success plus the PARC-adjacent metrics that can be reproduced locally.

## Planned sequence

1. Reproduce pi0.5 LoRA smoke and verify GA semantics.
2. Inventory public metadata now; replace with organizer-data inventory when metadata is available.
3. Run Static Quality Analyzer V1 and create task-relative REVIEW queue.
4. Freeze safe integrity filtering and experimental statistical-pruning candidates.
5. Generate fixed eval holdout + V0/V1/V2 episode manifests.
6. Run Trajectory-group Leakage Gate. If FAIL, rebuild group-aware holdout before training.
7. Run cheap V0 Raw / V1 Clean / V2 sqrt-balanced screening with pi0.5 fixed only after the leakage Gate passes.
8. Send V0 + promising dataset variants to fixed local/simulator evaluation; do not promote from loss alone.
9. Add SmolVLA and OpenVLA-OFT on the same dataset/eval split.
10. Compare model choice and later V3/V4/V5 dataset ablations separately.
11. Research simulator-valid Track3 inverse/reversed demonstrations.
12. Promote only the strongest configuration to organizer-GPU training.
