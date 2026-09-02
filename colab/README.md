# Colab A100 benchmark workspace

`colab/` is the entry point for experiments that should be completed before spending the PARC organizer GPU budget.

## Goal

Use a Colab A100 for model comparison, data ablation, Track3 inverse-data research, and submission preflight. Keep the organizer GPU for the final candidate training/evaluation runs. Large public training datasets should be prefetched on a CPU runtime and persisted to Google Drive before allocating an A100.

## Repository roles

- `py_AI`: experiment definitions, manifests, benchmark outputs, PARC evaluation/submission code.
- `/content/vendor/*`: disposable upstream clones such as LeRobot/OpenVLA-OFT. Do not vendor them into this repository.
- `/content/cache/*`: Hugging Face/model caches and temporary datasets.
- `/content/parc2026/datasets/*`: fast local staging area used while training/evaluating in the current Colab runtime.
- Google Drive: persistent source for prefetched datasets/checkpoints/large artifacts. Do not train directly from Drive when local staging is practical.

## Notebook order

`00_a100_preflight.ipynb` remains the recommended first notebook, but the main notebooks are self-contained. For the public `lerobot/libero_plus` training proxy, use `48` on a CPU runtime first, then switch to A100 and use `49` before `50`.

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
   - legacy episode-level fixed splitを生成するV1
   - Static Analyzer出力がなければcompact public v3から自動生成
   - V0/V1/V2のfiltering / balancing定義を固定する
   - **trajectory leakageが確認されたため、このV1 splitを最終比較用evalとしては使わない**
6. `45_trajectory_group_leakage.ipynb`
   - self-contained preflight。GPU不要
   - compact public v3のmeta + trajectory parquetだけを使い、画像・動画はdownloadしない
   - task + state + action sequenceを量子化SHA256し、`exact_group` を作る
   - task + action sequenceだけの `action_group` もsoft signalとして作る
   - fixed evalと各training manifestのtrajectory-group overlapを検査
   - exact overlapが1件でもあれば `Trajectory Leakage Gate: FAIL`
7. `47_group_aware_ablation_manifests.ipynb`
   - self-contained preflight。GPU不要
   - 2 distinct exact trajectory groups / taskを固定evalとしてseed固定で選ぶ
   - 各groupから1 OK episodeだけをeval代表にする
   - 選ばれたgroupの全siblingsをすべてのtraining variantから除外する
   - V0/V1/V2をgroup exclusion後に再生成する
   - `--fail-on-exact-leakage` でもう一度45相当の検査を行う
   - `Group-aware Manifest Gate: PASS` と `Trajectory Leakage Gate: PASS` の両方を必須にする
8. `48_prefetch_training_dataset_to_drive.ipynb`
   - **CPU runtime専用**。A100を使わない
   - public `lerobot/libero_plus` のrevisionをpinし、meta/data/videosをGoogle Driveへresumable download
   - 既定revision: `f3f49f426d75030177b18778374005bc12ccd588`
   - 保存先: `MyDrive/parc2026-cache/datasets/lerobot_libero_plus_v3_train`
   - 全required fileの存在/sizeを検証して `.parc_prefetch_complete.json` を保存
   - `DRIVE PREFETCH GATE: PASS` がA100へ切り替える条件
9. `49_stage_training_dataset_from_drive.ipynb`
   - A100 runtime開始直後に実行
   - Drive completion manifestを検証してから `/content/parc2026/datasets/public_libero_plus_v3_train` へlocal staging
   - Hugging Face local-dir cache metadataもstageし、Notebook 50の大容量再downloadを避ける
   - Drive直読みではなくlocal diskから学習する
   - `LOCAL STAGE GATE: PASS` 後に同runtimeでNotebook 50へ進む
10. `50_pi05_dataset_ablation.ipynb`
   - π0.5 group-aware cheap screening本体
   - legacy `dataset_ablation_manifests_v1` をtrainingには使用せず、`dataset_ablation_manifests_v2_group_aware` のみ許可
   - fresh runtimeでもStatic Quality → trajectory grouping → group-aware manifestをself-containedで再生成可能
   - training直前にschema v2 / `group_aware=true` / dataset revision / protected group数を再検証
   - `check_trajectory_group_leakage.py --fail-on-exact-leakage` を必須Gateとして再実行
   - public proxyでは protected 674 episodes、V0=13,673 / Multi-flag=13,579 / All-review=13,301 / sqrt-balanced=10,758 を再現Gateにする
   - training用public fallbackはcompact `lerobot/libero_plus` v3をrevision pinして取得
   - `48 → 49` 済みならlocal-dir cache hitになり、大容量videoを再downloadしない
   - LeRobot v0.4.4 native `DatasetConfig.episodes` でdataset copy無しにvariantを切替
   - π0.5 LoRA / seed / optimizer steps / effective batchを固定してcheap screening
   - 初期値は `BS=4 / GA=8 / 150 optimizer steps`
   - `RUN_ABLATIONS=False` を安全gateとし、設定確認後に明示的にTrueへする
   - loss / wall time / peak VRAM / manifest hash / protected episode count / dataset revision / git SHAを保存する
   - training lossだけで最終選定しない

`10` はModel Selection側、`20`〜`50` はDataset Factory側です。モデル側のGateとDataset Factory側のinventory/quality/leakage/ablationを並列に進め、固定評価の同期点で合流します。

## Organizer vs public dataset

公開fallbackはColab開発を止めないためのものです。公開LIBERO-plusの結果を、運営 `libero_combined_20hz` の正本Inventoryや最終学習結果として扱ってはいけません。

公開LIBERO-plusでpipelineを先に完成させてよいですが、Run A固定前には運営combined datasetへ同じInventory / Static Quality / Manifest / Trajectory-group Leakage / Group-aware split pipelineを再適用します。`success / collision / replayability` はconverted LeRobot dataだけから推測せず、raw simulator stateを確保できた範囲でReplay Validatorを別途実行します。

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
5. Generate legacy fixed eval holdout + V0/V1/V2 episode manifests.
6. Run Trajectory-group Leakage Gate.
7. If legacy Gate FAILs, generate group-aware holdout and exclude all eval-group siblings from training.
8. Re-run Trajectory-group Leakage Gate and require PASS.
9. Prefetch the public training proxy to Drive on CPU runtime (`48`), then local-stage it on A100 (`49`).
10. Run group-aware π0.5 cheap V0 Raw / V1 Clean / V2 sqrt-balanced screening in Notebook 50.
11. Send V0 + promising dataset variants to fixed local/simulator evaluation; do not promote from loss alone.
12. Add SmolVLA and OpenVLA-OFT on the same dataset/eval split.
13. Compare model choice and later V3/V4/V5 dataset ablations separately.
14. Research simulator-valid Track3 inverse/reversed demonstrations.
15. Promote only the strongest configuration to organizer-GPU training.
