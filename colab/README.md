# Colab A100 benchmark workspace

`colab/` is the entry point for experiments that should be completed before spending the PARC organizer GPU budget.

## Goal

Use a Colab A100 for model comparison, data ablation, Track3 inverse-data research, and submission preflight. Keep the organizer GPU for final candidate training/evaluation. Large public training datasets should be prefetched on a CPU runtime and persisted to Google Drive before allocating an A100.

## Repository roles

- `py_AI`: experiment definitions, manifests, benchmark outputs, PARC evaluation/submission code.
- `/content/vendor/*`: disposable upstream clones such as LeRobot/OpenVLA-OFT.
- `/content/cache/*`: model caches, training venv, checkpoints, and logs in the current Colab runtime.
- `/content/parc2026/datasets/*`: fast local staging area used while training/evaluating.
- Google Drive: persistent source for prefetched datasets/checkpoints/large artifacts. Do not train directly from Drive when local staging is practical.

## Notebook order

`00_a100_preflight.ipynb` remains the recommended first notebook, but the main notebooks are self-contained. For the public `lerobot/libero_plus` training proxy, use **`48` on a CPU runtime, then switch to A100 and run `50` directly**. The former standalone `49_stage_training_dataset_from_drive.ipynb` has been folded into `50`.

1. `00_a100_preflight.ipynb`
   - GPU / Python / workspace確認
   - `/content/parc2026/py_AI` clone
2. `10_pi05_smoke_ga.ipynb`
   - π0.5 LoRA smoke / GA semantics validation
   - public fallback is only a pipeline proxy; organizer combined must be rechecked before Run A
3. `20_dataset_inventory.ipynb`
   - task / episode inventory
   - raw / uniform / sqrt-balanced sampling candidates
4. `30_static_quality_analyzer.ipynb`
   - compact public `lerobot/libero_plus` v3 fallback
   - EEF path/displacement, jerk, action RMS, idle ratio, gripper switches, timestamp/frame integrity
   - task-relative `REVIEW` queue; not automatic Reject
5. `40_dataset_ablation_manifests.ipynb`
   - legacy episode-level V0/V1/V2 manifests
   - trajectory leakageが確認されたためlegacy splitは最終比較用evalに使わない
6. `45_trajectory_group_leakage.ipynb`
   - task + state + action sequenceをhashしてtrajectory group leakageを検査
7. `47_group_aware_ablation_manifests.ipynb`
   - 2 distinct exact trajectory groups / taskをfixed evalへ
   - selected groupの全siblingsをtrainingから除外
   - `Group-aware Manifest Gate: PASS` / `Trajectory Leakage Gate: PASS` を必須化
8. `48_prefetch_training_dataset_to_drive.ipynb`
   - **CPU runtime専用**
   - public `lerobot/libero_plus` revisionをpin
   - meta/data/videosを `MyDrive/parc2026-cache/datasets/lerobot_libero_plus_v3_train` へresumable download
   - completion marker `.parc_prefetch_complete.json` を保存
   - `DRIVE PREFETCH GATE: PASS` がA100へ切り替える条件
9. `50_pi05_dataset_ablation.ipynb`
   - **旧49のDrive → `/content` stagingを内包**
   - local stage済みならcopyをskip
   - Drive prefetchが無い場合はA100上で約16GBのHF video downloadへフォールバックせず停止
   - completion manifestに列挙されたdataset本体だけをstageし、`.cache/huggingface` のlock/metadataはcopyしない
   - legacy manifestをtrainingには使用せず `dataset_ablation_manifests_v2_group_aware` のみ許可
   - Static Quality → trajectory grouping → group-aware manifestをself-containedで再生成可能
   - `check_trajectory_group_leakage.py --fail-on-exact-leakage` を必須Gateとして再実行
   - public proxyでは protected 674 episodes、V0=13,673 / Multi-flag=13,579 / All-review=13,301 / sqrt-balanced=10,758 を再現Gateにする
   - LeRobot v0.4.4 native `DatasetConfig.episodes` でvariantを切替
   - **GA=8 runtime trace Gateも50内で実行**し、24 backward micro-step → 3 optimizer updateを検証
   - cheap screening default: `BS=4 / GA=8 / 150 optimizer steps / seed=1000`
   - `RUN_ABLATIONS=False` が初期値。全Gate確認後だけTrueへする
   - loss / wall time / peak VRAM / manifest hash / protected episode count / dataset revision / git SHAを保存
   - training lossだけで最終選定しない

`10` はModel Selection側、`20`〜`50` はDataset Factory側です。モデル側のGateとDataset Factory側のinventory/quality/leakage/ablationを並列に進め、固定評価の同期点で合流します。

## Organizer vs public dataset

公開fallbackはColab開発を止めないためのproxyです。公開LIBERO-plusの結果を、運営 `libero_combined_20hz` の正本Inventoryや最終学習結果として扱ってはいけません。

Run A固定前には運営combined datasetへ同じInventory / Static Quality / Manifest / Trajectory-group Leakage / Group-aware split pipelineを再適用します。`success / collision / replayability` はconverted LeRobot dataだけから推測せず、raw simulator stateを確保できた範囲でReplay Validatorを別途実行します。

## Comparison contract

Every comparable run must pin:

1. git SHA / upstream model revision,
2. dataset manifest,
3. eval split,
4. seed,
5. effective batch size and optimizer-step semantics,
6. wall time, peak VRAM, latency and task metrics.

Do not select a model or dataset using training loss alone. Prefer simulator success plus PARC-adjacent metrics that can be reproduced locally.

## Planned sequence

1. Inventory / Static Quality / group-aware leakage pipelineをpublic proxyで完成。
2. `48` をCPU runtimeで実行し `DRIVE PREFETCH GATE: PASS` を確認。
3. A100へ切り替え、`50` を先頭から実行。local staging → group-aware / leakage / GA Gateを確認。
4. `RUN_ABLATIONS=True` にしてV0 Raw / V1 Clean / V2 sqrt-balanced cheap screening。
5. V0 + promising variantsを固定local/simulator evalへ送る。lossだけでpromoteしない。
6. SmolVLA / OpenVLA-OFTを同じdataset/eval splitで比較。
7. Track3 inverse/reversed demonstrationをsimulator-validに設計。
8. 最強構成だけをorganizer GPUのRun A / 必要ならRun Bへ昇格。
