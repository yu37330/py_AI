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

`00_a100_preflight.ipynb` remains the recommended first notebook, but `10` and `20` are now **self-contained**. Each notebook repeats the essential preflight steps at the beginning: runtime/GPU check, `/content/parc2026` workspace creation, and `py_AI` clone/update. Therefore a fresh Colab runtime can open `10` or `20` directly without failing because `00` was not executed in the same runtime.

1. `00_a100_preflight.ipynb`
   - GPU / Python / workspace確認
   - `/content/parc2026/py_AI` clone
2. `10_pi05_smoke_ga.ipynb`
   - self-contained preflight
   - Python 3.10学習環境をColab側へ構築
   - dataset priority: explicit `PI05_DATASET_ROOT` → organizer combined → public `Sylvest/libero_plus_lerobot`
   - public fallbackは `meta/ + data/ + videos/` をColab一時領域へ取得するため大容量downloadになる
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

`10` と `20` は並列レーンです。モデル側のGateとDataset Factory側のinventoryを独立に進め、両方が揃った時点でcheap ablationへ進みます。

## Organizer vs public dataset

公開fallbackはColab開発を止めないためのものです。公開LIBERO-plusの結果を、運営 `libero_combined_20hz` の正本Inventoryや最終学習結果として扱ってはいけません。

次に運営GPUを本来の目的で起動したタイミングで、少なくとも `libero_combined_20hz/meta/` を外部へ保存し、`20_dataset_inventory.ipynb` を `PARC_DATASET_ROOT` 指定で再実行します。Run A固定前には `10` も運営combined datasetで短い再確認を行います。

## Comparison contract

Every comparable run must pin:

1. git SHA / upstream model revision,
2. dataset manifest,
3. eval split,
4. seed,
5. effective batch size and optimizer-step semantics,
6. wall time, peak VRAM, latency and task metrics.

Do not select a model using training loss alone. Prefer simulator success plus the PARC-adjacent metrics that can be reproduced locally.

## Planned sequence

1. Reproduce pi0.5 LoRA smoke and verify GA semantics.
2. Inventory public metadata now; replace with organizer-data inventory when metadata is available.
3. Establish V0 Raw / task-sampling candidates.
4. Run cheap V0/V1/V2 dataset ablations with pi0.5 fixed.
5. Add SmolVLA and OpenVLA-OFT on the same dataset/eval split.
6. Compare model choice and later V3/V4/V5 dataset ablations separately.
7. Research simulator-valid Track3 inverse/reversed demonstrations.
8. Promote only the strongest configuration to organizer-GPU training.
