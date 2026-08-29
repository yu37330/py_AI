# Colab A100 benchmark workspace

`colab/` is the entry point for experiments that should be completed before spending the PARC organizer GPU budget.

## Goal

Use a Colab A100 for model comparison, data ablation, Track3 inverse-data research, and submission preflight. Keep the organizer GPU for the final candidate training/evaluation runs.

## Repository roles

- `py_AI`: experiment definitions, manifests, benchmark outputs, PARC evaluation/submission code.
- `/content/vendor/*`: disposable upstream clones such as LeRobot/OpenVLA-OFT. Do not vendor them into this repository.
- `/content/cache/*`: Hugging Face/model caches and temporary datasets.
- Google Drive: optional persistence for checkpoints and large generated artifacts only.

## Notebook order

1. `00_a100_preflight.ipynb`
   - GPU / Python / workspace確認
   - `/content/parc2026/py_AI` clone
2. `10_pi05_smoke_ga.ipynb`
   - Python 3.10学習環境をColab側へ構築
   - π0.5 LoRAの短いsmoke
   - runtime traceで `GA=8` が8 micro-stepごとに1 optimizer updateになることを検証
   - 20-step + mergeは明示的に有効化した場合だけ実行
3. `20_dataset_inventory.ipynb`
   - LeRobot metadataからtask / episode構成をCSV化
   - raw / uniform / sqrt-balancedのsampling候補を生成
   - success/collision/replayabilityはこの段階では推測しない

`10` と `20` は並列レーンです。モデル側のGateとDataset Factory側のinventoryを独立に進め、両方が揃った時点でcheap ablationへ進みます。

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
2. Inventory organizer data and establish V0 Raw / task-sampling candidates.
3. Run cheap V0/V1/V2 dataset ablations with pi0.5 fixed.
4. Add SmolVLA and OpenVLA-OFT on the same dataset/eval split.
5. Compare model choice and later V3/V4/V5 dataset ablations separately.
6. Research simulator-valid Track3 inverse/reversed demonstrations.
7. Promote only the strongest configuration to organizer-GPU training.
