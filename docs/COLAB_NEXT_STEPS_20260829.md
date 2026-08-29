# Colab A100 次実行手順（2026-08-29）

PR #8で定義した `Model Selection` と `Dataset Factory` の並列レーンを、実行可能なNotebookへ落とした。

## Lane M — π0.5 GA Gate

`colab/10_pi05_smoke_ga.ipynb`

実行順:

1. preflight済みrepo / GPUを確認
2. `uv` でPython 3.10を用意
3. `PI05_DATASET_ROOT` をLeRobot dataset rootへ設定
4. `getpass` でHF tokenを入力
5. Colab一時領域へLeRobot v0.4.4 + patch環境を構築
6. BS=1 / GA=8 / 3 optimizer steps のprobeを実行
7. runtime traceを検証

合格条件:

- backward call = 24
- accelerated optimizer step call = 24
- `sync_gradients=True` = 3
- underlying optimizer step = 3
- training中scheduler step = 3
- instrumentation error = 0

上記PASS後だけ、必要に応じて20-step smoke + LoRA mergeを有効化する。

## Lane D — Dataset Inventory Gate

`colab/20_dataset_inventory.ipynb`

対象datasetの `meta/info.json` / `meta/episodes.jsonl` / `meta/tasks.jsonl` を読み、以下を生成する。

- `episode_inventory.csv`
- `task_inventory.csv`
- `task_sampling_candidates.csv`
- `dataset_inventory_summary.json`

この段階で確定するもの:

- task数
- episode数
- frame数
- fps
- taskごとのepisode/frame数
- episode長分布
- raw / uniform / sqrt-balanced sampling候補
- source labelがmetadataに残っているか

この段階ではsuccess / collision / replayabilityを推測しない。これらはsimulator replay可能なsource dataを確保した後のReplay Validatorで扱う。

## 次の同期点

Lane MとLane Dの両方がPASSした後、π0.5を固定してDataset V0/V1/V2のcheap ablationへ進む。

- V0: Organizer Raw
- V1: Static Quality AnalyzerによるClean候補
- V2: Task Balanced

その後、best datasetを固定してSmolVLA / OpenVLA-OFTを同じ評価契約へ接続する。
