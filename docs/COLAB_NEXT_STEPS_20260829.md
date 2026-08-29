# Colab A100 次実行手順（2026-08-29）

PR #8で定義した `Model Selection` と `Dataset Factory` の並列レーンを、実行可能なNotebookへ落とした。

## 2026-08-30 status

Dataset Factory Lane の `20_dataset_inventory.ipynb` は、公開 fallback `Sylvest/libero_plus_lerobot` の metadata に対して PASS した。

確認結果:

- 20 Hz
- 40 tasks
- 14,347 episodes
- 2,238,036 frames
- `info.json` reported 値と episode metadata 再集計値が一致
- task episode 数は 146〜500
- basket / plate 系 pick-and-place が episode ベースで約 64.6%
- explicit inverse task coverage は薄い
- `success / collision / replayability` は metadata から推測しない

詳細: `docs/PARC2026_PUBLIC_LIBERO_PLUS_INVENTORY_RESULTS_20260830.md`

次の Dataset Factory 実装は `Static Quality Analyzer V1` とし、その後 π0.5 固定で V0 / V1 / V2 cheap ablation を行う。

## Lane M — π0.5 GA Gate

`colab/10_pi05_smoke_ga.ipynb`

実行順:

1. self-contained preflight でrepo / GPUを確認
2. `uv` でPython 3.10を用意
3. `PI05_DATASET_ROOT` をLeRobot dataset rootへ解決。organizer dataset が無い場合は公開 fallback を使用
4. Colab Secrets または `getpass` でHF tokenを入力
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

公開 fallback 上のPASSは GA / pipeline 動作確認として扱い、Run A 固定前には運営 `libero_combined_20hz` で短い再確認を行う。

## Lane D — Dataset Inventory Gate

`colab/20_dataset_inventory.ipynb`

対象datasetの `meta/info.json` / `meta/episodes.jsonl` / `meta/tasks.jsonl` を読み、以下を生成する。

- `episode_inventory.csv`
- `task_inventory.csv`
- `task_sampling_candidates.csv`
- `dataset_inventory_summary.json`
- `task_metadata_raw.json`

公開 fallback での Inventory Gate は PASS 済み。

この段階で確定したもの:

- task数
- episode数
- frame数
- fps
- taskごとのepisode/frame数
- episode長分布
- raw / uniform / sqrt-balanced sampling候補
- public metadata では source label が `unknown` であること

この段階ではsuccess / collision / replayabilityを推測しない。これらはsimulator replay可能なsource dataを確保した後のReplay Validatorで扱う。

## 次の同期点

### Dataset Lane

1. `Static Quality Analyzer V1`
   - duration
   - movement
   - path length
   - displacement / path efficiency
   - jerk
   - action magnitude
   - idle ratio
   - gripper switches
2. π0.5固定で V0 / V1 / V2 cheap ablation
   - V0: Raw
   - V1: Clean candidate
   - V2: Task Balanced
3. best dataset recipe を仮固定

### Model Lane

1. π0.5 GA=8 Gate を完了
2. best dataset recipe を固定後、π0.5 / SmolVLA / OpenVLA-OFT を同一評価契約へ接続
3. equal-data / equal-wall-time で比較

### Final sync

shortlist model のみに V3 augmentation / V4 public supplemental / V5 Track3 inverse を追加し、Run A exact recipe を決める。

Run A 固定前には運営配布 `libero_combined_20hz` 19,533 episodes を同じ Inventory pipeline に通し、公開 fallbackとの差を再確認する。
