# PARC2026 Public LIBERO-plus Dataset Inventory Results

更新日: 2026-08-30

## 1. 目的

Colab A100 側の Dataset Factory Lane で実行した `20_dataset_inventory.ipynb` の初回結果を記録する。

今回の対象は運営配布 `libero_combined_20hz` ではなく、Colab 側で fallback として取得した公開 `Sylvest/libero_plus_lerobot` の metadata である。したがって、本ドキュメントの数値は **公開 LIBERO-plus の事前分析結果** として扱い、Run A の最終 dataset recipe を固定する前に運営配布 combined dataset で同じ Inventory を再実行する。

## 2. 今回確認できた事実

`dataset_inventory_summary.json` の出力は以下で整合した。

| 項目 | 結果 |
| --- | ---: |
| FPS | 20 Hz |
| task 数 | 40 |
| episode 数 | 14,347 |
| frame 数 | 2,238,036 |
| episode metadata から再計算した frame 数 | 2,238,036 |
| inventory source | `public_metadata:Sylvest/libero_plus_lerobot` |
| source label | `unknown` |

`info.json` の reported 値と episode metadata から再集計した値が一致しており、Inventory V1 の metadata 集計 pipeline は正常に動作したと判断する。

注意点として、この変換済み metadata だけでは元データの source provenance が復元できず、`source` は `unknown` のままである。また `success / collision / replayability` は simulator-capable な raw state が必要なため、Inventory V1 では推測しない。

## 3. Episode 長の分布

公開 metadata から、episode length と 20 Hz を用いて `duration_sec_approx = frames / fps` を算出した。

| 指標 | frame 数 | duration |
| --- | ---: | ---: |
| 平均 | 155.99 | 7.80 sec |
| 中央値 | 138 | 6.90 sec |
| 25 percentile | 112 | 5.60 sec |
| 75 percentile | 170 | 8.50 sec |
| 最小 | 75 | 3.75 sec |
| 最大 | 505 | 25.25 sec |

短い単一操作 task と、複数 object / drawer / microwave を含む複合 task で episode 長が大きく異なる。Raw frame sampling を行う場合、episode 数だけでなく trajectory 長の差でも task ごとの学習寄与が変わる点に注意する。

## 4. Task 数と episode 数の偏り

40 task の episode 数は 146〜500 の範囲で、最大 / 最小は約 3.4 倍の差がある。

多い例:

- `pick up the chocolate pudding and place it in the basket`: 500 episodes
- `turn on the stove`: 500 episodes
- `put both the cream cheese box and the butter in the basket`: 480 episodes
- `put the bowl on the plate`: 480 episodes
- `put the bowl on the stove`: 460 episodes

少ない例:

- `put both moka pots on the stove`: 146 episodes
- `put the yellow and white mug in the microwave and close it`: 160 episodes
- `open the top drawer and put the bowl inside`: 200 episodes
- `put the white mug on the plate and put the chocolate pudding to the right of the plate`: 230 episodes
- `put both the alphabet soup and the tomato sauce in the basket`: 240 episodes

このため、Raw sampling のみを採用すると頻出 task が optimizer update を支配する可能性がある。

## 5. 操作カテゴリの偏り

task 名を操作カテゴリとして粗く分類すると、basket / plate への pick-and-place 系が大部分を占める。

今回の簡易集計では:

- basket-place: 5,175 episodes
- plate-place: 4,090 episodes
- 上記 2 系統合計: 9,265 / 14,347 episodes ≒ **64.6%**

一方、drawer、microwave、push、rack、caddy、stove 操作などは相対的に少ない。

これは「14,347 episodes」という総量だけでは dataset の skill coverage を判断できないことを示す。Data Factory では task / skill distribution を明示的に扱う必要がある。

## 6. 複合 task

複数 object や複数 sub-action を含む task も存在する。

例:

- `put both the cream cheese box and the butter in the basket`
- `turn on the stove and put the moka pot on it`
- `put both the alphabet soup and the cream cheese box in the basket`
- `put the white mug on the left plate and put the yellow and white mug on the right plate`
- `put the black bowl in the bottom drawer of the cabinet and close it`
- `open the top drawer and put the bowl inside`
- `put the yellow and white mug in the microwave and close it`
- `put both moka pots on the stove`

文字列ベースの簡易抽出では 10 task / 2,736 episodes、全 episode の約 19.1% が複合操作候補に該当した。

複合 task は平均 episode 長も長くなりやすく、frame sampling 時には一層大きな重みを持つ可能性がある。

## 7. Track3 inverse coverage に関する初期所見

公開 40 task には以下の forward skill が含まれる。

- drawer を開く
- drawer に bowl を入れる
- drawer に bowl を入れて閉じる
- microwave に mug を入れて閉じる
- stove を点ける
- object を stove に置く

一方、task 名一覧上では、それらに対する明示的な inverse task、例えば以下は確認できない。

- drawer を閉じるだけの task
- drawer から bowl を取り出す task
- microwave を開けて object を取り出す task
- stove を消す task
- stove から object を取り除く task

したがって、**通常の公開 LIBERO-plus だけでは inverse / reversed skill coverage が薄い可能性が高い**。これは配布 baseline の参考値で Track3 が 0.000 だった事実とも整合するが、因果関係は未確認である。

Track3 対策としては action sequence の単純 reverse は採用せず、正しい inverse initial state を用意し、simulator 上で成立する expert / scripted / learned policy trajectory を生成して成功判定する方針を維持する。

## 8. Task balancing 候補

Inventory V1 では 3 種類の sampling probability を生成した。

1. `raw_prob`
   - episode 数に比例
   - 元 dataset の分布を維持
2. `uniform_prob`
   - 全 task を同確率
   - 少数 task を強く oversample するリスクあり
3. `sqrt_balanced_prob`
   - `sqrt(episode_count)` に比例
   - Raw と Uniform の中間案

現時点では `sqrt-balanced` を有力な比較候補とする。ただし final recipe としては未決定で、π0.5 を固定した cheap ablation で Raw / Uniform / Sqrt-balanced を比較する。

## 9. 運営資料との関係

PARC2026 本選 v1.1 では、運営側の `~/dataset/` に以下が配布されると説明されている。

- `IPEC-COMMUNITY/`: LIBERO standard suite
- `Sylvest/`: camera / lighting / texture 等の摂動を加えた LIBERO-plus
- `lerobot/`: baseline 学習用 `libero_combined_20hz`
- `v2.1/`, `v2.0/`: 各形式の派生 dataset

またデータ使用ルールとして、講座提供データだけでなく、公開 dataset、独自収集・生成データ、simulator / teleoperation / scripted policy / augmentation / generative model を用いたデータ生成が許可されている。一方で、評価結果を利用した追加学習や評価観測を保存して後続学習へ使用することは禁止されている。

したがって Dataset Factory はルールの範囲内で以下を比較対象にできる。

- V0: Organizer Raw
- V1: Quality-filtered Clean
- V2: Task Balanced
- V3: Safe Augmentation
- V4: Public Supplemental
- V5: Track3 Inverse generated data

## 10. 今回まだ分からないこと

今回の公開 metadata Inventory だけでは以下は確認できない。

- 各 episode が成功 demo か
- harmful collision の有無
- replayability
- MuJoCo state drift
- trajectory / object state の再現性
- LIBERO / LIBERO-plus の元 source provenance
- 運営 combined 19,533 episodes の実際の task distribution

これらを推測で埋めない。

## 11. 次の実装順

### Gate D1: Static Quality Analyzer V1

公開 full dataset の state / action を用いて episode 単位で以下を CSV 化する。

- duration
- Cartesian movement
- path length
- displacement
- path efficiency
- jerk
- action RMS / max
- idle ratio
- gripper switch count

目的は成功判定ではなく、異常値・長すぎる trajectory・振動的 trajectory・task 内 outlier を機械的に抽出すること。

### Gate D2: V0 / V1 / V2 cheap ablation

π0.5 を固定し、短い学習で以下を比較する。

- V0 Raw
- V1 Clean
- V2 Raw + task balancing

最初から augmentation / public supplemental / inverse を全部盛りしない。Quality filtering と sampling の寄与を先に分離する。

### Gate D3: Model Selection との同期

V0 / V1 / V2 から有力 dataset recipe を 1 つ選び、その dataset を固定して π0.5 / SmolVLA / OpenVLA-OFT を equal-data / equal-wall-time protocol で比較する。

### Gate D4: V3 / V4 / V5

shortlist model のみに対して以下を追加する。

- Safe visual augmentation
- 不足 task / variation を補う公開 data
- simulator-valid Track3 inverse data

### Gate D5: Organizer combined で再確認

Run A 固定前に、運営配布 `libero_combined_20hz` の metadata を同じ Inventory pipeline に通す。

確認項目:

- task 数
- task 別 episode / frame 数
- task distribution
- public fallback と organizer combined の差
- inverse skill coverage
- sampling recipe の再計算

この結果を最終 dataset manifest に固定してから配布 GPU の本番学習へ進む。

## 12. Decision

現時点では以下を採用する。

1. `20_dataset_inventory.ipynb` の Inventory V1 は PASS
2. 公開 LIBERO-plus は Data Factory 開発用の proxy dataset として使用する
3. 公開 metadata からは success / collision / replayability を推測しない
4. Dataset の量より task / skill distribution を重視する
5. Raw / Uniform / Sqrt-balanced を cheap ablation で比較する
6. Track3 inverse coverage は別途 dataset generation の対象とする
7. 最終 Run A recipe は運営 `libero_combined_20hz` で再集計してから確定する
8. 次の実装は `Static Quality Analyzer V1` とする

## 13. Evidence files

Colab `20_dataset_inventory.ipynb` の今回の出力:

- `dataset_inventory_summary.json`
- `task_inventory.csv`
- `episode_inventory.csv`
- `task_sampling_candidates.csv`
- `task_metadata_raw.json`

これらは実験 evidence であり、大容量 raw data / model weight と同様に Git の正本には直接含めず、必要な集計結果・Decision を本ドキュメントと experiment manifest に残す。
