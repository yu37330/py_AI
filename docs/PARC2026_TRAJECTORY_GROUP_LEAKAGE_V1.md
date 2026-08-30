# PARC2026 Trajectory-group Leakage Gate V1

## 背景

Static Quality Analyzer V1では、同一task内で複数episodeが非常に近いtrajectory統計を共有することが確認された。LIBERO-plusのdomain / visual perturbationでは、同じ非visual control trajectoryを複数episodeへ展開している可能性がある。

episode単位だけでfixed evalを作ると、同じ元trajectoryの別perturbationがtrainとevalへ分かれ、offline比較が楽になる可能性がある。π0.5 Dataset Ablationを回す前にこれを検査する。

## V1のgroup定義

`tools/data/check_trajectory_group_leakage.py` は画像・動画を使わず、各episodeのLeRobot parquetから次をhashする。

### exact group

- task index
- sequence length
- `observation.state` 全sequence
- `action` 全sequence

state/actionはfloat dtype差や極小serialization差を吸収するため、小数6桁へ量子化してからSHA256化する。

同じexact groupがfixed evalとtraining manifestの両方にあれば **Leakage Gate = FAIL** とする。

### action group

- task index
- sequence length
- `action` 全sequence

stateが異なってもcontrol sequenceが同じepisodeを拾う。これは「同一元trajectory」の候補にはなるが、別initial stateで同じactionが妥当な場合もあるのでV1ではWARN扱いとし、exact Gateは落とさない。

## 出力

`colab/45_trajectory_group_leakage.ipynb` は以下を生成する。

- `trajectory_group_leakage_summary.json`
- `manifest_leakage_report.csv`
- `manifest_leakage_members.csv`
- `trajectory_group_members.csv`
- `exact_trajectory_groups.csv`
- `action_trajectory_groups.csv`

Static Quality CSVが存在する場合は `quality_review_candidate` もjoinし、REVIEW episodeがduplicate trajectory groupへどれだけ含まれるかをcross-checkする。

## 実行順

```text
30 Static Quality
  ↓
40 Dataset Manifest
  ↓
45 Trajectory-group Leakage Gate
  ↓
PASS → 50 π0.5 cheap ablation
FAIL → group-aware fixed evalへmanifestを再設計 → 45を再実行
```

`45` はself-containedであり、Colabのfresh runtimeでもpublic `lerobot/libero_plus` v3のmeta + parquetだけを取得して動作する。画像・動画はdownloadしない。

## 重要な制約

- このGateはtask success / collision / replayabilityを判定しない。
- exact hash一致は非visual trajectoryの強い重複証拠だが、元データ生成pipelineのprovenanceそのものを証明するものではない。
- action-only一致だけでepisodeを自動除外しない。
- public LIBERO-plusで完成させたpipelineは、Run A固定前に運営 `libero_combined_20hz` へ再適用する。

## FAIL時の次段

FAILした場合、現在の80 episode fixed evalをそのまま使わず、trajectory groupを単位にtrain exclusionを作る。evalで代表episodeを2本/taskに保つ場合でも、その代表episodeと同じtrajectory groupの全siblingsをtraining manifestから除外する。

これによりevaluation episode数とtrain exclusion数を分離し、task coverageを保ちながらgroup leakageを0にする。
