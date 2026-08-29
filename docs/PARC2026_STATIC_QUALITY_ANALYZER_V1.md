# PARC2026 Static Quality Analyzer V1

## Purpose

Dataset Factory の `V0 Raw -> V1 Clean候補` を作るため、LeRobot形式の各episodeを simulator replayなしで静的解析する。

V1は「悪いデータを自動削除する仕組み」ではない。全episodeを同じ指標で数値化し、task内分布から外れたものを `REVIEW` queueへ送る仕組みとする。

## Input contract

- LeRobot-style dataset root
- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `data/**/*.parquet`
- fps: metadataを正本とする。現 public LIBERO-plus は20Hz。

LIBERO 8D stateは次のlayoutとして扱う。

```text
observation.state = [eef_xyz(3), eef_axis_angle(3), gripper_qpos(2)]
action            = [eef_delta_pose(6), gripper(1)]
```

このlayout assumptionは出力summaryにも記録する。別schemaのdatasetへ黙って適用しない。

## Episode metrics

1 episode = 1 rowとして以下を出す。

### Trajectory

- `frames`
- `duration_sec`
- `eef_path_m`
- `eef_displacement_m`
- `path_efficiency`
- `eef_step_rms_m`
- `axis_angle_path_l2`
- `gripper_state_path_l2`

`path_efficiency = net displacement / path length` はtask successや一般的な効率を意味しない。往復動作では低い値が正常な場合があるため、descriptive/outlier signalとしてのみ利用する。

### Jerk

- `rms_cart_jerk_raw`
- `max_cart_jerk_raw`
- `rms_cart_jerk_smooth`
- `max_cart_jerk_smooth`

20Hzのpositionから三階差分を取るためraw jerkはノイズに敏感。監査用にrawを残し、主な分布確認には5-frame moving average後の値を使う。jerk単独でRejectしない。

### Action / idle / gripper

- `action_rms`
- `motion_action_rms`
- `xyz_action_rms`
- `action_max_abs`
- `idle_ratio`
- `gripper_switches`

`idle_ratio` は6D motion action normが初期threshold以下のframe比率。thresholdはdataset/action convention依存なので、普遍的な品質基準として扱わない。

### Integrity

- `timestamp_nonmonotonic_count`
- `timestamp_dt_mae_sec`
- `frame_gap_count`
- `invalid_value_count`

NaN/Inf、timestamp逆行、frame index gapはtrajectory qualityとは別のdata integrity signalとして扱う。

## Task-relative REVIEW flags

global固定閾値ではなく、taskごとに median / MAD を使った robust-z を計算する。

対象:

- duration
- EEF path
- smoothed jerk
- idle ratio
- motion action RMS

V1 defaultは `abs(robust-z) > 5` をoutlier flagとする。integrity issueは常にflagする。

```text
quality_status_v1 = OK | REVIEW
```

`REVIEW` はRejectではない。taskごとのtrajectory形状が本質的に違うため、次段で動画spot-check / replay / task semanticsを確認してからV1 Cleanの除外ルールを決める。

## Outputs

- `episode_quality_metrics.csv`: 全episodeの静的指標
- `quality_review_candidates.csv`: REVIEWのみ
- `task_quality_summary.csv`: task別medianとREVIEW率
- `static_quality_summary.json`: dataset/config/schema assumption/limitations
- `ok_episode_ids.json`
- `review_episode_ids.json`

## Public development dataset vs organizer dataset

Colabでは開発を止めないため `Sylvest/libero_plus_lerobot` をfallbackとして使う。Static Quality Analyzerはtrajectory parquetだけを必要とするため、公開fallbackでは `meta + data/**/*.parquet` のみdownloadし、videosは取得しない。

公開データで得たthresholdやREVIEW率を運営datasetへそのまま流用しない。Run A固定前に organizer `libero_combined_20hz` へ同じAnalyzerを再実行し、task別分布を再計算する。

## What V1 does not do

以下はconverted LeRobot trajectoryだけから推測しない。

- task success
- collision / harmful collision
- replayability
- simulator state drift
- final object placement correctness

これらはraw simulator state / environmentを確保した後のReplay Validatorで追加する。

## Next gate

1. public LIBERO-plus全14,347 episodeでV1を完走
2. task別REVIEW率と上位outlierを確認
3. REVIEWから少量動画spot-check
4. V1 Clean ruleを固定
5. π0.5固定で `V0 Raw / V1 Clean / V2 sqrt-balanced` cheap ablation
6. best datasetを固定してSmolVLA/OpenVLA-OFT比較へ進む
