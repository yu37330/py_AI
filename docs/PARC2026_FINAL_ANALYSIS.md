# PARC2026 本戦環境・公式ルール v1.1 分析メモ

## 目的

本書は、PARC2026 本戦配布リポジトリの実装と、公式資料 `PARC2026開発コンペティション_本選_v1.1` を突き合わせ、確認できた事実と攻略上の示唆を分けて整理したものです。実装と公式資料が異なるように見える場合は、役割を分けて解釈します。学習用GPUは RTX PRO 6000 Blackwell、採点は単一 NVIDIA L4 24GB です。

---

## 1. 公式 v1.1 で確定した重要事項

### 1.1 スケジュールと勝ち残り

- 本選1: 2026/8/24 18:00 ～ 9/17 23:59
- 本選2: 9/28 ～ 10/22（予定）
- 本選3: 10/28 ～ 11/24（予定）
- 本選1: 200人 → 100人
- 本選2: 100人 → 50人
- 本選3は最終評価ランキングとレポートを踏まえて優秀者を選出

### 1.2 Track

| Track | 意味 |
|---|---|
| Track 1 | 同一タスク・同一ドメイン |
| Track 2 | 同一タスク・未知ドメイン。位置・視点・ノイズ変更等 |
| Track 3 | 既知タスクの組み合わせ・未知ドメイン。言語指示で新規構成されたタスク |

Trackごとに別モデルを提出しても、同一モデルを提出してもよい。

### 1.3 評価式は success rate 単独ではない

公式 v1.1 では、成功判定と衝突判定をゲートとして、以下の正規化済み指標を重み付きで評価することが明示されています。

- time / steps
- jerk
- SPARC
- trajectory length
- EEF rotation
- collision penalty

概念的には以下です。

`Total Score = average(success * (1 - collision_penalty) * smooth_metrics)`

`smooth_metrics` は time, jerk, SPARC, trajectory, rotation の非公開重み付き正規化値です。正規化式と重みは非公開です。

したがって、公開ローカル scorer の success-rate ベース `overall_score` は開発用指標であり、本番スコアそのものではありません。成功率を最重要ゲートとして維持しつつ、成功軌道の滑らかさ・短さ・安全性も改善対象です。

Track 1/2/3 の Total Score の合計が各本選スコアです。最終評価フォームに未提出の Track は 0 点です。

### 1.4 学習GPUと採点GPUは別

#### 配布学習環境

- NVIDIA RTX PRO 6000 Blackwell
- 本選1: 60 GPU時間
- 本選2: 75 GPU時間
- 本選3: 120 GPU時間
- ステージごとに時間はリセット、繰越なし
- Notebook は完全アイドル1時間で停止
- 起動から12時間で自動停止

#### 採点環境

- 単一 NVIDIA L4
- VRAM 24GB
- 1 inference 10秒以内
- リーダーボードの Track 評価は依存インストールから評価完了まで wall-clock 1時間以内

結論: RTX PRO 6000 Blackwell の大容量VRAMを前提に提出モデルを設計してはいけません。学習はBlackwell、最終推論設計はL4 24GB基準です。

### 1.5 採点Docker

公式資料では、採点環境と同一構成の Docker（CUDA 13.0、PyTorch 2.11.0+cu130 等）が配布GPU環境に用意され、提出zipの end-to-end 検証をそのDocker内で行うことが推奨されています。

独自Dockerはサポート対象外です。提出前検証は運営配布Dockerを正とします。

### 1.6 Policy server interface

`submission_template/policy_server.py` の `MyPolicy` を変更します。

- `__init__`: `model_weights/` からモデルをロード
- `get_action(obs)`: 観測から action を生成
- `reset(instruction="")`: エピソード開始時に言語指示を受け、キャッシュ・履歴等をクリア

Action:

```text
float32 shape (7,)
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

1リクエスト10秒以内です。

### 1.7 提出とリーダーボード

- リーダーボードは Track ごとに1日1回採点可能
- エラーでも回数を消費
- アップロード完了時刻が提出時刻
- 最終評価はリーダーボード提出とは別の専用フォーム
- Track 1/2/3 の3フォームすべてに提出する
- 期限内なら最終フォームは何度でも差し替え可能で、最後にアップロード完了したzipが対象
- 最終評価ではリーダーボードよりタスク種類・Episode数を増やす
- 最終提出枠では動作確認されず、エラー救済は基本なし

本選1最終提出期限は 9/17 23:59（アップロード完了基準）。

### 1.8 データセット

配布環境には少なくとも以下があります。

- LIBERO standard: spatial / object / goal / 10 / 90（no_noops）
- LIBERO-plus: camera viewpoint、lighting、texture 等の摂動データ
- LeRobot 統合20Hz `libero_combined_20hz`
- LeRobot v2.1 / v3.0 系

LIBERO / LIBERO-plus データの学習利用は公式Q&Aで許可されています。また配布ローカル評価環境を使った強化学習も禁止されていません。

### 1.9 π0.5 baseline

公式配布 `examples/pi05_libero_finetune` には以下が含まれます。

- 学習環境セットアップ
- LeRobot patch（gradient accumulation 等）
- π0.5 LoRA training
- LoRA merge script
- submission policy server example

step 5000 checkpoint が配布 baseline で、公式参考 leaderboard score は:

| Track | Score |
|---|---:|
| Track 1 | 0.286 |
| Track 2 | 0.165 |
| Track 3 | 0.000 |

これはモデル選定時の最低比較ラインとして扱います。

### 1.10 Sandbox / offline / submission 制約

評価中は外部通信が遮断されるため、モデル重み等はzipに同梱します。配布リポジトリ実装からも、評価側秘密領域へのアクセス制限、requirementsの外部URL禁止、zip validation等が確認できます。

提出物は自己完結させます。

---

## 2. 禁止事項と許可される改善

### 禁止

- 外部・手続き的な task-level planner（CaPX等）
- task固有FSM
- hard-coded action sequence
- scene ID / task ID / evaluation seed をキーとした action table
- 成功条件・報酬を直接参照する planner
- 評価環境専用 if 文
- hidden task fingerprinting
- 学習済みモデルを実質使用しない fallback policy
- nested archive、zip bomb
- symlink / hardlink
- path traversal
- 不要な難読化

### 公式Q&Aで許可

- 全タスク共通・決定論的な action 後段補正
- 配布ローカル評価環境を使った強化学習
- LIBERO / LIBERO-plus データ学習
- `lerobot` を requirements.txt に記載

この境界は重要です。後処理を入れるなら task-specific rule ではなく、全タスク共通の安全・滑らかさ改善として設計します。

---

## 3. 攻略方針のアップデート

### 3.1 最適化目標

以前の「success rate 最優先」は方向としては維持しますが、公式 v1.1 により次の二段階最適化に修正します。

1. 成功ゲートを最大化する
2. 成功を落とさない範囲で collision / jerk / SPARC / steps / trajectory / rotation を改善する

成功しなければゲートでスコアを失うため、滑らかさだけを先に最適化するのは誤りです。一方、同程度の成功率なら、短く滑らかで衝突しないpolicyが本番では有利です。

### 3.2 Track別戦略

#### Track 1

- task competence と安定性
- action normalization / gripper convention の完全一致
- 成功後の不要動作を抑え steps / trajectory を短縮

#### Track 2

- LIBERO-plus を中心に visual domain robustness を強化
- camera / lighting / texture / position variation
- agentview と wrist image の使い分け検証

#### Track 3

baseline が 0.000 なので最大の差別化候補です。

- compositional language understanding
- multi-task / skill composition
- instruction paraphrase
- history / action chunk の設計
- standard LIBERO と LIBERO-plus を混ぜた skill coverage

ただし本選1は3 Track合算なので、Track3だけに賭けず Track1/2 の確実な点も取りに行きます。

### 3.3 モデル選定

比較対象を最低限以下に固定します。

1. π0.5 official baseline
2. SmolVLA
3. OpenVLA-OFT+

比較条件:

- 同一のTrack evaluation
- L4 24GBで実測
- success / official smooth metrics相当
- p50/p95/p99 inference latency
- VRAM peak
- package size
- task別failure mode

RTX PRO 6000 Blackwellで動くことは採用条件ではありません。L4 24GBで安定動作することが必須条件です。

### 3.4 推論後段補正

公式Q&Aにより、全タスク共通・決定論的な action 後段補正は利用可能です。候補:

- action magnitude clipping
- delta translation / rotation のrate limit
- temporal smoothing
- gripper hysteresis
- 異常値防止

ただし success rate を落とさないことをA/B testで確認します。task名やsceneに依存する条件分岐は入れません。

---

## 4. GPU時間60hを前提にした本選1実験計画

60hを無計画なfull trainingに使わず、stage-gate方式にします。

### Gate A: baseline再現 6h

- π0.5 official checkpointを配布Dockerで検証
- Track1/2/3 smoke evaluation
- latency / VRAM計測
- submission zip validation

### Gate B: 候補モデル比較 10h

- π0.5
- SmolVLA
- OpenVLA-OFT+

短時間fine-tuneまたは既存checkpointで比較し、明確に弱い候補を落とします。

### Gate C: 勝ちモデル学習 28h

- LIBERO + LIBERO-plus mix
- Track2 augmentation
- Track3 compositionを意識したsampling
- checkpointを複数保存

### Gate D: scoring改善 8h

成功率上位checkpointに対してのみ、共通action post-processingをA/B testします。

### Gate E: 最終検証 8h

- 運営Dockerでclean end-to-end
- 3 Track zipを作成
- `validate_submission.py`
- dependency installから評価完了まで確認
- 10 sec/inference margin確認
- hash / model metadata記録

合計60h。実験が早く終わればGate Cへ戻します。

---

## 5. Submission acceptance criteria

- model weightsをzip内に同梱
- 外部通信不要
- L4 24GBで起動
- `/health` PASS
- `/reset` PASS
- `/act` = float32 `(7,)`
- NaN / Infなし
- inference < 10 sec、目標p99は十分な余裕を持たせる
- Track全体wall-clock < 1h
- 運営配布Dockerでend-to-end PASS
- Track1/2/3を最終フォームへすべて提出
- 最終提出zipとレポート記載hashを一致させる

---

## 6. 現時点の優先順位

1. **π0.5 official baselineを完全再現** — 公式0.286 / 0.165 / 0.000を基準化
2. **OpenVLA-OFT+を本選I/Oへadapter化してL4 benchmark**
3. **SmolVLAを同条件benchmark**
4. **Track2向けLIBERO-plus学習mix最適化**
5. **Track3 composition学習を重点検証**
6. **成功率を維持した共通action smoothing / clipping**
7. **3 Track別checkpoint採用判断**
8. **運営Dockerによる最終zip hardening**

---

## 7. 重要な判断

本選1の勝ち筋は「最大モデルをRTX PRO 6000で学習すること」ではありません。

**Blackwellを学習速度のために使い、L4 24GBで確実に動くVLAを作り、Track1/2の成功率を確保しながらTrack3のcomposition汎化で差を作り、最後に成功を壊さない範囲で滑らかさ・効率・安全性を改善する**、という構成が公式v1.1に最も整合します。

また、リーダーボードは1日1回/Trackしか使えずエラーでも消費するため、Omnicampusをデバッグ環境にしてはいけません。ローカル評価と運営Dockerを主な開発ループとし、leaderboard submissionは仮説検証の高価な測定点として扱います。
