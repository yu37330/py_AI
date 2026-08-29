# PARC2026 本戦環境・公式ルール v1.1 + Slack 更新分析メモ

## 目的

本書は、PARC2026 本戦配布リポジトリの実装、公式資料 `PARC2026開発コンペティション_本選_v1.1`、および公式 Slack の本選アナウンスを突き合わせ、確認できた事実と攻略上の示唆を分けて整理したものです。資料とSlackで差分がある場合は、より新しい運営アナウンスを優先します。学習用GPUは RTX PRO 6000 Blackwell、採点は単一 NVIDIA L4 24GB です。

---

## 0. Slackで追加・更新された最重要事項

### 0.1 Track 3 の設計が公式に開示された

2026/8/28、運営は参加者向け配布物に内部検査スクリプト `check_no_leaks.py` を誤って含めていたことを公表しました。同ファイルに Track 3 の設計情報が含まれていたため、公平性確保のため Track 3 の中核設計が全参加者に公式開示されました。

**Track 3 は「LIBERO の既存タスクを逆順にしたタスク」で構成されます。**

例:

- 順方向: A を B に置く
- Track 3: B から A の元の位置へ戻す

運営は本選1では Track 3 を変更せず、この設計のまま実施すると明言しています。一方、個別タスク内容とタスク数は引き続き非公開です。

これは従来資料の「既知タスクの組み合わせ／未知ドメイン」という抽象説明より具体的であり、Track 3 戦略を大きく更新する情報です。

### 0.2 Track 3 戦略を「composition」から「inverse skill generalization」へ修正

従来は compositional language understanding を主軸候補としていましたが、Slack公式開示後は次を優先します。

1. LIBERO既存タスクの順方向と逆方向を対にした学習データ生成
2. goal state → original/init region への復帰操作の学習
3. grasp / lift / transport / place の primitive を逆方向にも安定実行
4. instruction の forward / inverse pair augmentation
5. standard LIBERO task の BDDL / init region を使った逆操作データ生成の検討
6. Track 3 example を用いた inverse-task evaluation suite 作成

重要なのは task ID や hidden task fingerprint を使うことではなく、**学習済みモデル自身に逆操作一般化能力を持たせること**です。

### 0.3 最終評価フォームが開設済み

Slack公式アナウンスでは、本選1の最終評価用提出フォームが Track 1 / 2 / 3 ごとに開設されています。

- 提出期限: 2026/09/17 23:59（アップロード完了基準）
- リーダーボード提出とは別に最終評価フォームへの提出が必須
- 期限内なら何度でも再提出可能
- 最後にアップロード完了した zip が評価対象
- 最終評価ではリーダーボードよりタスク種類・タスク数・評価 Episode 数を増やす
- 最終提出枠そのものでは動作確認を行わない

したがって、最終提出を validation 用に使ってはいけません。リーダーボードとローカル運営Dockerで事前に完全検証してから最終zipを置く運用にします。

### 0.4 Slack参加者報告でも採点環境を実測確認

参加者の成功提出ログとして以下が共有されています。

- Python 3.10.12
- PyTorch 2.11.0+cu130
- CUDA 13.0
- NVIDIA L4
- EGL rendering

これは配布リポジトリ／公式資料から確認した採点環境と一致しており、L4 + Python 3.10 + CUDA13 を提出互換性の基準として扱う判断を補強します。

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

| Track | 現時点での確定理解 |
|---|---|
| Track 1 | 同一タスク・同一ドメイン |
| Track 2 | 同一タスク・未知ドメイン。位置・視点・ノイズ変更等 |
| Track 3 | **LIBERO既存タスクの逆操作**。個別タスク・タスク数は非公開 |

Trackごとに別モデルを提出しても、同一モデルを提出してもよい。

### 1.3 評価式は success rate 単独ではない

公式 v1.1 では、成功判定と衝突判定をゲートとして、以下の正規化済み指標を重み付きで評価します。

- time / steps
- jerk
- SPARC
- trajectory length
- EEF rotation
- collision penalty

概念的には以下です。

`Total Score = average(success * (1 - collision_penalty) * smooth_metrics)`

`smooth_metrics` は time, jerk, SPARC, trajectory, rotation の非公開重み付き正規化値です。正規化式と重みは非公開です。

公開ローカル scorer の success-rate ベース `overall_score` は開発用指標であり、本番スコアそのものではありません。成功率を最重要ゲートとして維持しつつ、成功軌道の滑らかさ・短さ・安全性も改善対象です。

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
- Python 3.10.12
- CUDA 13.0
- PyTorch 2.11.0+cu130
- EGL rendering
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

Track 3 の公式開示を受け、既存データをそのまま使うだけでなく、**forward trajectory から inverse-task training data を作れるか**を最優先で検証します。ただし単純な action sequence の時間反転が物理的に正しいとは限らないため、環境状態・gripper・collisionを含めて再生成／rolloutする方式を優先します。

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

Track 3 が0であることと逆操作設計の開示を合わせると、**inverse-task fine-tuning は本選1で最も明確な改善仮説の一つ**です。

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

Track 3 の設計が公開されたことは、task-specific hard coding が許可されたことを意味しません。逆操作は training/evaluation distribution の設計に利用し、推論時は学習済みpolicyから action を生成します。

---

## 3. 攻略方針のアップデート

### 3.1 最適化目標

1. 成功ゲートを最大化する
2. 成功を落とさない範囲で collision / jerk / SPARC / steps / trajectory / rotation を改善する

成功しなければゲートでスコアを失います。一方、同程度の成功率なら、短く滑らかで衝突しないpolicyが本番では有利です。

### 3.2 Track別戦略

#### Track 1

- task competence と安定性
- action normalization / gripper convention の完全一致
- 成功後の不要動作を抑え steps / trajectory を短縮

#### Track 2

- LIBERO-plus を中心に visual domain robustness を強化
- camera / lighting / texture / position variation
- agentview と wrist image の使い分け検証

#### Track 3 — 最優先仮説を更新

Track 3 は LIBERO 既存タスクの逆操作です。したがって従来の漠然とした composition 対策より、次を優先します。

- forward / inverse task pair dataset
- original init region を goal にした reverse placement
- inverse instruction generation
- grasp-release の逆方向 skill coverage
- reverse task rollout による demonstration 生成
- LIBERO standard + LIBERO-plus の双方で inverse task を生成し未知visual domainにも対応
- Track3 example BDDLを基準にしたローカル inverse evaluation

π0.5 baseline Track3=0.000 のため、ここに成功率を作れれば差別化が大きい可能性があります。ただし3 Track合算なので Track1/2 を犠牲にしない multi-task mix とします。

### 3.3 モデル選定

比較対象:

1. π0.5 official baseline
2. SmolVLA
3. OpenVLA-OFT+

比較条件:

- 同一Track evaluation
- L4 24GBで実測
- success / official smooth metrics相当
- p50/p95/p99 inference latency
- VRAM peak
- package size
- task別failure mode
- **forward task / inverse task を分離した success rate**

### 3.4 推論後段補正

公式Q&Aで許可される全タスク共通・決定論的な候補:

- action magnitude clipping
- delta translation / rotation のrate limit
- temporal smoothing
- gripper hysteresis
- 異常値防止

success rate を落とさないことをA/B testします。task名やsceneに依存する条件分岐は入れません。

---

## 4. GPU時間60hを前提にした本選1実験計画 — Slack更新版

### Gate A: baseline再現 5h

- π0.5 official checkpointを配布Dockerで検証
- Track1/2/3 smoke evaluation
- latency / VRAM計測
- submission zip validation

### Gate B: Track3 inverse pipeline PoC 8h

- Track3 example BDDL解析
- forward → inverse task generator PoC
- inverse instruction生成
- demonstration生成方法を決定
- 小規模fine-tuneで inverse success が0から立ち上がるか確認

**Go条件:** inverse evaluationでbaselineより明確な改善が出ること。

### Gate C: 候補モデル比較 8h

- π0.5
- SmolVLA
- OpenVLA-OFT+

forward + inverse の両方で短時間比較し、勝ちモデルを選定します。

### Gate D: 勝ちモデル学習 27h

training mix:

- standard LIBERO forward
- LIBERO-plus forward
- inverse LIBERO
- inverse LIBERO-plus

Track1/2性能を維持しつつTrack3を立ち上げる比率を探索します。

### Gate E: scoring改善 5h

成功率上位checkpointのみ共通action post-processingをA/B testします。

### Gate F: 最終検証 7h

- 運営Docker clean end-to-end
- Track1/2/3 zip作成
- `validate_submission.py`
- dependency installから評価完了まで確認
- 10 sec/inference margin確認
- hash / model metadata記録
- リーダーボードで最終動作確認
- 最終評価フォームへのアップロードは十分な余裕を持って行う

合計60h。

---

## 5. Submission acceptance criteria

- model weightsをzip内に同梱
- 外部通信不要
- L4 24GBで起動
- Python 3.10.12 / CUDA13 / PyTorch2.11環境でPASS
- `/health` PASS
- `/reset` PASS
- `/act` = float32 `(7,)`
- NaN / Infなし
- inference < 10 sec、目標p99は十分な余裕を持たせる
- Track全体wall-clock < 1h
- 運営配布Dockerでend-to-end PASS
- Track1/2/3を最終フォームへすべて提出
- 最終提出zipとレポート記載hashを一致させる
- **最終評価フォームを動作確認用途に使わない**

---

## 6. 現時点の優先順位

1. **Track3 inverse-task generator / evaluation を作る** — Slack開示で最も大きく変わった点
2. **π0.5 official baselineを完全再現** — 0.286 / 0.165 / 0.000を基準化
3. **inverse dataでπ0.5を短時間fine-tuneしTrack3が0から改善するか検証**
4. **OpenVLA-OFT+を本選I/Oへ接続しL4 benchmark**
5. **SmolVLAを同条件比較**
6. 勝ちモデルに forward + inverse + LIBERO-plus を混ぜて本学習
7. successを落とさない範囲でtrajectory scoring改善
8. リーダーボードで動作確認後、最終評価フォームへ提出

---

## 7. 情報源の扱い

優先順位:

1. 最新の運営Slack公式アナウンス
2. `PARC2026開発コンペティション_本選_v1.1`
3. 本戦配布リポジトリ実装
4. 公式Q&A
5. Slack参加者の実測報告（補助Evidence）

参加者の推測・雑談は公式仕様とは区別します。特に今回の Track3 逆操作については参加者推測ではなく、2026/8/28の運営公式アナウンスによる確定情報として扱います。

---

## 8. 次に実装するもの

最優先実装を以下に変更します。

1. `Track3 inverse task manifest`
2. BDDL / init region から逆操作タスクを構成する generator の仕様調査
3. inverse demonstration 生成方式のPoC
4. forward / inverse 両対応 Dataset Manifest
5. π0.5 inverse fine-tune experiment
6. Track3 local evaluation runner
7. L4 latency / VRAM benchmark
8. 最終submission build + validation pipeline

Track3の設計が判明したため、単に大きなVLAへ切り替えるより、**既存LIBEROスキルを逆操作へ転用できる学習データ設計**が本選1で最も高い情報価値を持つ実験です。
