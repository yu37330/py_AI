# PARC2026 本戦環境 分析メモ

## 目的

本書は、PARC2026 本戦の配布リポジトリに含まれる README、評価パイプライン、提出バリデータ、提出テンプレート、参考実装を読み、確認できた事実と、そこから導く実装・学習上の示唆を分けて整理したものです。

---

## 1. 確認できた事実

### 1.1 本番採点環境

本番は GPU コンテナ上で実行され、配布 Dockerfile は本番と同一の依存構成を再現します。

| 項目 | 確認値 |
|---|---|
| GPU | NVIDIA L4 |
| OS | Ubuntu 22.04.5 LTS |
| Base image | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` |
| Python | 3.10.12 |
| CUDA | 13.0 |
| cuDNN | 9.14.0 |
| NCCL | 2.28.3+cuda13.0 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| NVIDIA driver | R580 系 |
| Rendering | `MUJOCO_GL=EGL` |

LIBERO-plus、LIBERO、assets もコミット固定です。

### 1.2 Track 構成

| Track | Suite | 意味 |
|---|---|---|
| Track 1 | `libero_t1` | 同一タスク・同一ドメイン |
| Track 2 | `libero_t2` | 同一タスク・未知ドメイン |
| Track 3 | `libero_t3` | 既知タスク組み合わせ・未知ドメイン |

本番では公開 example とは別の非公開タスクを含むタスクセットで採点されます。

### 1.3 観測仕様

評価側からポリシーへ送られる観測キーは以下です。

- `agentview_image`
- `robot0_eye_in_hand_image`
- `robot0_joint_pos`
- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_gripper_qpos`

画像解像度の既定値は 128 x 128 です。

### 1.4 Action 仕様

`POST /act` の応答は `float32`、shape `(7,)` である必要があります。

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

NaN / Inf は許容されません。

### 1.5 HTTP Interface

提出物は HTTP ポリシーサーバーとして実装し、以下のエンドポイントを持つ必要があります。

- `GET /health`
- `POST /reset`
- `POST /act`

`reset()` はエピソードごとに呼ばれます。action chunk cache、language instruction state、history buffer 等のエピソード固有状態はここでクリアする必要があります。

### 1.6 Timeout

`/act` の 1 リクエストあたり timeout は 10 秒です。

評価 config には以下も定義されています。

- max steps / episode: 300
- 配布環境の既定評価 episode 数: 20
- episode timeout: 120 秒
- GPU time limit: 3600 秒

本番の episode 数は公開されていません。

### 1.7 提出サイズ

- zip 上限: 20 GB
- 展開後上限: 40 GB
- model size 設定値: 20 GB
- エントリ数上限: 200,000

巨大 zip や異常な圧縮率は validation で拒否されます。

### 1.8 requirements.txt 制約

採点時は提出専用 venv が `--system-site-packages` 付きで作成されます。そのため、本番イメージにプリインストール済みのライブラリは requirements に再記載しなくても利用できます。

一方、以下は禁止されています。

- `git+https://...`
- `http://...`, `https://...` 等の外部 URL
- `--index-url`
- `--extra-index-url`
- `--find-links`
- editable install
- 外部 requirement / constraint の参照

つまり、採点時のネットワーク取得を前提にした依存やモデルロードは不可です。

### 1.9 Sandbox

評価時、参加者の policy server は可能な場合 `nobody` 等の非特権ユーザーに降格して起動されます。また以下の評価側秘密領域は filesystem hardening の対象です。

- `/workspace/LIBERO-plus`
- `/workspace/compe`
- `/workspace/pipeline`
- `/workspace/scoring_config.json`
- `/workspace/total_score_config.json`
- `/workspace/normalization_config.json`

提出コードから hidden task や scoring config を覗く前提の実装は成立しません。

### 1.10 公開ローカル scorer

配布 scorer の `overall_score` は各タスク成功率の平均、すなわち track の平均 success rate です。

軌道メトリクスとして以下も計算されます。

- Cartesian path length
- joint path length
- orientation path length
- average / RMS Cartesian jerk
- average / RMS joint jerk
- average steps to success
- episode time

ただし README で明記されている通り、リーダーボード最終順位を決定する scoring config は配布されていません。したがってローカル `overall_score` と本番最終スコアを同一視してはいけません。

### 1.11 運営の参考モデル

運営は参考例として少なくとも以下を配布しています。

1. SmolVLA LoRA
   - `lerobot/smolvla_libero_plus`
   - LIBERO-plus Spatial 10 task
   - 50 episode
   - 3,000 training steps
   - Colab T4 を想定した最小構成

2. pi0.5 LoRA
   - LIBERO 系データで LoRA fine-tune
   - LoRA を merge して submission に同梱
   - LeRobot v0.4.4 固定

運営自身も「training environment と evaluation environment を分ける」「本番は offline submission」と明確に設計しています。

---

## 2. ここから導ける実装上の示唆

以下は上記仕様からの推論・提案です。

### 2.1 最優先は success rate

公開 scorer では track score は task success rate の単純平均です。最終 scoring config は非公開ですが、少なくとも成功しない軌道の smoothness を改善しても勝ち筋にはなりません。

優先順位は原則として次です。

1. 成功率
2. 未知 domain に対する robustness
3. task composition robustness
4. 推論安定性 / timeout 回避
5. 軌道品質

### 2.2 Track 2 / 3 対策が本戦の差になりやすい

Track 1 は同一 task・同一 domain ですが、Track 2 は未知 domain、Track 3 は既知 task の組み合わせ + 未知 domain です。

したがって単純な train task memorization より、以下の方が価値が高い可能性があります。

- appearance / texture / lighting augmentation
- camera perturbation
- object placement variation
- instruction paraphrase
- task-balanced sampling
- composition を意識した multi-task learning

### 2.3 128x128 を本番条件として最適化する

参考 notebook の 256x256 evaluation と本番 128x128 は条件が異なります。

VLA の image preprocessing を 256x256 前提のまま考えるのではなく、128x128 observation からモデル入力へ resize した時の情報損失を含めて検証すべきです。

特に小物体操作では eye-in-hand image の寄与を再評価する価値があります。

### 2.4 Action chunking は有力だが reset 処理が必須

1 action request が 10 秒以内なら、毎 step 大型モデルをフル forward するより action chunk を生成してキャッシュする方法は latency 面で有利です。

ただし episode 切替時に `/reset` が呼ばれるため、chunk cache、instruction、history を確実にリセットする必要があります。

### 2.5 L4 を基準に推論設計する

モデル選定は「GPU に載るか」だけでは不十分です。

本番では L4 1枚上で、ロード後の `/act` を 10 秒以内に安定して返す必要があります。

したがって確認対象は最低でも以下です。

- cold start model load time
- VRAM peak
- steady-state `/act` latency
- action chunk length
- quantization の有無
- FlashAttention / SDPA 互換性
- PyTorch 2.11 + CUDA13 互換性

### 2.6 OpenVLA-OFT+ はそのままではなく本戦 I/O adapter が重要

OpenVLA 系を使う場合、本戦の observation / action interface に合わせた変換が必要です。

最低限、次を明示的に管理するべきです。

- agentview / wrist image の mapping
- state vector の mapping
- quaternion / rotation representation
- action normalization / unnormalization
- gripper convention
- action chunk slicing
- instruction encoding

本戦では normalization config が評価側の秘密領域として隠されるため、submission は自分の training 時 normalization 情報を完全に同梱して自己完結させる必要があります。

---

## 3. 推奨攻略ロードマップ

### Phase 0: Submission infra を先に固定

モデル改善より先に、最低限の submission server を本番互換に固定する。

Acceptance criteria:

- `validate_submission.py --static` PASS
- dynamic validation PASS
- `/health` PASS
- `/reset` PASS
- `/act` float32 `(7,)`
- NaN / Inf なし
- `/act` p99 latency < 10 sec
- Docker 本番互換環境で起動

### Phase 1: baseline を本戦評価器で再測定

比較する候補例:

- SmolVLA baseline / fine-tune
- pi0.5 LoRA
- 既存 OpenVLA-OFT+ checkpoint

すべて同じ `python -m pipeline` 条件で比較する。

記録する指標:

- Track 1 / 2 / 3 success rate
- task 別 success rate
- latency
- VRAM
- steps to success
- failed episode の failure mode

### Phase 2: Track 2 robustness

未知 domain に対して augmentation を重点的に導入する。

優先候補:

- color / brightness / contrast
- texture randomization
- object pose variation
- camera perturbation
- instruction paraphrase

### Phase 3: Track 3 composition

単一 task の性能だけでなく、既知 skill の組み合わせへ汎化できる training mix を設計する。

- balanced multi-task sampling
- compositional instruction
- skill boundary を跨ぐ episode
- history / chunk 長の検討

### Phase 4: inference optimization

- mixed precision
- SDPA / FlashAttention
- model weight merge
- unnecessary dependency removal
- action chunking
- model initialization cache

目標は success rate を落とさず latency margin を十分確保すること。

### Phase 5: submission hardening

最終 zip について以下を毎回自動確認する。

- zip < 20GB
- external dependency なし
- model weights 全同梱
- clean venv から再現可能
- Docker 本番環境で end-to-end evaluate
- Track 1 / 2 / 3 smoke test

---

## 4. 今回の重要ポイント

本戦では、単純に「より大きい VLA を入れる」よりも、以下の4点の掛け算で性能が決まる可能性が高いです。

1. VLA 自体の能力
2. LIBERO-plus / hidden domain への fine-tuning
3. 本番 I/O / action normalization の整合
4. L4 + 10秒制約内での安定推論

特に Track 2 / 3 があるため、Track 1 の公開 task への過適合は危険です。

本戦向けには「公開タスクを覚える」ではなく「既知 skill を未知 visual domain と composition に持ち出せる policy」を目標にするべきです。

---

## 5. 次に実施する具体作業

1. 既存 OpenVLA-OFT+ submission と本戦 `policy_server.py` interface の差分調査
2. observation / action adapter の仕様書化
3. 本番 L4 を想定した latency / VRAM benchmark script 作成
4. Track 1 / 2 / 3 共通 evaluation runner 作成
5. failure analysis 用 result parser 作成
6. SmolVLA / pi0.5 / OpenVLA-OFT+ の同条件比較
7. hidden-domain を意識した augmentation plan 作成
8. 最終 submission を `validate_submission.py` と `evaluate.py` で自動検証する build script 作成

この順序なら、モデル学習を先に進めて最後に提出仕様で詰まるリスクを抑えつつ、本戦の評価軸に沿って改善できます。
