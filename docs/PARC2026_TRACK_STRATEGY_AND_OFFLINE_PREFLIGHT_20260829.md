# PARC2026 Track戦略と本番学習前オフライン検証（2026-08-29）

## 1. Decision: 最初からTrack別に3本フル学習しない

Track1〜3は別モデルを提出しても同一モデルを提出してもよいが、配布GPUは60時間しかないため、最初から3本を独立フル学習する方針は採らない。

現時点の基本戦略は次のとおり。

```text
π0.5 base
  ↓
Run A: 3Track共通モデル
  ├─ Track1: 原則そのまま提出候補
  ├─ Track2: 原則そのまま提出候補
  └─ Track3: まず共通モデルで評価
                 ↓
        Track3が明確に弱い場合のみ
                 ↓
        Run B: Track3特化continuation
```

### Run A

Run AはTrack1専用ではなく、Track1〜3すべての基盤となる共通LoRAモデルとする。

狙い:

- Track1: 同task / 同domainで基礎成功率を上げる
- Track2: 視覚・domain変化へのgeneralizationを維持する
- Track3: inverse / reversed taskにも最低限対応できる汎用操作能力を持たせる

Track1/2専用モデルは、leaderboardまたはローカル評価で共通モデルからの改善Evidenceがある場合だけ検討する。

### Run B

Run Bは「2本目をとりあえず学習」ではなく、Run Aの評価結果に基づく目的付き追加学習とする。

第一候補はTrack3特化continuation。

理由:

- 公式baseline参考値ではTrack3が最も弱い
- Track3は既存LIBERO taskのinverse / reversed方向で、通常のforward task中心学習だけでは不足する可能性がある
- Track1/2は共通表現・操作能力を共有しやすい一方、Track3はデータ分布と目標方向の差が大きい

ただし単純なaction列の時間反転をそのまま正解データとみなさない。物体配置・接触・gripper状態・初期状態の整合が崩れるため、inverse task用データはsimulator上で成立性を確認したtrajectory、またはルール上許可された学習データとして構築する。

---

## 2. 本番学習前に「運営GPUを使わず」完了させること

ここでいうオフラインは「運営配布GPUを消費しない」という意味。CPUだけに限定せず、手元GPU、Colab、外部クラウドGPU、外部L4等を使ってよい。

### Gate O1: gradient accumulation semanticsを確定

最優先。`PI05_GA=8` が本当に8 micro-stepごとに1 optimizer updateになっていることを検証する。

確認項目:

- GA=1 / GA=2 / GA=8でforward/backward回数を計測
- `optimizer.step()` 回数を明示ログ化
- `scheduler.step()` 回数を明示ログ化
- `accelerator.sync_gradients` のtrue周期を確認
- checkpointのstep番号がoptimizer update数と一致
- resume後もstep semanticsが維持される
- effective batch = `batch_size × GA × world_size` をログ表示する

合格条件:

- GA=8時、8 micro-stepにつきoptimizer/schedulerが1回だけ進む
- `PI05_STEPS=20` が20 optimizer updatesを意味する

### Gate O2: 学習scriptを固定して再現性を作る

本学習開始前に以下をmanifestへ固定する。

- Git commit SHA
- LeRobot revision
- transformers fork revision
- base checkpoint revision
- PaliGemma tokenizer revision
- dataset path / version / manifest / episode数
- LoRA rank
- LR / scheduler
- BS / GA
- seed
- save interval
- model merge手順
- submission packaging手順

「本番GPUを起動してからコードを直す」を避ける。

### Gate O3: smoke merged modelでsubmission round-trip

本学習済みモデルでなくてよい。既に作成した20-step smoke merged checkpointを使い、提出経路を先に完成させる。

```text
merged checkpoint
→ submission/model_weights
→ requirements確認
→ zip作成
→ static validate_submission
→ organizer Docker相当環境
→ policy server startup
→ inference request
→ Track1少数episode
```

可能なら外部のNVIDIA L4 24GBで実行する。本番評価GPUと同容量なので、98GBの学習GPU上で動くことより価値が高い。

確認するもの:

- zip <= 20GB
- expanded <= 40GB
- weights <= 20GB
- 起動120秒以内
- 推論10秒以内
- L4 24GBでOOMしない
- tokenizer / merged checkpointの相対pathが壊れない
- `embed_tokens.weight` warningが実推論結果へ影響しない

### Gate O4: Track1/2/3共通モデル用データ方針を固定

Run Aの学習対象episodeを事前にmanifest化する。

最低限:

- task別episode数
- suite別episode数
- 重複の有無
- 失敗trajectory混入の有無
- action / state / image shape
- fps / timestamp整合
- `drop_n_last_frames=49`の影響
- train / local-eval分離

Track1だけへ過適合するsamplingにしない。Track2のdomain generalizationを落としにくい構成を優先する。

### Gate O5: Track3 inverseデータ案を小さく検証

Run B用のデータを本番後に考え始めない。

外部環境で先に以下を比較する。

- 共通Run A相当のデータのみ
- inverse task用trajectoryを少量追加
- inverse sample比率を変えた場合

評価はlossだけでなく、inverse taskの実successを優先する。

注意:

- naiveなaction sequence reversalは採用しない
- runtimeでtask名を読んでFSM/hardcoded action tableに分岐する方式は使わない
- Track3対策はtraining-timeのデータ / model adaptationとして行う

### Gate O6: 小規模hyperparameter比較

運営GPU上でr16/r32、LR、augmentationを総当たりしない。

外部GPUで少数stepまたは短縮runを行い、候補を1〜2個まで絞る。

優先度:

1. r16 baseline
2. LR 5e-5周辺
3. augmentation有無
4. 必要ならr32

短縮runのtraining lossだけでは決めず、固定されたlocal eval subsetで比較する。

### Gate O7: 24GB inference制約を先に潰す

学習GPUは約98GBだが、本番/leaderboard側の推論GPUはL4 24GB。

本番学習前に、少なくともsmoke merged modelで以下を計測する。

- policy server peak VRAM
- startup time
- first inference latency
- steady inference latency
- action chunk生成時間

学習が成功してもL4に載らなければ提出できないため、このGateは学習より前に通す。

### Gate O8: storage / restore手順を軽量化

今回 `data-push` が約93.8GiBになったため、次回は以下を分離する。

`~/data`:

- repo
- checkpoints
- merged model
- logs
- manifests
- submission zip

`/opt/dlami/nvme`:

- dataset展開
- venv
- HF cache
- model download cache
- 再生成可能な中間物

次回起動後に手作業で迷わないよう、dataset展開・venv構築・HF_HOME設定をscript化しておく。

---

## 3. オフラインで用意しておく成果物

運営GPUを再起動する時点で、以下が揃っている状態を目標とする。

- [ ] PR #4相当のLoRA probe修正がmainへ入っている
- [ ] GA=8 verification log
- [ ] effective batch表示修正または確認script
- [ ] Run A training manifest
- [ ] Run A exact command
- [ ] Run A resume command
- [ ] Run A merge command
- [ ] smoke merged submission zip
- [ ] static validator PASS
- [ ] 外部L4 24GBでstartup / inference PASS（可能なら）
- [ ] Track1固定local eval subset
- [ ] Track2評価方針
- [ ] Track3 inverse eval subset
- [ ] Track3用追加データ案
- [ ] Run Bを発動する判定条件
- [ ] storage配置script
- [ ] data-push前チェックlist

---

## 4. 60時間の使い方をTrack戦略込みで更新

| 用途 | 上限 | 方針 |
|---|---:|---|
| 再起動 / restore / 最終smoke | 2h | ここで開発しない |
| Run A: 3Track共通LoRA | 23h | 最優先 |
| merge / package / 3Track評価 | 5h | AをTrack1/2/3すべてで測る |
| Run B: Track3特化continuation | 15h | AでTrack3弱点が確認された場合のみ |
| 障害 / resume / 最終確認buffer | 15h | 未使用なら追加改善へ回す |
| 合計 | 60h | |

Run AをTrack1/2/3へ共通提出候補として評価した後、Run Bの必要性を判断する。

Run B発動の例:

- Track1/2は改善したがTrack3だけ明確に失敗
- inverse local evalで共通モデルがbaseline近辺のまま
- Track3用追加データの外部小規模検証で改善Evidenceがある

逆にEvidenceがなければRun Bは実行せず、bufferをRun A continuationや再評価に残す。

---

## 5. 本番GPU再開Gate

以下がすべてYesになるまで20k本学習を開始しない。

| Gate | 必須 |
|---|---|
| GA=8 semantics確認 | Yes |
| Run A command固定 | Yes |
| dataset manifest固定 | Yes |
| checkpoint resume確認 | Yes |
| merge確認 | Yes |
| submission static validation | Yes |
| L4 24GB inference確認 | Strongly recommended |
| Track3方針決定 | Yes |
| Run B発動条件決定 | Yes |
| storage / sync運用固定 | Yes |

このGateを通した後、運営GPUは「検証環境」ではなく「本番checkpoint生成環境」として使う。
