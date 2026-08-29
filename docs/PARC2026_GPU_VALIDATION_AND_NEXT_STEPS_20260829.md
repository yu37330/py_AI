# PARC2026 π0.5 GPU検証まとめと今後の対応方針（2026-08-29）

## 1. 目的

本書は、P&R2026本選環境で実施した以下の検証結果を整理し、60時間の配布GPU枠を無駄にしないための次アクションを定義する。

- 運営Dockerでの公式π0.5 baseline評価
- π0.5学習環境の構築
- LIBERO学習データの読込
- LoRA smoke training / merge
- batch size別VRAM probe
- 永続化 / Stop Server運用
- 今後の60時間GPU配分方針

既存の全体分析は `docs/PARC2026_FINAL_ANALYSIS.md` を参照。

---

## 2. 今回確認できたこと

### 2.1 公式baselineの評価経路は動作する

運営配布の `pi05_step005000_submission_py310.zip` を使い、運営Docker相当環境で以下を確認した。

- static validation: PASS
- Track1 end-to-end評価: 完走
- fresh Docker再試行では policy server ready 約11秒
- 4 task × 1 episodeで2成功 / 4試行
- local overall = 0.500

この `0.500` はローカルのsuccess-rateベースの値であり、非公開の本番Total Scoreや公式leaderboard参考値と直接比較しない。

最初の試行では120秒のserver startup timeoutが1回発生したが、その後の300秒診断およびfresh Docker / 120秒制限では再現しなかった。したがって現時点では「baseline自体が起動不能」ではない。

### 2.2 学習環境は構築できた

`examples/pi05_libero_finetune/scripts/setup_train.sh` により以下を確認した。

- Python 3.10.12
- FFmpeg 4.4.2
- LeRobot v0.4.4
- torch 2.10.0+cu128
- CUDA available
- torchcodec利用可
- gradient accumulation patch適用済み
- π0.5 config patch適用済み

評価環境と学習環境は依存バージョンが異なるため、提出前は必ずルート側の評価用環境 / 運営Dockerで再検証する。

### 2.3 LIBEROデータセットを正常に読めた

運営配布の約20GB tarを、一時NVMeへ展開して利用した。

展開先:

```text
/opt/dlami/nvme/parc_work/libero_combined_20hz
```

構造:

```text
data/
meta/
videos/
```

学習時の実測:

- frames: 3,028,708
- episodes: 19,533

`~/dataset` はread-onlyのため、学習用tarは `/opt/dlami/nvme` の一時領域へ展開する方針で問題ない。

### 2.4 Hugging Face gated modelへのアクセスを確認した

`HF_TOKEN` を設定し、`google/paligemma-3b-pt-224` のmodel info取得に成功した。

トークン値はログやリポジトリへ保存しない。シェル変数として都度設定する。

### 2.5 π0.5 LoRA smoke trainingからmergeまで完走した

20 effective stepsのsmokeを実行し、以下まで成功した。

```text
base model load
→ LIBERO data load
→ LoRA training
→ checkpoint save
→ LoRA merge
→ PaliGemma tokenizer同梱
→ merged checkpoint検証
```

実測:

- steps: 20
- batch size: 2
- LoRA rank: 16
- learnable params: 1,287,168
- total params: 3,618,044,688
- peak VRAM: 12,601 MiB
- total wall time: 85 sec
- merged checkpoint生成: 成功

base weight読込時に以下のwarningは出ている。

```text
Missing key(s) in state_dict:
model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
```

ただしsmoke training / checkpoint save / mergeは最後まで完走したため、現時点ではfatal errorではない。提出round-trip時に改めて確認する。

---

## 3. VRAM probeで判明した重要事項

### 3.1 最初のBS=16 probeはLoRAではなくfull fine-tuningになっていた

最初の `probe_pi05_bs.sh` には `--peft.method_type=LORA` / `--peft.r` がなく、実際にはほぼ全パラメータが学習対象になっていた。

誤ったprobeの実測:

- BS=16
- learnable params: 3,616,757,520
- peak VRAM: 79,949 MiB

これは本番LoRA条件のVRAM値として使用してはいけない。

この問題は PR #4 `fix: pi0.5 VRAM probeをLoRA本学習条件に合わせる` で修正中。

### 3.2 修正版LoRA probeの実測

#### BS=16 / LoRA r16

- learnable params: 1,287,168
- total params: 3,618,044,688
- peak VRAM: 44,249 MiB
- rate: 1.47 step/s（30 step終了時表示）
- steady-state update_s: 約0.51 sec / micro-step
- OOM: なし

#### BS=32 / LoRA r16

- learnable params: 1,287,168
- total params: 3,618,044,688
- peak VRAM: 80,435 MiB
- rate: 1.23 sec/step（30 step終了時表示）
- steady-state update_s: 約1.06 sec / micro-step
- OOM: なし

### 3.3 現時点ではBS=16を第一候補とする

full trainingの既定effective batchは128である。

候補:

| 条件 | Peak VRAM | 1 micro-step概算 | effective batch 128あたり概算 |
|---|---:|---:|---:|
| BS16 × GA8 | 44.2GB | 約0.51s | 約4.1s |
| BS32 × GA4 | 80.4GB | 約1.06s | 約4.2s |

BS32にしても今回の実測では明確な高速化が見られず、VRAM余裕だけが大きく減る。

したがって現時点の第一候補は:

```text
PI05_BS=16
PI05_GA=8
LoRA r=16
```

とする。

ただし、20k step本学習を開始する前にgradient accumulationの実動作確認を必須Gateとする。

---

## 4. gradient accumulationについて残っている確認事項

`grad-accum-env-var.patch` は以下の方針で実装されている。

- `LEROBOT_GRAD_ACCUM` を `Accelerator(gradient_accumulation_steps=...)` に渡す
- `accelerator.accumulate(policy)` を使用
- `accelerator.sync_gradients` のときだけeffective stepをincrement
- schedulerもoptimizer sync時のみ進める

実装上はGAに対応している。

一方、学習ログの `Effective batch size: batch_size x num_processes` 表示にはGA倍率が含まれていないため、ログ表示だけではGAが正しく効いているか判断しにくい。

### 本学習前の必須検証

配布GPUではなく、可能なら外部GPU / 別環境で以下を確認する。

1. `GA=1` と `GA=8` で同じeffective step数を実行
2. forward/backward micro-step回数を数える
3. optimizer.step / scheduler.stepが8 micro-stepごとに1回であることを確認
4. checkpoint resume後もstep semanticsが崩れないことを確認
5. 20k stepsが「20k micro-step」ではなく「20k optimizer update」であることを明確にする

この検証なしに約23時間規模の本学習を開始しない。

---

## 5. ストレージ運用で得た知見

今回 `parc-home-sync data-push` を実行したところ、約93.8GiBを同期し、完了まで約219秒かかった。

原因は `~/data` 配下に、保存必須でない以下の再生成可能データも置いていたため。

- LeRobot `.venv`
- Python site-packages
- Hugging Face cache
- 一部の大きな中間ファイル

### 次回以降の推奨配置

#### 永続化対象: `~/data`

- Git repository
- 学習checkpoint
- merged checkpoint
- training logs
- submission artifact
- 実験結果 / manifest

#### 一時領域: `/opt/dlami/nvme/...`

- 展開済みdataset
- `.venv`
- Hugging Face cache
- model download cache
- 再生成可能な一時ファイル

これにより `data-push` を軽量化し、停止前の時間消費を削減する。

---

## 6. GPU停止までの運用確認

今回の終了時には以下を実施した。

1. `parc-home-sync data-push` 完了
2. `File → Hub Control Panel`
3. `Stop Server`
4. JupyterHubが `Start My Server` 表示になったことを確認

この状態を「GPUサーバー停止完了」の確認方法とする。

ブラウザタブを閉じるだけでは停止扱いにしない。

---

## 7. 60時間GPU枠に対する方針変更

### 結論

**現時点では20k step本学習を開始しない。**

配布GPUで必要な「動くか / VRAMに載るか / mergeできるか」は十分確認できた。

今後は、比較検証やバグ確認を可能な限り外部環境で行い、60時間枠は「本番候補checkpointを作る処理」と「運営環境でしか意味のない最終評価」に集中させる。

---

## 8. 配布GPUを再開する前に外部で詰めること

優先順位順:

### Gate A: GA=8の実動作検証

上記4章の確認を完了する。

### Gate B: submission round-tripを完成させる

smokeで作成したmerged checkpoint相当を使い、以下を通す。

```text
merged checkpoint
→ submission/model_weights
→ zip
→ validate_submission
→ organizer Docker
→ policy server startup
→ Track1 1 episode
```

本学習後に提出形式の問題が発覚することを防ぐ。

### Gate C: 学習条件の比較

配布GPUではなく外部GPU等で、小規模比較を行う。

候補:

- LoRA rank: r16 / r32
- LR: 5e-5周辺
- steps / scheduler
- augmentation有無
- seed差

短いlossだけで決めず、可能なら同じ小規模LIBERO評価で比較する。

### Gate D: Track別のデータ戦略を確定

1 checkpointですべてのTrackを狙うか、track別checkpointを持つかを決める。

- Track1: 同task / 同domain性能
- Track2: unseen domainへのgeneralization
- Track3: inverse / reversed taskへの対応

特にTrack3は既存taskの逆操作であるため、通常LIBEROだけの追加学習で十分かを事前に検証する。

---

## 9. 60時間の暫定予算

BS16 × GA8のprobeから、20k optimizer stepsは純学習だけで約23時間規模になる可能性がある。

実測前提がまだGA検証前なので、この値は暫定見積もりとする。

配布GPU再開後の予算上限案:

| 用途 | 上限 |
|---|---:|
| 再起動 / dataset展開 / 環境確認 | 2h |
| Run A: 本命LoRA学習 | 23h |
| merge / packaging / organizer Docker評価 | 5h |
| Run B: evidenceがある場合のみ追加学習 / track特化 | 15h |
| 障害対応・再実行・最終確認buffer | 15h |
| 合計 | 60h |

Run Bは「とりあえず2本目をフル学習」には使わない。

Run A評価で改善が確認できた場合に、以下のような目的が明確な追加学習へ使う。

- checkpoint continuation
- Track2 generalization強化
- Track3 inverse data追加
- 明確に優位だった別LoRA条件

---

## 10. 次回GPU起動時の手順

再開時は、GPU時間を消費する前提で最短経路にする。

1. `~/data` restore状態確認
2. `/opt/dlami/nvme` にdataset tar展開
3. venv / HF cacheは一時NVMeへ構築
4. `HF_TOKEN` をシェルへ設定
5. GA検証済みscript / PRをmainへ反映
6. 2〜5 stepの最終smoke
7. Gate条件が全部OKならRun A開始
8. checkpointを定期保存
9. 学習後すぐmerge
10. organizer Dockerでsubmission round-trip
11. 結果保存後 `data-push`
12. Hub Control PanelからStop Server

---

## 11. 現時点のDecision

### 採用

- π0.5を継続して本命候補として評価
- LoRA方式
- BS=16を第一候補
- datasetは一時NVMeへ展開
- persistent storageはcheckpoint / logs / repo中心に限定
- 60時間GPU枠は本学習と公式環境評価へ温存

### 保留 / 要検証

- GA=8の実step semantics
- 20k stepsが最適か
- LoRA r16が最適か
- Track1/2/3共通checkpointでよいか
- warningの `embed_tokens.weight` が提出時に影響しないか
- merged checkpointのorganizer Docker round-trip

### やらない

- 検証なしで20k本学習を開始
- BS32を「VRAMに載るから」という理由だけで採用
- `/opt/dlami/nvme` の一時データを永続前提にする
- `.venv` / HF cacheを毎回約100GB規模で `data-push` する運用

---

## 12. 関連PR

- PR #3: baseline実測結果と学習setup進捗（merged）
- PR #4: `probe_pi05_bs.sh` をLoRA本学習条件へ修正（open）

次の開発作業は、PR #4を反映したうえで、GA検証とsubmission round-tripを優先する。
