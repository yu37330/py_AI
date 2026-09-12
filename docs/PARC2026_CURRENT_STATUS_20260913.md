# PARC2026 現状まとめ・次アクション（2026-09-13）

## 1. この文書の位置づけ

本書は、2026-09-13 時点の PARC2026 / M3 系の**現在地を短時間で把握するための進捗スナップショット**である。

長期の障害履歴・再発防止・PR台帳は以下を参照する。

- `docs/PARC2026_M3_OPERATIONAL_CAUTIONS_20260912.md`
- `docs/PARC2026_M3_PREEXECUTION_AUDIT_ADDENDUM_20260912.md`
- 過去の current status: `docs/PARC2026_CURRENT_STATUS_20260906.md`

本書は「いま何が完了していて、何が未完了で、次に何をすべきか」を正本として扱う。

---

## 2. 2026-09-13 時点の結論

現在地は以下。

```text
Dataset / D10
  V2_SQRT_BALANCED_RAW 固定
  group-aware manifest 固定
  D10 hash 固定
  69b RLDS smoke PASS
  69c streaming bridge validation PASS

M3 readiness
  71 preflight PASS
  72a π0.5 batch probe PASS
  72b SmolVLA batch probe PASS
  72c OpenVLA-OFT batch probe PASS
  72d controller PASS
  73 A100 training smoke PASS

Simulator validation
  74 minimal simulator smoke
    attempt 1〜6: runtime/evaluator incompatibilityを段階的に修正
    attempt 6: π0.5 libero_spatial 10/10 successまで到達
    attempt 7〜8: Colab storage exhaustionで停止
    74全体 60 episodes PASS はまだ未達

Full M3 benchmark
  75a / 75b: 未実行
  76a / 76b: 未実行
  77 promotion: 未実行
  final 800-episode evaluation: 未実行
```

**重要:** 1800秒の本番M3 benchmark trainingはまだ開始していない。

---

## 3. 固定済み Dataset / D10 contract

選択dataset:

```text
variant: V2_SQRT_BALANCED_RAW
dataset: lerobot/libero_plus
revision: f3f49f426d75030177b18778374005bc12ccd588
selected episodes: 10,758
selected frames: 1,620,614
selected episode IDs SHA256:
73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239
```

V2 manifest file SHA256:

```text
cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395
```

Drive backup:

```text
/content/drive/MyDrive/parc2026-cache/pi05-ablation-group-aware-v2/
  dataset_ablation_manifests_v2_group_aware
```

この選択は今後のM3実行で再選定しない。manifestを再生成して別入力に変えることも禁止する。

---

## 4. 69b / 69c の成立状況

### 69b

RLDS smoke は PASS 済み。

確認済み:

```text
smoke episodes: 8
smoke frames: 1,206
full projected size: 56.61 GiB
threshold: 35 GiB
recommendation: STREAMING_BRIDGE_RECOMMENDED
```

したがって full RLDS materialization は避ける。

### 69c

Streaming bridge validation は PASS 済み。

```text
status: READY_FOR_RUNNER
```

OpenVLA-OFTは full RLDS 展開ではなく、確定済みのstreaming pathを使う。

---

## 5. M3 preflight / batch probe の成立状況

以下は完了済み。

```text
71  preflight                     PASS
72a π0.5 A100 batch probe         PASS
72b SmolVLA A100 batch probe      PASS
72c OpenVLA-OFT A100 batch probe  PASS
72d controller                    PASS
```

主な確定事項:

- A100 >= 38 GiB をM3基準とする
- effective batch size = 32
- 3モデルは同一D10を使う
- forward / reverse fairnessを維持する
- 72 probe evidenceを不要に再実行しない

---

## 6. Notebook 73: training smoke

Notebook 73 は A100 で PASS 済み。

対象:

- π0.5
- SmolVLA
- OpenVLA-OFT

確認したもの:

- 実checkpoint生成
- 2 optimizer updates
- effective batch 32
- D10固定
- OpenVLA streaming training path
- fresh runtime再構成

明確に区別すること:

```text
73 = training smoke
75a / 75b = full M3 benchmark training
```

73 PASS は本番1800秒trainingを開始済みという意味ではない。

---

## 7. Notebook 74: minimal simulator smoke の進捗

### 契約

```text
suite: libero_spatial
models: π0.5 / SmolVLA / OpenVLA-OFT
seeds: 20260906 / 20260907
10 tasks × 2 seeds × 3 models = 60 episodes
max steps: 300
promotion evidence: false
```

Notebook 73の smoke checkpointのみを使う。

### attempt履歴

| Attempt | 到達点 / 原因 | 対応 |
|---:|---|---|
| 1 | hf-libero configが対話入力を要求 | non-interactive `LIBERO_CONFIG_PATH` を追加 |
| 2 | Matplotlib / EGL / GPU runtime差異 | `MPLBACKEND=Agg`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` |
| 3 | OpenVLAの固定LIBERO namespace import不整合 | pinned checkoutを `.pth` 経由で公開しimport pathを検証 |
| 4 | runtime setup PASS後、MuJoCo 3.13 + robosuite 1.4で失敗 | `mujoco==3.3.1` に固定 |
| 5 | π0.5 checkpoint load後、camera feature名不一致 | checkpoint `config.json` を読み `image/image2 -> front/wrist` rename map |
| 6 | π0.5で10 tasks完走、10/10 success | post-rolloutの `upstream/` 出力dir生成を追加 |
| 7 | child failure後、診断中にも `Errno 28` | disk diagnosticsを強化 |
| 8 | headroom preflight後も実行中に `No space left on device` | Colab storage容量が現在の主blocker |

### attempt 6 で実機確認できたこと

π0.5について、以下は実際に到達済み。

```text
runtime setup
checkpoint load
camera rename
LIBERO environment creation
model inference
simulator stepping
libero_spatial 10 tasks completion
10 / 10 success
```

したがって74の問題は、初期の「simulatorそのものが動かない」段階からは大きく前進している。

### attempt 8 の現状

実行pin:

```text
1b983910509020dc2a01f1474140d72bf5664e8e
```

今回確認されたエラー:

```text
OSError: [Errno 28] No space left on device
```

診断対象pathはSmolVLA側まで進んでいたが、診断ログ自体の読み取りも容量不足で失敗しているため、**attempt 8の完了episode数や元のchild exceptionを断定しない**。

74全体については、まだ次のPASSを得ていない。

```text
=== 74 M3 MINIMAL SIMULATOR SMOKE: PASS ===
=== 74 COMPLETE ===
```

---

## 8. 現在の主要blocker: Colab storage

現在のblockerはGPU演算能力そのものより**Colab runtime / mounted storageの容量不足**。

attempt 8では実行前に以下を追加済み。

- `/content` free space確認
- Drive free space確認
- uv / pip / apt の再生成可能cacheのみ削除
- D10 / checkpoint / HF cache / 過去evidenceは保護
- bounded diagnostics
- partial episode evidence保存

それでも実行中に `Errno 28` が発生した。

したがって、同じColab構成でattempt番号だけ変えて再実行するのは避ける。

---

## 9. 運営側GPUへの移行候補

Colab容量枯渇を受け、運営側GPUで74以降を実行する案を検討中。

**2026-09-13時点では、運営側GPUの以下の仕様はまだこのrepo上で確定していない。**

確認が必要:

```text
GPU model
VRAM
OS
NVIDIA driver / CUDA
local SSD total/free capacity
persistent storage capacity/path
internet access
GitHub access
Hugging Face access / HF_TOKEN handling
Python / uv / Docker利用可否
session timeout / job timeout
```

最低限、M3 runtime guardとの整合上:

```text
A100 class or equivalent
VRAM >= 38 GiB
十分なlocal SSD headroom
```

を確認する。

### 移行時に守ること

- D10を再選定しない
- manifestを再生成しない
- 73 checkpointを別物へ置換しない
- 72 probeを自動で再実行しない
- 74はまずminimal smokeのまま実行する
- organizer GPUで74がPASSするまで75本番へ進まない
- filesystem path差分はenv/config化し、benchmark contractを変えない

`PARC_DRIVE_ROOT` など現在Driveを指している保存rootは、運営環境のpersistent storageへ置き換える設計を優先する。

---

## 10. PR / branch の現在地

### PR #67

```text
#67 feat: add guarded M3 LIBERO evaluation executor
state: OPEN
merged: false
branch: feat/m3-libero-executor
```

Notebook 74、76 screening、77 promotionにつながる主要評価PR。

**74 PASS前にマージ完了扱いにしない。**

### PR #66

M3 model adapters / training実装の主系統。mainへ統合済み。

### PR #70

alternate training implementation。主系統として採用しない。

### PR #68

final candidate → 800 episode evaluation → artifact freeze / submission側の実装。

実M3 evidence待ちであり、現時点でfinal executionへ進めない。

### PR #77 / #78

M3運用注意事項とaudit/handoff ledgerのドキュメント整備。

mainへ統合済み。

---

## 11. 75a / 75b は現状のまま実行禁止

75a / 75b は本番M3 training。

契約:

```text
sample budget: 4,800 / model
equal wall time: 1,800 sec / model
3 models
forward + reverse
D10 fixed
effective batch size: 32
```

ただし現在の75 notebookは古いruntime pinを持つため、**74 PASS直後にそのまま実行しない**。

実行前に必ず:

1. 73で確定したfresh-runtime / OpenVLA training修正を取り込む
2. 74 PASSを明示gateに追加
3. fresh machine / fresh session bootstrapを確認
4. D10 hashを再検証
5. 1800秒training開始前にpreflightで止められる構造にする

を行う。

---

## 12. 76a / 76b / 77 への伝播事項

74で見つかったevaluation修正は76 screeningでも必要。

必須:

- non-interactive LIBERO config
- headless Agg + EGL
- pinned LIBERO namespace
- `mujoco==3.3.1`
- camera feature rename
- output directory precreation
- OpenVLA JIT merge / cleanup
- disk headroom / partial evidence

76の古いコードをそのまま走らせない。

77 promotionでは simulator success rateをprimary metricとし、最大2モデルまで昇格する。

---

## 13. ここからの推奨順序

### Path A: 運営側GPUへ移行する場合

```text
1. 運営GPU仕様を確認
2. persistent storage容量を確認
3. 74 runtimeをgeneric filesystem対応へ調整
4. D10 / 73 checkpoint / manifestを安全に配置
5. 74 minimal simulator smokeを再実行
6. 60 episode records + PASSを確認
7. #67 final review / merge判断
8. 75a / 75bを最新版へrefresh
9. 75a / 75b full M3 benchmark
10. 76a / 76b screening
11. 77 promotion
12. selected candidateのみ800 episode final evaluation
13. artifact freeze / fresh-session reproduction / submission
```

### Path B: Colabを継続する場合

まず容量原因を定量化し、十分な空き容量が確保できる構成へ変更してから74を再実行する。

同一構成でattempt番号だけ増やして再試行しない。

---

## 14. 絶対に壊さないもの

```text
D10 selected episode IDs
D10 SHA256
V2 group-aware manifest
Notebook 73 PASS checkpoint evidence
72a/72b/72c/72d PASS evidence
69b / 69c PASS evidence
past 74 attempt evidence
```

失敗attemptを成功扱いに書き換えない。

---

## 15. 現在のGo / No-Go

| Stage | Status | Go? |
|---|---|---|
| D10 selection | PASS / frozen | GO |
| 69b RLDS smoke | PASS | GO |
| 69c streaming bridge | PASS | GO |
| 71 preflight | PASS | GO |
| 72a-d probes/controller | PASS | GO |
| 73 training smoke | PASS | GO |
| 74 minimal simulator smoke | **NOT PASS** | **NO-GO** |
| 75a/75b full benchmark | NOT STARTED | NO-GO |
| 76 screening | NOT STARTED | NO-GO |
| 77 promotion | NOT STARTED | NO-GO |
| final 800 eval | NOT STARTED | NO-GO |

現在の唯一の正しい次Gateは、**十分なstorageを持つ実行環境で74を60 episodesまでPASSさせること**。

---

## 16. Snapshot metadata

```text
snapshot date: 2026-09-13
main at document creation: 86d74afe9cc6ce634266a5f03eff80322a8c732f
PR #67 head observed: 4bc24f5e9b4c9bbb092bbf6ea303cca395260994
Notebook 74 attempt 8 runtime pin: 1b983910509020dc2a01f1474140d72bf5664e8e
D10 hash: 73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239
```

この文書の次回更新タイミング:

- 運営側GPUの仕様が確定したとき
- Notebook 74がPASSしたとき
- 75a / 75bのrefreshが完了したとき
- full M3 benchmarkを開始/完了したとき
