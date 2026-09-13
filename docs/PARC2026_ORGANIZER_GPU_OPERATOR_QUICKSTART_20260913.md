# PARC2026 運営GPU Operator Quickstart（2026-09-13）

この文書は、配布GPUを起動したあとに迷わず実行するための**操作専用チートシート**。
設計・根拠は `PARC2026_ORGANIZER_GPU_EXECUTION_RUNBOOK_V2_20260913.md` を正本とする。

## 結論: Notebookセルではなく JupyterLab Terminal を使う

運営環境はJupyterHub/JupyterLabで起動するが、M3の実行はNotebookセルではなく **Terminal** を正本にする。

理由:

- 74/75/76は長時間処理で、shell logと終了codeを残した方が安全
- `source` した環境変数を同一terminalで維持できる
- session再開時の手順が明確
- notebook kernel restartの影響を受けない
- wrapperがpersistent logを自動保存する

NotebookはJSON/result/logの閲覧や簡単な確認に使ってよいが、**学習・screeningの起動元にはしない**。

`tmux` が既に入っていれば任意で使ってよい。

```bash
command -v tmux && tmux new -As parc2026
```

入っていなければ追加installは不要。JupyterLab Terminalをそのまま使う。

---

# A. GPUを起動する

ブラウザで公式JupyterHubを開く。

```text
https://parc-playground.omnicamp.us/hub/login
```

1. `Sign in with Auth0`
2. Omnicampus accountでlogin
3. GPU allocation完了まで待つ
4. JupyterLabが表示されたら起動完了

**この時点からGPU時間を消費していると考える。**

---

# B. Terminalを開く

JupyterLabで以下のどちらか。

```text
Launcher -> Terminal
```

または

```text
File -> New -> Terminal
```

以後、特記がなければ全コマンドをこのTerminalで実行する。

---

# C. 初回だけ: repositoryを取得する

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"

git clone --filter=blob:none https://github.com/yu37330/py_AI.git py_AI
cd py_AI

git fetch origin feat/organizer-gpu-runtime
git checkout --detach origin/feat/organizer-gpu-runtime

git status --short --branch
git rev-parse HEAD
```

ここでdetached HEADになるのは意図通り。GPU実行中にbranchをpullしてコードを変えない。

---

# D. 毎session: environmentを読み込む

```bash
cd "$HOME/src/py_AI"
source tools/benchmark/organizer_gpu_env.sh
```

Hugging Face tokenはterminalへだけ設定する。

```bash
export HF_TOKEN='hf_...'
```

token値はrepo / notebook / logへ書かない。

確認:

```bash
bash tools/benchmark/organizer_gpu_cli.sh status
```

見るべきもの:

```text
GPU = NVIDIA RTX PRO 6000 Blackwell
VRAM >= 90,000 MiB
repo SHAが表示される
scratch = /opt/dlami/nvme/parc2026
persistent = ~/data/parc2026-cache
HF_TOKEN = set (hidden)
```

GPU名/VRAMが違う場合はSTOP。閾値をその場で変更しない。

---

# E. 初回session: handoff bundleを置く

GPU起動前にColab側で作成した2ファイルだけを運ぶ。

```text
m3-organizer-handoff-v1.tar
m3-organizer-handoff-v1.tar.sha256.json
```

dataset / venv / HF cacheは運ばない。

Terminalで保存先を作る。

```bash
mkdir -p "$HOME/data/incoming"
```

JupyterLab左側File Browserで `data -> incoming` を開き、Uploadボタンから上記2ファイルをuploadする。

upload後:

```bash
ls -lh "$HOME/data/incoming"
```

**upload直後に永続化**する。

```bash
bash tools/benchmark/organizer_gpu_cli.sh push
```

---

# F. 初回session: preflight

```bash
bash tools/benchmark/organizer_gpu_cli.sh preflight
```

ここではbenchmark trainingは開始しない。

PASS marker:

```text
hardware PASS
M3 DISK HEADROOM: PASS
OPERATOR PREFLIGHT: PASS
```

FAILしたら74へ進まない。

---

# G. 初回session: handoff restore

```bash
bash tools/benchmark/organizer_gpu_cli.sh restore
```

PASS marker:

```text
M3 ORGANIZER HANDOFF INTEGRITY: PASS
M3 ORGANIZER HANDOFF RESTORE: PASS
```

restore後にsource Notebook73 artifactsを書き換えない。

---

# H. 74: 60 episode smoke

```bash
bash tools/benchmark/organizer_gpu_cli.sh 74
```

このcommandがruntime再構築も行う。

最終PASS:

```text
ORGANIZER GPU 74 EQUIVALENT: PASS
```

74 PASS前にdataset/75へ進まない。

---

# I. exact datasetをNVMeへ作る

```bash
bash tools/benchmark/organizer_gpu_cli.sh dataset
```

PASS:

```text
M3 ORGANIZER DATASET: PASS
```

保存先はNVMe。Server停止で消えてよい。

---

# J. Blackwell training compatibility

```bash
bash tools/benchmark/organizer_gpu_cli.sh compat
```

PASS:

```text
M3 ORGANIZER TRAINING COMPATIBILITY: PASS
ready_for_75=true
```

ここで初めてBlackwell上で3モデルのbounded training compatibilityが確認される。

---

# K. 75 forward

```bash
bash tools/benchmark/organizer_gpu_cli.sh 75-forward
```

PASS:

```text
M3 ORGANIZER FORWARD BENCHMARK: PASS
```

終わったら必ず:

```bash
bash tools/benchmark/organizer_gpu_cli.sh stop-ready
```

`stop-ready` は `parc-home-sync data-push` まで行う。
その後、ブラウザで:

```text
File -> Hub Control Panel -> Stop Server
```

`Start My Server` 表示を確認する。

**browser tabを閉じるだけでは停止にならない。**

---

# L. 2回目以降のsession開始

GPUを再度起動し、Terminalを開く。

最初にpersistent dataを戻す。

```bash
cd "$HOME/src/py_AI"
source tools/benchmark/organizer_gpu_env.sh
export HF_TOKEN='hf_...'

bash tools/benchmark/organizer_gpu_cli.sh pull
bash tools/benchmark/organizer_gpu_cli.sh preflight
```

`$HOME` のrepoは自動persistentなので、**git pullしない**。
同じdetached SHAを使い続ける。

確認:

```bash
git rev-parse HEAD
echo "$PARC_EXPECTED_SOURCE_SHA"
```

一致必須。

NVMe datasetは消えているので再生成する。

```bash
bash tools/benchmark/organizer_gpu_cli.sh dataset
```

75 wrapperは必要なtraining runtimeをNVMeへ再構築するため、個別にvenv setup commandを打つ必要はない。

---

# M. 75 reverse

```bash
bash tools/benchmark/organizer_gpu_cli.sh 75-reverse
```

PASS:

```text
M3 ORGANIZER REVERSE BENCHMARK: PASS
```

その後:

```bash
bash tools/benchmark/organizer_gpu_cli.sh stop-ready
```

---

# N. 76: 1 checkpoint = 80 episodesずつ実行

76ではdatasetは不要。各unit commandがsimulator runtimeをNVMeへ再構築する。

## forward / equal_data / pi05

```bash
bash tools/benchmark/organizer_gpu_cli.sh 76-unit \
  forward equal_data pi05 unit-1
```

## forward / equal_data / smolvla

```bash
bash tools/benchmark/organizer_gpu_cli.sh 76-unit \
  forward equal_data smolvla unit-1
```

## forward / equal_data / openvla_oft

```bash
bash tools/benchmark/organizer_gpu_cli.sh 76-unit \
  forward equal_data openvla_oft unit-1
```

## forward / equal_wall

同様に3モデル。

```bash
bash tools/benchmark/organizer_gpu_cli.sh 76-unit forward equal_wall pi05 unit-1
bash tools/benchmark/organizer_gpu_cli.sh 76-unit forward equal_wall smolvla unit-1
bash tools/benchmark/organizer_gpu_cli.sh 76-unit forward equal_wall openvla_oft unit-1
```

6 unit揃ったら:

```bash
bash tools/benchmark/organizer_gpu_cli.sh 76-finalize forward
```

reverseも同じ。

```bash
bash tools/benchmark/organizer_gpu_cli.sh 76-unit reverse equal_data openvla_oft unit-1
bash tools/benchmark/organizer_gpu_cli.sh 76-unit reverse equal_data smolvla unit-1
bash tools/benchmark/organizer_gpu_cli.sh 76-unit reverse equal_data pi05 unit-1
bash tools/benchmark/organizer_gpu_cli.sh 76-unit reverse equal_wall openvla_oft unit-1
bash tools/benchmark/organizer_gpu_cli.sh 76-unit reverse equal_wall smolvla unit-1
bash tools/benchmark/organizer_gpu_cli.sh 76-unit reverse equal_wall pi05 unit-1

bash tools/benchmark/organizer_gpu_cli.sh 76-finalize reverse
```

1 unitが途中失敗した場合、同じ `unit-1` を消さない。
次だけ新しいattempt名にする。

```bash
bash tools/benchmark/organizer_gpu_cli.sh 76-unit \
  forward equal_data pi05 unit-2
```

成功済みunitは再実行されない。

各unitの合間に12時間残量をJupyterHub UIで確認する。
残時間が怪しければ新unitを開始せず:

```bash
bash tools/benchmark/organizer_gpu_cli.sh stop-ready
```

---

# O. 77 promotion

forward/reverse finalizerが両方PASSした後:

```bash
bash tools/benchmark/organizer_gpu_cli.sh 77
```

expected:

```text
77 M3 FORWARD/REVERSE PROMOTION: READY
```

その後:

```bash
bash tools/benchmark/organizer_gpu_cli.sh stop-ready
```

---

# P. statusはいつでも実行してよい

迷ったら処理を始める前に:

```bash
bash tools/benchmark/organizer_gpu_cli.sh status
```

これはtraining/evaluationを開始しない。

persistent operator log:

```text
~/data/parc2026-cache/operator-logs/
```

すべてのCLI commandはstdout/stderrをここへteeする。

---

# Q. session別の最短コマンド

## Session A

```bash
cd ~/src/py_AI
source tools/benchmark/organizer_gpu_env.sh
export HF_TOKEN='hf_...'

bash tools/benchmark/organizer_gpu_cli.sh preflight
bash tools/benchmark/organizer_gpu_cli.sh restore
bash tools/benchmark/organizer_gpu_cli.sh 74
bash tools/benchmark/organizer_gpu_cli.sh dataset
bash tools/benchmark/organizer_gpu_cli.sh compat
bash tools/benchmark/organizer_gpu_cli.sh 75-forward
bash tools/benchmark/organizer_gpu_cli.sh stop-ready
```

## Session B

```bash
cd ~/src/py_AI
source tools/benchmark/organizer_gpu_env.sh
export HF_TOKEN='hf_...'

bash tools/benchmark/organizer_gpu_cli.sh pull
bash tools/benchmark/organizer_gpu_cli.sh preflight
bash tools/benchmark/organizer_gpu_cli.sh dataset
bash tools/benchmark/organizer_gpu_cli.sh 75-reverse
bash tools/benchmark/organizer_gpu_cli.sh stop-ready
```

## Session C以降

```bash
cd ~/src/py_AI
source tools/benchmark/organizer_gpu_env.sh
export HF_TOKEN='hf_...'

bash tools/benchmark/organizer_gpu_cli.sh pull
bash tools/benchmark/organizer_gpu_cli.sh preflight

# 76-unitを残時間に応じて1つずつ実行
# 終了時:
bash tools/benchmark/organizer_gpu_cli.sh stop-ready
```

---

# R. やらないこと

GPU起動後に以下を即興で行わない。

```text
git pull / branch切替
D10再生成
manifest差し替え
72/73再実行
pip installを手動で足す
CUDA/Torchを手動変更
VRAM gateを下げる
failed evidence削除
full RLDS materialization
独自Docker構築
```

依存/runtime修正が必要になった場合は、GPU上で場当たりpatchせず、失敗evidenceを保存してrepo側で修正する。
