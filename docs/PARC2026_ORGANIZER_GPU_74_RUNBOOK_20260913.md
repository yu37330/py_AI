# PARC2026 運営GPU Notebook 74 移行Runbook（2026-09-13）

## 1. 目的

PR #79 / #80 / #81で整理した方針に基づき、Colabでstorage exhaustionしているNotebook 74相当のminimal simulator smokeを、60時間の運営GPU / JupyterHub環境で安全に実行するための手順を定義する。

このRunbookの対象は **74の60 episode smokeだけ**。

以下は開始しない。

- 72a〜72d probeの再実行
- 75a / 75bの1800秒benchmark training
- 76 screening
- 77 promotion
- final 800 episode evaluation

---

## 2. 固定contract

```text
D10:
73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239

suite:
libero_spatial

models:
pi05 / smolvla / openvla_oft

seeds:
20260906 / 20260907

10 tasks x 2 seeds x 3 models = 60 episodes
promotion evidence = false
max steps = 300
```

Notebook 73のforward/equal-data smoke checkpointだけを使用する。

---

## 3. 運営GPUで使うstorage role

推奨:

```text
高速一時領域:
/opt/dlami/nvme/parc2026

永続領域:
~/data/parc2026-cache
```

役割:

### local scratch

- source checkout
- venv
- Hugging Face cache
- torch cache
- LIBERO checkout
- OpenVLA runtime
- 再生成可能な一時データ

### persistent storage

- Notebook 73 PASS summary
- Notebook 73 train_result
- exact smoke checkpoints
- 74 runtime/eval logs
- 74 episode records
- 74 summary
- organizer migration inventory

`~/data` にvenvやHF cacheを置かない。

---

## 4. 事前に転送するartifact

最低限、Colab / Drive側の次のtreeを同じrelative layoutでpersistent rootへ転送する。

```text
model-benchmark-v1/
  m3-training-smoke-v1/
    m3_training_smoke_summary.json
    forward/
      equal_data/
        seed-20260906/
          pi05/
          smolvla/
          openvla_oft/
```

各model配下では `train_result.json` と、その `checkpoint_ref` が指していたcheckpoint本体を保持する。

重要:

- 元の `train_result.json` は書き換えない
- Drive絶対pathの `checkpoint_ref` はそのまま保存する
- organizer launcherが `model-benchmark-v1/...` のrelative suffixで移行先を検証し、attempt-local runtime viewのみを書き換える
- D10 hash / source_ref / sample count / optimizer updateはfail-closedで検証する

監査用として以下もpersistent storageへ残すことを推奨する。

- D10 manifest
- 69b / 69c PASS evidence
- 72a〜72d PASS evidence

ただし74のためにこれらを再生成しない。

---

## 5. GPU起動直後

まずコードを動かす前に確認する。

```bash
nvidia-smi
df -h /opt/dlami/nvme ~/data
df -i /opt/dlami/nvme ~/data
```

launcherも同じinventoryをpersistent storageへ保存する。

既存の74 runtime setupは現在もA100 >= 38,000 MiBを要求する。

したがって:

- A100なら既存contractのまま進む
- A100以外なら、その場で閾値を下げない
- GPU名 / VRAM / driverを保存してSTOPし、hardware compatibilityを別変更としてレビューする

L4 24GBへ無理に合わせない。L4は最終submission互換性確認用の評価環境として扱う。

---

## 6. source checkout

GPU時間を使って大きな調査をしない。

実行対象branchをexact checkoutする。

```bash
export PARC_LOCAL_SCRATCH_ROOT=/opt/dlami/nvme/parc2026
export PARC_PERSIST_ROOT=$HOME/data/parc2026-cache

mkdir -p "$PARC_LOCAL_SCRATCH_ROOT"
cd "$PARC_LOCAL_SCRATCH_ROOT"

git clone https://github.com/yu37330/py_AI.git py_AI
cd py_AI
git fetch origin feat/organizer-gpu-runtime
git checkout --detach origin/feat/organizer-gpu-runtime

git rev-parse HEAD
```

PRが更新された場合は、実行直前のhead SHAを記録する。

---

## 7. 環境変数

`HF_TOKEN` はshell変数のみで設定し、repo / logへ保存しない。

```bash
export PY_AI_REPO=$PARC_LOCAL_SCRATCH_ROOT/py_AI
export PARC_M3_EXECUTE=1
export PARC_M3_SIM_SMOKE_ATTEMPT=organizer-1
export PARC_EXPECTED_SOURCE_SHA=$(git -C "$PY_AI_REPO" rev-parse HEAD)
export HF_TOKEN='...'
```

launcherが内部で以下を設定する。

```text
PARC_ROOT -> PARC_LOCAL_SCRATCH_ROOT
PARC_DRIVE_ROOT -> PARC_PERSIST_ROOT  (legacy helper alias)
PARC_M3_ALLOW_ARTIFACT_REBASE=1
LIBERO_CONFIG_PATH -> local scratch
HF_HOME / TORCH_HOME / XDG_CACHE_HOME -> local scratch
MPLBACKEND=Agg
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

---

## 8. 74相当を実行

```bash
cd "$PY_AI_REPO"
python -u tools/benchmark/run_m3_organizer_simulator_smoke.py
```

launcherの順序:

```text
1. source SHA確認
2. GPU / disk inventory保存
3. Notebook73 handoff identity検証
4. local / persistent headroom preflight
5. non-interactive LIBERO config
6. simulator runtime再構築
7. MuJoCo 3.3.1 compatibility
8. headroom再確認
9. 60 episode minimal simulator smoke
10. PASS summary検証
```

---

## 9. expected PASS

最終的に以下が必要。

```text
=== M3 DISK HEADROOM: PASS ===
=== M3 SIMULATOR RUNTIME SETUP: PASS ===
=== 74 M3 MINIMAL SIMULATOR SMOKE: PASS ===
=== ORGANIZER GPU 74 EQUIVALENT: PASS ===
```

74 summary:

```text
status = PASS
episode_count = 60
promotion_evidence = false
benchmark_training_started = false
selected_episode_ids_sha256 = fixed D10 hash
```

また、Driveから移したcheckpointの場合は `artifact_rebased_models` に対象modelが記録され、元のNotebook73 `train_result.json` は変更されない。

---

## 10. failure時

同じattemptを消してやり直さない。

```bash
export PARC_M3_SIM_SMOKE_ATTEMPT=organizer-2
```

として新しいattemptへ進む。

保存対象:

```text
model-benchmark-v1/m3-organizer-migration-v1/attempt-*/
model-benchmark-v1/m3-simulator-minimal-smoke-v1/attempt-*/
```

原因がhardware incompatibilityなら、72a〜72dの既存PASSを上書きせず、organizer hardware compatibility evidenceとして別管理する。

---

## 11. PASS後

74 PASS後に初めて:

1. organizer GPU対応PRをPR #67へ統合
2. PR #67 final review
3. mainへ統合判断
4. 75a / 75bを最新runtimeへrefresh
5. 74 PASSを75 runner-level mandatory gateへ追加
6. hardware-specific micro-batch / gradient accumulation compatibilityを必要最小限で確認
7. full M3 benchmarkへ進む

74 PASSだけを理由に、古い75 notebookをそのまま実行しない。

---

## 12. 停止前

必要なevidenceがpersistent rootへあることを確認する。

```bash
parc-home-sync data-push
```

完了後:

```text
File -> Hub Control Panel -> Stop Server
```

`Start My Server` 表示まで確認する。

ブラウザを閉じるだけでは停止完了と扱わない。
