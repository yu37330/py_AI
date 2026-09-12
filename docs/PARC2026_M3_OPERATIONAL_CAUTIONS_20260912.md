# PARC2026 / M3 運用注意事項・修正履歴（2026-09-12）

## 0. この文書の目的

この文書は、PARC2026 の Dataset Gate から M3 3-model comparison、Notebook 73/74 の実機デバッグまでに発生した不具合と、その修正から得た **再発防止上の注意事項**を一か所にまとめるものです。

対象は主に以下です。

- group-aware dataset / D10 固定
- 69b / 69c OpenVLA bridge
- 72a〜72d runtime probe
- Notebook 73 A100 training smoke
- Notebook 74 LIBERO simulator smoke
- 75a / 75b benchmark training へ進む前の注意
- 76a / 76b screening、77 promotion、final evaluation への引き継ぎ

この文書は「過去の実行を再現するための完全な監査ログ」ではなく、**同じ失敗を繰り返さないための運用ガード**です。実際のコード状態は `main`、実行中の評価系は未マージ PR、Drive 上の固定 artifact を合わせて確認してください。

---

## 1. 最重要: これだけは壊さない

### 1.1 D10 は再選択・再生成しない

現在の M3 比較で固定している dataset は次です。

```text
variant: V2_SQRT_BALANCED_RAW
source dataset: lerobot/libero_plus
source revision: f3f49f426d75030177b18778374005bc12ccd588
episodes: 10,758
frames: 1,620,614
tasks: 40
episode_ids_sha256: 73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239
manifest_file_sha256: cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395
```

固定 manifest の Drive バックアップ:

```text
/content/drive/MyDrive/parc2026-cache/pi05-ablation-group-aware-v2/
  dataset_ablation_manifests_v2_group_aware/
```

manifest が見つからない場合は **fail closed** とし、同じ seed で作り直して代用してはいけません。

69b の recovery 文書には、episode ID hash が一致していても「過去の original D10 と歴史的に完全同一だったか」は recovery 時点で未証明だった旨が残っています。したがって今後は、現在固定している manifest 本体と provenance をそのまま保存し、再生成で置き換えないでください。

### 1.2 legacy episode-level split は使わない

旧 fixed-eval split は trajectory sibling leakage が確認され、比較用 split として廃止済みです。

Group-aware V2 では eval group の sibling を全 training variant から除外し、以下の Gate を通したものだけを正とします。

```text
Group-aware Manifest Gate: PASS
Trajectory Leakage Gate: PASS
```

**legacy V1 manifest に戻さないこと。**

### 1.3 72a〜72d probe は原則再実行しない

72 系は runtime / batch feasibility を確認するための probe evidence です。

現在の downstream はその結果を前提に組まれています。明示的に probe artifact が invalidated された場合を除き、73/74/75 のトラブル解決のために 72a〜72d を再実行しないでください。

### 1.4 full RLDS materialization はしない

69b で selected pool の full RLDS projection が **56.61 GiB** と見積もられ、35 GiB threshold を超えました。

そのため OpenVLA は 69c の LeRobot streaming bridge を使用します。

```text
full RLDS: 禁止
streaming bridge: 使用
```

容量問題を解決するために full RLDS を作る、という方向へ戻さないでください。

### 1.5 partial evidence を勝手に消さない

73 / 74 / benchmark / screening は途中失敗の evidence 自体が重要です。

- PASS result があるものは identity を再検証して reuse
- incomplete tree に real files / symlinks がある場合は自動削除しない
- 74 は attempt 番号を進めて新しい evidence root を使う
- 原因調査のために失敗 attempt を保存する

「やり直したいからフォルダを消す」は原則禁止です。

---

## 2. Source of Truth の優先順位

### 2.1 merged training-side fixes

2026-09-12 時点の `main` は PR #76 までを含みます。

```text
main: d074a26881af14dfc44c5c0085b1b9da5b400d14
```

Notebook 73 の最終 A100 smoke PASS pin:

```text
61f3978e55e28d1e0e83f8ab1d721124fb8d17e4
```

### 2.2 evaluation-side fixes

LIBERO evaluator / Notebook 74 / screening 系は PR #67 で進行中です。

```text
PR #67: feat/m3-libero-executor
status: open / unmerged
```

74 PASS 前に #67 を「動いたはず」で merge しないでください。

また 2026-09-12 の確認時点では GitHub が #67 を `mergeable=false` と返したタイミングがあります。74 PASS 後は、必ず current main との差分・競合・head SHA を再確認してから merge 判断してください。

### 2.3 PR #70 は別系統の実装

PR #70 `feat: implement guarded M3 3-model training runners after 72d` は open のまま残っています。

PR #66〜#76 で現在の canonical training path が進んでいるため、#70 を安易に merge / close しないでください。別実装として扱い、明示判断なしに混ぜないこと。

---

## 3. Dataset / 69b / 69c で得た注意事項

### 3.1 69b: dependency failure は parent `CalledProcessError` だけを見ない

69b では parent Notebook から見ると `CalledProcessError` でしたが、実原因は dependency resolver 側でした。

具体的には:

```text
tensorflow-datasets==4.9.6
  -> promise が必要
  -> global binary-only policy では usable wheel がない
```

修正:

```text
promise==2.3
```

のみ source build を許可して先に入れ、その他の binary-only policy は維持しました。

固定した dependency line:

```text
Python 3.10
TensorFlow CPU 2.17.1
TFDS 4.9.6
PyAV 12.3.0
promise 2.3
```

注意:

- PyAV 等まで source build 許可へ広げない
- dependency smoke PASS を full conversion の許可と解釈しない
- 69b smoke PASS は M3 training 開始許可ではない

### 3.2 69c: OpenVLA bridge contract を崩さない

OpenVLA streaming bridge の固定 contract:

```text
action_dim: 7
proprio_dim: 8
action_chunk: 8
normalization: bounds_q99
OpenVLA-OFT source: small-zeng/openvla-oft@e4287e94541f459edc4feabc4e181f537cd569a8
```

変換:

```text
gripper = 1 - clip(raw_action[6], 0, 1)
proprio = concat(state[:6], state[-2:])
future relative action padding = 0
future absolute gripper padding = last value repeat
```

selected 10,758 episodes / 1,620,614 frames の statistics を使います。

**OpenVLA だけ別 sampling pool / 別 normalization にしないこと。**

---

## 4. M3 fairness contract

training 比較は以下を固定します。

```text
training schedule seed: 20260906
eval seeds: 20260906, 20260907
effective batch size: 32
```

### Equal Data

```text
4,800 actual samples
150 optimizer updates
```

### Equal Wall Time

```text
1,800 sec train-loop time
停止は safe optimizer boundary
setup / download / conversion / checkpoint serialization / eval は timer 外
```

### Promotion

主指標:

```text
simulator success rate
```

補助:

```text
steps-to-success
episode duration
inference latency
peak inference VRAM
train wall time
peak train VRAM
```

**training loss だけで promotion を決めないこと。**

forward / reverse は experimental order-control です。比較の公平性のため同じ canonical sample schedule を使います。

---

## 5. Notebook 73 A100 training smoke: 修正履歴と再発防止

Notebook 73 は最終的に PASS しましたが、実機で複数の fresh-Colab / child-process 問題が見つかりました。

### PR #66: model adapter runtime の基本 contract

- π0.5 / SmolVLA / OpenVLA-OFT の guarded training adapter
- equal-data / equal-wall contract
- OpenVLA は full RLDS ではなく 69c streaming
- benchmark と smoke の `benchmark_training_started` を明確に分離
- OpenVLA merged 7B は training 中に作らず、eval 直前に JIT materialize

### PR #71: fresh Colab を前提にする

問題:

- 72 で作った `/content` runtime が残っている前提だった
- Colab は session が変われば `/content` が消える

修正:

- π0.5 runtime を再構築
- SmolVLA Python 3.12 runtime を再構築
- OpenVLA Python 3.10 runtime を再構築
- setup-only とし 72 probe は再実行しない

OpenVLA compatibility set の例:

```text
wandb=0.16.6
protobuf=3.20.3
pandas=2.2.3
pyarrow=17.0.0
av=12.3.0
```

注意:

**Colab の system Python に偶然入っている package を前提にしないこと。**

### PR #72: child stdout/stderr を Drive に残す

問題:

parent Notebook に `CalledProcessError` だけ出て、child traceback が消える。

修正:

- smoke orchestrator stdout+stderr を Drive に tee
- status / log tail / latest plan / latest result を残す
- validated OpenVLA Python 3.10 で orchestration

注意:

**長い subprocess は「終了コードだけ」ではなく、persistent log を必須にする。**

### PR #73: direct script invocation と PYTHONPATH

問題:

```text
ModuleNotFoundError: No module named 'tools'
```

原因:

`tools/benchmark/*.py` を absolute file path で直接起動すると、`sys.path[0]` は script directory になり repo root が見えない。

修正:

- exact checkout repo root を `PYTHONPATH` の先頭へ注入
- child が継承

注意:

- repo package import がある script は `python -m ...` を優先
- file-path invocation を使う場合は repo root の importability を明示検証

### PR #74: LeRobot training output directory を pre-create しない

問題:

```text
FileExistsError: Output directory .../checkpoint already exists and resume is False
```

原因:

adapter が actual output directory を作ったあと、LeRobot trainer が「既存ディレクトリ」を拒否した。

修正:

- `output_dir.parent` のみ作る
- actual checkpoint dir は LeRobot に作らせる

注意:

**training output ownership は upstream trainer に合わせる。**

### PR #75: OpenVLA scheduled adapter symbol mismatch

問題:

```text
ImportError: cannot import name 'to_upstream_window'
```

原因:

69c adapter の validated API は `to_openvla_rlds_batch()` だったが、scheduled loader が `to_upstream_window` を import していた。

修正:

- validated converter を壊さず compatibility wrapper を追加
- completed π0.5 / SmolVLA result の safe resume を追加

注意:

**validated bridge API を rename して壊さず、compatibility alias で吸収する。**

### PR #76: OpenVLA model logic sync は cwd 依存

問題:

```text
AttributeError: 'PrismaticVisionBackbone' object has no attribute 'set_num_images_in_input'
```

直前に:

```text
modeling_prismatic.py is not found anywhere in the current directory
configuration_prismatic.py is not found anywhere in the current directory
```

原因:

OpenVLA-OFT upstream helper が `./prismatic/` を cwd 基準で探索する。Notebook 73 が py_AI repo cwd から呼んだため、pinned OpenVLA model logic が HF snapshot へ同期されなかった。

修正:

- model-logic sync の間だけ exact OpenVLA source checkout へ cwd を変更
- 必ず元の cwd に戻す
- `modeling_prismatic.py` / `configuration_prismatic.py` を確認
- load 後に multi-image API を fail-fast 検証

必須 API:

```text
set_num_images_in_input
get_num_images_in_input
get_num_patches
```

注意:

**upstream helper が cwd に依存していないか必ず確認する。**

---

## 6. OpenVLA checkpoint lifecycle

OpenVLA は training 後に persistent artifact として次を保持します。

```text
lora_adapter/
action_head--*checkpoint.pt
proprio_projector--*checkpoint.pt
dataset_statistics.json
```

merged 7B weights は常設しません。

原則:

1. train-loop timer 停止
2. evaluation 直前に JIT merge
3. evaluator 実行
4. この attempt が作った merged weights のみ dematerialize
5. LoRA / components / stats は残す

注意:

- merge/serialization を equal-wall timer に含めない
- 既に存在していた merged model を「今回作ったもの」と誤認して削除しない
- cleanup failure は evidence として残す
- disk headroom を merge 前に確認する

---

## 7. Notebook 74 LIBERO simulator smoke: attempt 1〜8

Notebook 74 の目的は benchmark 前に evaluator integration を壊さず確認することです。

固定 smoke contract:

```text
suite: libero_spatial
10 tasks
2 eval seeds
3 models
1 trial/task/seed
20 episodes/model
60 episodes total
max steps: 300
promotion evidence: false
```

### attempt 1: LIBERO interactive config prompt

失敗:

```text
Do you want to specify a custom path for the dataset folder? (Y/N):
EOFError
```

原因:

fresh environment に LIBERO config がなく、noninteractive subprocess で `input()` が呼ばれた。

修正:

- `LIBERO_CONFIG_PATH` を明示
- config を事前生成
- π0.5 / SmolVLA / OpenVLA が同じ task/assets path を使う

注意:

**CI / Colab subprocess で interactive config generator を踏ませない。**

### attempt 2: Colab Matplotlib backend leakage

失敗:

```text
MPLBACKEND=module://matplotlib_inline.backend_inline
ValueError: invalid backend
```

原因:

Colab 親環境の inline backend が隔離 venv へ継承された。

修正:

```text
MPLBACKEND=Agg
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

注意:

- Notebook wrapper だけでなく evaluator child にも強制
- `setdefault()` ではなく inherited bad value を上書きする

### attempt 3: OpenVLA LIBERO namespace import

失敗:

```text
ModuleNotFoundError: No module named 'libero'
```

原因:

固定 LIBERO revision の legacy `setup.py/find_packages()` と namespace-style source layout の組み合わせで、editable install の metadata はできても import path が有効にならなかった。

修正:

- pinned checkout 自体は改変しない
- OpenVLA venv の site-packages に `.pth` を置き source root を公開
- `libero.libero.__file__` が exact pinned checkout を指すことを検証

注意:

**「pip install 成功」だけで importable と判断しない。実 import path を検証する。**

### attempt 4: MuJoCo dependency drift

実際の simulator construction まで到達し、robosuite 1.4.x と最新 MuJoCo 3.13 系の incompatibility が露出。

修正:

```text
mujoco==3.3.1
```

を 3 evaluator runtime で固定。

注意:

**hf-libero の transitive dependency resolver に MuJoCo version を任せない。**

### attempt 5: camera feature name mismatch

環境側:

```text
observation.images.image
observation.images.image2
```

π0.5 checkpoint 側:

```text
observation.images.front
observation.images.wrist
```

LeRobot consistency guard で停止。

修正:

- checkpoint `config.json` から exact input feature names を読む
- canonical 2-camera layout の場合だけ:

```text
image  -> front
image2 -> wrist
```

- official `rename_map` path を使用
- unknown layout は推測せず fail closed

注意:

**camera order/name を hard-code だけで済ませず checkpoint contract と照合する。**

### attempt 6: simulator rollout 成功後の output directory

π0.5 は `libero_spatial` 10 tasks を最後まで rollout し、10/10 success まで到達。

その後 upstream LeRobot が:

```text
.../upstream/eval_info.json
```

を書こうとして、`upstream/` が無いため停止。

修正:

- evaluator 開始前に専用 upstream output dir を作成

注意:

これは simulator / inference failure ではなく post-rollout persistence failure。

また attempt 6 の 100% は smoke checkpoint の小規模評価であり、**promotion evidence ではない**。

### attempt 7: disk full + diagnostics masking

確認できたエラー:

```text
OSError: [Errno 28] No space left on device
```

重要:

- child `run_m3_minimal_simulator_smoke.py` は先に non-zero exit
- その後 Notebook が diagnostic file を読む途中でも disk full になった
- diagnostics failure が original child exception を隠した

したがって、**attempt 7 の child root cause を disk full と断定してはいけない**。disk exhaustion が発生していたことだけが確定。

### attempt 8: disk headroom hardening

attempt 8 では以下を追加。

- setup 前に disk headroom check
- eval 直前にも再check
- local `/content`: 最低 24 GiB free
- Drive: 最低 20 GiB free
- cleanup 対象は disposable cache のみ
  - uv cache
  - pip cache
  - apt cache
-削除禁止:
  - Hugging Face model cache
  - D10 dataset / manifest
  - checkpoints
  - prior attempt evidence
- diagnostics は bounded tail で読む
- read failure を catch
- failure 時に `df -h` / `df -i` を出す
- episode/task progress を small partial record として残す

Notebook 74 attempt 8 runtime pin:

```text
1b983910509020dc2a01f1474140d72bf5664e8e
```

expected marker:

```text
74 simulator smoke attempt: 8
=== M3 DISK HEADROOM: PASS ===
```

PASS はまだ未確認。

---

## 8. Notebook pin の運用

Colab は browser tab が古い Notebook revision を保持することがあります。

実行前に必ず最初の marker を確認してください。

例:

```text
74 simulator smoke code: <expected commit>
74 simulator smoke attempt: <expected attempt>
```

期待 pin と違ったら停止してください。

**GitHub branch の最新 head SHA と Notebook 内の runtime PIN は同じとは限りません。**

Notebook は「その実行で検証する固定 commit」を checkout するため、PR branch に test-only commit が後続していても runtime pin が意図的に一つ前の commit になる場合があります。

---

## 9. 75a / 75b は現状のまま実行しない

2026-09-12 の確認では、PR #67 branch 上の:

```text
colab/75a_m3_forward_training_benchmark.ipynb
colab/75b_m3_reverse_training_benchmark.ipynb
```

はどちらも古い runtime pin:

```text
b5c2d150ba8d361ca22da874677827f8924f41fc
```

を参照しています。

この pin は Notebook 73 実機デバッグで確定した PR #76 の OpenVLA model-logic cwd fix より前です。

さらに現行 75 は:

- 73 PASS だけを gate にしている
- 74 PASS を要求しない
- fresh Colab runtime bootstrap を Notebook 自身では行わない

ため、**74 PASS の直後に現行 75a / 75b をそのまま回してはいけません。**

75 実行前に最低限以下を反映すること。

1. code pin を PR #76 以降の validated training path へ更新
2. 74 PASS summary を gate に追加
3. exact 60 episode / D10 hash を検証
4. fresh Colab runtime bootstrap を追加
5. equal-data / equal-wall schedule hash を再確認
6. attempt-scoped output を使用
7. partial evidence を自動削除しない
8. benchmark 開始前に disk headroom を確認

---

## 10. 76a / 76b screening へ引き継ぐもの

76 screening は simulator evaluation なので、74 で見つかった fix をすべて引き継ぐ必要があります。

最低限:

```text
LIBERO_CONFIG_PATH explicit
MPLBACKEND=Agg
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
mujoco==3.3.1
pinned OpenVLA LIBERO .pth
checkpoint-driven camera rename_map
LeRobot upstream output dir pre-create
no video recording
batch_size=1
300 step cap
partial episode evidence
disk headroom preflight
```

**74 にだけ fix が入って 76 generic evaluator に入っていない状態を作らないこと。**

screening は 80 episodes/checkpoint。final 800 を screening で使わない。

---

## 11. 77 promotion / final evaluation

Promotion は全 expected screening record が揃ったあとに行います。

- forward / reverse
- equal-data / equal-wall
- 3 models
- screening success evidence

最大 2 models まで。

primary metric は simulator success rate。

**loss を primary promotion metric にしない。**

final 800 episodes は selected final candidate にだけ使います。

---

## 12. Disk / cache 運用

OpenVLA は base snapshot、venv、source checkout、JIT merged weights が重なると Colab local / Drive の両方を圧迫します。

安全に削除可能なのは「再生成できる cache」に限定します。

候補:

```text
uv cache
pip cache
apt cache
```

原則削除禁止:

```text
/content/drive/MyDrive/parc2026-cache/pi05-ablation-group-aware-v2/
D10 manifest
m3-training-smoke-v1 PASS artifacts
benchmark attempt evidence
simulator smoke attempts
OpenVLA lora_adapter / action head / proprio projector / stats
HF cache（base model の再downloadコストと pin provenance のため通常保持）
```

容量不足時に「Drive の古い attempt を全部消す」は禁止。

---

## 13. Error diagnosis の基本ルール

### parent exception と child root cause を分ける

以下は root cause ではなく wrapper のことが多いです。

```text
CalledProcessError
```

必ず child log / status / persisted result を見る。

### warnings と fatal error を分ける

例:

- robosuite private macro warning
- TensorFlow cuDNN/cuFFT duplicate registration warning
- Gym maintenance warning
- tied-weight load warning

は、それだけで今回の停止理由とは限りません。

**最後の fatal traceback と、どの stage まで PASS したかを優先する。**

### diagnostic code 自体も failure-safe にする

attempt 7 で diagnostic read が disk full により二次失敗しました。

今後:

- bounded tail
- read exception catch
- filesystem status を先に出す
- diagnostic failure で original exception を上書きしない

---

## 14. Current execution order

2026-09-12 時点の安全な順序:

```text
73 A100 training smoke               PASS
  ↓
74 minimal simulator smoke attempt 8 現在地
  ↓ 60 episodes PASS 必須
#67 を current main と再照合
  ↓ explicit approval 後のみ merge
75a forward benchmark を更新して実行
75b reverse benchmark を更新して実行
  ↓
76a / 76b screening
  ↓
77 promotion
  ↓
selected candidate only: final 800 episodes
  ↓
artifact freeze
fresh-session reproduction
submission
```

75/76 を先に走らせないでください。

---

## 15. PR / commit 履歴の索引

### Dataset / bridge

- `docs/PARC2026_GROUP_AWARE_HOLDOUT_V2.md`
  - legacy split leakage と group-aware V2 contract
- `docs/69b-promise-dependency-recovery.md`
  - 69b dependency recovery / manifest preservation
- PR #43 `feat: validate selected OpenVLA streaming bridge before M3`
  - 69c streaming bridge
  - full RLDS 回避

### M3 training

- PR #66 `feat: wire frozen M3 schedules into model adapter runtimes`
- PR #71 `fix: make Notebook 73 self-contained on fresh Colab`
- PR #72 `fix: persist Notebook 73 smoke orchestrator diagnostics`
- PR #73 `fix: keep repo imports visible in Notebook 73 child processes`
- PR #74 `fix: stop pre-creating LeRobot smoke output directory`
- PR #75 `fix: repair OpenVLA scheduled adapter and resume validated smoke`
- PR #76 `fix: sync pinned OpenVLA model logic before M3 smoke`

### M3 simulator / screening

- PR #67 `feat: add guarded M3 LIBERO evaluation executor`
  - open / unmerged
  - Notebook 74 attempt history
- PR #70
  - secondary training implementation
  - open / do not mix automatically

---

## 16. 最終チェックリスト

高価な A100 実行前に確認:

- [ ] Notebook の printed code pin が期待値と一致
- [ ] attempt ID が新しい
- [ ] D10 episode IDs SHA が一致
- [ ] manifest file SHA が一致
- [ ] 72 probe を再実行していない
- [ ] full RLDS を作ろうとしていない
- [ ] A100 >= 38 GiB gate
- [ ] fresh runtime bootstrap がある
- [ ] exact source revisions を verify
- [ ] child stdout/stderr が persistent log に残る
- [ ] local / Drive disk headroom PASS
- [ ] partial evidence を削除していない
- [ ] benchmark / promotion / final が意図せず自動開始しない
- [ ] OpenVLA merge は train timer 外
- [ ] evaluation headless contract が有効
- [ ] camera feature contract が checkpoint と一致
- [ ] simulator result count を stage 終了時に検証

---

## 17. 更新ルール

今後新しい A100 / LIBERO 実機不具合が出たら、単にコードを直すだけではなく、この文書にも以下を追記してください。

1. どの stage / attempt で発生したか
2. root cause と wrapper error を分ける
3. 何を修正したか
4. どの downstream notebook にも反映が必要か
5. 再発防止の fail-fast / regression test
6. 固定 artifact を変更したか否か

特に 74 で見つかった evaluator fix は 76 screening / final evaluator まで伝播させ、73 で見つかった training fix は 75a / 75b の本番 runner まで伝播させてください。
