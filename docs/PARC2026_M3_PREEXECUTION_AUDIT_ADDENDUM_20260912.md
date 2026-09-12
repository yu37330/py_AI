# PARC2026 / M3 downstream pre-execution audit addendum (2026-09-12)

この文書は `PARC2026_M3_OPERATIONAL_CAUTIONS_20260912.md` の追補監査です。

主文書の作成後に、PR #67 上の 76a / 76b / 77 と generic evaluator wrapper、open PR #68 まで実コードを追加確認した結果、**74 attempt 8 の実行を止めるものではないが、74 PASS後にそのまま downstream を実行すると危険な点**が見つかったため、明示的な blocker として残します。

---

## 1. 76a / 76b は現状のまま実行しない

PR #67 branch の現在の Notebook:

```text
colab/76a_m3_forward_screening_evaluation.ipynb
colab/76b_m3_reverse_screening_evaluation.ipynb
```

は、どちらも runtime pin が:

```text
23fea09284db3ab5caa0225c9e08c6a3496c453e
```

のままです。

この pin は Notebook74 attempt 1〜8 で確定した後続修正より古いため、**74 PASS後でもそのまま76を開いて実行してはいけません。**

さらに current 76a/76b Notebook は、fresh Colab上で直接:

```text
run_m3_screening_evaluation_order.py
```

を呼びますが、Notebook74で追加した以下を自前ではbootstrapしていません。

```text
prepare_m3_libero_noninteractive_config.py
prepare_m3_simulator_runtimes.py
prepare_m3_mujoco_compat.py / equivalent pinned mujoco setup
prepare_m3_disk_headroom.py
MPLBACKEND=Agg
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

したがって 76 refresh 時には、少なくとも以下を必須化してください。

1. current validated #67 codeへ pin 更新
2. fresh-Colab simulator runtime bootstrap
3. explicit noninteractive `LIBERO_CONFIG_PATH`
4. `mujoco==3.3.1`
5. pinned OpenVLA LIBERO checkout + `.pth`
6. checkpoint-driven camera `rename_map`
7. LeRobot upstream output dir pre-create
8. no-video evaluation
9. Agg + EGLをNotebookとchildの両方で強制
10. local / Drive disk headroom preflight
11. partial episode evidence preservation
12. A100 >= 38 GiB gate

---

## 2. Screening 80 episodes と final 800 episodes の実装上の注意

`run_m3_screening_evaluation_order.py` は 1 checkpoint あたり:

```text
4 LIBERO suites
10 tasks/suite
2 seeds
1 trial/task
= 80 episodes/checkpoint
```

を実行します。

重要なのは、このrunnerが generic wrapper に:

```text
--mode smoke
```

を渡して 1 trial/task を選んでいる点です。

一方 `m3_libero_executor.py` の generic `benchmark` mode は:

```text
4 suites
10 tasks/suite
2 seeds
10 trials/task
= 800 episodes/checkpoint
```

です。

したがって **「screeningなのにmode名がsmokeなのは変だからbenchmarkへ直す」という変更をしてはいけません。** それをすると6 checkpoint/orderで意図せず巨大な800-episode評価を開始します。

運用契約:

```text
screening: 80 episodes/checkpoint
final selected candidate only: 800 episodes
```

を維持してください。

将来的には mode名を `screening` / `final` に分離した方が読みやすいですが、締切前に大きなrefactorを行う必要はありません。まず現契約をtestで固定することを優先します。

---

## 3. Generic evaluator direct-script import risk

`tools/benchmark/run_m3_libero_evaluation.py` の executor command は現在:

```text
python -u <repo>/tools/benchmark/m3_libero_executor.py ...
```

です。

しかし `m3_libero_executor.py` は module import時に:

```python
from tools.benchmark.m3_runner_core import ...
```

を使用します。

fresh Pythonで absolute script path invocation を行う場合、`sys.path[0]` は `tools/benchmark/` となり、repo rootが import path に無いと `tools` packageが見えない可能性があります。

これは Notebook73 の PR #73 で実際に発生した:

```text
ModuleNotFoundError: No module named 'tools'
```

と同じ失敗クラスです。

現時点では 76 generic wrapperで live failureを観測したわけではないため、**confirmed bugではなく static-audit risk** とします。

76を実行する前に:

```text
python -m tools.benchmark.m3_libero_executor
```

へ切り替えるか、exact repo rootを `PYTHONPATH` に注入し、fresh-process regression testを追加してください。

Notebook74 minimal smokeはこのgeneric wrapperを通らないので、attempt 8を止める理由ではありません。

---

## 4. 77 promotion Notebook も old pin のまま

PR #67 branch の:

```text
colab/77_m3_forward_reverse_promotion_controller.ipynb
```

も現在:

```text
PIN = 23fea09284db3ab5caa0225c9e08c6a3496c453e
```

です。

promotion controllerはCPU処理なので74のMuJoCo修正自体は不要ですが、76側のpromotion record schema / hash / attempt運用を最新版で確定した後に **77も同じvalidated revisionへrefresh** してください。

77で最低限再検証するもの:

```text
forward 6 promotion records
reverse 6 promotion records
12 records total
exact D10
forward/reverse × equal-data/equal-wall completeness
same source_ref per model
simulator_success_rate primary
max 2 promoted models
training loss not in ranking key
final evaluation not auto-started
```

---

## 5. PR #68 は final evidenceができた後にcurrent mainへ再照合する

PR #68 `feat: orchestrate Track3 final candidate artifact freeze` は usefulな final-freeze implementationですが、現在openで、作成時のbaseは現mainより古いです。

このため:

```text
#68 is implemented
```

を:

```text
#68 is ready to merge/run now
```

と解釈しないでください。

使用前に:

1. #67 の evaluator fixes をcurrent mainへ統合済みであること
2. refreshed 75 benchmark evidence
3. refreshed 76 screening evidence
4. 77 promotion summary
5. selected final candidate / selected budget
6. final evaluator 800-episode contract

をそろえてから #68 を current main と比較し、必要なrebase / integration repairを行います。

OpenVLAがfinal selected candidateの場合、#68の契約どおり final 800 episode中だけ:

```text
PARC_M3_KEEP_MERGED_OPENVLA=1
```

を使って evaluator-ready merged artifactをartifact freezeまで保持します。中間screeningではmerged 7Bを保持しません。

---

## 6. 75 / 76 / 77 / #68 の downstream propagation checklist

Notebook74 PASS後の安全な更新順:

```text
74 PASS (60 episodes)
  ↓
#67 current main再照合 + merge判断
  ↓
75a/75b refresh
  - PR #76 training fixes
  - fresh runtime bootstrap
  - 74 PASS runner-level gate
  - disk headroom
  ↓
76a/76b refresh
  - all Notebook74 evaluator fixes
  - generic wrapper import fix
  - fresh simulator runtime bootstrap
  - 80 episodes/checkpoint test
  ↓
77 refresh
  - latest promotion record schema/pins
  - 12-record completeness
  ↓
#68 refresh/review
  - current main
  - selected candidate only final 800
  - immutable final provenance / freeze
```

**Notebook pinだけ更新して downstream contractを更新し忘れないこと。**

---

## 7. 現時点での結論

今回の追加監査で、74 attempt 8そのものを止める新しいknown blockerは見つかっていません。

ただし、74 PASS後に現行75/76/77を連続実行するのは安全ではありません。

特に:

```text
75: old pin + no 74 runner gate
76: old pin + no fresh simulator bootstrap + generic import risk
77: old pin
#68: old-base open implementation, real evidence待ち
```

を明示的な pre-execution blockers として扱います。

このaddendumと主文書を合わせて、今後のruntime修正・PR merge・artifact freezeの引き継ぎ資料とします。
