# PARC2026 Track3 Submission Critical Path

Status: **P0 / submission-critical**  
Official upload deadline: **2026-09-17 23:59 JST**  
Internal target: **2026-09-16 23:59 JST**  
Final artifact freeze target: **2026-09-15 end of day**  
Tracking issue: #50

## Objective

The goal for the remaining submission window is not unlimited optimization. The goal is to produce a **reproducible, uploadable Track3 submission by September 15**, complete a dry-run / practical submission by September 16, and reserve September 17 only for integrity checks and final upload confirmation.

After September 15, reproducibility and submission correctness take priority over further performance improvements.

## Non-negotiable invariants

Do not change the frozen D10 dataset identity:

- variant: `V2_SQRT_BALANCED_RAW`
- selected episodes: `10758`
- selected frames: `1620614`
- episode IDs SHA256: `73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239`
- no D10 reselection
- no full RLDS materialization
- do not delete existing venvs
- benchmark execution remains explicitly guarded

## Required execution-order control

Track3 model comparison must be executed in both directions to detect execution-order, cache, temperature, runtime, or setup bias.

### Forward order

1. π0.5
2. SmolVLA
3. OpenVLA-OFT

### Reverse order

1. OpenVLA-OFT
2. SmolVLA
3. π0.5

Forward and reverse runs must preserve the same:

- D10 episode pool
- sampling policy
- seed set
- equal-data sample budget
- equal-wall timing semantics
- metric definitions
- model recipes and frozen source revisions

Promotion decisions must use evidence aggregated across **forward + reverse**, not one direction alone.

## Benchmark contract

### Equal-data track

- exactly `4800` actual samples per model
- target effective batch size: `32`
- nominally `150` optimizer updates when framework semantics permit
- exact actual sample count must be recorded; no overshoot is allowed

### Equal-wall track

- train-loop budget: `1800 sec` per model
- timer starts immediately before the first train-loop batch fetch
- setup, download, conversion, checkpoint merge, and evaluation are excluded
- stop only at a safe optimizer boundary at or after the target duration
- record actual train-loop wall time and actual samples consumed

### Required metrics

- simulator success rate
- steps to success
- episode duration
- inference latency
- peak inference VRAM
- train wall time
- peak train VRAM

Training loss alone is not sufficient for promotion.

## Submission schedule

### September 9 — close the batch-probe gate

P0:

- [ ] review / merge the active 72c compatibility fix
- [ ] 72c OpenVLA-OFT batch probe PASS
- [ ] 72d controller PASS
- [ ] reach `READY_FOR_M3_RUNNER_IMPLEMENTATION`
- [ ] record selected micro-batch, gradient accumulation, and peak VRAM for all runnable models

Cut line:

If OpenVLA remains blocked by non-model infrastructure after one additional focused repair cycle, stop broad dependency debugging. By **September 10 noon**, freeze the runnable model candidate set and proceed with documented exclusion evidence rather than consuming the submission week.

### September 10 — implement and freeze the M3 runner

P0:

- [ ] exact sample counter for equal-data execution
- [ ] enforce `4800` actual samples per model
- [ ] enforce effective batch size `32`
- [ ] implement 1800-second train-loop-only timer
- [ ] exclude setup/download/conversion/eval from equal-wall timing
- [ ] implement deterministic common sampling schedule and seed handling
- [ ] encode forward/reverse order explicitly in the runner contract
- [ ] preserve `PARC_M3_EXECUTE=1` as the explicit execution guard
- [ ] run only short per-model smoke tests first
- [ ] confirm no full benchmark starts automatically from smoke

Completion gate:

`READY_FOR_M3_BENCHMARK_EXECUTION`

### September 11 — forward benchmark

Run in this order:

`π0.5 → SmolVLA → OpenVLA-OFT`

P0:

- [ ] forward equal-data track
- [ ] forward equal-wall track
- [ ] persist samples, optimizer updates, train-loop wall time, peak train VRAM, and loss
- [ ] preserve immutable run metadata and logs

Failure policy:

A deterministic/config/runtime failure receives one focused repair attempt. If still blocked, preserve failure evidence and continue with the remaining models. Do not spend a full day debugging one model.

### September 12 — reverse benchmark

Run in this order:

`OpenVLA-OFT → SmolVLA → π0.5`

P0:

- [ ] reverse equal-data track
- [ ] reverse equal-wall track
- [ ] same seeds
- [ ] same sampling schedule
- [ ] same metric definitions
- [ ] compare forward/reverse deltas for order effects

Hard cut:

By end of September 12, benchmark data collection is considered complete even if one model is marked failed or excluded.

### September 13 — simulator evaluation and promotion

P0:

- [ ] simulator success rate
- [ ] steps to success
- [ ] episode duration
- [ ] inference latency
- [ ] peak inference VRAM
- [ ] train wall time
- [ ] peak train VRAM
- [ ] aggregate forward + reverse evidence
- [ ] inspect order-effect deltas
- [ ] freeze one promotion summary JSON

Promotion contract:

- maximum promoted models: `2`
- primary metric: `simulator_success_rate`
- training loss alone is insufficient

By end of September 13, freeze the final-candidate decision.

### September 14 — final candidate run and minimum robustness

P0:

- [ ] choose the final submission candidate, or top two only if the Track3 submission format explicitly allows it
- [ ] execute one reproducible final fine-tune/evaluation pass with the frozen recipe
- [ ] perform only the minimum robustness checks needed to support the final result

P1 only after all P0 items are green:

- [ ] second seed repeat
- [ ] one small high-value ablation
- [ ] additional evaluation episodes

Do **not** start broad ablation or hyperparameter sweeps after September 14.

### September 15 — submission artifact freeze

P0:

- [ ] final checkpoint / adapter artifact fixed
- [ ] final config fixed
- [ ] final source commit fixed
- [ ] D10 manifest and hashes recorded
- [ ] forward/reverse benchmark summary fixed
- [ ] final simulator metrics fixed
- [ ] README / reproduction commands fixed
- [ ] required Track3 upload files generated
- [ ] generated files can be opened/read successfully
- [ ] file size, name, and format verified against the Omnicampus form

Target gate:

`SUBMISSION_ARTIFACT_FROZEN`

No non-blocking experiment should begin after this gate.

### September 16 — dry-run submission day

No new experiment unless a submission-blocking defect is discovered.

P0:

- [ ] fresh-session reproduction sanity check
- [ ] calculate hashes of final upload artifacts
- [ ] create backup copies in Drive/local storage
- [ ] reopen the Track3 Omnicampus assignment and verify exact upload requirements
- [ ] upload the final artifact if replacement remains possible
- [ ] record screenshot / evidence of upload state

Target: be **submission-ready at least 24 hours before the official deadline**.

### September 17 — buffer and final upload

Hard rules:

- no architecture changes
- no dataset changes
- no benchmark-design changes
- no optional ablations
- no broad dependency work

Only:

- [ ] final integrity check
- [ ] final upload / replacement if necessary
- [ ] verify upload completion before **22:00 JST**
- [ ] preserve upload confirmation evidence

Official hard deadline: **2026-09-17 23:59 JST**.

## Priority tiers

### P0 — must ship

1. 72c / 72d gate
2. runnable M3 benchmark
3. forward + reverse execution-order control
4. simulator-success evaluation
5. final candidate selection
6. reproducible final artifact
7. upload verification

### P1 — only if all P0 work is green

- second seed / extra evaluation episodes
- one or two high-value ablations
- richer plots and tables

### P2 — cut immediately if the schedule slips

- broad hyperparameter search
- many ablations
- adding new model families
- full RLDS materialization
- cosmetic refactors
- performance tuning that risks artifact stability

## Schedule-slip rules

- **Sep 10 noon:** runnable model candidate set freezes
- **Sep 12 end:** benchmark collection freezes
- **Sep 13 end:** promotion / final-candidate decision freezes
- **Sep 15 end:** submission artifact freezes
- after Sep 15: correctness and reproducibility beat additional performance

## Final submission evidence checklist

- [ ] exact final source SHA
- [ ] exact dataset manifest SHA
- [ ] exact episode IDs SHA
- [ ] final model/checkpoint SHA or immutable reference
- [ ] final model config
- [ ] forward results
- [ ] reverse results
- [ ] order-effect comparison
- [ ] required simulator metrics
- [ ] reproduction command
- [ ] upload file checksum
- [ ] Omnicampus upload confirmation / screenshot

## Decision principle

The submission week is managed by a simple rule:

> A slightly weaker but fully reproducible, fully evaluated, successfully uploaded Track3 result is preferable to a theoretically stronger experiment that is incomplete at the deadline.
