# PARC2026 Static Quality Analyzer V1 results — public LIBERO-plus

Date: 2026-08-30

## Scope

This result is from the public Colab fallback dataset, not the organizer `libero_combined_20hz` source of truth.

- dataset: `lerobot/libero_plus`
- pinned revision: `f3f49f426d75030177b18778374005bc12ccd588`
- LeRobot format: v3.0
- fps: 20
- tasks: 40
- episodes: 14,347
- trajectory parquet files: 2
- analyzer: `tools/data/static_quality_analyzer.py`
- robust-z threshold: 5.0, computed task-relative
- smoothing window: 5 frames

Run A promotion still requires the same analyzer to be rerun on organizer `libero_combined_20hz`.

## Analyzer completion

The analyzer completed all 14,347 requested episodes.

- episodes analyzed: 14,347 / 14,347
- missing episodes: 0
- OK: 13,975
- REVIEW: 372
- REVIEW rate: 2.5929%
- integrity failures: 0

The current public dataset therefore does not show obvious static corruption such as NaN/Inf, non-monotonic timestamps, or frame gaps. Most REVIEW labels are distributional outliers, not hard failures.

## REVIEW reason breakdown

Flags are task-relative and may overlap.

| Flag | Episodes |
| --- | ---: |
| duration outlier | 215 |
| EEF path outlier | 167 |
| smoothed cartesian jerk outlier | 49 |
| idle-ratio outlier | 15 |
| motion-action RMS outlier | 20 |
| integrity failure | 0 |

Unique REVIEW episodes = 372.

Severity by number of flags:

| quality_flag_count | Episodes |
| --- | ---: |
| 0 | 13,975 |
| 1 | 278 |
| 2 | 94 |
| 3+ | 0 |

The 94 two-flag episodes are the strongest static REVIEW candidates, but even these are not automatically invalid demonstrations.

## REVIEW concentration by task

Highest task-relative REVIEW rates observed:

| Task | Episodes | REVIEW | Rate |
| --- | ---: | ---: | ---: |
| push the plate to the front of the stove | 241 | 27 | 11.20% |
| put the cream cheese in the bowl | 315 | 30 | 9.52% |
| pick up the salad dressing and place it in the basket | 440 | 40 | 9.09% |
| pick up the black bowl on the wooden cabinet and place it on the plate | 395 | 30 | 7.59% |
| put both moka pots on the stove | 146 | 10 | 6.85% |
| pick up the orange juice and place it in the basket | 405 | 25 | 6.17% |
| open the middle drawer of the cabinet | 370 | 20 | 5.41% |
| pick up the black bowl between the plate and the ramekin and place it on the plate | 375 | 20 | 5.33% |
| pick up the alphabet soup and place it in the basket | 380 | 20 | 5.26% |
| open the top drawer and put the bowl inside | 200 | 10 | 5.00% |

This concentration is evidence against treating every robust-z outlier as bad data. Some tasks naturally have wider motion or duration distributions, and LIBERO-plus itself introduces perturbations/variation.

## Decision: do not equate REVIEW with Reject

The organizer score gates smoothness / efficiency metrics behind success and collision checks. Removing valid but difficult trajectories just because they have longer paths or unusual jerk can reduce task success, which is likely worse than preserving a hard but successful demonstration.

Therefore Static Quality Analyzer V1 remains a triage layer, not a hard cleaner.

### Candidate dataset variants for controlled ablation

Keep filtering and task balancing as separate experiment axes.

1. `V0_RAW`
   - all 14,347 episodes
   - reference

2. `V1_INTEGRITY_ONLY`
   - remove only hard integrity failures (NaN/Inf, frame/timestamp integrity)
   - current public dataset: 14,347 episodes because hard integrity failures = 0
   - this is the safest production cleaning policy from static evidence alone

3. `V1_MULTI_FLAG_PRUNED_EXPERIMENTAL`
   - exclude only `quality_flag_count >= 2`
   - removes 94 episodes
   - retains 14,253 episodes (99.34%)
   - use as the conservative statistical filtering ablation

4. `V1_ALL_REVIEW_PRUNED_EXPERIMENTAL`
   - exclude all 372 REVIEW episodes
   - retains 13,975 episodes (97.41%)
   - aggressive filtering ablation only; do not promote without simulator/video evidence or clear evaluation gain

5. `V2_SQRT_BALANCED_RAW`
   - task-balanced sampling on V0 data using sqrt episode-count weighting
   - keep separate from filtering so the effect of balancing is measurable

If filtering and balancing both help independently, add a combined candidate after the first ablation rather than mixing both changes from the start.

## Cheap ablation order

Use a fixed pi0.5 configuration and fixed local eval split:

1. V0_RAW
2. V1_MULTI_FLAG_PRUNED_EXPERIMENTAL
3. V1_ALL_REVIEW_PRUNED_EXPERIMENTAL
4. V2_SQRT_BALANCED_RAW

Compare at minimum:

- simulator success rate
- collision rate if available
- jerk / trajectory distance / rotation
- episode steps / time
- training wall time
- inference latency / VRAM

Do not select a dataset variant from training loss or static quality metrics alone.

## Spot-check queue

Before calling any statistical outlier a true bad demonstration, inspect a small stratified set rather than all 372:

- two-flag episodes: highest priority
- one-flag duration outliers
- one-flag path outliers
- jerk outliers
- idle / motion-action outliers
- highest-REVIEW-rate tasks, especially task 36 / 1 / 35 / 38 / 33

A useful first manual pass is about 30–50 episodes, stratified by flag and task. Replay validation remains the preferred hard evidence when raw simulator state is available.

## Dataset Factory status

- D1 Inventory: PASS on public proxy
- D2 Static Quality Analyzer: PASS on public proxy
- D3 V1 cleaning policy: PARTIAL / experimental variants defined
- D4 Task balancing: ready for cheap ablation
- D5 Replay / success / collision validation: pending raw simulator-state path
- Organizer source-of-truth rerun: pending

## Next synchronization with Model Selection

Model Selection and Dataset Factory continue in parallel:

- Model lane: pi0.5 smoke / gradient-accumulation gate, then SmolVLA / OpenVLA-OFT comparison
- Data lane: generate V0/V1/V2 episode manifests and run cheap pi0.5 dataset ablations

The next synchronization point is after the first dataset ablation and model shortlist, before spending organizer GPU budget on Run A.
