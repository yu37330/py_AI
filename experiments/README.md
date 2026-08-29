# Experiment layout

This directory defines reproducible cross-model experiments for PARC2026.

## Directories

- `configs/`: model/training definitions. Keep upstream repository + revision explicit.
- `manifests/datasets/`: immutable descriptions of the training data used by a run.
- `manifests/eval_splits/`: fixed task/seed/episode selections for local comparison.
- `results/`: small human-readable summaries only. Large checkpoints/videos stay outside git.

## Fair comparison

Use two complementary protocols instead of assuming that identical optimizer steps are fair across architectures:

### Protocol A: equal data exposure

Hold the dataset manifest, eval split, seeds and approximate number of training samples constant. This measures sample efficiency and task performance.

### Protocol B: equal wall-time budget

Give each candidate the same A100 training-time budget and evaluate with the same split. This measures practical return per GPU-hour.

For both protocols record:

- success rate per task / track-like group,
- steps-to-success when available,
- trajectory/path/smoothness metrics when available,
- policy inference latency,
- peak VRAM,
- train wall time,
- exact source revisions and command.

Training loss by itself is not a promotion criterion.

## Promotion rule

A candidate is promoted to organizer-GPU training only when it beats the current candidate on the fixed local evaluation and still fits the PARC submission/runtime constraints.
