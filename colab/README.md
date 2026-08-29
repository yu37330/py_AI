# Colab A100 benchmark workspace

`colab/` is the entry point for experiments that should be completed before spending the PARC organizer GPU budget.

## Goal

Use a Colab A100 for model comparison, data ablation, Track3 inverse-data research, and submission preflight. Keep the organizer GPU for the final candidate training/evaluation runs.

## Repository roles

- `py_AI`: experiment definitions, manifests, benchmark outputs, PARC evaluation/submission code.
- `/content/vendor/*`: disposable upstream clones such as LeRobot/OpenVLA-OFT. Do not vendor them into this repository.
- `/content/cache/*`: Hugging Face/model caches and temporary datasets.
- Google Drive: optional persistence for checkpoints and large generated artifacts only.

## First notebook

Open `00_a100_preflight.ipynb` in Colab. It checks the GPU, prepares `/content/parc2026`, clones `py_AI`, and creates the expected workspace.

After preflight, run model-specific experiments using the configs under `experiments/configs/`.

## Comparison contract

Every comparable run must pin:

1. git SHA / upstream model revision,
2. dataset manifest,
3. eval split,
4. seed,
5. effective batch size and optimizer-step semantics,
6. wall time, peak VRAM, latency and task metrics.

Do not select a model using training loss alone. Prefer simulator success plus the PARC-adjacent metrics that can be reproduced locally.

## Planned sequence

1. Reproduce pi0.5 LoRA smoke and verify GA semantics.
2. Add SmolVLA on the same dataset/eval split.
3. Add OpenVLA-OFT on the same dataset/eval split.
4. Compare model choice and dataset ablations separately.
5. Research Track3 inverse/reversed demonstrations.
6. Promote only the strongest configuration to organizer-GPU training.
