# PARC2026 post-screening notebook sequence

This sequence implements `experiments/plans/parallel_model_data_v2.yaml`. It intentionally does not jump from the π0.5 dataset screen directly to 20k training.

## Order

1. `65_pi05_top2_tiebreak_l4.ipynb`
   - rerun only V1_MULTI vs V2_SQRT with independent eval seed `20260906`
   - same Track1 / 4 tasks / 8 episodes / max 300-step contract
   - writes `provisional_best_dataset_recipe.json` only as `DECIDED` when simulator success breaks the tie
   - if still tied, records `STILL_TIED` and stops automatic promotion
   - current D10 result: `V2_SQRT_BALANCED_RAW` selected

2. `69_model_benchmark_bringup_smoke.ipynb`
   - lightweight preflight before allocating an A100 to M2/M3
   - CPU / L4 / A100 are all accepted because no model-weight training or inference is started
   - verifies D10, Drive metadata/q01/q99, registry and pinned sources
   - does not copy the ~15GB dataset locally or load large model weights
   - current result: preflight PASS; selected-subset OpenVLA RLDS remains missing

3. `69b_openvla_selected_rlds_smoke.ipynb`
   - **Recovery version: run its only code cell once. Do not run the old 2/5, 3/5 or 4/5 cells.**
   - the notebook uses a separate, pinned code checkout and invokes `tools/colab/rlds_smoke_recovery.py`
   - CPU / L4 / A100; no model training, no full RLDS materialisation
   - requires the D10 `V2_SQRT_BALANCED_RAW` decision and an exact group-aware training manifest with valid episode IDs/hash
   - preserves existing source/data/decision/manifest artifacts; never silently reselects or regenerates a different pool
   - reuses a verified Python 3.10 conversion environment, otherwise installs binary-only pinned dependencies (`av==12.3.0`)
   - converts only 8 exact selected episodes (`--max-episodes 8`) from LeRobot v3/20Hz into OpenVLA-style RLDS/TFDS
   - uses front/wrist images, state8/action7 and task language; no extra no-op filtering
   - conversion runs in a dedicated new local attempt directory; a failed attempt cannot be mistaken for a previously successful report
   - verifies report provenance/counts and prepared TFDS; existing artifacts are reused only after matching checks
   - prints the child error tail on failure and saves complete logs under `openvla-rlds-selected-v1/logs/`
   - the authoritative status is `openvla-rlds-selected-v1/bridge_smoke_status.json`; only `status=PASS` means 69b completed
   - success writes `bridge_smoke_report.json` and `bridge_capacity_decision.json`; failure leaves capacity `BLOCKED` and preserves the previous capacity result under `history/`
   - if projected full materialisation exceeds 35 GiB, prefer a validated streaming bridge; this threshold is a planning heuristic, not an automatic conversion command
   - the 8-episode smoke never creates a full `conversion_contract.json`
   - if the exact original manifest is unavailable, restore it from the completed dataset-screening artifacts; do not silently create a new dataset selection

4. M3 screening budget
   - frozen in `experiments/plans/model_benchmark_budget_v1.json`
   - Equal Data Exposure: 4,800 training examples/model, matching the proven 150 optimizer-step × effective batch 32 dataset screen
   - Equal Wall Time: 1,800 seconds/model of training-loop time only
   - setup/download/dataset conversion/checkpoint merge/simulator evaluation are excluded from the wall-time clock
   - this is a screening budget, not Run A full training

5. `70_model_benchmark_a100.ipynb`
   - requires D10 `DECIDED`
   - candidates are π0.5 / SmolVLA / OpenVLA-OFT
   - pins exact upstream/base revisions in `experiments/model_registry_v1.json`
   - requires the same group-aware selected episode pool and evaluation contract
   - comparison protocols are Equal Data Exposure and Equal Wall Time
   - OpenVLA-OFT remains blocked until the exact selected episode pool has a provenance-matched full RLDS conversion or an explicitly validated streaming-equivalent bridge
   - **69b success does not unblock M3 by itself.** The actual three-model training and inference adapters still require implementation and runtime verification.
   - target promotion is at most two models

6. `75_generalization_screening_a100.ipynb`
   - requires M3 shortlist (max two models)
   - baseline best recipe vs safe visual augmentation / targeted public supplemental data
   - candidates: brightness, contrast, mild color jitter, mild crop-resize, provenance-gated supplemental data
   - Track1 regression gate; Track2 is the primary generalization target
   - writes `generalization_result.json` only after comparable candidate results exist

7. `80_track3_inverse_factory.ipynb`
   - can prepare the registry before M3, but inverse ablation requires the M3 shortlist
   - naive action/time reversal is forbidden
   - valid inverse demos require inverse task definition + inverse initial state + simulator success + replay validation + LeRobot 20Hz conversion
   - preregisters inverse mix ratios 0 / 5 / 10 / 20 percent
   - Track3 success is primary; Track1/2 are regression gates

8. Reapply the final recipe to organizer `libero_combined_20hz`
   - rerun inventory / integrity / trajectory-group leakage gates
   - public `lerobot/libero_plus` is only the screening proxy
   - this execution depends on the selected M3/G1/T3 recipe and remains a later organizer-side step

9. `90_run_a_freeze.ipynb`
   - requires D10, M3, G1, T3 and organizer-source gates
   - only then writes `run_a_recipe.json` with `status=READY`
   - Run A is the common Track1/2/3 candidate
   - Run B remains conditional on Track3 being clearly weaker after Run A and inverse screening being positive

10. 20k Run A full training
   - only after `90` reports READY
   - evaluate intermediate checkpoints (5k / 10k / 15k / 20k)
   - merge, evaluate Track1/2/3, then validate L4 24GB latency/VRAM and offline package

## Hard rules

- Do not start 20k before model comparison and pre-training gates.
- Do not choose a dataset/model from training loss alone.
- Do not use the legacy leaky episode-level split.
- Do not use naive reversed action sequences as Track3 demonstrations.
- Do not substitute OpenVLA's upstream RLDS dataset for the selected common episode pool when claiming a fair M3 comparison.
- Do not blindly materialise the full V2 RLDS copy before measuring its storage cost.
- Keep model-specific training environments separate and normalize only the comparison contract, not the implementation stack.
