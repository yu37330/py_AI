# PARC2026 post-screening notebook sequence

This sequence implements `experiments/plans/parallel_model_data_v2.yaml`. It intentionally does not jump from the π0.5 dataset screen directly to 20k training.

## Order

1. `65_pi05_top2_tiebreak_l4.ipynb`
   - rerun only V1_MULTI vs V2_SQRT with independent eval seed `20260906`
   - same Track1 / 4 tasks / 8 episodes / max 300-step contract
   - writes `provisional_best_dataset_recipe.json` only as `DECIDED` when simulator success breaks the tie
   - if still tied, records `STILL_TIED` and stops automatic promotion

2. `69_model_benchmark_bringup_smoke.ipynb`
   - lightweight preflight before allocating an A100 to M2/M3
   - CPU / L4 / A100 are all accepted because no model-weight training or inference is started
   - verifies the D10 winner, Drive-prefetched dataset metadata/q01/q99, exact model registry, and pinned SmolVLA/OpenVLA-OFT source revisions
   - does not copy the ~15GB dataset to local storage and does not load large model weights
   - reports the expected remaining blockers: selected-subset OpenVLA RLDS conversion and explicit Equal Data Exposure / Equal Wall Time budgets
   - writes `model-benchmark-smoke-v1/smoke_report.json` to Drive

3. `70_model_benchmark_a100.ipynb`
   - requires D10 `DECIDED`
   - candidates are π0.5 / SmolVLA / OpenVLA-OFT
   - pins exact upstream/base revisions in `experiments/model_registry_v1.json`
   - requires the same group-aware selected episode pool and evaluation contract
   - comparison protocols are Equal Data Exposure and Equal Wall Time
   - comparison budgets must be explicitly frozen; the notebook does not invent them
   - OpenVLA-OFT is blocked until the exact selected LeRobot subset has a provenance-matched RLDS conversion
   - target promotion is at most two models

4. `75_generalization_screening_a100.ipynb`
   - requires M3 shortlist (max two models)
   - baseline best recipe vs safe visual augmentation / targeted public supplemental data
   - candidates: brightness, contrast, mild color jitter, mild crop-resize, provenance-gated supplemental data
   - Track1 regression gate; Track2 is the primary generalization target
   - writes `generalization_result.json` only after comparable candidate results exist

5. `80_track3_inverse_factory.ipynb`
   - can prepare the registry before M3, but inverse ablation requires the M3 shortlist
   - naive action/time reversal is forbidden
   - valid inverse demos require inverse task definition + inverse initial state + simulator success + replay validation + LeRobot 20Hz conversion
   - preregisters inverse mix ratios 0 / 5 / 10 / 20 percent
   - Track3 success is primary; Track1/2 are regression gates

6. Reapply the final recipe to organizer `libero_combined_20hz`
   - rerun inventory / integrity / trajectory-group leakage gates
   - public `lerobot/libero_plus` is only the screening proxy
   - this execution depends on the selected M3/G1/T3 recipe and remains a later organizer-side step

7. `90_run_a_freeze.ipynb`
   - requires D10, M3, G1, T3 and organizer-source gates
   - only then writes `run_a_recipe.json` with `status=READY`
   - Run A is the common Track1/2/3 candidate
   - Run B remains conditional on Track3 being clearly weaker after Run A and inverse screening being positive

8. 20k Run A full training
   - only after `90` reports READY
   - evaluate intermediate checkpoints (5k / 10k / 15k / 20k)
   - merge, evaluate Track1/2/3, then validate L4 24GB latency/VRAM and offline package

## Hard rules

- Do not start 20k before model comparison and pre-training gates.
- Do not choose a dataset/model from training loss alone.
- Do not use the legacy leaky episode-level split.
- Do not use naive reversed action sequences as Track3 demonstrations.
- Do not substitute OpenVLA's upstream RLDS dataset for the selected common episode pool when claiming a fair M3 comparison.
- Keep model-specific training environments separate and normalize only the comparison contract, not the implementation stack.
