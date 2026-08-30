# PARC2026 Group-aware Fixed Eval V2

## Why V1 must not be used for dataset ablation

The trajectory-group leakage check on the public `lerobot/libero_plus` proxy found:

- 14,347 episodes hashed
- 1,681 exact state+action trajectory groups
- every episode belongs to a duplicated exact group
- the legacy 80-episode fixed eval spans only 79 exact groups
- all 80 eval episodes have an exact-group sibling in training
- V0 / V1 training contain 642 sibling episodes across those 79 leaked groups
- V2 sqrt-balanced still contains 489 sibling episodes across the same 79 leaked groups
- Leakage Gate = FAIL for every V0/V1/V2 manifest

This is enough to reject the old episode-level split for comparative offline evaluation. It does not by itself prove why the duplicates exist; visual/domain perturbation siblings are one plausible explanation, but the Gate only relies on the observed non-visual state+action equality.

## V2 split contract

`tools/data/build_group_aware_ablation_manifests.py` changes the sampling unit of the fixed eval split from **episode** to **exact trajectory group**.

For every task:

1. consider Static Quality `OK` episodes only for eval selection,
2. deduplicate them by `exact_group_hash`,
3. choose two distinct groups with seed `20260830`,
4. use one deterministic representative episode from each chosen group for eval,
5. mark every episode in each chosen group as protected,
6. remove all protected episodes from every training variant before filtering or balancing.

The eval set therefore remains compact at 2 episodes/task while the training pool cannot contain an exact state+action sibling of any eval episode.

## Manifest schema additions

`FIXED_EVAL_HOLDOUT.json` uses schema version 2 and adds:

- `group_aware: true`
- `group_key: exact_group_hash`
- `exact_group_hashes`
- `protected_episode_ids`
- hashes for those lists
- `protected_episode_count`
- per-selected-group rows

Every training manifest also carries `group_aware: true`, `group_key`, and the number of protected eval groups/episodes.

## Variant order is unchanged

After group exclusion the same dataset ablation axes are rebuilt:

- `V0_RAW`
- `V1_INTEGRITY_ONLY`
- `V1_MULTI_FLAG_PRUNED_EXPERIMENTAL`
- `V1_ALL_REVIEW_PRUNED_EXPERIMENTAL`
- `V2_SQRT_BALANCED_RAW`

Filtering and task balancing therefore remain separable from leakage prevention.

## Required Gate

Run `colab/47_group_aware_ablation_manifests.ipynb`.

It regenerates the group-aware manifests and then runs `check_trajectory_group_leakage.py` again with `--fail-on-exact-leakage`.

Do not start π0.5 dataset ablation until both lines appear:

```text
Group-aware Manifest Gate: PASS
Trajectory Leakage Gate: PASS
```

After this Gate passes, update the training notebook to accept only `dataset_ablation_manifests_v2_group_aware` and refuse the legacy V1 split.

## Organizer-data requirement

The public LIBERO-plus result is only a Data Factory proxy. Before fixing Run A, regenerate trajectory groups and the group-aware split on the organizer `libero_combined_20hz` dataset. Group counts and protected fractions must not be assumed to match the public proxy.
