# 69b dependency recovery

## Confirmed failure

Attempt `343b6686d693` passed preflight and failed at `rlds-conversion-deps`. uv reported that `tensorflow-datasets==4.9.6` requires `promise`, for which no usable wheel was available under the global binary-only policy. The Python 3.10 installation was successful. The parent Notebook's `CalledProcessError` was only a wrapper around the child failure.

## Fix and boundaries

Install pinned `promise==2.3` without runtime dependencies, allowing its source build, before installing the remaining pinned dependencies with the existing wheel-only policy. Verify the installed promise version. Keep Python 3.10, TensorFlow CPU 2.17.1, TFDS 4.9.6, and PyAV 12.3.0 pinned. Do not relax binary restrictions for PyAV or the other dependencies. Do not regenerate the dataset, retrain, or change the D10 decision.

The regenerated V2 manifest is backed up at `pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/`. The user-confirmed episode IDs SHA256 is `73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239`. This identifies the recovered selection; historical identity with the original D10 training pool remains unproven. Preserve the manifest and its provenance record.

## Verification and execution

Run the dependency contract tests without downloading the dataset. Actual dependency installation and the eight-episode TFDS smoke require a Colab runtime and are not implied by unit-test success. After the PR is merged, use the updated Notebook revision; its pinned checkout must contain this fix. Do not reuse the old Notebook that checks out `5f7cbb4ed4b054b381a3a0074ab54b419b09a5de`.

A successful 69b must report `status=PASS` in `openvla-rlds-selected-v1/bridge_smoke_status.json` and produce a provenance-matched smoke report and capacity decision. A dependency failure leaves capacity BLOCKED. The smoke does not authorize full conversion or M3 training.
