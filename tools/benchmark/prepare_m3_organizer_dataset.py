#!/usr/bin/env python3
"""Materialize the exact frozen LIBERO-plus source on organizer NVMe scratch.

The dataset is deliberately excluded from the persistent handoff bundle. It is
reconstructible from the exact Hugging Face revision and belongs on
/opt/dlami/nvme rather than the organizer's 100 GiB persistent ~/data volume.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
D10_HASH = "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239"
D10_MANIFEST_SHA256 = "cd69c47224a84fec57886776f3981bb207c8c5e2308cebb64b5300bf96e46395"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to materialize the frozen dataset")
    if not os.environ.get("PARC_LOCAL_SCRATCH_ROOT"):
        raise RuntimeError("PARC_LOCAL_SCRATCH_ROOT is required")
    if not os.environ.get("PARC_PERSIST_ROOT"):
        raise RuntimeError("PARC_PERSIST_ROOT is required")

    repo = Path(os.environ.get("PY_AI_REPO", Path(__file__).resolve().parents[2])).expanduser().resolve()
    scratch = Path(os.environ["PARC_LOCAL_SCRATCH_ROOT"]).expanduser().resolve()
    persist = Path(os.environ["PARC_PERSIST_ROOT"]).expanduser().resolve()
    target = Path(
        os.environ.get(
            "PARC_DATASET_ROOT",
            str(scratch / "datasets/lerobot_libero_plus_v3_train"),
        )
    ).expanduser().resolve()
    evidence = persist / "model-benchmark-v1/m3-organizer-migration-v1/dataset_readiness.json"
    manifest = (
        persist
        / "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json"
    )

    if not (repo / ".git").is_dir():
        raise FileNotFoundError(f"PY_AI_REPO is not a Git checkout: {repo}")
    if not persist.is_dir():
        raise FileNotFoundError(persist)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if _sha256(manifest) != D10_MANIFEST_SHA256:
        raise RuntimeError("D10 manifest file SHA mismatch")
    manifest_data = _load(manifest)
    if manifest_data.get("episode_ids_sha256") != D10_HASH:
        raise RuntimeError("D10 manifest episode hash mismatch")

    target.parent.mkdir(parents=True, exist_ok=True)
    hf_home = Path(os.environ.get("HF_HOME", scratch / "cache/huggingface")).expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(target),
        token=os.environ["HF_TOKEN"],
        resume_download=True,
    )

    info = target / "meta/info.json"
    if not info.is_file():
        raise FileNotFoundError(f"dataset meta/info.json missing after download: {info}")
    for required_dir in ("meta", "data"):
        if not (target / required_dir).is_dir():
            raise FileNotFoundError(f"dataset directory missing: {target / required_dir}")

    # If the current Python environment already has the streaming dependencies,
    # perform a stronger selected-episode metadata check. Otherwise the pinned
    # training runtime will repeat this validation before training starts.
    selected_episode_metadata_validated = False
    metadata_validation_error = None
    try:
        sys.path.insert(0, str(repo))
        from tools.data.openvla_lerobot_streaming import load_episode_metadata  # noqa: PLC0415

        episode_ids = [int(x) for x in manifest_data["episode_ids"]]
        metadata = load_episode_metadata(target, episode_ids)
        if len(metadata) != len(episode_ids):
            raise RuntimeError(f"selected episode metadata count mismatch: {len(metadata)} != {len(episode_ids)}")
        selected_episode_metadata_validated = True
    except (ImportError, ModuleNotFoundError) as exc:
        metadata_validation_error = f"deferred_to_pinned_runtime: {type(exc).__name__}: {exc}"

    usage = shutil.disk_usage(scratch)
    payload = {
        "schema_version": 1,
        "stage": "M3_organizer_dataset_readiness",
        "status": "PASS",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_root": str(target),
        "dataset_meta_info_sha256": _sha256(info),
        "selected_episode_ids_sha256": D10_HASH,
        "d10_manifest_sha256": D10_MANIFEST_SHA256,
        "selected_episode_metadata_validated": selected_episode_metadata_validated,
        "metadata_validation_note": metadata_validation_error,
        "scratch_free_gib_after": round(usage.free / (1024**3), 2),
        "persistent_dataset_copy_created": False,
        "benchmark_training_started": False,
    }
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== M3 ORGANIZER DATASET: PASS ===", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
