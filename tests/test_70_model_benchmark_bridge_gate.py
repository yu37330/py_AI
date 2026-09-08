"""Notebook 70 must accept validated streaming without regenerating D10 input."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab/70_model_benchmark_a100.ipynb"


def code_source():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        line
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )


def test_notebook70_uses_fixed_manifest_without_reselection():
    source = code_source()
    assert "run_pi05_group_aware_ablation.py" not in source
    assert "pi05-ablation-group-aware-v2/dataset_ablation_manifests_v2_group_aware/V2_SQRT_BALANCED_RAW.json" in source
    assert "73ed0d3b0c5e73c745c0aa2e81517ce1fa40240c75f9eb040d65b6876ba08239" in source
    assert "EPISODE_COUNT==10758" in source
    assert "FRAME_COUNT==1620614" in source


def test_notebook70_accepts_full_or_validated_streaming_bridge():
    source = code_source()
    assert "conversion_contract.json" in source
    assert "openvla-streaming-selected-v1/streaming_bridge_contract.json" in source
    assert "candidate.get('bridge_type')=='lerobot_streaming'" in source
    assert "e4287e94541f459edc4feabc4e181f537cd569a8" in source
    assert "candidate.get('source_episode_ids_sha256')==selected['episode_ids_sha256']" in source
    assert "oc.get('action_chunk')==8" in source
    assert "oc.get('normalization_type')=='bounds_q99'" in source
    assert "selected_subset_bridge_contract" in source
    assert "READY_FOR_RUNNER" in source


def test_notebook70_remains_controller_only():
    source = code_source()
    # The upstream finetune.py path is checked for existence, but never launched.
    assert "torchrun" not in source
    assert "accelerate launch" not in source
    assert "--max_steps" not in source
    assert "Controller gate passed. Do not infer that training ran" in source
