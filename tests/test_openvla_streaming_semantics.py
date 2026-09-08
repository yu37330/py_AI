"""69c OpenVLA streaming semantics regression tests (no dataset/network required)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "openvla_lerobot_streaming",
    ROOT / "tools/data/openvla_lerobot_streaming.py",
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_libero_action_standardization_matches_pinned_openvla_contract():
    raw = np.array(
        [
            [1, 2, 3, 4, 5, 6, -1.0],
            [1, 2, 3, 4, 5, 6, 0.0],
            [1, 2, 3, 4, 5, 6, 1.0],
        ],
        dtype=np.float32,
    )
    got = module.standardize_action(raw)
    np.testing.assert_array_equal(got[:, :6], raw[:, :6])
    np.testing.assert_array_equal(got[:, 6], np.array([1.0, 1.0, 0.0], dtype=np.float32))


def test_libero_proprio_is_state_first6_plus_last2():
    state = np.arange(16, dtype=np.float32).reshape(2, 8)
    got = module.standardize_proprio(state)
    expected = np.concatenate([state[:, :6], state[:, -2:]], axis=1)
    np.testing.assert_array_equal(got, expected)
    assert got.shape == (2, 8)


def test_bounds_q99_uses_full_min_max_for_constant_dimension_gate():
    values = np.array([[1.0, 9.0]], dtype=np.float32)
    got = module.bounds_q99_normalize(
        values,
        q01=np.array([1.0, 0.0], dtype=np.float32),
        q99=np.array([1.0, 10.0], dtype=np.float32),
        min_values=np.array([0.0, 9.0], dtype=np.float32),
        max_values=np.array([2.0, 9.0], dtype=np.float32),
    )
    # Dim 0 has q01==q99 but is not truly constant -> do not force to zero.
    assert got[0, 0] == -1.0
    # Dim 1 is truly constant by full min/max -> upstream maps it to zero.
    assert got[0, 1] == 0.0


def test_action_chunk_uses_relative_zero_and_absolute_gripper_padding():
    actions = np.array(
        [
            [1, 2, 3, 4, 5, 6, 0.25],
            [7, 8, 9, 10, 11, 12, 0.75],
        ],
        dtype=np.float32,
    )
    got = module.make_action_chunk(actions, 1, chunk_size=4)
    np.testing.assert_array_equal(got[0], actions[1])
    np.testing.assert_array_equal(got[1:, :6], np.zeros((3, 6), dtype=np.float32))
    np.testing.assert_array_equal(got[1:, 6], np.full((3,), 0.75, dtype=np.float32))


def test_openvla_streaming_constants_are_frozen():
    assert module.ACTION_DIM == 7
    assert module.PROPRIO_DIM == 8
    assert module.ACTION_CHUNK == 8
    assert module.NORMALIZATION_TYPE == "bounds_q99"
    assert module.ABSOLUTE_ACTION_MASK == [False] * 6 + [True]
    assert module.ACTION_NORMALIZATION_MASK == [True] * 6 + [False]
