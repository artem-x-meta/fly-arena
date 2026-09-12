import numpy as np

from fly_arena.brain import advance, BrainConfig, MotorDecoder
from fly_arena.data import fill_csr, map_ids


def reference(ptr, posts, weights, v, current, refractory, queue, clock, drive, steps):
    """Slow independent dense reference for delivery order and membrane dynamics."""
    matrix = np.zeros((len(v), len(v)), np.float32)
    for pre in range(len(v)):
        for e in range(ptr[pre], ptr[pre + 1]):
            matrix[pre, posts[e]] += weights[e]
    counts = np.zeros(len(v), np.int32)
    for tick in range(steps):
        slot = (clock + tick) % 3
        current[:] = current * np.exp(-.2) + queue[slot]
        queue[slot] = 0
        refractory_mask = refractory > 0
        refractory[refractory_mask] -= 1
        candidate = np.exp(-.05) * v + (1 - np.exp(-.05)) * (drive + current)
        v[:] = np.maximum(-30, candidate)
        v[refractory_mask] = 0
        fired = (~refractory_mask) & (v >= 7)
        counts += fired
        v[fired] = 0
        refractory[fired] = 2
        queue[(clock + tick + 2) % 3] += matrix[fired].sum(axis=0)
    return counts


def fixture():
    # Includes recurrent excitation, inhibition, a self-edge and duplicate targets.
    ptr = np.array([0, 3, 4, 6], np.int64)
    posts = np.array([0, 1, 1, 2, 0, 1], np.int32)
    weights = np.array([.7, 12, 3, -9, 4, 1], np.float32)
    state = [np.array([6.8, 2, 4], np.float32), np.zeros(3, np.float32),
             np.zeros(3, np.int16), np.zeros((3, 3), np.float32)]
    drive = np.array([15, 9, 10], np.float32)
    return ptr, posts, weights, state, drive


def test_sparse_kernel_matches_dense_dynamics():
    ptr, posts, weights, state, drive = fixture()
    dense_state = [x.copy() for x in state]
    actual = advance(ptr, posts, weights, *state, 0, drive, 500)
    expected = reference(ptr, posts, weights, *dense_state, 0, drive, 500)
    np.testing.assert_array_equal(actual, expected)
    for actual_state, expected_state in zip(state, dense_state):
        np.testing.assert_allclose(actual_state, expected_state, atol=2e-5)


def test_splitting_calls_preserves_delays_and_state():
    ptr, posts, weights, state, drive = fixture()
    split = [x.copy() for x in state]
    total = advance(ptr, posts, weights, *state, 0, drive, 100)
    part = advance(ptr, posts, weights, *split, 0, drive, 37)
    part += advance(ptr, posts, weights, *split, 37, drive, 63)
    np.testing.assert_array_equal(total, part)
    for a, b in zip(state, split):
        np.testing.assert_array_equal(a, b)


def test_motor_readout_has_no_hidden_baseline():
    ports = {"DNp09": {"L": [0], "R": [1]}, "DNa02": {"L": [2], "R": [3]},
             "MDN": {"L": [], "R": []}}
    decoder = MotorDecoder(ports, BrainConfig())
    np.testing.assert_array_equal(decoder.update(np.zeros(4), .01), [0, 0])
    forward = decoder.update(np.array([1, 1, 0, 0]), .01)
    assert forward[0] == forward[1] > 0
    left = decoder.update(np.array([1, 1, 1, 0]), .01)
    assert left[1] > left[0]


def test_id_mapping_does_not_attach_unknown_segments():
    ids = np.array([10, 40, 90])
    idx, keep = map_ids(ids, np.array([0, 10, 39, 40, 91]))
    np.testing.assert_array_equal(keep, [False, True, False, True, False])
    np.testing.assert_array_equal(idx[keep], [0, 1])


def test_csr_preserves_weak_self_and_repeated_edges():
    rows = np.array([[1, 1, 1], [0, 1, 7], [1, 0, 2], [0, 1, 1]], np.int32)
    ptr = np.array([0, 2, 4], np.int64)
    cursor = ptr[:-1].copy()
    posts = np.empty(4, np.int32)
    counts = np.empty(4, np.uint32)
    fill_csr(rows, cursor, posts, counts)
    np.testing.assert_array_equal(cursor, ptr[1:])
    np.testing.assert_array_equal(posts, [1, 1, 1, 0])
    np.testing.assert_array_equal(counts, [7, 1, 1, 2])
