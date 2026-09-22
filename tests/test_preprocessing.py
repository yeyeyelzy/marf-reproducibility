import numpy as np

from marc.preprocessing.graph import build_cooccurrence_graph, divide_transition_groups, generate_neighbors


def test_graph_support_and_row_normalization():
    adjacency = build_cooccurrence_graph([[0, 1, 1], [1, 2], [0, 2]], code_num=3, threshold=0.01)
    assert np.all(np.diag(adjacency) == 0.0)
    active = adjacency.sum(axis=1) > 0
    assert np.allclose(adjacency.sum(axis=1)[active], 1.0)
    assert np.array_equal(adjacency > 0, (adjacency > 0).T)


def test_neighbors_and_transition_groups():
    code_x = np.asarray([[[1, 1, 0], [0, 1, 1]]], dtype=bool)
    adjacency = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    lengths = np.asarray([2], dtype=np.int64)
    neighbors = generate_neighbors(code_x, lengths, adjacency)
    divided = divide_transition_groups(code_x, neighbors, lengths)
    assert neighbors.shape == code_x.shape
    assert divided.shape == code_x.shape + (3,)
    assert divided[0, 0, 0, 0]
