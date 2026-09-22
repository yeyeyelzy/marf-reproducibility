"""Training-only co-occurrence graph construction used by MARC."""

from typing import Iterable, Sequence

import numpy as np


def normalize_rows(adjacency: np.ndarray) -> np.ndarray:
    """Row-normalize a non-negative adjacency; zero rows remain zero."""
    row_sum = adjacency.sum(axis=-1, keepdims=True).astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    return adjacency.astype(np.float64) / row_sum


def build_cooccurrence_graph(visits: Iterable[Sequence[int]], code_num: int, threshold: float = 0.01) -> np.ndarray:
    """Count bidirectional within-visit pairs, prune low row shares, then normalize.

    Raw pair support is symmetric and has no self loops.  The final row-normalized
    matrix can be asymmetric, which is the matrix consumed by graph propagation.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")
    raw = np.zeros((code_num, code_num), dtype=np.int64)
    for visit in visits:
        codes = np.unique(np.asarray(visit, dtype=np.int64))
        if np.any(codes < 0) or np.any(codes >= code_num):
            raise ValueError("Visit contains an out-of-vocabulary code ID.")
        for left_index in range(max(codes.size - 1, 0)):
            left = codes[left_index]
            right = codes[left_index + 1 :]
            raw[left, right] += 1
            raw[right, left] += 1
    preliminary = normalize_rows(raw)
    active_rows = raw.sum(axis=-1, keepdims=True) > (1.0 / threshold)
    raw[np.logical_and(preliminary < threshold, active_rows)] = 0
    np.fill_diagonal(raw, 0)
    return normalize_rows(raw).astype(np.float32)


def generate_neighbors(code_x: np.ndarray, visit_lens: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    neighbors = np.zeros_like(code_x, dtype=bool)
    for patient_index, visits in enumerate(code_x):
        for visit_index in range(int(visit_lens[patient_index])):
            codes = set(np.where(visits[visit_index] > 0)[0].tolist())
            candidate = set()
            for code in codes:
                candidate.update(np.where(adjacency[code] > 0)[0].tolist())
            candidate.difference_update(codes)
            if candidate:
                neighbors[patient_index, visit_index, list(candidate)] = True
    return neighbors


def divide_transition_groups(code_x: np.ndarray, neighbors: np.ndarray, visit_lens: np.ndarray) -> np.ndarray:
    divided = np.zeros(code_x.shape + (3,), dtype=bool)
    for patient_index, visits in enumerate(code_x):
        length = int(visit_lens[patient_index])
        divided[patient_index, 0, :, 0] = visits[0] > 0
        for visit_index in range(1, length):
            current = set(np.where(visits[visit_index] > 0)[0].tolist())
            previous = set(np.where(visits[visit_index - 1] > 0)[0].tolist())
            previous_neighbors = set(np.where(neighbors[patient_index, visit_index - 1] > 0)[0].tolist())
            for group, members in enumerate((current & previous, current & previous_neighbors, current - previous - previous_neighbors)):
                if members:
                    divided[patient_index, visit_index, list(members), group] = True
    return divided
