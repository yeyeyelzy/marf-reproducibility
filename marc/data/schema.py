"""Names and shape checks for private MARC assets."""

from typing import Dict, Tuple


SPARSE_ARRAY_KEYS = ("idx", "values", "shape")
SPLIT_SPARSE_FILES = ("code_x", "divided", "neighbors", "time_weight")
SPLIT_DENSE_FILES = ("visit_lens", "hf_y")


def expected_split_files() -> Dict[str, Tuple[str, ...]]:
    """Return the public contract without accessing private assets."""
    return {
        "sparse": tuple(name + ".npz" for name in SPLIT_SPARSE_FILES),
        "dense": tuple(name + ".npz" for name in SPLIT_DENSE_FILES),
        "optional": ("code_x_timestamp.npz",),
    }
