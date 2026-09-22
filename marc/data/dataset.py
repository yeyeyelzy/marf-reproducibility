"""Data contract for privately generated MIMIC-derived assets.

No data or generated assets are distributed by this repository.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_sparse(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as payload:
        if set(payload.files) != {"idx", "values", "shape"}:
            raise ValueError("Sparse asset must contain idx, values, and shape: {}".format(path))
        indices, values, shape = payload["idx"], payload["values"], tuple(payload["shape"].tolist())
    matrix = np.zeros(shape, dtype=values.dtype)
    matrix[tuple(indices)] = values
    return matrix


def load_adjacency(data_root: str, device: torch.device) -> torch.Tensor:
    adjacency_path = Path(data_root) / "code_adj.npz"
    if not adjacency_path.is_file():
        raise FileNotFoundError("Missing training-only adjacency asset: {}".format(adjacency_path))
    return torch.from_numpy(load_sparse(adjacency_path)).to(device=device, dtype=torch.float32)


def _timestamp_to_day(value, previous: int) -> int:
    if value is None or value == 0:
        return previous
    if isinstance(value, datetime):
        return value.toordinal()
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[D]").astype(np.int64))
    return int(value)


class HeartFailureDataset:
    """A compact batch dataset for the formal TCEM/MARC asset schema."""

    required_sparse = ("code_x", "divided", "neighbors")

    def __init__(
        self,
        data_root: str,
        split: str,
        batch_size: int,
        shuffle: bool,
        device: torch.device,
        time_weight_mode: str = "split_specific",
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test.")
        self.root = Path(data_root)
        self.path = self.root / split
        if not self.path.is_dir():
            raise FileNotFoundError("Missing split directory: {}".format(self.path))
        self.batch_size, self.shuffle, self.device = batch_size, shuffle, device
        arrays = {name: load_sparse(self.path / (name + ".npz")) for name in self.required_sparse}
        self.code_x, self.divided, self.neighbors = arrays["code_x"], arrays["divided"], arrays["neighbors"]
        with np.load(str(self.path / "visit_lens.npz"), allow_pickle=False) as payload:
            self.visit_lens = np.asarray(payload["lens"], dtype=np.int64)
        with np.load(str(self.path / "hf_y.npz"), allow_pickle=False) as payload:
            self.labels = np.asarray(payload["hf_y"], dtype=np.float32).reshape(-1)
        weight_path = self.path / "time_weight.npz" if time_weight_mode == "split_specific" else self.root / "time_weight.npz"
        self.time_weight = load_sparse(weight_path)
        if self.time_weight.shape[0] < self.code_x.shape[0]:
            raise ValueError("Time weights do not cover this data split.")
        if time_weight_mode == "split_specific" and self.time_weight.shape[0] != self.code_x.shape[0]:
            raise ValueError("Split-specific time weights must align exactly with the split.")
        self.timestamps = self._load_timestamps()
        self.indices = np.arange(self.code_x.shape[0])
        self._validate()

    def _load_timestamps(self):
        candidate = self.path / "code_x_timestamp.npz"
        if not candidate.is_file():
            return None
        with np.load(str(candidate), allow_pickle=True) as payload:
            return payload["time"]

    def _validate(self) -> None:
        n, max_visits, code_num = self.code_x.shape
        if self.divided.shape != (n, max_visits, code_num, 3):
            raise ValueError("divided must have shape [N, T, C, 3].")
        if self.neighbors.shape != self.code_x.shape:
            raise ValueError("neighbors must have the same shape as code_x.")
        if self.visit_lens.shape != (n,) or self.labels.shape != (n,):
            raise ValueError("visit_lens and hf_y must have one value per patient.")
        if np.any(self.visit_lens < 1) or np.any(self.visit_lens > max_visits):
            raise ValueError("visit_lens must be within [1, max_visits].")
        if self.time_weight.shape[1:] != (code_num, 1):
            raise ValueError("time_weight must have shape [N, C, 1].")

    def __len__(self) -> int:
        return int(np.ceil(self.size / self.batch_size))

    @property
    def size(self) -> int:
        return int(self.code_x.shape[0])

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _visit_gaps(self, rows: np.ndarray) -> np.ndarray:
        max_visits = self.code_x.shape[1]
        if self.timestamps is None:
            return np.zeros((len(rows), max_visits), dtype=np.int64)
        result = np.zeros((len(rows), max_visits), dtype=np.int64)
        for output_index, row in enumerate(rows):
            previous = 0
            days = np.zeros(max_visits, dtype=np.int64)
            for visit_index, timestamp in enumerate(np.asarray(self.timestamps[row], dtype=object)):
                days[visit_index] = _timestamp_to_day(timestamp, previous)
                previous = days[visit_index]
            result[output_index, 1:] = days[1:] - days[:-1]
        return result

    def __getitem__(self, step: int):
        start, end = step * self.batch_size, min((step + 1) * self.batch_size, self.size)
        rows = self.indices[start:end]
        return (
            torch.from_numpy(self.code_x[rows]).to(self.device, dtype=torch.float32),
            torch.from_numpy(self.visit_lens[rows]).to(self.device, dtype=torch.long),
            torch.from_numpy(self.divided[rows]).to(self.device, dtype=torch.float32),
            torch.from_numpy(self.labels[rows]).to(self.device, dtype=torch.float32),
            torch.from_numpy(self.neighbors[rows]).to(self.device, dtype=torch.float32),
            torch.from_numpy(self.time_weight[rows]).to(self.device, dtype=torch.float32),
            torch.from_numpy(self._visit_gaps(rows)).to(self.device, dtype=torch.long),
        )
