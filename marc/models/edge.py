"""State-aware edge calibration that never changes graph support."""

import torch
from torch import nn


class EdgeCalibration(nn.Module):
    def __init__(self, node_size: int, state_size: int, hidden_size: int, use_state: bool = True):
        super().__init__()
        self.use_state = bool(use_state)
        input_size = 2 * node_size + (state_size if self.use_state else 0)
        self.score = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.register_buffer("_source", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_target", torch.empty(0, dtype=torch.long), persistent=False)
        self._shape = None

    def _edge_index(self, adjacency: torch.Tensor):
        if self._shape != tuple(adjacency.shape) or not self._source.numel() or self._source.device != adjacency.device:
            if adjacency.is_sparse:
                indices = adjacency.coalesce().indices()
                values = adjacency.coalesce().values()
                keep = values > 0
                self._source, self._target = indices[0, keep], indices[1, keep]
            else:
                indices = torch.nonzero(adjacency > 0, as_tuple=False)
                self._source, self._target = indices[:, 0], indices[:, 1]
            self._shape = tuple(adjacency.shape)
        return self._source, self._target

    @staticmethod
    def _weights(adjacency: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if adjacency.is_sparse:
            dense = adjacency.to_dense()
            return dense[source, target]
        return adjacency[source, target]

    def forward(self, adjacency: torch.Tensor, node_representations: torch.Tensor, patient_state: torch.Tensor, return_gates: bool = False):
        source, target = self._edge_index(adjacency)
        if not source.numel():
            result = adjacency.coalesce() if adjacency.is_sparse else adjacency.to_sparse().coalesce()
            return (result, source, target, source.new_empty((0,), dtype=torch.float32)) if return_gates else result
        inputs = [node_representations[source], node_representations[target]]
        if self.use_state:
            inputs.append(patient_state.reshape(1, -1).expand(source.numel(), -1))
        gates = self.score(torch.cat(inputs, dim=-1)).squeeze(-1)
        weights = self._weights(adjacency, source, target) * gates
        conditioned = torch.sparse_coo_tensor(
            torch.stack((source, target)), weights, adjacency.shape, dtype=adjacency.dtype, device=adjacency.device
        ).coalesce()
        return (conditioned, source, target, gates) if return_gates else conditioned
