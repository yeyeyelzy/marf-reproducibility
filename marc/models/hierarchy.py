"""CCS hierarchy calibration used by MARC H."""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn


class HierarchyCalibration(nn.Module):
    """Attention or uniform aggregation of a code and its CCS ancestors.

    The required NPZ contains integer ``ancestor_ids`` and Boolean ``ancestor_mask``.
    Row zero is padding; each remaining row begins with its own one-based code ID.
    Generated hierarchy assets are intentionally not distributed with this repository.
    """

    def __init__(
        self,
        code_num: int,
        embedding_dim: int,
        ontology_path: str,
        attention_size: int = 128,
        residual_init: float = 0.10,
        aggregation: str = "attention",
        writeback: str = "residual",
    ):
        super().__init__()
        asset = Path(ontology_path)
        if not asset.is_file():
            raise FileNotFoundError("CCS hierarchy asset was not found: {}".format(asset))
        if aggregation not in {"attention", "uniform", "self_only"}:
            raise ValueError("Unsupported hierarchy aggregation: {}".format(aggregation))
        if writeback not in {"residual", "replace"}:
            raise ValueError("Unsupported hierarchy write-back: {}".format(writeback))
        if not 0.0 < residual_init < 1.0:
            raise ValueError("residual_init must be in (0, 1).")

        with np.load(str(asset), allow_pickle=False) as payload:
            if set(payload.files) != {"ancestor_ids", "ancestor_mask"}:
                raise ValueError("Hierarchy asset must contain ancestor_ids and ancestor_mask only.")
            ids = np.asarray(payload["ancestor_ids"])
            mask = np.asarray(payload["ancestor_mask"], dtype=bool)
        self._validate_asset(ids, mask, code_num)
        ids, mask, compact, original_ids = self._compact_ancestor_ids(ids[1:], mask[1:], code_num)

        self.code_num = code_num
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.writeback = writeback
        self.max_depth = int(ids.shape[1])
        self.register_buffer("ancestor_ids", torch.from_numpy(ids))
        self.register_buffer("ancestor_mask", torch.from_numpy(mask))
        self.register_buffer("ancestor_compact_index", torch.from_numpy(compact))
        self.register_buffer("original_ancestor_ids", torch.from_numpy(original_ids))

        self.ancestor_embedding = nn.Embedding(int(original_ids.size), embedding_dim)
        self.attention_hidden = nn.Linear(2 * embedding_dim, attention_size)
        self.attention_projection = nn.Linear(attention_size, 1, bias=False)
        self.rho_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))
        nn.init.uniform_(self.ancestor_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.attention_hidden.weight, -0.1, 0.1)
        nn.init.zeros_(self.attention_hidden.bias)
        nn.init.uniform_(self.attention_projection.weight, -0.1, 0.1)

    @staticmethod
    def _validate_asset(ids: np.ndarray, mask: np.ndarray, code_num: int) -> None:
        if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != code_num + 1:
            raise ValueError("Hierarchy arrays must have shape [code_num + 1, max_depth].")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("ancestor_ids must be integer-valued.")
        if np.any(ids[0] != 0) or not mask[0, 0] or np.any(mask[0, 1:]):
            raise ValueError("Invalid hierarchy padding row.")
        expected_self = np.arange(1, code_num + 1, dtype=np.int64)
        if not np.all(mask[1:, 0]) or not np.array_equal(ids[1:, 0], expected_self):
            raise ValueError("Each code row must begin with its own active one-based ID.")
        if np.any(~mask & (ids != 0)):
            raise ValueError("Masked hierarchy slots must use ID zero.")
        if np.any(mask[1:, 1:] & (ids[1:, 1:] <= code_num)):
            raise ValueError("Non-self active hierarchy IDs must be ancestor concepts.")

    @staticmethod
    def _compact_ancestor_ids(ids: np.ndarray, mask: np.ndarray, code_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del code_num
        ids = ids.astype(np.int64, copy=True)
        mask = mask.astype(bool, copy=True)
        original = np.unique(ids[:, 1:][mask[:, 1:]])
        original.sort()
        compact = np.zeros_like(ids, dtype=np.int64)
        if original.size:
            compact[:, 1:][mask[:, 1:]] = np.searchsorted(original, ids[:, 1:][mask[:, 1:]])
        return ids, mask, compact, original

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.rho_logit)

    def _candidates(self, leaves: torch.Tensor) -> torch.Tensor:
        repeated_leaves = leaves.unsqueeze(1).expand(-1, self.max_depth, -1)
        if self.ancestor_embedding.num_embeddings == 0:
            return repeated_leaves
        ancestors = self.ancestor_embedding(self.ancestor_compact_index)
        ancestor_slots = self.ancestor_mask.clone()
        ancestor_slots[:, 0] = False
        return torch.where(ancestor_slots.unsqueeze(-1), ancestors, repeated_leaves)

    def forward(self, leaves: torch.Tensor, return_attention: bool = False):
        if tuple(leaves.shape) != (self.code_num, self.embedding_dim):
            raise ValueError("Leaf embeddings have an incompatible shape.")
        candidates = self._candidates(leaves)
        expanded_leaves = leaves.unsqueeze(1).expand_as(candidates)
        if self.aggregation == "attention":
            hidden = torch.tanh(self.attention_hidden(torch.cat((expanded_leaves, candidates), dim=-1)))
            scores = self.attention_projection(hidden).squeeze(-1)
            scores = scores.masked_fill(~self.ancestor_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
        elif self.aggregation == "uniform":
            weights = self.ancestor_mask.to(dtype=leaves.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        else:
            weights = torch.zeros_like(self.ancestor_mask, dtype=leaves.dtype)
            weights[:, 0] = 1.0
        aggregate = torch.sum(weights.unsqueeze(-1) * candidates, dim=1)
        output = aggregate if self.writeback == "replace" else leaves + self.rho * (aggregate - leaves)
        return (output, weights, aggregate) if return_attention else output
