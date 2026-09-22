import math

import torch
from torch import nn


class SingleHeadAttention(nn.Module):
    """The transition-layer attention used by the retained TCEM backbone."""

    def __init__(self, query_size: int, key_size: int, value_size: int, attention_size: int):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.dense_q(query)
        k = self.dense_k(key)
        v = self.dense_v(value)
        scores = torch.matmul(q, k.T) / math.sqrt(self.attention_size)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * v, dim=-2)


class TemporalAttention(nn.Module):
    def __init__(self, value_size: int, attention_size: int):
        super().__init__()
        self.context = nn.Parameter(nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, visits: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.dense(visits), self.context).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(visits * weights.unsqueeze(-1), dim=-2)
