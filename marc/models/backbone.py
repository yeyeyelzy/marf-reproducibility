from typing import Optional

import torch
from torch import nn

from .attention import SingleHeadAttention


class CodeEmbeddingLayer(nn.Module):
    """Separate centre and neighbour leaf embeddings, plus unrelated-code embeddings."""

    def __init__(self, code_num: int, code_size: int, graph_size: int):
        super().__init__()
        self.c_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.u_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adjacency: torch.Tensor, code_size: int, graph_size: int):
        super().__init__()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix.")
        self.register_buffer("adjacency", adjacency)
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    @staticmethod
    def _matmul(adjacency: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adjacency, embeddings) if adjacency.is_sparse else adjacency @ embeddings

    def forward(
        self,
        code_x: torch.Tensor,
        neighbor: torch.Tensor,
        c_embeddings: torch.Tensor,
        n_embeddings: torch.Tensor,
        time_weight: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ):
        graph = self.adjacency if adjacency is None else adjacency
        centre_codes = code_x.unsqueeze(-1)
        neighbour_codes = neighbor.unsqueeze(-1)
        centre = centre_codes * c_embeddings
        neighbour = neighbour_codes * n_embeddings
        centre_propagated = self._matmul(graph, centre * time_weight)
        neighbour_propagated = self._matmul(graph, neighbour * time_weight)
        centre_output = self.activation(
            self.dense(centre + centre_codes * centre_propagated + centre_codes * neighbour_propagated)
        )
        neighbour_output = self.activation(
            self.dense(neighbour + neighbour_codes * neighbour_propagated + neighbour_codes * centre_propagated)
        )
        return centre_output, neighbour_output


class TransitionLayer(nn.Module):
    def __init__(self, code_num: int, graph_size: int, hidden_size: int, attention_size: int):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(graph_size, hidden_size)
        self.attention = SingleHeadAttention(graph_size, graph_size, hidden_size, attention_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        time_index: int,
        centre_embeddings: torch.Tensor,
        divided: torch.Tensor,
        previous_neighbour_embeddings: Optional[torch.Tensor],
        unrelated_embeddings: torch.Tensor,
        hidden_state: Optional[torch.Tensor],
    ):
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
        m1_index, m2_index, m3_index = (torch.where(mask > 0)[0] for mask in (m1, m2, m3))
        new_hidden = centre_embeddings.new_zeros((self.code_num, self.hidden_size))
        output_m1 = centre_embeddings.new_zeros((self.hidden_size,))
        output_m23 = centre_embeddings.new_zeros((self.hidden_size,))
        if m1_index.numel():
            h = None if hidden_state is None else hidden_state[m1_index]
            h_m1 = self.gru(centre_embeddings[m1_index], h)
            new_hidden[m1_index] = h_m1
            output_m1 = h_m1.max(dim=-2).values
        if time_index > 0 and m2_index.numel() + m3_index.numel():
            if previous_neighbour_embeddings is None:
                raise RuntimeError("Previous neighbour embeddings are required after the first visit.")
            query = torch.vstack((previous_neighbour_embeddings[m2_index], unrelated_embeddings[m3_index]))
            value = torch.vstack((centre_embeddings[m2_index], centre_embeddings[m3_index]))
            h_m23 = self.activation(self.attention(query, query, value))
            new_hidden[m2_index] = h_m23[: m2_index.numel()]
            new_hidden[m3_index] = h_m23[m2_index.numel() :]
            output_m23 = h_m23.max(dim=-2).values
        if not m1_index.numel():
            output = output_m23
        elif not (m2_index.numel() + m3_index.numel()):
            output = output_m1
        else:
            output = torch.vstack((output_m1, output_m23)).max(dim=-2).values
        return output, new_hidden
