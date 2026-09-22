import torch
from torch import nn


class VisitGapEncoder(nn.Module):
    def __init__(self, bins: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(bins, embedding_size)

    def forward(self, gaps: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(gaps.long(), 0, self.embedding.num_embeddings - 1)
        return self.embedding(clipped)


class PatientStateUpdater(nn.Module):
    """Causal GRUCell update over current and preceding visits only."""

    def __init__(self, visit_size: int, state_size: int, gap_size: int):
        super().__init__()
        self.gru = nn.GRUCell(visit_size + gap_size, state_size)

    def forward(self, visit_representations: torch.Tensor, gap_representations: torch.Tensor) -> torch.Tensor:
        state = visit_representations.new_zeros((1, self.gru.hidden_size))
        states = []
        for visit, gap in zip(visit_representations, gap_representations):
            state = self.gru(torch.cat((visit, gap)).unsqueeze(0), state)
            states.append(state.squeeze(0))
        return torch.stack(states)
