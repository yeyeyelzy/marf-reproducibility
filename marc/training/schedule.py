from typing import List


def paper_learning_rate(epoch_index: int) -> float:
    """Zero-based epoch schedule: 0.01, 0.001, 0.0001 (3--19), then 0.00001."""
    if epoch_index == 0:
        return 0.01
    if epoch_index == 1:
        return 0.001
    if epoch_index < 19:
        return 0.0001
    return 0.00001


def learning_rate_sequence(max_epochs: int) -> List[float]:
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive.")
    return [paper_learning_rate(epoch) for epoch in range(max_epochs)]
