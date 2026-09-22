"""Class-aware patient-relation calibration used only during training."""

import torch
import torch.nn.functional as functional


def _upper_relu_cosine(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("features must retain a batch dimension.")
    normalized = functional.normalize(features.reshape(features.shape[0], -1), p=2, dim=1)
    matrix = functional.relu(normalized @ normalized.T)
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1, device=features.device)
    return matrix[indices[0], indices[1]]


def relation_loss(
    embeddings: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    detach_prediction: bool = False,
    target: str = "prediction_x_label",
):
    """Mean-squared relation loss over unordered, non-self patient pairs."""
    probabilities = probabilities.reshape(-1)
    labels = labels.reshape(-1)
    if embeddings.shape[0] != probabilities.numel() or probabilities.numel() != labels.numel():
        raise ValueError("embeddings, probabilities, and labels must share batch size.")
    feature_edges = _upper_relu_cosine(embeddings)
    epsilon = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp(epsilon, 1.0 - epsilon)
    prediction_edges = _upper_relu_cosine(torch.stack((1.0 - probabilities, probabilities), dim=1))
    match = (labels[:, None] == labels[None, :]).to(feature_edges.dtype)
    indices = torch.triu_indices(labels.numel(), labels.numel(), offset=1, device=labels.device)
    label_edges = match[indices[0], indices[1]]
    if target == "prediction_x_label":
        target_edges = prediction_edges * label_edges
    elif target == "label_only":
        target_edges = label_edges
    elif target == "prediction_only":
        target_edges = prediction_edges
    else:
        raise ValueError("Unsupported relation target: {}".format(target))
    if detach_prediction:
        target_edges = target_edges.detach()
    loss = embeddings.sum() * 0.0 if not feature_edges.numel() else torch.mean((feature_edges - target_edges) ** 2)
    return loss, {
        "pair_count": int(feature_edges.numel()),
        "same_label_pair_count": int(label_edges.detach().sum().item()),
        "raw_loss": float(loss.detach().item()),
    }
