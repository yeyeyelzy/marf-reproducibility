"""Validation-set representation geometry analysis for the R mechanism."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from marc.data import HeartFailureDataset, load_adjacency
from marc.models import MARC
from marc.models.relation import relation_loss
from marc.training.engine import evaluate, load_checkpoint
from scripts.common import load_run_config, select_device


def mean_pair_similarity(embeddings: np.ndarray, labels: np.ndarray):
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), np.finfo(np.float64).eps)
    values = np.maximum(normalized @ normalized.T, 0.0)
    upper = np.triu_indices(labels.size, k=1)
    same = labels[upper[0]] == labels[upper[1]]
    same_values, different_values = values[upper][same], values[upper][~same]
    if not same_values.size or not different_values.size:
        raise ValueError("Geometry analysis requires at least one same-class and one cross-class patient pair.")
    return float(same_values.mean()), float(different_values.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Private CSV output outside the repository.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, training, _ = load_run_config(args.config)
    device = select_device(args.device)
    adjacency = load_adjacency(args.data_root, device)
    data = HeartFailureDataset(args.data_root, "valid", training.batch_size, False, device)
    model = MARC(data.code_x.shape[-1], adjacency, config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    result = evaluate(model, data, torch.nn.BCELoss(), return_embeddings=True)
    same, different = mean_pair_similarity(result["embeddings"], result["labels"])
    embedding = torch.from_numpy(result["embeddings"]).to(device)
    probabilities = torch.from_numpy(result["probabilities"]).to(device)
    labels = torch.from_numpy(result["labels"]).to(device)
    relation_mse, _ = relation_loss(embedding, probabilities, labels, config.relation_detach_prediction, config.relation_target)
    row = {"same_class_similarity": same, "different_class_similarity": different, "geometry_gap": same - different, "relation_mse": float(relation_mse.item())}
    row.update(result["metrics"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
