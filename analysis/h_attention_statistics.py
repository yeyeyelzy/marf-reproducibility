"""Export aggregate H attention and residual statistics without patient-level data."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from marc.data import HeartFailureDataset, load_adjacency
from marc.models import MARC
from marc.training.engine import load_checkpoint
from scripts.common import load_run_config, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Private CSV output outside the repository.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, training, _ = load_run_config(args.config)
    if not config.use_hierarchy:
        raise ValueError("H statistics require use_hierarchy=True.")
    device = select_device(args.device)
    data = HeartFailureDataset(args.data_root, "valid", training.batch_size, False, device)
    model = MARC(data.code_x.shape[-1], load_adjacency(args.data_root, device), config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    with torch.no_grad():
        leaves, _, _ = model.embedding_layer()
        _, weights, _ = model.hierarchy(leaves, return_attention=True)
    weights = weights.detach().cpu().numpy()
    self_mass = weights[:, 0]
    ancestor_mass = 1.0 - self_mass
    entropy = -(weights * np.log(np.maximum(weights, np.finfo(np.float64).eps))).sum(axis=1)
    row = {
        "rho": float(model.hierarchy.rho.detach().item()),
        "mean_self_attention": float(self_mass.mean()),
        "mean_ancestor_attention": float(ancestor_mass.mean()),
        "mean_attention_entropy": float(entropy.mean()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
