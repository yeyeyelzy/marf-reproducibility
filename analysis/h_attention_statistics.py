"""Export aggregate H attention and residual statistics without patient-level data."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from marc.data import HeartFailureDataset, load_adjacency
from marc.models import MARC
from marc.training.engine import load_checkpoint
from scripts.common import apply_private_overrides, load_run_config, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ontology-path", default=None)
    parser.add_argument("--output", required=True, help="Private CSV output outside the repository.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, training, _ = load_run_config(args.config)
    apply_private_overrides(config, args.ontology_path)
    if not config.use_hierarchy:
        raise ValueError("H statistics require use_hierarchy=True.")
    device = select_device(args.device)
    data = HeartFailureDataset(args.data_root, "valid", training.batch_size, False, device, training.time_weight_mode)
    model = MARC(data.code_x.shape[-1], load_adjacency(args.data_root, device), config).to(device)
    load_checkpoint(model, args.checkpoint, device, config, training)
    model.eval()
    with torch.no_grad():
        center, neighbor, _ = model.embedding_layer()
        _, center_weights, _ = model.hierarchy(center, return_attention=True)
        _, neighbor_weights, _ = model.hierarchy(neighbor, return_attention=True)
    row = {"rho": float(model.hierarchy.rho.detach().item())}
    for role, tensor in (("center", center_weights), ("neighbor", neighbor_weights)):
        weights = tensor.detach().cpu().numpy()
        self_mass = weights[:, 0]
        entropy = -(weights * np.log(np.maximum(weights, np.finfo(np.float64).eps))).sum(axis=1)
        row[role + "_mean_self_attention"] = float(self_mass.mean())
        row[role + "_mean_ancestor_attention"] = float((1.0 - self_mass).mean())
        row[role + "_mean_attention_entropy"] = float(entropy.mean())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
