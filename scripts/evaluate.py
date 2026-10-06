import argparse
from pathlib import Path

import numpy as np
import torch

from marc.data import HeartFailureDataset, load_adjacency
from marc.models import MARC
from marc.training.engine import evaluate, load_checkpoint

from .common import apply_private_overrides, load_run_config, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a validation-selected MARC checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ontology-path", default=None, help="Private CCS hierarchy NPZ; overrides the YAML placeholder.")
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output-dir", required=True, help="Private output directory outside the repository.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, training, _ = load_run_config(args.config)
    apply_private_overrides(config, args.ontology_path)
    device = select_device(args.device)
    adjacency = load_adjacency(args.data_root, device)
    dataset = HeartFailureDataset(args.data_root, args.split, training.batch_size, False, device, training.time_weight_mode)
    model = MARC(dataset.code_x.shape[-1], adjacency, config).to(device)
    load_checkpoint(model, args.checkpoint, device, config, training)
    details = evaluate(model, dataset, torch.nn.BCELoss(), return_embeddings=True)
    write_json(Path(args.output_dir) / (args.split + "_metrics.json"), {"metrics": details["metrics"]})
    predictions = (details["probabilities"] > 0.5).astype("int64")
    np.savez_compressed(
        str(Path(args.output_dir) / (args.split + "_predictions.npz")),
        patient_index=details["patient_index"],
        label=details["labels"],
        probability=details["probabilities"],
        prediction=predictions,
        embedding_z=details["embeddings"],
    )
    print(details["metrics"])


if __name__ == "__main__":
    main()
