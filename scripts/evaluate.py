import argparse
from pathlib import Path

import torch

from marc.data import HeartFailureDataset, load_adjacency
from marc.models import MARC
from marc.training.engine import evaluate, load_checkpoint

from .common import load_run_config, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a validation-selected MARC checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output-dir", required=True, help="Private output directory outside the repository.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, training, _ = load_run_config(args.config)
    device = select_device(args.device)
    adjacency = load_adjacency(args.data_root, device)
    dataset = HeartFailureDataset(args.data_root, args.split, training.batch_size, False, device)
    model = MARC(dataset.code_x.shape[-1], adjacency, config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    details = evaluate(model, dataset, torch.nn.BCELoss())
    write_json(Path(args.output_dir) / (args.split + "_metrics.json"), {"metrics": details["metrics"]})
    print(details["metrics"])


if __name__ == "__main__":
    main()
