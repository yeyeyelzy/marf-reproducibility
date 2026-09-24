"""Aggregate E gate statistics; raw patient--visit--edge values remain local and untracked."""

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
    parser.add_argument("--reservoir-size", type=int, default=200000)
    args = parser.parse_args()
    if args.reservoir_size < 1:
        raise ValueError("reservoir-size must be positive.")
    config, training, _ = load_run_config(args.config)
    if not config.use_edge_calibration:
        raise ValueError("E gate statistics require use_edge_calibration=True.")
    device = select_device(args.device)
    data = HeartFailureDataset(args.data_root, "valid", training.batch_size, False, device)
    model = MARC(data.code_x.shape[-1], load_adjacency(args.data_root, device), config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    count, total, total_square = 0, 0.0, 0.0
    reservoir = np.empty(args.reservoir_size, dtype=np.float64)
    random = np.random.RandomState(0)
    with torch.no_grad():
        for batch_index in range(len(data)):
            code_x, lens, divided, _, neighbors, weights, gaps = data[batch_index]
            _, _, internal = model(code_x, divided, neighbors, lens, weights, gaps, return_internal=True)
            for patient in internal["patients"]:
                for gate in patient["gates"]:
                    values = gate["gamma"].detach().cpu().numpy().astype(np.float64, copy=False)
                    total += values.sum()
                    total_square += np.square(values).sum()
                    for value in values:
                        if count < reservoir.size:
                            reservoir[count] = value
                        else:
                            replacement = random.randint(0, count + 1)
                            if replacement < reservoir.size:
                                reservoir[replacement] = value
                        count += 1
    if count == 0:
        raise ValueError("The adjacency has no positive edges, so there are no E gates to summarize.")
    used = reservoir[: min(count, reservoir.size)]
    variance = max(total_square / max(count, 1) - (total / max(count, 1)) ** 2, 0.0)
    row = {"edge_visit_count": count, "gamma_mean": total / max(count, 1), "gamma_std": variance ** 0.5}
    for quantile, name in ((0.05, "p05"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")):
        row["gamma_" + name] = float(np.quantile(used, quantile))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
