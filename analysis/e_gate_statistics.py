"""Aggregate E gate statistics; raw patient--visit--edge values remain local and untracked."""

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
    parser.add_argument("--reservoir-size", type=int, default=200000)
    args = parser.parse_args()
    if args.reservoir_size < 1:
        raise ValueError("reservoir-size must be positive.")
    config, training, _ = load_run_config(args.config)
    apply_private_overrides(config, args.ontology_path)
    if not config.use_edge_calibration:
        raise ValueError("E gate statistics require use_edge_calibration=True.")
    device = select_device(args.device)
    data = HeartFailureDataset(args.data_root, "valid", training.batch_size, False, device, training.time_weight_mode)
    adjacency = load_adjacency(args.data_root, device)
    model = MARC(data.code_x.shape[-1], adjacency, config).to(device)
    load_checkpoint(model, args.checkpoint, device, config, training)
    model.eval()
    count, total, total_square = 0, 0.0, 0.0
    gamma_min, gamma_max, max_weight_error = float("inf"), float("-inf"), 0.0
    support_violations = 0
    reservoir = np.empty(args.reservoir_size, dtype=np.float64)
    random = np.random.RandomState(0)
    with torch.no_grad():
        for batch_index in range(len(data)):
            code_x, lens, divided, _, neighbors, weights, gaps = data[batch_index]
            _, _, internal = model(code_x, divided, neighbors, lens, weights, gaps, return_internal=True)
            for patient in internal["patients"]:
                for gate in patient["gates"]:
                    values = gate["gamma"].detach().cpu().numpy().astype(np.float64, copy=False)
                    source, target = gate["source"], gate["target"]
                    base = adjacency[source, target]
                    dynamic = gate["weight"]
                    max_weight_error = max(
                        max_weight_error,
                        float(torch.max(torch.abs(dynamic - base * gate["gamma"])).item()),
                    )
                    support_violations += int(source.numel() != int(torch.count_nonzero(adjacency).item()))
                    gamma_min = min(gamma_min, float(values.min()))
                    gamma_max = max(gamma_max, float(values.max()))
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
    row = {
        "edge_visit_count": count,
        "gamma_mean": total / max(count, 1),
        "gamma_std": variance ** 0.5,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "support_violations": support_violations,
        "max_abs_A_tilde_minus_A_gamma": max_weight_error,
    }
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
