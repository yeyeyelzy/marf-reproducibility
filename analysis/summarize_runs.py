"""Summarize private per-run JSON metrics as mean and sample standard deviation."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


METRICS = ("auroc", "auprc", "f1", "brier_score")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Private test_metrics.json files.")
    parser.add_argument("--output", required=True, help="Private JSON output outside this repository.")
    args = parser.parse_args()
    records: List[Dict[str, float]] = []
    for location in args.inputs:
        with Path(location).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        record = payload.get("metrics", payload)
        records.append({name: float(record[name]) for name in METRICS})
    summary = {
        name: {
            "mean": float(np.mean([record[name] for record in records])),
            "sample_std": float(np.std([record[name] for record in records], ddof=1)),
        }
        for name in METRICS
    }
    summary["run_count"] = len(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
