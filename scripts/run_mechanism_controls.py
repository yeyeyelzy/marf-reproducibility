import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import yaml

from .common import load_run_config


def write(path: Path, model, training) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"model": asdict(model), "training": asdict(training)}, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the H/E/R mechanism-control configurations.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    model, training, _ = load_run_config(args.base_config)
    model.validate()
    training.validate()
    if not (model.use_hierarchy and model.use_edge_calibration and model.use_relation_calibration):
        raise ValueError("Mechanism controls must be generated from a full-MARC configuration with H, E, and R enabled.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    h_uniform = deepcopy(model); h_uniform.hierarchy_aggregation = "uniform"
    h_replace = deepcopy(model); h_replace.hierarchy_writeback = "replace"
    e_endpoint = deepcopy(model); e_endpoint.edge_use_state = False
    r_stopgrad = deepcopy(model); r_stopgrad.relation_detach_prediction = True
    r_label = deepcopy(model); r_label.relation_target = "label_only"
    r_nomask = deepcopy(model); r_nomask.relation_target = "prediction_only"
    for name, current in (("h_uniform", h_uniform), ("h_no_residual", h_replace), ("e_no_state", e_endpoint), ("r_stop_gradient", r_stopgrad), ("r_label_only", r_label), ("r_no_mask", r_nomask)):
        write(output / (name + ".yaml"), current, training)


if __name__ == "__main__":
    main()
