import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import yaml

from .common import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the eight H/E/R factorial configurations.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    model, training, _ = load_run_config(args.base_config)
    model.validate()
    training.validate()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for h in (False, True):
        for e in (False, True):
            for r in (False, True):
                name = "a{}{}{}".format(int(h), int(e), int(r))
                current = deepcopy(model)
                current.use_hierarchy, current.use_edge_calibration, current.use_relation_calibration = h, e, r
                if not h:
                    current.ontology_path = None
                with (output / (name + ".yaml")).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump({"model": asdict(current), "training": asdict(training)}, handle, sort_keys=False)


if __name__ == "__main__":
    main()
