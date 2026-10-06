import argparse
from dataclasses import asdict
from pathlib import Path

from marc.data import HeartFailureDataset, load_adjacency
from marc.models import MARC
from marc.training.engine import seed_everything, train

from .common import apply_private_overrides, load_run_config, private_asset_manifest, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one MARC/TCEM run on private local assets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, help="Private directory containing code_adj.npz and train/valid/test.")
    parser.add_argument("--output-dir", required=True, help="Private result directory outside the source repository.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ontology-path", default=None, help="Private CCS hierarchy NPZ; overrides the YAML placeholder.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config, training, raw = load_run_config(args.config)
    apply_private_overrides(config, args.ontology_path)
    if args.seed is not None:
        training.seed = args.seed
    # Match the formal implementation: seed once before model construction.
    seed_everything(training.seed)
    device = select_device(args.device)
    adjacency = load_adjacency(args.data_root, device)
    train_data = HeartFailureDataset(args.data_root, "train", training.batch_size, True, device, training.time_weight_mode)
    valid_data = HeartFailureDataset(args.data_root, "valid", training.batch_size, False, device, training.time_weight_mode)
    model = MARC(train_data.code_x.shape[-1], adjacency, config).to(device)
    output = Path(args.output_dir)
    write_json(
        output / "run_config.json",
        {
            "model": asdict(config),
            "training": asdict(training),
            "source_config": raw,
            "private_asset_manifest": private_asset_manifest(args.data_root, config),
        },
    )
    result = train(model, train_data, valid_data, config, training, str(output))
    print("best_epoch={} best_valid_auroc={:.6f}".format(result["best_epoch"], result["best_valid_auroc"]))


if __name__ == "__main__":
    main()
