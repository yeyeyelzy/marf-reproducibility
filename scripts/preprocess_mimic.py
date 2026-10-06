import argparse

from marc.preprocessing.mimic import build_private_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build private MIMIC-III or MIMIC-IV MARC assets."
    )
    parser.add_argument("--dataset", required=True, choices=("mimic3", "mimic4"))
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--destination", required=True, help="Private directory outside the repository.")
    parser.add_argument("--gem-path", help="Required for MIMIC-IV; CSV or CDC fixed-width GEM.")
    parser.add_argument("--seed", type=int, default=6669)
    parser.add_argument("--sample-num", type=int)
    parser.add_argument("--train-num", type=int)
    parser.add_argument("--test-num", type=int)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument(
        "--split-manifest",
        help="Optional private JSON with train/valid/test patient ID lists.",
    )
    args = parser.parse_args()
    build_private_assets(
        dataset=args.dataset,
        mimic_root=args.mimic_root,
        destination=args.destination,
        gem_path=args.gem_path,
        seed=args.seed,
        sample_num=args.sample_num,
        train_num=args.train_num,
        test_num=args.test_num,
        threshold=args.threshold,
        split_manifest=args.split_manifest,
    )


if __name__ == "__main__":
    main()
