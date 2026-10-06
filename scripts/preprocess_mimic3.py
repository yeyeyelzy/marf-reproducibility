import argparse

from marc.preprocessing.mimic3 import build_private_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build private MIMIC-III MARC assets from authorized raw tables."
    )
    parser.add_argument(
        "--mimic-root",
        required=True,
        help="Private directory containing ADMISSIONS and DIAGNOSES_ICD CSV files.",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Private output directory outside the source repository.",
    )
    parser.add_argument("--seed", type=int, default=6669)
    parser.add_argument("--train-num", type=int, default=6000)
    parser.add_argument("--test-num", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    build_private_assets(
        args.mimic_root,
        args.destination,
        seed=args.seed,
        train_num=args.train_num,
        test_num=args.test_num,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
