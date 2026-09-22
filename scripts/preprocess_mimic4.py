import argparse

from marc.preprocessing.mimic4 import build_private_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build private MIMIC-IV MARC assets from authorized raw tables.")
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--gem-csv", required=True)
    parser.add_argument("--destination", required=True, help="Private output directory outside the repository.")
    parser.add_argument("--seed", type=int, default=6669)
    args = parser.parse_args()
    build_private_assets(args.mimic_root, args.gem_csv, args.destination, args.seed)


if __name__ == "__main__":
    main()
