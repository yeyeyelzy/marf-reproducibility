import argparse

from marc.preprocessing.ccs import build_hierarchy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private CCS hierarchy NPZ from public CCS and a local code map.")
    parser.add_argument("--code-map", required=True)
    parser.add_argument("--ccs-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoding", default="latin-1")
    args = parser.parse_args()
    print(build_hierarchy(args.code_map, args.ccs_csv, args.output, args.encoding))


if __name__ == "__main__":
    main()
