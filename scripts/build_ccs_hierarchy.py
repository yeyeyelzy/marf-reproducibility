import argparse
import json
from pathlib import Path

from marc.preprocessing.ccs import build_hierarchy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private CCS hierarchy NPZ from public CCS and a local code map.")
    parser.add_argument("--code-map", required=True)
    parser.add_argument("--ccs-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoding", default="latin-1")
    args = parser.parse_args()
    audit = build_hierarchy(args.code_map, args.ccs_csv, args.output, args.encoding)
    output = Path(args.output)
    audit_path = output.with_name(output.stem + "_audit.json")
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    print(audit)


if __name__ == "__main__":
    main()
