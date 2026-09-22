"""Check a candidate release tree for files that must remain private.

This is a conservative preflight aid, not a substitute for reviewing the
staged Git file list before publication.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List


BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".pickle", ".parquet",
    ".h5", ".hdf5", ".csv", ".tsv", ".zip",
}
BLOCKED_ROOT_DIRECTORIES = {"data", "assets", "results", "outputs", "checkpoints", "logs"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".sh"}
IDENTITY_MARKERS = ("github.com/yeyey", "lin ziy", "linz")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def candidate_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        if any(part == ".git" for part in path.parts):
            continue
        yield path


def find_issues(root: Path) -> List[str]:
    issues: List[str] = []
    for path in candidate_files(root):
        relative = path.relative_to(root)
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            issues.append("private-looking suffix: {}".format(relative))
        if relative.parts and relative.parts[0].lower() in BLOCKED_ROOT_DIRECTORIES:
            issues.append("private-looking directory: {}".format(relative))
        if relative.as_posix() == "scripts/check_release.py":
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                content = path.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                issues.append("non-UTF8 text candidate: {}".format(relative))
                continue
            markers = [marker for marker in IDENTITY_MARKERS if marker in content]
            if markers:
                issues.append("identity marker {} in {}".format(", ".join(markers), relative))
            if EMAIL_PATTERN.search(content):
                issues.append("email-like text in {}".format(relative))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    issues = find_issues(root)
    if issues:
        print("Release check found {} issue(s):".format(len(issues)))
        for issue in issues:
            print("- {}".format(issue))
        raise SystemExit(1)
    print("Release check passed: no blocked assets or simple identity markers found.")


if __name__ == "__main__":
    main()
