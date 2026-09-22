"""Build the private CCS hierarchy asset from public CCS and a local code map."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def normalize_icd9(value: str) -> str:
    return "".join(str(value).strip().strip("'\"").upper().replace("D_", "").replace(".", "").split())


def _load_ccs(path: Path, encoding: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with path.open("r", encoding=encoding, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            code = normalize_icd9(row[0])
            if not code or "ICD" in code and "CODE" in code:
                continue
            descriptions = [str(row[index]).strip().strip("'\"") for index in (2, 4, 6, 8) if index < len(row) and str(row[index]).strip()]
            if descriptions:
                mapping[code] = descriptions
    if not mapping:
        raise ValueError("No CCS diagnoses could be read from {}".format(path))
    return mapping


def build_hierarchy(code_map_path: str, ccs_csv: str, output: str, encoding: str = "latin-1") -> Dict[str, object]:
    with Path(code_map_path).open("r", encoding="utf-8") as handle:
        code_map = json.load(handle)
    ordered: List[Tuple[str, int]] = sorted(((str(code), int(index)) for code, index in code_map.items()), key=lambda item: item[1])
    if [index for _, index in ordered] != list(range(len(ordered))):
        raise ValueError("code_map IDs must be contiguous from zero.")
    hierarchy = _load_ccs(Path(ccs_csv), encoding)
    root_id, next_id = len(ordered) + 1, len(ordered) + 2
    concepts = {"__root__": root_id}
    chains = [[0]]
    matched = 0
    for code, index in ordered:
        chain = [index + 1, root_id]
        descriptions = hierarchy.get(normalize_icd9(code))
        if descriptions:
            matched += 1
            for description in descriptions:
                concepts.setdefault(description, next_id)
                if concepts[description] == next_id:
                    next_id += 1
                chain.append(concepts[description])
        chains.append(chain)
    depth = max(map(len, chains))
    ids = np.zeros((len(ordered) + 1, depth), dtype=np.int64)
    mask = np.zeros_like(ids, dtype=bool)
    for row, chain in enumerate(chains):
        ids[row, : len(chain)] = chain
        mask[row, : len(chain)] = True
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), ancestor_ids=ids, ancestor_mask=mask)
    return {"vocabulary_size": len(ordered), "matched_codes": matched, "coverage": matched / float(len(ordered)), "maximum_depth": depth}
