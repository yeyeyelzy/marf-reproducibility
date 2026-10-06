"""MIMIC-III preprocessing compatible with the formal TCEM/MARC runs.

Only user-authorized raw tables are read.  Every generated patient tensor,
code map, split, and graph remains private and must be written outside this
source repository.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .graph import build_cooccurrence_graph
from .mimic4 import _build_split, _encode, save_sparse, split_patients, standard_icd9


def _read_table(root: Path, name: str, columns: List[str]) -> pd.DataFrame:
    for suffix in (".csv", ".csv.gz"):
        candidate = root / (name + suffix)
        if candidate.is_file():
            return pd.read_csv(candidate, usecols=columns, dtype=str)
    raise FileNotFoundError("Cannot find {}(.csv|.csv.gz) under {}".format(name, root))


def load_patient_visits(mimic_root: str) -> OrderedDict:
    """Parse eligible MIMIC-III admissions and normalized ICD-9 codes.

    The retained protocol keeps patients with at least two admissions and
    removes a patient if any retained admission has no diagnosis row.
    Diagnosis order is preserved while duplicates within one admission are
    removed.
    """
    root = Path(mimic_root)
    admissions = _read_table(
        root,
        "ADMISSIONS",
        ["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
    )
    diagnoses = _read_table(
        root,
        "DIAGNOSES_ICD",
        ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    )
    admissions["SUBJECT_ID"] = admissions["SUBJECT_ID"].astype(np.int64)
    admissions["HADM_ID"] = admissions["HADM_ID"].astype(np.int64)
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="raise")
    diagnoses["SUBJECT_ID"] = diagnoses["SUBJECT_ID"].astype(np.int64)
    diagnoses["HADM_ID"] = diagnoses["HADM_ID"].astype(np.int64)
    diagnoses["ICD9_CODE"] = diagnoses["ICD9_CODE"].fillna("").map(standard_icd9)
    diagnoses = diagnoses[diagnoses["ICD9_CODE"] != ""]

    diagnosis_by_admission: Dict[int, List[str]] = OrderedDict()
    for admission_id, group in diagnoses.groupby("HADM_ID", sort=False):
        diagnosis_by_admission[int(admission_id)] = list(
            dict.fromkeys(group["ICD9_CODE"].tolist())
        )

    patients: OrderedDict = OrderedDict()
    for patient_id, group in admissions.groupby("SUBJECT_ID", sort=False):
        if len(group) < 2:
            continue
        visits = []
        for _, admission in group.sort_values("ADMITTIME").iterrows():
            admission_id = int(admission["HADM_ID"])
            codes = diagnosis_by_admission.get(admission_id)
            if not codes:
                visits = []
                break
            visits.append((admission["ADMITTIME"], codes))
        if len(visits) >= 2:
            patients[int(patient_id)] = visits
    return patients


def build_private_assets(
    mimic_root: str,
    destination: str,
    seed: int = 6669,
    train_num: int = 6000,
    test_num: int = 1000,
    threshold: float = 0.01,
) -> None:
    """Create private MIMIC-III assets with the formal preprocessing scope.

    The vocabulary follows the retained TCEM protocol and is defined over the
    eligible cohort.  The co-occurrence graph uses every admission belonging
    to training-split patients, including their held-out target admission;
    validation and test patients never contribute to graph counts.  Patient
    ``code_x`` tensors still contain historical admissions only.
    """
    patients = load_patient_visits(mimic_root)
    if train_num + test_num >= len(patients):
        raise ValueError("train_num + test_num must leave at least one validation patient.")
    code_map = _encode(patients)
    train_ids, valid_ids, test_ids = split_patients(
        patients, code_map, train_num, test_num, seed
    )
    train_visits = (
        [code_map[code] for code in codes]
        for patient_id in train_ids
        for _, codes in patients[int(patient_id)]
    )
    adjacency = build_cooccurrence_graph(
        train_visits, len(code_map), threshold=threshold
    )

    output = Path(destination)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Private output directory is not empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    save_sparse(output / "code_adj.npz", adjacency)
    with (output / "code_map.json").open("w", encoding="utf-8") as handle:
        json.dump(code_map, handle, indent=2, sort_keys=True)

    max_visits = max(len(visits) for visits in patients.values())
    for name, identifiers in (
        ("train", train_ids),
        ("valid", valid_ids),
        ("test", test_ids),
    ):
        split = output / name
        split.mkdir(exist_ok=True)
        code_x, code_y, labels, lens, divided, neighbors, weights, timestamps = _build_split(
            identifiers, patients, code_map, max_visits, adjacency
        )
        save_sparse(split / "code_x.npz", code_x)
        save_sparse(split / "code_y.npz", code_y)
        save_sparse(split / "divided.npz", divided)
        save_sparse(split / "neighbors.npz", neighbors)
        save_sparse(split / "time_weight.npz", weights)
        np.savez_compressed(str(split / "visit_lens.npz"), lens=lens)
        np.savez_compressed(str(split / "hf_y.npz"), hf_y=labels)
        np.savez_compressed(str(split / "code_x_timestamp.npz"), time=timestamps)
