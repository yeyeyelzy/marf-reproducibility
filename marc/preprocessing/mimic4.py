"""MIMIC-IV v3.1 preprocessing for the paper's private local asset build.

This module reads user-authorized PhysioNet files only. It never uploads or
ships the resulting data, split IDs, graph, or patient-level artifacts.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .graph import build_cooccurrence_graph, divide_transition_groups, generate_neighbors


def standard_icd9(code: str) -> str:
    code = str(code).strip()
    if not code:
        return ""
    position = 4 if code.startswith("E") else 3
    return code[:position] + ("." + code[position:] if len(code) > position else "")


def _read_table(root: Path, name: str) -> pd.DataFrame:
    for suffix in (".csv.gz", ".csv"):
        candidate = root / "hosp" / (name + suffix)
        if candidate.is_file():
            return pd.read_csv(candidate)
    raise FileNotFoundError("Cannot find hosp/{}(.csv|.csv.gz) under {}".format(name, root))


def _load_gem(path: Path) -> Dict[str, str]:
    """Read either the standard two-column CSV or CDC's fixed-width GEM file."""
    try:
        table = pd.read_csv(path, dtype=str)
    except pd.errors.ParserError:
        table = None
    if table is not None:
        columns = {name.upper(): name for name in table.columns}
        if "ICD10" in columns and "ICD9" in columns:
            return {
                str(row[columns["ICD10"]]).strip(): str(row[columns["ICD9"]]).strip()
                for _, row in table.iterrows()
                if str(row[columns["ICD10"]]).strip()
            }
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="ascii") as source:
        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            icd10, icd9 = line[0:7].strip(), line[8:13].strip()
            if icd10 and icd9:
                mapping[icd10] = icd9
    if not mapping:
        raise ValueError("GEM input must be a CSV with ICD10/ICD9 columns or a CDC fixed-width I10 GEM file.")
    return mapping


def _map_icd(code: str, version: int, gem: Dict[str, str]) -> str:
    if int(version) == 9:
        return standard_icd9(code)
    mapped = gem.get(str(code), gem.get(str(code) + "1", ""))
    return "" if mapped == "NoDx" else standard_icd9(mapped)


def load_patient_visits(mimic_root: str, gem_csv: str) -> OrderedDict:
    root = Path(mimic_root)
    patients = _read_table(root, "patients")
    admissions = _read_table(root, "admissions")
    diagnoses = _read_table(root, "diagnoses_icd")
    gem = _load_gem(Path(gem_csv))
    patients["anchor_year_group_start"] = patients["anchor_year_group"].astype(str).str.slice(0, 4).astype(int)
    year_offset = dict(zip(patients["subject_id"], patients["anchor_year"] - patients["anchor_year_group_start"]))
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions = admissions[admissions.apply(lambda row: row["admittime"].year - year_offset[row["subject_id"]] > 2012, axis=1)]
    valid_admissions = set(admissions["hadm_id"].tolist())
    diagnosis_by_admission: Dict[int, List[str]] = {}
    for _, row in diagnoses.iterrows():
        hadm_id = row["hadm_id"]
        if hadm_id not in valid_admissions:
            continue
        code = _map_icd(row["icd_code"], row["icd_version"], gem)
        if code:
            diagnosis_by_admission.setdefault(hadm_id, []).append(code)
    grouped: OrderedDict = OrderedDict()
    for subject_id, group in admissions.sort_values("admittime").groupby("subject_id", sort=False):
        visits = []
        for _, admission in group.iterrows():
            codes = sorted(set(diagnosis_by_admission.get(admission["hadm_id"], [])))
            if not codes:
                visits = []
                break
            visits.append((admission["admittime"], codes))
        if len(visits) >= 2:
            grouped[int(subject_id)] = visits
    return grouped


def split_patients(patients: OrderedDict, code_map: Dict[str, int], train_num: int, test_num: int, seed: int):
    """Use the retained TCEM split rule: train covers the observed vocabulary."""
    identifiers = np.asarray(list(patients.keys()))
    if train_num + test_num >= identifiers.size:
        raise ValueError("Requested train/test sizes leave no validation patients.")
    maximum_visits, longest_patient = 0, None
    for patient_id, visits in patients.items():
        if len(visits) > maximum_visits:
            maximum_visits, longest_patient = len(visits), patient_id
    required = set()
    uncovered = set(code_map)
    for patient_id, visits in patients.items():
        patient_codes = {code for _, codes in visits for code in codes}
        newly_covered = uncovered.intersection(patient_codes)
        if newly_covered:
            required.add(patient_id)
            uncovered.difference_update(newly_covered)
        if not uncovered:
            break
    if uncovered:
        raise ValueError("Code map entries lack supporting patients: {}".format(len(uncovered)))
    required.add(longest_patient)
    if len(required) > train_num:
        raise ValueError("Vocabulary coverage requires more patients than the requested train split.")
    random = np.random.RandomState(seed)
    remaining = np.asarray(sorted(patient_id for patient_id in identifiers if patient_id not in required))
    random.shuffle(remaining)
    valid_num = identifiers.size - train_num - test_num
    train = np.asarray(sorted(required.union(set(remaining[: train_num - len(required)].tolist()))))
    valid_start = train_num - len(required)
    valid = remaining[valid_start : valid_start + valid_num]
    test = remaining[valid_start + valid_num :]
    return train, valid, test


def _encode(patients: OrderedDict):
    code_map: Dict[str, int] = {}
    for visits in patients.values():
        for _, codes in visits:
            for code in codes:
                code_map.setdefault(code, len(code_map))
    return code_map


def _build_split(patient_ids: Iterable[int], patients: OrderedDict, code_map: Dict[str, int], max_visits: int, adjacency: np.ndarray):
    patient_ids = list(patient_ids)
    code_num = len(code_map)
    code_x = np.zeros((len(patient_ids), max_visits, code_num), dtype=bool)
    code_y = np.zeros((len(patient_ids), code_num), dtype=bool)
    labels = np.zeros(len(patient_ids), dtype=np.int64)
    lens = np.zeros(len(patient_ids), dtype=np.int64)
    timestamps = np.zeros((len(patient_ids), max_visits), dtype=np.int64)
    for row, patient_id in enumerate(patient_ids):
        visits = patients[int(patient_id)]
        history, target = visits[:-1], visits[-1][1]
        lens[row] = len(history)
        for visit_index, (time, codes) in enumerate(history):
            code_x[row, visit_index, [code_map[code] for code in codes]] = True
            timestamps[row, visit_index] = int(time.toordinal())
        code_y[row, [code_map[code] for code in target]] = True
        labels[row] = any(code.startswith("428") for code in target)
    neighbors = generate_neighbors(code_x, lens, adjacency)
    divided = divide_transition_groups(code_x, neighbors, lens)
    time_weight = build_time_weight(code_x, lens, timestamps)
    return code_x, code_y, labels, lens, divided, neighbors, time_weight, timestamps


def build_time_weight(code_x: np.ndarray, lens: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    weights = np.zeros((code_x.shape[0], code_x.shape[2], 1), dtype=np.float64)
    for patient in range(code_x.shape[0]):
        length = int(lens[patient])
        days = timestamps[patient, :length] - timestamps[patient, 0]
        scale = days / max(days.max(), 1)
        scale = np.exp(scale)
        scale = scale / max(scale.max(), 1.0)
        active_visits, active_codes = np.where(code_x[patient, :length] > 0)
        for code in np.unique(active_codes):
            weights[patient, code, 0] = scale[active_visits[active_codes == code]].sum()
        maximum = weights[patient].max()
        if maximum > 0:
            weights[patient] /= maximum
    return weights


def save_sparse(path: Path, array: np.ndarray) -> None:
    indices = np.where(array > 0)
    np.savez_compressed(str(path), idx=indices, values=array[indices], shape=np.asarray(array.shape, dtype=np.int64))


def build_private_assets(mimic_root: str, gem_csv: str, destination: str, seed: int = 6669) -> None:
    """Create private MIMIC-IV assets; destination must be outside the Git repository."""
    patients = load_patient_visits(mimic_root, gem_csv)
    if len(patients) < 10000:
        raise ValueError("The formal MIMIC-IV protocol requires at least 10,000 eligible patients.")
    random = np.random.RandomState(seed)
    selected_ids = random.choice(np.asarray(list(patients.keys())), 10000, replace=False)
    selected = OrderedDict((int(identifier), patients[int(identifier)]) for identifier in selected_ids)
    code_map = _encode(selected)
    train_ids, valid_ids, test_ids = split_patients(selected, code_map, 8000, 1000, seed)
    train_visits = ([code_map[code] for code in codes] for patient in train_ids for _, codes in selected[int(patient)])
    adjacency = build_cooccurrence_graph(train_visits, len(code_map), threshold=0.01)
    output = Path(destination)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Private output directory is not empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    save_sparse(output / "code_adj.npz", adjacency)
    with (output / "code_map.json").open("w", encoding="utf-8") as handle:
        json.dump(code_map, handle, indent=2, sort_keys=True)
    # The retained TCEM preprocessing allocates the maximum total admission
    # count.  Since the last admission is the target, one trailing all-zero
    # history slot is possible and is ignored through ``visit_lens``.
    max_visits = max(len(visits) for visits in selected.values())
    for name, ids in (("train", train_ids), ("valid", valid_ids), ("test", test_ids)):
        split = output / name
        split.mkdir(exist_ok=True)
        code_x, code_y, labels, lens, divided, neighbors, weights, timestamps = _build_split(
            ids, selected, code_map, max_visits, adjacency
        )
        save_sparse(split / "code_x.npz", code_x)
        save_sparse(split / "code_y.npz", code_y)
        save_sparse(split / "divided.npz", divided)
        save_sparse(split / "neighbors.npz", neighbors)
        save_sparse(split / "time_weight.npz", weights)
        np.savez_compressed(str(split / "visit_lens.npz"), lens=lens)
        np.savez_compressed(str(split / "hf_y.npz"), hf_y=labels)
        np.savez_compressed(str(split / "code_x_timestamp.npz"), time=timestamps)
