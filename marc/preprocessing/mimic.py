"""Private MIMIC-III/MIMIC-IV preprocessing used by the MARC experiments.

The module reads only user-authorized tables and writes every patient-level
artifact outside the source repository.  Dataset-specific behavior is selected
explicitly through ``dataset``; graph construction and tensor serialization are
shared so that both cohorts follow one auditable implementation path.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .graph import build_cooccurrence_graph, divide_transition_groups, generate_neighbors


DATASET_DEFAULTS = {
    "mimic3": {"sample_num": None, "train_num": 6000, "test_num": 1000},
    "mimic4": {"sample_num": 10000, "train_num": 8000, "test_num": 1000},
}


def standard_icd9(code: str) -> str:
    code = str(code).strip()
    if not code:
        return ""
    position = 4 if code.startswith("E") else 3
    return code[:position] + ("." + code[position:] if len(code) > position else "")


def _read_csv(root: Path, names: Iterable[str], columns=None) -> pd.DataFrame:
    for name in names:
        for suffix in (".csv.gz", ".csv"):
            candidate = root / (name + suffix)
            if candidate.is_file():
                return pd.read_csv(candidate, usecols=columns, dtype=str)
    raise FileNotFoundError("Cannot find any of {} under {}".format(list(names), root))


def _load_gem(path: Path) -> Dict[str, str]:
    """Read a two-column CSV or CDC fixed-width ICD-10-CM GEM file."""
    try:
        table = pd.read_csv(path, dtype=str)
    except (pd.errors.ParserError, UnicodeDecodeError):
        table = None
    if table is not None:
        columns = {name.upper(): name for name in table.columns}
        if "ICD10" in columns and "ICD9" in columns:
            return {
                str(row[columns["ICD10"]]).replace(".", "").strip().upper():
                str(row[columns["ICD9"]]).strip()
                for _, row in table.iterrows()
                if str(row[columns["ICD10"]]).strip()
            }
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="ascii") as source:
        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if line:
                icd10, icd9 = line[0:7].strip(), line[8:13].strip()
                if icd10 and icd9:
                    mapping[icd10.replace(".", "").upper()] = icd9
    if not mapping:
        raise ValueError("GEM must be an ICD10/ICD9 CSV or CDC fixed-width file.")
    return mapping


def _map_icd(code: str, version: int, gem: Mapping[str, str]) -> str:
    if int(version) == 9:
        return standard_icd9(code)
    normalized = str(code).replace(".", "").strip().upper()
    mapped = gem.get(normalized, gem.get(normalized + "1", ""))
    return "" if mapped == "NoDx" else standard_icd9(mapped)


def _load_mimic3_visits(mimic_root: str) -> OrderedDict:
    root = Path(mimic_root)
    admissions = _read_csv(
        root, ["ADMISSIONS", "admissions"], ["SUBJECT_ID", "HADM_ID", "ADMITTIME"]
    )
    diagnoses = _read_csv(
        root, ["DIAGNOSES_ICD", "diagnoses_icd"],
        ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    )
    admissions["SUBJECT_ID"] = admissions["SUBJECT_ID"].astype(np.int64)
    admissions["HADM_ID"] = admissions["HADM_ID"].astype(np.int64)
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="raise")
    diagnoses["HADM_ID"] = diagnoses["HADM_ID"].astype(np.int64)
    diagnoses["ICD9_CODE"] = diagnoses["ICD9_CODE"].fillna("").map(standard_icd9)
    diagnoses = diagnoses[diagnoses["ICD9_CODE"] != ""]
    by_admission = OrderedDict()
    for admission_id, group in diagnoses.groupby("HADM_ID", sort=False):
        by_admission[int(admission_id)] = list(dict.fromkeys(group["ICD9_CODE"].tolist()))
    patients = OrderedDict()
    for patient_id, group in admissions.groupby("SUBJECT_ID", sort=False):
        if len(group) < 2:
            continue
        visits = []
        for _, admission in group.sort_values("ADMITTIME").iterrows():
            codes = by_admission.get(int(admission["HADM_ID"]))
            if not codes:
                visits = []
                break
            visits.append((admission["ADMITTIME"], codes))
        if len(visits) >= 2:
            patients[int(patient_id)] = visits
    return patients


def _load_mimic4_visits(mimic_root: str, gem_path: str) -> OrderedDict:
    root = Path(mimic_root)
    table_root = root / "hosp" if (root / "hosp").is_dir() else root
    patients = _read_csv(table_root, ["patients"])
    admissions = _read_csv(table_root, ["admissions"])
    diagnoses = _read_csv(table_root, ["diagnoses_icd"])
    gem = _load_gem(Path(gem_path))
    patients["anchor_year_group_start"] = (
        patients["anchor_year_group"].astype(str).str.slice(0, 4).astype(int)
    )
    patients["anchor_year"] = patients["anchor_year"].astype(int)
    year_offset = dict(
        zip(patients["subject_id"].astype(int), patients["anchor_year"] - patients["anchor_year_group_start"])
    )
    admissions["subject_id"] = admissions["subject_id"].astype(int)
    admissions["hadm_id"] = admissions["hadm_id"].astype(int)
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="raise")
    admissions = admissions[
        admissions.apply(
            lambda row: row["admittime"].year - year_offset[int(row["subject_id"])] > 2012,
            axis=1,
        )
    ]
    diagnoses["hadm_id"] = diagnoses["hadm_id"].astype(int)
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(admissions["hadm_id"])]
    codes = diagnoses["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    is_icd10 = diagnoses["icd_version"].astype(int) == 10
    mapped = codes.map(gem)
    missing = mapped.isna() & is_icd10
    mapped.loc[missing] = (codes.loc[missing] + "1").map(gem)
    mapped = mapped.fillna("").replace("NoDx", "")
    codes.loc[is_icd10] = mapped.loc[is_icd10]
    diagnoses = diagnoses.copy()
    diagnoses["normalized_code"] = codes.map(standard_icd9)
    diagnoses = diagnoses[diagnoses["normalized_code"] != ""]
    by_admission = OrderedDict(
        (int(admission_id), group["normalized_code"].tolist())
        for admission_id, group in diagnoses.groupby("hadm_id", sort=False)
    )
    grouped = OrderedDict()
    # Preserve the first patient occurrence in the admissions table, then sort
    # only that patient's visits.  This is the ordering used by EHRParser.
    for patient_id, group in admissions.groupby("subject_id", sort=False):
        visits = []
        for _, admission in group.sort_values("admittime").iterrows():
            # Keep diagnosis-row order here: code IDs are assigned on first
            # occurrence.  Pair counting and tensors deduplicate downstream.
            codes = by_admission.get(int(admission["hadm_id"]), [])
            if not codes:
                visits = []
                break
            visits.append((admission["admittime"], codes))
        if len(visits) >= 2:
            grouped[int(patient_id)] = visits
    return grouped


def load_patient_visits(dataset: str, mimic_root: str, gem_path: Optional[str] = None) -> OrderedDict:
    dataset = dataset.lower()
    if dataset == "mimic3":
        return _load_mimic3_visits(mimic_root)
    if dataset == "mimic4":
        if not gem_path:
            raise ValueError("MIMIC-IV preprocessing requires --gem-path.")
        return _load_mimic4_visits(mimic_root, gem_path)
    raise ValueError("dataset must be 'mimic3' or 'mimic4'.")


def _encode(patients: OrderedDict) -> Dict[str, int]:
    code_map: Dict[str, int] = {}
    for visits in patients.values():
        for _, codes in visits:
            for code in codes:
                code_map.setdefault(code, len(code_map))
    return code_map


def _required_training_patients(patients: OrderedDict, code_map: Mapping[str, int]) -> set:
    """Mirror the formal TCEM per-code scan and insertion order."""
    required = set()
    for code in code_map:
        for patient_id, visits in patients.items():
            if any(code in codes for _, codes in visits):
                required.add(patient_id)
                break
    longest = max(patients, key=lambda patient_id: len(patients[patient_id]))
    required.add(longest)
    return required


def split_patients(
    patients: OrderedDict,
    code_map: Mapping[str, int],
    train_num: int,
    test_num: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the original TCEM split rule used by the formal server run.

    The source implementation used Python sets before shuffling.  Consequently,
    an already reported split should be preserved with a private split manifest;
    rerunning this function under another Python build can change membership.
    """
    identifiers = list(patients.keys())
    if train_num + test_num >= len(identifiers):
        raise ValueError("Requested train/test sizes leave no validation patients.")
    required = _required_training_patients(patients, code_map)
    if len(required) > train_num:
        raise ValueError("Vocabulary coverage exceeds the requested training size.")
    remaining = np.asarray(list(set(identifiers).difference(required)))
    random = np.random.RandomState(seed)
    random.shuffle(remaining)
    valid_num = len(identifiers) - train_num - test_num
    added = set(remaining[: train_num - len(required)].tolist())
    train = np.asarray(list(required.union(added)))
    start = train_num - len(required)
    valid = remaining[start : start + valid_num]
    test = remaining[start + valid_num :]
    return train, valid, test


def _load_split_manifest(path: str, patients: Mapping[int, object]):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    arrays = tuple(np.asarray(payload[name], dtype=np.int64) for name in ("train", "valid", "test"))
    flattened = np.concatenate(arrays)
    if len(np.unique(flattened)) != len(flattened):
        raise ValueError("Split manifest contains duplicate patient IDs.")
    if set(flattened.tolist()) != set(patients):
        raise ValueError("Split manifest does not exactly cover the selected cohort.")
    return arrays


def build_time_weight(code_x: np.ndarray, lens: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    weights = np.zeros((code_x.shape[0], code_x.shape[2], 1), dtype=np.float64)
    for patient in range(code_x.shape[0]):
        length = int(lens[patient])
        days = timestamps[patient, :length] - timestamps[patient, 0]
        scale = np.exp(days / max(days.max(), 1))
        scale = scale / max(scale.max(), 1.0)
        active_visits, active_codes = np.where(code_x[patient, :length] > 0)
        for code in np.unique(active_codes):
            weights[patient, code, 0] = scale[active_visits[active_codes == code]].sum()
        maximum = weights[patient].max()
        if maximum > 0:
            weights[patient] /= maximum
    return weights


def _build_split(patient_ids, patients, code_map, max_visits, adjacency):
    patient_ids = list(patient_ids)
    code_x = np.zeros((len(patient_ids), max_visits, len(code_map)), dtype=bool)
    code_y = np.zeros((len(patient_ids), len(code_map)), dtype=bool)
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
    return (
        code_x, code_y, labels, lens, divided, neighbors,
        build_time_weight(code_x, lens, timestamps), timestamps,
    )


def save_sparse(path: Path, array: np.ndarray) -> None:
    indices = np.where(array > 0)
    np.savez_compressed(
        str(path), idx=indices, values=array[indices],
        shape=np.asarray(array.shape, dtype=np.int64),
    )


def build_private_assets(
    dataset: str,
    mimic_root: str,
    destination: str,
    gem_path: Optional[str] = None,
    seed: int = 6669,
    sample_num: Optional[int] = None,
    train_num: Optional[int] = None,
    test_num: Optional[int] = None,
    threshold: float = 0.01,
    split_manifest: Optional[str] = None,
) -> None:
    """Build private tensors and the formal-scope training-patient graph."""
    dataset = dataset.lower()
    if dataset not in DATASET_DEFAULTS:
        raise ValueError("dataset must be 'mimic3' or 'mimic4'.")
    defaults = DATASET_DEFAULTS[dataset]
    sample_num = defaults["sample_num"] if sample_num is None else sample_num
    train_num = defaults["train_num"] if train_num is None else train_num
    test_num = defaults["test_num"] if test_num is None else test_num
    patients = load_patient_visits(dataset, mimic_root, gem_path)
    if sample_num is not None:
        if len(patients) < sample_num:
            raise ValueError("Eligible cohort is smaller than sample_num.")
        random = np.random.RandomState(seed)
        selected_ids = random.choice(np.asarray(list(patients.keys())), sample_num, replace=False)
        patients = OrderedDict((int(identifier), patients[int(identifier)]) for identifier in selected_ids)
    code_map = _encode(patients)
    if split_manifest:
        train_ids, valid_ids, test_ids = _load_split_manifest(split_manifest, patients)
    else:
        train_ids, valid_ids, test_ids = split_patients(
            patients, code_map, int(train_num), int(test_num), seed
        )
    train_visits = (
        [code_map[code] for code in codes]
        for patient_id in train_ids
        for _, codes in patients[int(patient_id)]
    )
    adjacency = build_cooccurrence_graph(train_visits, len(code_map), threshold)
    output = Path(destination)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Private output directory is not empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    save_sparse(output / "code_adj.npz", adjacency)
    with (output / "code_map.json").open("w", encoding="utf-8") as handle:
        json.dump(code_map, handle, indent=2, sort_keys=True)
    with (output / "split_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"train": train_ids.tolist(), "valid": valid_ids.tolist(), "test": test_ids.tolist()},
            handle,
        )
    max_visits = max(len(visits) for visits in patients.values())
    for name, identifiers in (("train", train_ids), ("valid", valid_ids), ("test", test_ids)):
        split = output / name
        split.mkdir(exist_ok=True)
        arrays = _build_split(identifiers, patients, code_map, max_visits, adjacency)
        for filename, array in zip(
            ("code_x.npz", "code_y.npz", None, None, "divided.npz", "neighbors.npz", "time_weight.npz", None),
            arrays,
        ):
            if filename:
                save_sparse(split / filename, array)
        np.savez_compressed(str(split / "hf_y.npz"), hf_y=arrays[2])
        np.savez_compressed(str(split / "visit_lens.npz"), lens=arrays[3])
        np.savez_compressed(str(split / "code_x_timestamp.npz"), time=arrays[7])
