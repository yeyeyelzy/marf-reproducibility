import numpy as np

from marc.data.dataset import load_sparse
from marc.preprocessing.graph import build_cooccurrence_graph, divide_transition_groups, generate_neighbors
from marc.preprocessing.mimic import build_private_assets, load_patient_visits


def test_graph_support_and_row_normalization():
    adjacency = build_cooccurrence_graph([[0, 1, 1], [1, 2], [0, 2]], code_num=3, threshold=0.01)
    assert np.all(np.diag(adjacency) == 0.0)
    active = adjacency.sum(axis=1) > 0
    assert np.allclose(adjacency.sum(axis=1)[active], 1.0)
    assert np.array_equal(adjacency > 0, (adjacency > 0).T)


def test_neighbors_and_transition_groups():
    code_x = np.asarray([[[1, 1, 0], [0, 1, 1]]], dtype=bool)
    adjacency = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    lengths = np.asarray([2], dtype=np.int64)
    neighbors = generate_neighbors(code_x, lengths, adjacency)
    divided = divide_transition_groups(code_x, neighbors, lengths)
    assert neighbors.shape == code_x.shape
    assert divided.shape == code_x.shape + (3,)
    assert divided[0, 0, 0, 0]


def test_mimic3_builder_matches_formal_target_visit_graph_scope(tmp_path):
    raw = tmp_path / "authorized_raw"
    output = tmp_path / "private_assets"
    raw.mkdir()
    admission_lines = ["SUBJECT_ID,HADM_ID,ADMITTIME"]
    diagnosis_lines = ["SUBJECT_ID,HADM_ID,ICD9_CODE"]
    for patient_id in range(1, 6):
        first_admission, target_admission = patient_id * 10, patient_id * 10 + 1
        admission_lines.extend(
            [
                "{},{},2020-01-01 00:00:00".format(patient_id, first_admission),
                "{},{},2020-01-03 00:00:00".format(patient_id, target_admission),
            ]
        )
        diagnosis_lines.extend(
            [
                "{},{},25000".format(patient_id, first_admission),
                "{},{},4019".format(patient_id, first_admission),
                "{},{},4019".format(patient_id, target_admission),
            ]
        )
        if patient_id == 1:
            diagnosis_lines.append("{},{},4280".format(patient_id, target_admission))
    (raw / "ADMISSIONS.csv").write_text("\n".join(admission_lines) + "\n", encoding="ascii")
    (raw / "DIAGNOSES_ICD.csv").write_text("\n".join(diagnosis_lines) + "\n", encoding="ascii")

    patients = load_patient_visits("mimic3", str(raw))
    assert list(patients) == [1, 2, 3, 4, 5]
    assert patients[1][0][1] == ["250.00", "401.9"]
    build_private_assets(
        "mimic3",
        str(raw),
        str(output),
        seed=6669,
        train_num=3,
        test_num=1,
    )

    import json

    code_map = json.loads((output / "code_map.json").read_text(encoding="utf-8"))
    adjacency = load_sparse(output / "code_adj.npz")
    assert adjacency[code_map["428.0"], code_map["401.9"]] > 0
    train_x = load_sparse(output / "train" / "code_x.npz")
    train_y = load_sparse(output / "train" / "code_y.npz")
    with np.load(output / "train" / "visit_lens.npz", allow_pickle=False) as payload:
        visit_lens = payload["lens"]
    assert train_x.shape == (3, 2, 3)
    assert train_y.shape == (3, 3)
    assert visit_lens.max() == 1
    assert not np.any(train_x[:, 1])


def test_mimic4_builder_uses_formal_sampling_split_and_graph_scope(tmp_path):
    raw = tmp_path / "authorized_mimic4" / "hosp"
    output = tmp_path / "private_assets"
    raw.mkdir(parents=True)
    patient_lines = ["subject_id,anchor_year,anchor_year_group"]
    admission_lines = ["subject_id,hadm_id,admittime"]
    diagnosis_lines = ["subject_id,hadm_id,seq_num,icd_code,icd_version"]
    for patient_id in range(1, 6):
        first_admission, target_admission = patient_id * 10, patient_id * 10 + 1
        patient_lines.append("{},2015,2010 - 2012".format(patient_id))
        admission_lines.extend(
            [
                "{},{},2020-01-01 00:00:00".format(patient_id, first_admission),
                "{},{},2020-01-03 00:00:00".format(patient_id, target_admission),
            ]
        )
        diagnosis_lines.extend(
            [
                "{},{},1,I10,10".format(patient_id, first_admission),
                "{},{},1,4019,9".format(patient_id, target_admission),
            ]
        )
        if patient_id == 1:
            diagnosis_lines.append("{},{},2,4280,9".format(patient_id, target_admission))
    (raw / "patients.csv").write_text("\n".join(patient_lines) + "\n", encoding="ascii")
    (raw / "admissions.csv").write_text("\n".join(admission_lines) + "\n", encoding="ascii")
    (raw / "diagnoses_icd.csv").write_text("\n".join(diagnosis_lines) + "\n", encoding="ascii")
    gem = tmp_path / "gem.csv"
    gem.write_text("ICD10,ICD9\nI10,4019\n", encoding="ascii")

    build_private_assets(
        "mimic4",
        str(raw.parent),
        str(output),
        gem_path=str(gem),
        sample_num=5,
        train_num=3,
        test_num=1,
    )
    assert (output / "split_ids.json").is_file()
    adjacency = load_sparse(output / "code_adj.npz")
    assert adjacency.shape == (2, 2)
    assert np.allclose(adjacency.sum(axis=1), 1.0)
