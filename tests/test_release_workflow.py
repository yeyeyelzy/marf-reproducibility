import sys

import numpy as np
import pytest
import torch

from analysis.r_geometry import mean_pair_similarity
from marc.data.dataset import load_sparse
from marc.preprocessing.mimic4 import _load_gem, save_sparse
from scripts.evaluate import main as evaluate_main
from scripts.train import main as train_main


def _write_split(root, name, labels):
    split = root / name
    split.mkdir()
    code_x = np.asarray(
        [
            [[1, 1, 0], [0, 1, 1]],
            [[0, 1, 1], [1, 0, 1]],
            [[1, 0, 1], [1, 1, 0]],
            [[1, 1, 0], [1, 0, 1]],
        ],
        dtype=bool,
    )
    divided = np.zeros(code_x.shape + (3,), dtype=bool)
    divided[:, :, :, 0] = code_x
    save_sparse(split / "code_x.npz", code_x)
    save_sparse(split / "divided.npz", divided)
    save_sparse(split / "neighbors.npz", np.zeros_like(code_x, dtype=bool))
    save_sparse(split / "time_weight.npz", np.ones((4, 3, 1), dtype=np.float32))
    np.savez_compressed(split / "visit_lens.npz", lens=np.asarray([2, 2, 2, 2], dtype=np.int64))
    np.savez_compressed(split / "hf_y.npz", hf_y=np.asarray(labels, dtype=np.float32))


def _checkpoint(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def test_train_and_evaluate_cli_are_seed_reproducible(tmp_path, monkeypatch):
    data_root = tmp_path / "private_assets"
    data_root.mkdir()
    adjacency = np.asarray([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=np.float32)
    save_sparse(data_root / "code_adj.npz", adjacency)
    for split in ("train", "valid", "test"):
        _write_split(data_root, split, labels=[0, 1, 0, 1])
    config = tmp_path / "config.yaml"
    config.write_text(
        """model:
  code_size: 4
  graph_size: 3
  hidden_size: 5
  temporal_attention_size: 3
  gap_embed_size: 2
  state_size: 3
  use_hierarchy: false
  use_relation_calibration: false
training:
  max_epochs: 2
  batch_size: 2
  patience: 1
  seed: 41
""",
        encoding="utf-8",
    )
    run_one, run_two = tmp_path / "run_one", tmp_path / "run_two"
    for output in (run_one, run_two):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--config",
                str(config),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
            ],
        )
        train_main()
    first, second = _checkpoint(run_one / "best_valid_auc.pt"), _checkpoint(run_two / "best_valid_auc.pt")
    assert first["best_epoch"] == second["best_epoch"]
    for name, tensor in first["model_state_dict"].items():
        assert torch.equal(tensor, second["model_state_dict"][name])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(run_one / "best_valid_auc.pt"),
            "--split",
            "test",
            "--output-dir",
            str(run_one),
            "--device",
            "cpu",
        ],
    )
    evaluate_main()
    assert (run_one / "test_metrics.json").is_file()


def test_sparse_loader_rejects_invalid_indices(tmp_path):
    path = tmp_path / "invalid_sparse.npz"
    np.savez_compressed(
        path,
        idx=np.asarray([[0, 3], [0, 1]], dtype=np.int64),
        values=np.asarray([1.0, 1.0], dtype=np.float32),
        shape=np.asarray([3, 3], dtype=np.int64),
    )
    try:
        load_sparse(path)
    except ValueError as error:
        assert "out-of-bounds" in str(error)
    else:
        raise AssertionError("Invalid sparse coordinates must be rejected.")


def test_gem_loader_accepts_csv_and_cdc_fixed_width_formats(tmp_path):
    csv_path = tmp_path / "gem.csv"
    csv_path.write_text("ICD10,ICD9\nA000,001.0\n", encoding="ascii")
    fixed_width_path = tmp_path / "2018_I10gem.txt"
    fixed_width_path.write_text("A000    00100  0\n", encoding="ascii")
    assert _load_gem(csv_path) == {"A000": "001.0"}
    assert _load_gem(fixed_width_path) == {"A000": "00100"}


def test_geometry_requires_both_pair_types():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="same-class and one cross-class"):
        mean_pair_similarity(embeddings, np.asarray([0, 1], dtype=np.int64))
