import numpy as np
import torch

from marc.config import MARCConfig
from marc.models.edge import EdgeCalibration
from marc.models.marc import MARC
from marc.models.relation import relation_loss


def make_hierarchy(path, code_num):
    # Zero is a padding row.  Each code has itself plus one shared ancestor.
    ids = np.zeros((code_num + 1, 2), dtype=np.int64)
    mask = np.zeros((code_num + 1, 2), dtype=bool)
    mask[0, 0] = True
    for code in range(code_num):
        ids[code + 1] = [code + 1, code_num + 1]
        mask[code + 1] = [True, True]
    np.savez_compressed(path, ancestor_ids=ids, ancestor_mask=mask)


def make_batch(code_num=4):
    code_x = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 1, 0, 1]],
            [[0, 1, 1, 0], [1, 0, 0, 1]],
            [[1, 0, 1, 0], [1, 1, 0, 0]],
        ],
        dtype=torch.float32,
    )
    divided = torch.zeros((3, 2, code_num, 3), dtype=torch.float32)
    divided[:, 0, :, 0] = code_x[:, 0]
    divided[:, 1, :, 0] = code_x[:, 1]
    neighbors = torch.zeros_like(code_x)
    adjacency = torch.tensor(
        [[0.0, 0.6, 0.4, 0.0], [0.2, 0.0, 0.0, 0.8], [0.3, 0.0, 0.0, 0.7], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    lens = torch.tensor([2, 2, 2], dtype=torch.long)
    time_weight = torch.ones((3, code_num, 1), dtype=torch.float32)
    gaps = torch.tensor([[0, 3], [0, 4], [0, 2]], dtype=torch.long)
    return code_x, divided, neighbors, lens, time_weight, gaps, adjacency


def test_full_marc_forward_and_backward(tmp_path):
    torch.manual_seed(0)
    hierarchy = tmp_path / "hierarchy.npz"
    make_hierarchy(hierarchy, code_num=4)
    config = MARCConfig(
        code_size=6,
        graph_size=5,
        hidden_size=7,
        temporal_attention_size=4,
        gap_embed_size=3,
        state_size=5,
        ontology_path=str(hierarchy),
    )
    code_x, divided, neighbors, lens, weight, gaps, adjacency = make_batch()
    model = MARC(4, adjacency, config)
    probabilities, embeddings, internal = model(
        code_x, divided, neighbors, lens, weight, gaps, return_internal=True
    )
    assert probabilities.shape == (3, 1)
    assert embeddings.shape == (3, 7)
    assert len(internal["patients"]) == 3
    assert 0.0 < internal["hierarchy"]["rho"].item() < 1.0
    loss = probabilities.sum() + embeddings.square().mean()
    loss.backward()
    assert model.edge_calibration.score[0].weight.grad is not None


def test_edge_calibration_preserves_support_and_bounds():
    torch.manual_seed(1)
    adjacency = torch.tensor([[0.0, 0.5, 0.0], [0.25, 0.0, 0.75], [0.0, 0.0, 0.0]])
    calibrator = EdgeCalibration(node_size=4, state_size=3, hidden_size=5, use_state=True)
    dynamic, source, target, gates = calibrator(adjacency, torch.randn(3, 4), torch.randn(3), return_gates=True)
    dense = dynamic.to_dense()
    assert torch.equal(dense == 0.0, adjacency == 0.0)
    assert torch.all(dense[source, target] <= adjacency[source, target])
    assert torch.all(gates > 0.0)
    assert torch.all(gates < 1.0)


def test_relation_controls_use_unordered_pairs_only():
    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    probability = torch.tensor([0.2, 0.8, 0.3], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0])
    loss, details = relation_loss(embedding, probability, labels)
    assert details["pair_count"] == 3
    assert details["same_label_pair_count"] == 1
    loss.backward()
    assert probability.grad is not None
    assert embedding.grad is not None

