"""Training/evaluation for the heart-failure task and paper protocol."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from ..config import MARCConfig
from ..models.relation import relation_loss
from .metrics import heart_failure_metrics
from .schedule import learning_rate_sequence


@dataclass
class TrainingConfig:
    max_epochs: int = 40
    batch_size: int = 32
    weight_decay: float = 0.0
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 123
    time_weight_mode: str = "split_specific"

    def validate(self) -> None:
        if self.max_epochs < 1 or self.batch_size < 1:
            raise ValueError("max_epochs and batch_size must be positive.")
        if self.weight_decay != 0.0:
            raise ValueError("The formal MARC protocol uses weight_decay=0.")
        if self.patience < 1 or self.min_delta < 0.0:
            raise ValueError("patience must be positive and min_delta non-negative.")
        if self.time_weight_mode != "split_specific":
            raise ValueError("The reported MARC experiments use split_specific time weights.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataset, loss_fn: nn.Module, return_embeddings: bool = False) -> Dict[str, object]:
    model.eval()
    total_loss, total_count = 0.0, 0
    labels, probabilities, embeddings, sample_indices = [], [], [], []
    for step in range(len(dataset)):
        code_x, lens, divided, y, neighbors, time_weight, gaps = dataset[step]
        if return_embeddings:
            output, batch_embeddings = model(code_x, divided, neighbors, lens, time_weight, gaps, return_embedding=True)
            embeddings.append(batch_embeddings.detach().cpu().numpy())
        else:
            output = model(code_x, divided, neighbors, lens, time_weight, gaps)
        probability = output.reshape(-1)
        loss = loss_fn(probability, y)
        total_loss += loss.item() * y.numel()
        total_count += y.numel()
        labels.append(y.detach().cpu().numpy())
        probabilities.append(probability.detach().cpu().numpy())
        start = step * dataset.batch_size
        end = min((step + 1) * dataset.batch_size, dataset.size)
        sample_indices.append(np.asarray(dataset.indices[start:end], dtype=np.int64))
    labels_np = np.concatenate(labels)
    probabilities_np = np.concatenate(probabilities)
    metrics = heart_failure_metrics(labels_np, probabilities_np)
    metrics["loss"] = float(total_loss / max(total_count, 1))
    result: Dict[str, object] = {
        "metrics": metrics,
        "labels": labels_np,
        "probabilities": probabilities_np,
        "patient_index": np.concatenate(sample_indices),
    }
    if return_embeddings:
        result["embeddings"] = np.vstack(embeddings)
    return result


def _checkpoint_payload(model, optimizer, epoch: int, metrics: Dict[str, float], best_epoch: int, config: MARCConfig, training: TrainingConfig):
    return {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_metrics": metrics,
        "marc_config": asdict(config),
        "training_config": asdict(training),
    }


def train(model, train_data, valid_data, config: MARCConfig, training: TrainingConfig, output_dir: str) -> Dict[str, object]:
    """Train one independent run; checkpoint selection uses validation AUROC only."""
    config.validate()
    training.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=training.weight_decay)
    loss_fn = nn.BCELoss()
    best_auc, best_epoch, stale_epochs = float("-inf"), 0, 0
    history = []
    for epoch, learning_rate in enumerate(learning_rate_sequence(training.max_epochs), start=1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        running_total, running_bce, running_relation, sample_count = 0.0, 0.0, 0.0, 0
        for step in range(len(train_data)):
            code_x, lens, divided, y, neighbors, time_weight, gaps = train_data[step]
            optimizer.zero_grad()
            if config.use_relation_calibration:
                probabilities, embeddings = model(code_x, divided, neighbors, lens, time_weight, gaps, return_embedding=True)
                probabilities = probabilities.reshape(-1)
                bce = loss_fn(probabilities, y)
                relation, _ = relation_loss(
                    embeddings,
                    probabilities,
                    y,
                    detach_prediction=config.relation_detach_prediction,
                    target=config.relation_target,
                )
                objective = bce + config.relation_loss_weight * relation
            else:
                probabilities = model(code_x, divided, neighbors, lens, time_weight, gaps).reshape(-1)
                bce = loss_fn(probabilities, y)
                relation, objective = bce.detach() * 0.0, bce
            objective.backward()
            optimizer.step()
            count = y.numel()
            running_total += objective.detach().item() * count
            running_bce += bce.detach().item() * count
            running_relation += relation.detach().item() * count
            sample_count += count
        train_data.on_epoch_end()
        validation = evaluate(model, valid_data, loss_fn)
        valid_metrics = validation["metrics"]
        improved = valid_metrics["auroc"] > best_auc + training.min_delta
        if improved:
            best_auc, best_epoch, stale_epochs = valid_metrics["auroc"], epoch, 0
            torch.save(
                _checkpoint_payload(model, optimizer, epoch, valid_metrics, best_epoch, config, training),
                str(output / "best_valid_auc.pt"),
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": running_total / max(sample_count, 1),
                "train_bce": running_bce / max(sample_count, 1),
                "train_relation": running_relation / max(sample_count, 1),
                "valid": valid_metrics,
                "best_epoch": best_epoch,
                "improved": improved,
            }
        )
        if stale_epochs >= training.patience:
            break
    with (output / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "best_valid_auroc": best_auc, "history": history}, handle, indent=2)
    return {"best_epoch": best_epoch, "best_valid_auroc": best_auc, "history": history}


def load_checkpoint(
    model,
    path: str,
    device: torch.device,
    expected_config: Optional[MARCConfig] = None,
    expected_training: Optional[TrainingConfig] = None,
) -> Dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if expected_config is not None:
        observed = dict(checkpoint.get("marc_config", {}))
        requested = asdict(expected_config)
        observed.pop("ontology_path", None)
        requested.pop("ontology_path", None)
        if observed != requested:
            raise ValueError("Checkpoint MARC configuration does not match the requested run configuration.")
    if expected_training is not None:
        observed_mode = dict(checkpoint.get("training_config", {})).get("time_weight_mode", "split_specific")
        if observed_mode != expected_training.time_weight_mode:
            raise ValueError("Checkpoint time-weight mode does not match the requested run configuration.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint
