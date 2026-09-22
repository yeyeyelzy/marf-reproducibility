import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml

from marc.config import MARCConfig
from marc.training.engine import TrainingConfig


def load_yaml(path: str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML configuration must be a mapping.")
    return payload


def _construct(cls, values: Dict[str, object]):
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError("Unsupported {} fields: {}".format(cls.__name__, sorted(unknown)))
    return cls(**values)


def load_run_config(path: str) -> Tuple[MARCConfig, TrainingConfig, Dict[str, object]]:
    payload = load_yaml(path)
    model = _construct(MARCConfig, dict(payload.get("model", {})))
    training = _construct(TrainingConfig, dict(payload.get("training", {})))
    return model, training, payload


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
