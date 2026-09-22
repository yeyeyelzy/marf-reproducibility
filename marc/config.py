"""Configuration for the paper's MARC/TCEM model family."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class MARCConfig:
    """Only parameters used by MARC, its factorial ablation, and mechanism controls."""

    code_size: int = 48
    graph_size: int = 32
    hidden_size: int = 150
    temporal_attention_size: int = 32
    output_size: int = 1
    dropout_rate: float = 0.0

    gap_bins: int = 16
    gap_embed_size: int = 32
    state_size: int = 64

    use_hierarchy: bool = True
    ontology_path: Optional[str] = None
    hierarchy_attention_size: int = 128
    hierarchy_residual_init: float = 0.10
    hierarchy_aggregation: Literal["attention", "uniform", "self_only"] = "attention"
    hierarchy_writeback: Literal["residual", "replace"] = "residual"

    use_edge_calibration: bool = True
    edge_use_state: bool = True

    use_relation_calibration: bool = True
    relation_loss_weight: float = 1.0
    relation_detach_prediction: bool = False
    relation_target: Literal["prediction_x_label", "label_only", "prediction_only"] = "prediction_x_label"

    @property
    def ontology_file(self) -> Optional[Path]:
        return None if self.ontology_path is None else Path(self.ontology_path)

    def validate(self) -> None:
        if self.code_size < 1 or self.graph_size < 1 or self.hidden_size < 1:
            raise ValueError("Embedding dimensions must be positive.")
        if self.gap_bins < 1 or self.gap_embed_size < 1 or self.state_size < 1:
            raise ValueError("Gap and state dimensions must be positive.")
        if not 0.0 < self.hierarchy_residual_init < 1.0:
            raise ValueError("hierarchy_residual_init must be in (0, 1).")
        if self.hierarchy_aggregation not in {"attention", "uniform", "self_only"}:
            raise ValueError("Unsupported hierarchy aggregation mode.")
        if self.hierarchy_writeback not in {"residual", "replace"}:
            raise ValueError("Unsupported hierarchy write-back mode.")
        if self.relation_loss_weight < 0.0:
            raise ValueError("relation_loss_weight must be non-negative.")
        if self.relation_target not in {"prediction_x_label", "label_only", "prediction_only"}:
            raise ValueError("Unsupported relation target mode.")
        if self.use_hierarchy and not self.ontology_path:
            raise ValueError("ontology_path is required when use_hierarchy=True.")
