"""Retained MARC prediction path: H -> base encoder -> E -> conditional encoder -> R loss."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ..config import MARCConfig
from .attention import TemporalAttention
from .backbone import CodeEmbeddingLayer, GraphLayer, TransitionLayer
from .edge import EdgeCalibration
from .hierarchy import HierarchyCalibration
from .temporal import PatientStateUpdater, VisitGapEncoder


class MARC(nn.Module):
    def __init__(self, code_num: int, adjacency: torch.Tensor, config: MARCConfig):
        super().__init__()
        config.validate()
        self.code_num = int(code_num)
        self.config = config
        self.embedding_layer = CodeEmbeddingLayer(code_num, config.code_size, config.graph_size)
        self.graph_layer = GraphLayer(adjacency, config.code_size, config.graph_size)
        self.transition_layer = TransitionLayer(
            code_num, config.graph_size, config.hidden_size, config.temporal_attention_size
        )
        self.temporal_attention = TemporalAttention(config.hidden_size, config.temporal_attention_size)
        self.classifier = nn.Sequential(nn.Dropout(config.dropout_rate), nn.Linear(config.hidden_size, config.output_size), nn.Sigmoid())
        self.gap_encoder = VisitGapEncoder(config.gap_bins, config.gap_embed_size)
        self.state_updater = PatientStateUpdater(config.hidden_size, config.state_size, config.gap_embed_size)
        self.edge_calibration = (
            EdgeCalibration(config.graph_size, config.state_size, config.graph_size, use_state=config.edge_use_state)
            if config.use_edge_calibration
            else None
        )
        self.hierarchy = (
            HierarchyCalibration(
                code_num=code_num,
                embedding_dim=config.code_size,
                ontology_path=config.ontology_path,
                attention_size=config.hierarchy_attention_size,
                residual_init=config.hierarchy_residual_init,
                aggregation=config.hierarchy_aggregation,
                writeback=config.hierarchy_writeback,
            )
            if config.use_hierarchy
            else None
        )

    def _transition_pass(
        self,
        code_x: torch.Tensor,
        divided: torch.Tensor,
        neighbors: torch.Tensor,
        length: int,
        time_weight: torch.Tensor,
        c_embeddings: torch.Tensor,
        n_embeddings: torch.Tensor,
        u_embeddings: torch.Tensor,
        conditioned_adjacencies: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        outputs, centres, neighbours = [], [], []
        previous_neighbour, hidden = None, None
        for time_index in range(length):
            adjacency = None if conditioned_adjacencies is None else conditioned_adjacencies[time_index]
            centre, neighbour = self.graph_layer(
                code_x[time_index], neighbors[time_index], c_embeddings, n_embeddings, time_weight, adjacency
            )
            output, hidden = self.transition_layer(
                time_index, centre, divided[time_index], previous_neighbour, u_embeddings, hidden
            )
            outputs.append(output)
            centres.append(centre)
            neighbours.append(neighbour)
            previous_neighbour = neighbour
        return outputs, centres, neighbours

    def _encode_one(
        self,
        code_x: torch.Tensor,
        divided: torch.Tensor,
        neighbors: torch.Tensor,
        length: int,
        time_weight: torch.Tensor,
        visit_gaps: torch.Tensor,
        c_embeddings: torch.Tensor,
        n_embeddings: torch.Tensor,
        u_embeddings: torch.Tensor,
        return_internal: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        base_outputs, base_centres, _ = self._transition_pass(
            code_x, divided, neighbors, length, time_weight, c_embeddings, n_embeddings, u_embeddings
        )
        base_visits = torch.stack(base_outputs)
        states = self.state_updater(base_visits, self.gap_encoder(visit_gaps[:length]))
        gates: List[Dict[str, torch.Tensor]] = []
        if self.edge_calibration is None:
            conditional_outputs = base_outputs
        else:
            conditioned = []
            for visit_index in range(length):
                if return_internal:
                    dynamic, source, target, gamma = self.edge_calibration(
                        self.graph_layer.adjacency, base_centres[visit_index], states[visit_index], return_gates=True
                    )
                    gates.append({"visit_index": visit_index, "source": source, "target": target, "gamma": gamma, "weight": dynamic.values()})
                else:
                    dynamic = self.edge_calibration(self.graph_layer.adjacency, base_centres[visit_index], states[visit_index])
                conditioned.append(dynamic)
            conditional_outputs, _, _ = self._transition_pass(
                code_x, divided, neighbors, length, time_weight, c_embeddings, n_embeddings, u_embeddings, conditioned
            )
        embedding = self.temporal_attention(torch.stack(conditional_outputs))
        internal = {"states": states, "gates": gates, "base_visits": base_visits}
        return embedding, internal

    def encode_patients(
        self,
        code_x: torch.Tensor,
        divided: torch.Tensor,
        neighbors: torch.Tensor,
        visit_lens: torch.Tensor,
        time_weight: torch.Tensor,
        visit_gaps: Optional[torch.Tensor] = None,
        return_internal: bool = False,
    ):
        c_embeddings, n_embeddings, u_embeddings = self.embedding_layer()
        hierarchy_info = None
        if self.hierarchy is not None:
            if return_internal:
                c_embeddings, c_alpha, _ = self.hierarchy(c_embeddings, return_attention=True)
                n_embeddings, n_alpha, _ = self.hierarchy(n_embeddings, return_attention=True)
                hierarchy_info = {"center_attention": c_alpha, "neighbor_attention": n_alpha, "rho": self.hierarchy.rho}
            else:
                c_embeddings, n_embeddings = self.hierarchy(c_embeddings), self.hierarchy(n_embeddings)
        if visit_gaps is None:
            visit_gaps = torch.zeros(code_x.shape[:2], dtype=torch.long, device=code_x.device)
        embeddings, internals = [], []
        for patient_index in range(code_x.shape[0]):
            length = int(visit_lens[patient_index].item())
            if length < 1:
                raise ValueError("Every patient must have at least one historical visit.")
            embedding, internal = self._encode_one(
                code_x[patient_index, :length],
                divided[patient_index, :length],
                neighbors[patient_index, :length],
                length,
                time_weight[patient_index],
                visit_gaps[patient_index],
                c_embeddings,
                n_embeddings,
                u_embeddings,
                return_internal,
            )
            embeddings.append(embedding)
            if return_internal:
                internals.append(internal)
        result = torch.stack(embeddings)
        return (result, {"patients": internals, "hierarchy": hierarchy_info}) if return_internal else result

    def forward(
        self,
        code_x: torch.Tensor,
        divided: torch.Tensor,
        neighbors: torch.Tensor,
        visit_lens: torch.Tensor,
        time_weight: torch.Tensor,
        visit_gaps: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
        return_internal: bool = False,
    ):
        if return_embedding and return_internal:
            raise ValueError("Choose at most one of return_embedding and return_internal.")
        encoded = self.encode_patients(
            code_x, divided, neighbors, visit_lens, time_weight, visit_gaps, return_internal=return_internal
        )
        if return_internal:
            embeddings, internals = encoded
            return self.classifier(embeddings), embeddings, internals
        embeddings = encoded
        probabilities = self.classifier(embeddings)
        return (probabilities, embeddings) if return_embedding else probabilities
