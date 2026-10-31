"""Situation entity models and candidate causal hypotheses."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CausalHypothesis:
    """Hypothesis representing a candidate explanation for observed congestion."""

    hypothesis_type: str
    target_cell: int
    time_window: tuple[float, float]
    evidence_list: list[dict[str, Any]]
    vehicle_hours_recovered: float
    ci_lower: float
    ci_upper: float
    confidence_score: float
    recommended_action: str
    responsible_party: str
    is_hypothesis_label: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_type": self.hypothesis_type,
            "target_cell": self.target_cell,
            "time_window": list(self.time_window),
            "evidence_list": self.evidence_list,
            "vehicle_hours_recovered": round(self.vehicle_hours_recovered, 2),
            "ci_lower": round(self.ci_lower, 2),
            "ci_upper": round(self.ci_upper, 2),
            "confidence_score": round(self.confidence_score, 2),
            "recommended_action": self.recommended_action,
            "responsible_party": self.responsible_party,
            "is_hypothesis_label": self.is_hypothesis_label,
        }


@dataclass
class Situation:
    """Civic Situation entity aggregating evidence, clusters, and ranked causal hypotheses."""

    situation_id: str
    corridor_name: str
    observations_used: list[dict[str, Any]]
    jam_clusters: list[dict[str, Any]]
    ranked_hypotheses: list[CausalHypothesis]
    status: str
    simulated: bool
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "situation_id": self.situation_id,
            "corridor_name": self.corridor_name,
            "observation_count": len(self.observations_used),
            "jam_cluster_count": len(self.jam_clusters),
            "ranked_hypotheses": [h.to_dict() for h in self.ranked_hypotheses],
            "top_hypothesis": self.ranked_hypotheses[0].to_dict() if self.ranked_hypotheses else None,
            "status": self.status,
            "simulated": self.simulated,
            "config_hash": self.config_hash,
        }
