"""Situation builder constructing schema-validated civic entities and ranking hypotheses."""

from typing import Any
import numpy as np

from citysense.config.schemas import DiagnosisConfig, NetworkConfig
from citysense.counterfactual.types import CounterfactualLadderResult
from citysense.diagnosis.capacity import CapacityFitResult
from citysense.diagnosis.shockwave import ShockwaveOrigin
from citysense.diagnosis.types import JamCluster
from citysense.observation.types import ObservationRecord
from citysense.situations.entity import CausalHypothesis, Situation


def score_hypothesis_confidence(
    hypothesis_type: str,
    recovery_vhr: float,
    baseline_vhr: float,
    evidence_count: int,
    diag_cfg: DiagnosisConfig,
) -> float:
    """Computes deterministic confidence score bounded in [confidence_min, confidence_max]."""
    w_type = diag_cfg.hypothesis_weights.get(hypothesis_type, 1.0 / 2.0)
    denom = max(baseline_vhr, diag_cfg.counterfactual_noise_floor_vhr)
    rec_ratio = min(1.0, recovery_vhr / denom)

    # Pure deterministic score
    raw_score = w_type * rec_ratio + (1.0 / 2.0) * min(1.0, float(evidence_count) / 2.0)
    bounded_score = max(diag_cfg.confidence_min, min(diag_cfg.confidence_max, raw_score))
    return float(bounded_score)


def build_situation(
    situation_id: str,
    corridor_name: str,
    observations: list[ObservationRecord],
    jam_clusters: list[JamCluster],
    shockwave_origin: ShockwaveOrigin | None,
    capacity_fit: CapacityFitResult | None,
    ladder_result: CounterfactualLadderResult,
    diag_cfg: DiagnosisConfig,
    config_hash: str,
) -> Situation:
    """Builds a civic Situation entity with deterministically ranked causal hypotheses."""
    simulated_flag = any(r.simulated for r in observations) if observations else True
    target_cell = shockwave_origin.origin_cell if shockwave_origin else 0
    time_window = capacity_fit.time_window_sec if capacity_fit else (0.0, 0.0)
    evidence_trail = shockwave_origin.evidence_trail if shockwave_origin else []

    ranked_hypotheses: list[CausalHypothesis] = []

    for rung in ladder_result.rungs:
        r_name = rung.rung_name

        if r_name == diag_cfg.top_ranked_name_capacity:
            h_type = diag_cfg.cause_name_recurring
            resp_party = diag_cfg.responsible_party_capacity
            action_text = diag_cfg.action_template_capacity.format(cell_id=target_cell)
        elif r_name == diag_cfg.top_ranked_name_demand:
            h_type = diag_cfg.cause_name_demand_shift
            resp_party = diag_cfg.responsible_party_demand
            action_text = diag_cfg.action_template_demand
        elif r_name == diag_cfg.top_ranked_name_signal:
            h_type = "signal_coordination_fault"
            resp_party = diag_cfg.responsible_party_signal
            action_text = diag_cfg.action_template_signal.format(junction_id=target_cell)
        else:
            h_type = diag_cfg.top_ranked_name_control
            resp_party = diag_cfg.responsible_party_capacity
            action_text = diag_cfg.action_template_control

        conf = score_hypothesis_confidence(
            hypothesis_type=r_name,
            recovery_vhr=rung.mean_recovery_vhr,
            baseline_vhr=ladder_result.baseline_total_vhr,
            evidence_count=len(evidence_trail),
            diag_cfg=diag_cfg,
        )

        ranked_hypotheses.append(
            CausalHypothesis(
                hypothesis_type=h_type,
                target_cell=target_cell,
                time_window=time_window,
                evidence_list=evidence_trail,
                vehicle_hours_recovered=rung.mean_recovery_vhr,
                ci_lower=rung.ci_lower_vhr,
                ci_upper=rung.ci_upper_vhr,
                confidence_score=conf,
                recommended_action=action_text,
                responsible_party=resp_party,
                is_hypothesis_label=True,
            )
        )

    # Rank hypotheses descending by recovered vehicle-hours
    ranked_hypotheses.sort(key=lambda h: h.vehicle_hours_recovered, reverse=True)

    obs_dicts = [r.to_dict() for r in observations]
    cluster_dicts = [c.to_dict() for c in jam_clusters]

    return Situation(
        situation_id=situation_id,
        corridor_name=corridor_name,
        observations_used=obs_dicts,
        jam_clusters=cluster_dicts,
        ranked_hypotheses=ranked_hypotheses,
        status=diag_cfg.default_status,
        simulated=simulated_flag,
        config_hash=config_hash,
    )
