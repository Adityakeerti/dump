"""Gate tests for Module 10: Situation Records and Hypothesis Ranking."""

from pathlib import Path
import pytest

from citysense.config.loader import load_diagnosis_config
from citysense.counterfactual.types import (
    CounterfactualLadderResult,
    CounterfactualRungResult,
)
from citysense.diagnosis.capacity import CapacityFitResult
from citysense.diagnosis.shockwave import ShockwaveOrigin
from citysense.diagnosis.types import JamCluster
from citysense.observation.types import ObservationRecord
from citysense.situations.builder import build_situation, score_hypothesis_confidence
from citysense.situations.entity import Situation


DIAGNOSIS_PATH = Path("config/diagnosis.yaml")


def create_sample_situation_data():
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)

    obs = [
        ObservationRecord(
            cell_id=15,
            timestamp_sec=300.0,
            speed_mps=3.5,
            flow_vps=0.2,
            density_vpm=0.08,
            source="SIMULATED",
            simulated=True,
        ),
        ObservationRecord(
            cell_id=14,
            timestamp_sec=305.0,
            speed_mps=2.0,
            flow_vps=0.2,
            density_vpm=0.11,
            source="SIMULATED",
            simulated=True,
        ),
    ]

    cluster = JamCluster(
        cluster_id=0,
        cell_ids=[14, 15],
        timestamps_sec=[300.0, 305.0],
        tail_cell=14,
        head_cell=15,
        start_time_sec=300.0,
        end_time_sec=305.0,
        extent_m=30.0,
        duration_sec=5.0,
        mean_speed_mps=2.75,
    )

    trail = [
        {
            "cell_id": 15,
            "timestamp_sec": 300.0,
            "mean_speed_mps": 3.5,
            "mean_flow_vps": 0.2,
            "is_discharge_origin": True,
            "description": diag_cfg.action_template_capacity,
        }
    ]

    shockwave = ShockwaveOrigin(
        origin_cell=15,
        estimated_wave_speed_mps=5.56,
        earliest_time_sec=300.0,
        evidence_trail=trail,
        jam_cluster_id=0,
    )

    cap_fit = CapacityFitResult(
        cell_id=15,
        time_window_sec=(300.0, 350.0),
        nominal_capacity=0.5,
        estimated_capacity=0.2,
        capacity_ratio=0.4,
        confidence_interval=(0.35, 0.45),
        change_points=[300.0],
        significant_drop_detected=True,
    )

    rungs = [
        CounterfactualRungResult(
            rung_name=diag_cfg.top_ranked_name_capacity,
            mean_recovery_vhr=0.45,
            ci_lower_vhr=0.40,
            ci_upper_vhr=0.50,
            replicate_recoveries=[0.45],
            rank=1,
        ),
        CounterfactualRungResult(
            rung_name=diag_cfg.top_ranked_name_demand,
            mean_recovery_vhr=0.10,
            ci_lower_vhr=0.08,
            ci_upper_vhr=0.12,
            replicate_recoveries=[0.10],
            rank=2,
        ),
        CounterfactualRungResult(
            rung_name=diag_cfg.top_ranked_name_signal,
            mean_recovery_vhr=0.02,
            ci_lower_vhr=0.01,
            ci_upper_vhr=0.03,
            replicate_recoveries=[0.02],
            rank=3,
        ),
    ]

    ladder = CounterfactualLadderResult(
        scenario_name="recurring_capacity_drop",
        rungs=rungs,
        top_ranked_rung=diag_cfg.top_ranked_name_capacity,
        baseline_total_vhr=0.80,
        noise_floor_vhr=diag_cfg.counterfactual_noise_floor_vhr,
        explains_jam=True,
    )

    return obs, [cluster], shockwave, cap_fit, ladder, diag_cfg


def test_situation_schema_validation():
    """Verify Situation entity validates fields and propagates correct schema structure."""
    obs, clusters, shockwave, cap_fit, ladder, diag_cfg = create_sample_situation_data()

    sit = build_situation(
        situation_id="SIT-001",
        corridor_name="Metropolitan Corridor",
        observations=obs,
        jam_clusters=clusters,
        shockwave_origin=shockwave,
        capacity_fit=cap_fit,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash="abc123canonicalhash",
    )

    assert isinstance(sit, Situation)
    assert sit.situation_id == "SIT-001"
    assert len(sit.ranked_hypotheses) == 3

    # Hypotheses are always labelled hypotheses, never declared causes
    for hypo in sit.ranked_hypotheses:
        assert hypo.is_hypothesis_label is True

    d = sit.to_dict()
    assert "situation_id" in d
    assert "ranked_hypotheses" in d
    assert "top_hypothesis" in d
    assert d["top_hypothesis"]["hypothesis_type"] == diag_cfg.cause_name_recurring


def test_ranking_is_pure_function_of_inputs():
    """Verify ranking and scores are pure deterministic functions of inputs."""
    obs, clusters, shockwave, cap_fit, ladder, diag_cfg = create_sample_situation_data()

    sit1 = build_situation(
        situation_id="SIT-001",
        corridor_name="Metropolitan Corridor",
        observations=obs,
        jam_clusters=clusters,
        shockwave_origin=shockwave,
        capacity_fit=cap_fit,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash="hash1",
    )
    sit2 = build_situation(
        situation_id="SIT-001",
        corridor_name="Metropolitan Corridor",
        observations=obs,
        jam_clusters=clusters,
        shockwave_origin=shockwave,
        capacity_fit=cap_fit,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash="hash1",
    )

    assert sit1.to_dict() == sit2.to_dict()


def test_evidence_references_resolve_to_real_observations():
    """Verify evidence entries resolve to real observations in the situation."""
    obs, clusters, shockwave, cap_fit, ladder, diag_cfg = create_sample_situation_data()

    sit = build_situation(
        situation_id="SIT-001",
        corridor_name="Metropolitan Corridor",
        observations=obs,
        jam_clusters=clusters,
        shockwave_origin=shockwave,
        capacity_fit=cap_fit,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash="hash1",
    )

    obs_lookup = {
        (o["cell_id"], o["timestamp_sec"]) for o in sit.observations_used
    }

    for hypo in sit.ranked_hypotheses:
        for ev in hypo.evidence_list:
            key = (ev["cell_id"], ev["timestamp_sec"])
            assert key in obs_lookup, f"Evidence reference {key} not found in situation observations"


def test_simulated_flag_propagates():
    """Verify simulated flag propagates into Situation entity."""
    obs, clusters, shockwave, cap_fit, ladder, diag_cfg = create_sample_situation_data()

    sit = build_situation(
        situation_id="SIT-SIM",
        corridor_name="Corridor",
        observations=obs,
        jam_clusters=clusters,
        shockwave_origin=shockwave,
        capacity_fit=cap_fit,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash="hash",
    )
    assert sit.simulated is True
    assert sit.to_dict()["simulated"] is True


def test_no_field_is_filled_from_ground_truth():
    """Enforce Hard Rule 7: situation fields are never filled from ground truth."""
    obs, clusters, shockwave, cap_fit, ladder, diag_cfg = create_sample_situation_data()

    sit = build_situation(
        situation_id="SIT-TEST",
        corridor_name="Corridor",
        observations=obs,
        jam_clusters=clusters,
        shockwave_origin=shockwave,
        capacity_fit=cap_fit,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash="hash",
    )

    sit_dict = sit.to_dict()
    forbidden_terms = ["ground_truth", "planted_cause", "true_cell"]
    for term in forbidden_terms:
        assert term not in sit_dict
