"""Gate tests for Module 12: Outcome Tracker and Calibration."""

from pathlib import Path
import numpy as np
import pytest
import yaml

from citysense.config.loader import load_diagnosis_config
from citysense.situations.entity import CausalHypothesis, Situation
from citysense.tracker.calibration import CalibrationTable, OutcomeTracker


DIAGNOSIS_PATH = Path("config/diagnosis.yaml")
TOLERANCES_PATH = Path("tests/tolerances.yaml")


def load_tolerances() -> dict[str, float]:
    with open(TOLERANCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v["value"] for k, v in data.items()}


def create_dummy_situation(predicted_vhr: float, ci_half: float, simulated: bool = True) -> Situation:
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    hypo = CausalHypothesis(
        hypothesis_type=diag_cfg.top_ranked_name_capacity,
        target_cell=15,
        time_window=(300.0, 350.0),
        evidence_list=[],
        vehicle_hours_recovered=predicted_vhr,
        ci_lower=max(0.0, predicted_vhr - ci_half),
        ci_upper=predicted_vhr + ci_half,
        confidence_score=0.85,
        recommended_action="Inspect",
        responsible_party="DOT",
        is_hypothesis_label=True,
    )
    return Situation(
        situation_id="SIT-TEST",
        corridor_name="Metropolitan Corridor",
        observations_used=[],
        jam_clusters=[],
        ranked_hypotheses=[hypo],
        status=diag_cfg.default_status,
        simulated=simulated,
        config_hash="testhash",
    )


def test_observed_recovery_inside_predicted_interval_coverage():
    """Verify observed recovery falls inside predicted interval at tolerance coverage rate."""
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    tols = load_tolerances()
    tracker = OutcomeTracker(diag_cfg)

    rng = np.random.default_rng(42)
    num_runs = 20
    hits = 0

    pred_mean = 0.50
    ci_half = 0.10  # 95% interval [0.40, 0.60]

    for i in range(num_runs):
        # Simulate observed recovery from N(pred_mean, (ci_half/2)^2)
        obs_rec = float(rng.normal(pred_mean, ci_half / 2.0))
        sit = create_dummy_situation(pred_mean, ci_half)
        res = tracker.compare_outcome(sit, obs_rec)

        if res.inside_interval:
            hits += 1

    coverage_rate = hits / num_runs
    target_rate = tols["outcome_tracker_coverage"]
    assert coverage_rate >= target_rate, (
        f"Coverage rate {coverage_rate:.2f} below target {target_rate:.2f}"
    )


def test_calibration_update_is_bounded_and_reversible():
    """Verify calibration update is bounded by max step and fully reversible."""
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    calib = CalibrationTable(diag_cfg)
    h_type = diag_cfg.top_ranked_name_capacity

    initial_w = calib.get_weight(h_type)
    old_w, new_w = calib.update(h_type, confirmed=True)

    # Invariant 1: Bounded step
    step_size = abs(new_w - old_w)
    assert step_size <= diag_cfg.calibration_max_step_bound

    # Invariant 2: Within global bounds
    assert diag_cfg.min_calibrated_weight <= new_w <= diag_cfg.max_calibrated_weight

    # Invariant 3: Reversibility
    reverted_type, reverted_w = calib.revert_last()
    assert reverted_type == h_type
    assert reverted_w == pytest.approx(initial_w, abs=1e-12)
    assert calib.get_weight(h_type) == pytest.approx(initial_w, abs=1e-12)


def test_every_output_is_labelled_simulated():
    """Verify every outcome tracking output is explicitly labelled simulated."""
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    tracker = OutcomeTracker(diag_cfg)

    sit_sim = create_dummy_situation(0.4, 0.1, simulated=True)
    res = tracker.compare_outcome(sit_sim, 0.42)

    assert res.simulated is True
    res_dict = res.to_dict()
    assert res_dict["simulated"] is True
