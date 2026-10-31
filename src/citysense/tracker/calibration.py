"""Outcome tracker comparing predicted vs observed recovery and updating calibration weights."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from citysense.config.schemas import DiagnosisConfig
from citysense.situations.entity import Situation


@dataclass
class OutcomeComparisonResult:
    situation_id: str
    hypothesis_type: str
    predicted_recovery_vhr: float
    predicted_interval: tuple[float, float]
    observed_recovery_vhr: float
    prediction_error_vhr: float
    inside_interval: bool
    simulated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "situation_id": self.situation_id,
            "hypothesis_type": self.hypothesis_type,
            "predicted_recovery_vhr": round(self.predicted_recovery_vhr, 2),
            "predicted_interval": [
                round(self.predicted_interval[0], 2),
                round(self.predicted_interval[1], 2),
            ],
            "observed_recovery_vhr": round(self.observed_recovery_vhr, 2),
            "prediction_error_vhr": round(self.prediction_error_vhr, 2),
            "inside_interval": self.inside_interval,
            "simulated": self.simulated,
        }


class CalibrationTable:
    """Maintains hypothesis scoring weights and applies bounded, reversible updates."""

    def __init__(self, diag_config: DiagnosisConfig) -> None:
        self.diag_cfg = diag_config
        self.weights: dict[str, float] = {
            k: float(v) for k, v in diag_config.hypothesis_weights.items()
        }
        self.history: list[tuple[str, float, float]] = []

    def get_weight(self, hypothesis_type: str) -> float:
        return self.weights.get(hypothesis_type, self.diag_cfg.initial_calibration_weight)

    def update(self, hypothesis_type: str, confirmed: bool) -> tuple[float, float]:
        """Applies a bounded Bayesian-style weight adjustment."""
        old_w = self.get_weight(hypothesis_type)
        lr = self.diag_cfg.calibration_learning_rate
        max_bound = self.diag_cfg.calibration_max_step_bound

        if confirmed:
            raw_delta = lr * (self.diag_cfg.max_calibrated_weight - old_w)
        else:
            raw_delta = -lr * (old_w - self.diag_cfg.min_calibrated_weight)

        # Enforce strict step bounds
        bounded_delta = max(-max_bound, min(max_bound, raw_delta))
        new_w = old_w + bounded_delta
        new_w = max(
            self.diag_cfg.min_calibrated_weight,
            min(self.diag_cfg.max_calibrated_weight, new_w),
        )

        self.weights[hypothesis_type] = new_w
        self.history.append((hypothesis_type, old_w, new_w))
        return old_w, new_w

    def revert_last(self) -> tuple[str, float]:
        """Reverts the most recent weight adjustment."""
        if not self.history:
            return "", 0.0
        h_type, old_w, _ = self.history.pop()
        self.weights[h_type] = old_w
        return h_type, old_w


class OutcomeTracker:
    """Compares predicted counterfactual recovery against observed post-fix recovery."""

    def __init__(self, diag_config: DiagnosisConfig) -> None:
        self.diag_cfg = diag_config
        self.calibration_table = CalibrationTable(diag_config)

    def compare_outcome(
        self,
        situation: Situation,
        observed_recovery_vhr: float,
    ) -> OutcomeComparisonResult:
        if not situation.ranked_hypotheses:
            raise ValueError("Situation has no ranked hypotheses")

        top_h = situation.ranked_hypotheses[0]
        pred_rec = top_h.vehicle_hours_recovered
        ci = (top_h.ci_lower, top_h.ci_upper)

        inside = ci[0] <= observed_recovery_vhr <= ci[1]
        err = abs(observed_recovery_vhr - pred_rec)

        # Update calibration table
        self.calibration_table.update(top_h.hypothesis_type, confirmed=inside)

        return OutcomeComparisonResult(
            situation_id=situation.situation_id,
            hypothesis_type=top_h.hypothesis_type,
            predicted_recovery_vhr=pred_rec,
            predicted_interval=ci,
            observed_recovery_vhr=observed_recovery_vhr,
            prediction_error_vhr=err,
            inside_interval=inside,
            simulated=situation.simulated,
        )
