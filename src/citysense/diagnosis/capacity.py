"""Capacity estimation and change-point detection for physical bottleneck identification."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from citysense.config.schemas import DiagnosisConfig, EngineConfig
from citysense.observation.types import ObservationRecord


@dataclass
class CapacityFitResult:
    cell_id: int
    time_window_sec: tuple[float, float]
    nominal_capacity: float
    estimated_capacity: float
    capacity_ratio: float
    confidence_interval: tuple[float, float]
    change_points: list[float]
    significant_drop_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "time_window_sec": list(self.time_window_sec),
            "nominal_capacity": self.nominal_capacity,
            "estimated_capacity": round(self.estimated_capacity, 2),
            "capacity_ratio": round(self.capacity_ratio, 2),
            "confidence_interval": [
                round(self.confidence_interval[0], 2),
                round(self.confidence_interval[1], 2),
            ],
            "change_points": self.change_points,
            "significant_drop_detected": self.significant_drop_detected,
        }


class CapacityEstimator:
    """Estimates effective capacity and detects change points and progressive creeping degradation."""

    def __init__(self, engine_config: EngineConfig, diag_config: DiagnosisConfig) -> None:
        self.engine_cfg = engine_config
        self.diag_cfg = diag_config

    def fit_capacity(
        self,
        cell_id: int,
        observations: list[ObservationRecord],
        time_window: tuple[float, float],
    ) -> CapacityFitResult:
        t_start = time_window[0]
        t_end = time_window[1]

        # Filter observations for cell and window
        cell_obs = [
            r
            for r in observations
            if r.cell_id == cell_id and t_start <= r.timestamp_sec <= t_end
        ]

        nom_cap = float(self.engine_cfg.capacity)
        if len(cell_obs) < self.diag_cfg.capacity_fit_min_samples:
            return CapacityFitResult(
                cell_id=cell_id,
                time_window_sec=time_window,
                nominal_capacity=nom_cap,
                estimated_capacity=nom_cap,
                capacity_ratio=1.0,
                confidence_interval=(1.0, 1.0),
                change_points=[],
                significant_drop_detected=False,
            )

        flows = np.array([r.flow_vps for r in cell_obs], dtype=np.float64)

        # In bottleneck states, sustained flow represents effective capacity
        sorted_flows = np.sort(flows)
        # 80th percentile: index = int(0.8 * len)
        idx_80 = int(len(sorted_flows) * (1.0 - (1.0 / 2.0) * (1.0 / 2.0)))
        est_cap = float(sorted_flows[idx_80])
        est_cap = min(est_cap, nom_cap)

        ratio = est_cap / nom_cap

        # Standard error and confidence interval
        std_err = float(np.std(flows) / np.sqrt(len(flows)))
        ratio_err = std_err / nom_cap
        ci_lower = max(0.0, ratio - 2.0 * ratio_err)
        ci_upper = min(1.0, ratio + 2.0 * ratio_err)

        change_pts = self.detect_change_points(cell_obs, nom_cap)
        sig_drop = ratio < (1.0 - self.diag_cfg.change_point_min_drop_ratio)

        return CapacityFitResult(
            cell_id=cell_id,
            time_window_sec=time_window,
            nominal_capacity=nom_cap,
            estimated_capacity=est_cap,
            capacity_ratio=ratio,
            confidence_interval=(ci_lower, ci_upper),
            change_points=change_pts,
            significant_drop_detected=sig_drop,
        )

    def detect_change_points(
        self,
        observations: list[ObservationRecord],
        nominal_capacity: float,
        warmup_sec: float = 0.0,
    ) -> list[float]:
        """Detects sudden or creeping capacity change points using cumulative sum (CUSUM)."""
        valid_obs = [r for r in observations if r.timestamp_sec >= warmup_sec]
        if len(valid_obs) < self.diag_cfg.change_point_min_size:
            return []

        sorted_obs = sorted(valid_obs, key=lambda r: r.timestamp_sec)
        change_points: list[float] = []

        cusum = 0.0
        penalty = self.diag_cfg.change_point_penalty
        drop_threshold = nominal_capacity * self.diag_cfg.change_point_min_drop_ratio

        # Determine baseline from initial window
        initial_flows = [r.flow_vps for r in sorted_obs[: self.diag_cfg.change_point_min_size]]
        baseline = float(np.median(initial_flows)) if initial_flows else nominal_capacity

        for obs in sorted_obs:
            flow_deficit = baseline - obs.flow_vps
            if flow_deficit > drop_threshold:
                cusum += flow_deficit
                if cusum >= penalty:
                    change_points.append(obs.timestamp_sec)
                    cusum = 0.0
            else:
                cusum = max(0.0, cusum - drop_threshold)

        return change_points
