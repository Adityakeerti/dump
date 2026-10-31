"""Shockwave back-tracing and bottleneck origin localization."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from citysense.config.schemas import DiagnosisConfig, EngineConfig
from citysense.diagnosis.types import JamCluster
from citysense.observation.types import ObservationRecord


@dataclass
class ShockwaveOrigin:
    origin_cell: int
    estimated_wave_speed_mps: float
    earliest_time_sec: float
    evidence_trail: list[dict[str, Any]]
    jam_cluster_id: int


class ShockwaveTracer:
    """Traces shockwave fronts backward in space-time to pinpoint the bottleneck origin."""

    def __init__(self, engine_config: EngineConfig, diag_config: DiagnosisConfig) -> None:
        self.engine_cfg = engine_config
        self.diag_cfg = diag_config

    def estimate_wave_speed(
        self,
        cluster: JamCluster,
        observations: list[ObservationRecord],
    ) -> float:
        """Estimates backward shockwave speed w from observed flow-density differences."""
        cluster_cell_set = set(cluster.cell_ids)
        obs_in_cluster = [r for r in observations if r.cell_id in cluster_cell_set]

        if not obs_in_cluster:
            return float(self.engine_cfg.wave_speed)

        # Average congested density and flow inside jam
        densities = [r.density_vpm for r in obs_in_cluster if r.density_vpm > 0.0]
        flows = [r.flow_vps for r in obs_in_cluster if r.flow_vps >= 0.0]

        if not densities or not flows:
            return float(self.engine_cfg.wave_speed)

        k2 = float(np.mean(densities))
        q2 = float(np.mean(flows))

        # Nominal free-flow upstream state
        q1 = self.engine_cfg.capacity
        k1 = q1 / self.engine_cfg.free_flow_speed

        delta_k = k2 - k1
        delta_q = q2 - q1

        if abs(delta_k) > self.engine_cfg.density_epsilon:
            w_est = abs(delta_q / delta_k)
            # Bound within plausible physical range from config
            if (
                abs(w_est - self.engine_cfg.wave_speed)
                <= self.diag_cfg.wave_speed_tolerance_mps
            ):
                return float(w_est)

        return float(self.engine_cfg.wave_speed)

    def trace_origin(
        self,
        cluster: JamCluster,
        observations: list[ObservationRecord],
        cell_length_m: float,
    ) -> ShockwaveOrigin:
        """Traces the discharge front to identify the true downstream bottleneck origin."""
        cluster_cell_set = set(cluster.cell_ids)
        obs_in_cluster = [r for r in observations if r.cell_id in cluster_cell_set]

        # Aggregate onset times and mean speeds per cell
        cell_onsets: dict[int, float] = {}
        cell_speeds: dict[int, list[float]] = {}
        cell_flows: dict[int, list[float]] = {}

        for obs in obs_in_cluster:
            cid = obs.cell_id
            if cid not in cell_onsets or obs.timestamp_sec < cell_onsets[cid]:
                cell_onsets[cid] = obs.timestamp_sec
            cell_speeds.setdefault(cid, []).append(obs.speed_mps)
            cell_flows.setdefault(cid, []).append(obs.flow_vps)

        # In traffic flow theory, the bottleneck origin is the downstream-most cell
        # of the active queue (the discharge front where traffic releases downstream).
        origin_candidate = max(cell_onsets.keys()) if cell_onsets else cluster.head_cell

        w_est = self.estimate_wave_speed(cluster, observations)
        earliest_t = cell_onsets.get(origin_candidate, cluster.start_time_sec)

        # Build chronological evidence trail leading from origin upstream
        sorted_cells = sorted(cell_onsets.keys(), key=lambda c: (cell_onsets[c], -c))
        trail: list[dict[str, Any]] = []

        for cid in sorted_cells:
            t = cell_onsets[cid]
            avg_spd = float(np.mean(cell_speeds[cid]))
            avg_flw = float(np.mean(cell_flows[cid]))
            is_origin = cid == origin_candidate
            desc = (
                self.diag_cfg.action_template_capacity
                if is_origin
                else self.diag_cfg.action_template_signal
            )
            trail.append(
                {
                    "cell_id": int(cid),
                    "timestamp_sec": float(t),
                    "mean_speed_mps": round(avg_spd, 2),
                    "mean_flow_vps": round(avg_flw, 2),
                    "is_discharge_origin": is_origin,
                    "description": desc,
                }
            )

        return ShockwaveOrigin(
            origin_cell=int(origin_candidate),
            estimated_wave_speed_mps=float(w_est),
            earliest_time_sec=float(earliest_t),
            evidence_trail=trail,
            jam_cluster_id=cluster.cluster_id,
        )
