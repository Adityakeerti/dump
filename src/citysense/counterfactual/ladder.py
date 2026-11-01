"""Counterfactual simulation ladder evaluating causal hypotheses by vehicle-hours recovered."""

import copy
from typing import Any, Callable
import numpy as np

from citysense.config.schemas import DiagnosisConfig, EngineConfig
from citysense.config.units import SECONDS_PER_HOUR
from citysense.counterfactual.types import (
    CounterfactualLadderResult,
    CounterfactualRungResult,
)
from citysense.engine.adapter import EngineAdapter
from citysense.engine.types import EngineSnapshot
from citysense.network.corridor import CorridorNetwork
from citysense.network.demand import CorridorDemand


class CounterfactualLadder:
    """Evaluates causal counterfactual rungs by measuring congestion vehicle-hours recovered."""

    def __init__(self, engine_cfg: EngineConfig, diag_cfg: DiagnosisConfig) -> None:
        self.engine_cfg = engine_cfg
        self.diag_cfg = diag_cfg

    def _compute_congestion_hours(
        self,
        engine: EngineAdapter,
        network: CorridorNetwork,
        demand_fn: Callable[[float], float],
        horizon_steps: int,
        signal_controller: Any = None,
    ) -> float:
        """Runs the engine forward for horizon_steps and integrates congestion delay (hours)."""
        engine.set_demand_profile(demand_fn)
        engine.set_signal_controller(signal_controller)

        kc = self.engine_cfg.capacity / self.engine_cfg.free_flow_speed
        art_cells = network.arterial_cells
        dt_hours = self.engine_cfg.dt / SECONDS_PER_HOUR
        total_delay = 0.0

        for _ in range(horizon_steps):
            state = engine.step()
            art_dens = state.density[art_cells]
            excess_density = np.maximum(0.0, art_dens - kc)
            excess_veh = excess_density * network.cell_length[art_cells] * network.lanes[art_cells]
            total_delay += float(np.sum(excess_veh)) * dt_hours

        return total_delay

    def evaluate(
        self,
        engine: EngineAdapter,
        network: CorridorNetwork,
        base_demand: CorridorDemand,
        suspect_cell: int,
        snapshot: EngineSnapshot,
        horizon_steps: int,
        scenario_name: str,
        seed: int,
    ) -> CounterfactualLadderResult:
        """Evaluates each intervention rung over perturbed demand replicates."""
        rng = np.random.default_rng(seed)
        replicates = self.diag_cfg.counterfactual_replicates
        std_pert = self.diag_cfg.counterfactual_perturbation_std
        noise_floor = self.diag_cfg.counterfactual_noise_floor_vhr

        rungs = self.diag_cfg.ladder_ranking_order
        recoveries_by_rung: dict[str, list[float]] = {r: [] for r in rungs}
        baseline_vhs: list[float] = []

        for rep in range(replicates):
            # Demand perturbation multiplier
            delta = float(rng.normal(0.0, std_pert))
            mult = max(0.0, 1.0 + delta)

            def perturbed_demand(t: float, m=mult) -> float:
                return base_demand.get_rate(t) * m

            # 1. Baseline observed run (with existing disruptions)
            engine.restore(snapshot)
            vh_base = self._compute_congestion_hours(
                engine=engine,
                network=network,
                demand_fn=perturbed_demand,
                horizon_steps=horizon_steps,
            )
            baseline_vhs.append(vh_base)

            # 2. Rung: restore_capacity at suspect cell (searching within bottleneck_search_radius_cells downstream)
            engine.restore(snapshot)
            radius = self.diag_cfg.bottleneck_search_radius_cells
            cleared_windows = [
                w
                for w in engine.snapshot().capacity_windows
                if not any(0 <= (c - suspect_cell) <= radius for c in w.cell_ids)
            ]
            snap_cleared = copy.deepcopy(engine.snapshot())
            snap_cleared.capacity_windows = cleared_windows
            engine.restore(snap_cleared)

            vh_restore_cap = self._compute_congestion_hours(
                engine=engine,
                network=network,
                demand_fn=perturbed_demand,
                horizon_steps=horizon_steps,
            )
            rec_cap = max(0.0, vh_base - vh_restore_cap)
            recoveries_by_rung[self.diag_cfg.top_ranked_name_capacity].append(rec_cap)

            # 3. Rung: demand_peak_shift (retime peak surge)
            engine.restore(snapshot)

            # Retime only the peak window (CorridorDemand's peak_shift_sec). Shifting the whole curve,
            # get_rate(t - shift), also delayed the warm-up and produced negative demand for t < shift.
            retimed = CorridorDemand(
                base_demand.config,
                is_weekend=base_demand.is_weekend,
                peak_shift_sec=base_demand.peak_shift_sec + base_demand.config.peak_shift_sec,
            )

            def shifted_demand(t: float, m=mult, d=retimed) -> float:
                return d.get_rate(t) * m

            vh_demand_shift = self._compute_congestion_hours(
                engine=engine,
                network=network,
                demand_fn=shifted_demand,
                horizon_steps=horizon_steps,
            )
            rec_dem = max(0.0, vh_base - vh_demand_shift)
            recoveries_by_rung[self.diag_cfg.top_ranked_name_demand].append(rec_dem)

            # 4. Rung: rl_signal_control (arterial green coordination)
            engine.restore(snapshot)

            class GreenWaveController:
                def act(self, t: float, density: np.ndarray) -> dict[int, int]:
                    return {j: 0 for j in range(network.metadata.num_signalized_junctions)}

            vh_sig = self._compute_congestion_hours(
                engine=engine,
                network=network,
                demand_fn=perturbed_demand,
                horizon_steps=horizon_steps,
                signal_controller=GreenWaveController(),
            )
            rec_sig = max(0.0, vh_base - vh_sig)
            recoveries_by_rung[self.diag_cfg.top_ranked_name_signal].append(rec_sig)

        # Aggregate metrics across replicates
        rung_results: list[CounterfactualRungResult] = []
        for rung_name in rungs:
            recs = recoveries_by_rung[rung_name]
            mean_rec = float(np.mean(recs))
            std_rec = float(np.std(recs)) if len(recs) > 1 else 0.0
            ci_half = 2.0 * (std_rec / np.sqrt(len(recs))) if len(recs) > 0 else 0.0

            rung_results.append(
                CounterfactualRungResult(
                    rung_name=rung_name,
                    mean_recovery_vhr=mean_rec,
                    ci_lower_vhr=max(0.0, mean_rec - ci_half),
                    ci_upper_vhr=mean_rec + ci_half,
                    replicate_recoveries=recs,
                    rank=0,
                )
            )

        # Rank descending by mean recovery
        rung_results.sort(key=lambda r: r.mean_recovery_vhr, reverse=True)
        for rank_idx, r in enumerate(rung_results):
            r.rank = rank_idx + 1

        top_rung = rung_results[0]
        explains = top_rung.mean_recovery_vhr > noise_floor
        top_name = top_rung.rung_name if explains else self.diag_cfg.top_ranked_name_control

        return CounterfactualLadderResult(
            scenario_name=scenario_name,
            rungs=rung_results,
            top_ranked_rung=top_name,
            baseline_total_vhr=float(np.mean(baseline_vhs)),
            noise_floor_vhr=noise_floor,
            explains_jam=explains,
        )
