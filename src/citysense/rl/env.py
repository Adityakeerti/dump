"""Gymnasium-compatible multi-junction signal control environment."""

from typing import Any
import numpy as np

from citysense.config.schemas import EngineConfig, RLConfig
from citysense.engine.adapter import EngineAdapter
from citysense.network.corridor import CorridorNetwork
from citysense.network.demand import CorridorDemand
from citysense.rl.shield import SignalSafetyShield


class CorridorSignalEnv:
    """Traffic signal control environment for corridor networks."""

    def __init__(
        self,
        engine: EngineAdapter,
        network: CorridorNetwork,
        demand: CorridorDemand,
        rl_config: RLConfig,
        engine_config: EngineConfig,
    ) -> None:
        self.engine = engine
        self.network = network
        self.demand = demand
        self.rl_cfg = rl_config
        self.eng_cfg = engine_config
        self.num_junctions = network.metadata.num_signalized_junctions
        self.shield = SignalSafetyShield(rl_config, self.num_junctions)

        # Observation shape per junction:
        # [art_queue_norm, cross_queue_norm, phase, time_in_phase_norm, disruption_flag, left_nbr_queue, right_nbr_queue]
        self.obs_dim_per_junction = (
            2 * rl_config.state_feature_dim_per_approach
            + rl_config.state_extra_dim
            + 2
        )
        self.action_space_size = rl_config.action_space_size

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.engine.reset(seed)
        self.engine.set_demand_profile(self.demand.as_callable())
        self.shield.reset()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        s = self.engine.state()
        all_obs = np.zeros((self.num_junctions, self.obs_dim_per_junction), dtype=np.float32)

        norm = self.rl_cfg.queue_norm_factor
        max_cyc = self.rl_cfg.shield_max_cycle_sec

        for j in range(self.num_junctions):
            j_cells = self.network.junction_cells[j]
            art_c = j_cells["arterial_approach"]
            cross_cs = j_cells.get("cross_approaches", [j_cells["cross_approach"]])

            art_q = float(s.queue[art_c]) / norm
            cross_q = float(sum(s.queue[c] for c in cross_cs)) / norm

            sh_state = self.shield.states[j]
            phase_val = float(sh_state.current_phase)
            time_norm = float(sh_state.time_in_phase) / max_cyc

            # Disruption flag: 1.0 if speed drops below half of vf
            disrupt = 0.0
            if (
                s.speed[art_c]
                < self.network.vf[art_c] * (1.0 / 2.0)
            ):
                disrupt = 1.0

            # Neighbour queue features
            left_nbr_q = 0.0
            if j > 0:
                prev_art_c = self.network.junction_cells[j - 1]["arterial_approach"]
                left_nbr_q = float(s.queue[prev_art_c]) / norm

            right_nbr_q = 0.0
            if j < self.num_junctions - 1:
                next_art_c = self.network.junction_cells[j + 1]["arterial_approach"]
                right_nbr_q = float(s.queue[next_art_c]) / norm

            feat_list = [
                art_q,
                cross_q,
                phase_val,
                time_norm,
                disrupt,
                left_nbr_q,
                right_nbr_q,
            ]
            all_obs[j, : len(feat_list)] = feat_list

        return all_obs

    def step(self, actions: list[int] | dict[int, int]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if isinstance(actions, list):
            act_dict = {j: actions[j] for j in range(len(actions))}
        else:
            act_dict = actions

        executed_phases: dict[int, int] = {}
        for j in range(self.num_junctions):
            act = act_dict.get(j, self.rl_cfg.action_keep)
            phase, _ = self.shield.step(j, act, self.eng_cfg.dt)
            executed_phases[j] = phase

        state = self.engine.step(executed_phases)

        # Compute rewards: negative total queue + worst approach penalty
        queues: list[float] = []
        for j in range(self.num_junctions):
            j_cells = self.network.junction_cells[j]
            queues.append(float(state.queue[j_cells["arterial_approach"]]))
            queues.append(float(sum(state.queue[c] for c in j_cells.get("cross_approaches", [j_cells["cross_approach"]]))))

        total_q = float(np.sum(queues))
        worst_q = float(np.max(queues)) if queues else 0.0

        reward = (
            self.rl_cfg.reward_queue_weight * total_q
            + self.rl_cfg.reward_worst_approach_weight * worst_q
        )

        obs = self._get_obs()
        done = False
        info = {"total_vehicles": float(np.sum(state.queue))}

        return obs, reward, done, info
