"""Classical heuristic signal controllers: fixed-time, longest-queue, and max-pressure."""

from typing import Any
import numpy as np

from citysense.config.schemas import NetworkConfig, RLConfig
from citysense.engine.types import SimulationState
from citysense.network.corridor import CorridorNetwork


class FixedTimeController:
    """Fixed-time signal controller switching at regular half-cycle intervals."""

    def __init__(self, net_config: NetworkConfig, rl_config: RLConfig) -> None:
        self.net_cfg = net_config
        self.rl_cfg = rl_config
        self.half_cycle = net_config.signal_cycle_time * (1.0 / 2.0)
        self.timers: dict[int, float] = {}
        self.phases: dict[int, int] = {}

    def reset(self, num_junctions: int) -> None:
        self.timers = {j: 0.0 for j in range(num_junctions)}
        self.phases = {j: 0 for j in range(num_junctions)}

    def act(self, junction_id: int, dt: float) -> int:
        self.timers[junction_id] = self.timers.get(junction_id, 0.0) + dt
        if self.timers[junction_id] >= self.half_cycle:
            self.timers[junction_id] = 0.0
            return self.rl_cfg.action_switch
        return self.rl_cfg.action_keep


class LongestQueueController:
    """Greedy controller activating the phase with the largest incoming queue."""

    def __init__(self, rl_config: RLConfig) -> None:
        self.rl_cfg = rl_config

    def act(
        self,
        junction_id: int,
        current_phase: int,
        art_queue: float,
        cross_queue: float,
    ) -> int:
        if current_phase == 0 and cross_queue > art_queue:
            return self.rl_cfg.action_switch
        if current_phase == 1 and art_queue > cross_queue:
            return self.rl_cfg.action_switch
        return self.rl_cfg.action_keep


class MaxPressureController:
    """Max-pressure controller minimizing queue backpressure differential."""

    def __init__(self, rl_config: RLConfig) -> None:
        self.rl_cfg = rl_config

    def act(
        self,
        junction_id: int,
        current_phase: int,
        upstream_art_q: float,
        downstream_art_q: float,
        cross_q: float,
    ) -> int:
        art_pressure = upstream_art_q - downstream_art_q
        cross_pressure = cross_q

        if current_phase == 0 and cross_pressure > art_pressure:
            return self.rl_cfg.action_switch
        if current_phase == 1 and art_pressure > cross_pressure:
            return self.rl_cfg.action_switch
        return self.rl_cfg.action_keep
