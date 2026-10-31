"""Engine data structures for simulation state, capacity perturbations, and snapshots."""

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class SimulationState:
    density: np.ndarray
    flow: np.ndarray
    speed: np.ndarray
    queue: np.ndarray
    time_sec: float
    step_count: int
    total_entered: float
    total_exited: float
    queued_demand: float

    def copy(self) -> "SimulationState":
        return SimulationState(
            density=self.density.copy(),
            flow=self.flow.copy(),
            speed=self.speed.copy(),
            queue=self.queue.copy(),
            time_sec=self.time_sec,
            step_count=self.step_count,
            total_entered=self.total_entered,
            total_exited=self.total_exited,
            queued_demand=self.queued_demand,
        )


@dataclass
class CapacityFactorWindow:
    cell_ids: list[int]
    factor: float
    start_time_sec: float
    end_time_sec: float


@dataclass
class EngineSnapshot:
    state: SimulationState
    signal_phases: dict[int, int]
    signal_timers: dict[int, float]
    capacity_windows: list[CapacityFactorWindow]
    entrance_buffer: np.ndarray
    rng_state: Any
