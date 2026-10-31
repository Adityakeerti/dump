"""Native vectorized Cell Transmission Model (CTM) engine adhering strictly to Daganzo CTM theory."""

import copy
from typing import Any, Callable
import numpy as np

from citysense.config.schemas import EngineConfig
from citysense.engine.adapter import EngineAdapter
from citysense.engine.types import (
    CapacityFactorWindow,
    EngineSnapshot,
    SimulationState,
)


class NativeCTMEngine(EngineAdapter):
    """Vectorized macroscopic CTM simulation engine."""

    def __init__(
        self,
        config: EngineConfig,
        n_cells: int,
        upstream_cells: np.ndarray,
        downstream_cells: np.ndarray,
        lanes: np.ndarray | None = None,
        cell_length: np.ndarray | None = None,
        signal_controlled_cells: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        self.config = config
        self.dt = config.dt
        self.n_cells = n_cells
        self.upstream_cells = upstream_cells.astype(np.int64)
        self.downstream_cells = downstream_cells.astype(np.int64)

        if lanes is not None:
            self.lanes = lanes.astype(np.float64)
        else:
            self.lanes = np.ones(n_cells, dtype=np.float64)

        if cell_length is not None:
            self.length = cell_length.astype(np.float64)
        else:
            self.length = np.full(n_cells, config.cell_length, dtype=np.float64)

        self.vf = np.full(n_cells, config.free_flow_speed, dtype=np.float64)
        self.w = np.full(n_cells, config.wave_speed, dtype=np.float64)
        self.kj = np.full(n_cells, config.jam_density, dtype=np.float64)
        self.q_nom = np.full(n_cells, config.capacity, dtype=np.float64)

        self.signal_controlled_cells = signal_controlled_cells or {}
        self.capacity_windows: list[CapacityFactorWindow] = []
        self.demand_fn: Callable[[float], float] | None = None
        self.controller: Any = None

        self.signal_phases: dict[int, int] = {}
        self.signal_timers: dict[int, float] = {}

        self.source_mask = self.upstream_cells < 0
        self.sink_mask = self.downstream_cells < 0

        self.rng = np.random.default_rng(config.test_steps)
        self._init_state()

    def _init_state(self) -> None:
        self.density = np.zeros(self.n_cells, dtype=np.float64)
        self.vehicles = np.zeros(self.n_cells, dtype=np.float64)
        self.flow = np.zeros(self.n_cells, dtype=np.float64)
        self.speed = self.vf.copy()
        self.entrance_buffer = np.zeros(self.n_cells, dtype=np.float64)
        self.time_sec = 0.0
        self.step_count = 0
        self.total_entered = 0.0
        self.total_exited = 0.0

    def reset(self, seed: int | None = None) -> SimulationState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()
        self.capacity_windows.clear()
        self.signal_phases.clear()
        self.signal_timers.clear()
        return self.state()

    def set_capacity_factor(
        self, cell_ids: list[int] | int, factor: float, time_window: tuple[float, float]
    ) -> None:
        if isinstance(cell_ids, int):
            c_ids = [cell_ids]
        else:
            c_ids = list(cell_ids)
        start_t = time_window[0]
        end_t = time_window[1]
        self.capacity_windows.append(
            CapacityFactorWindow(
                cell_ids=c_ids,
                factor=factor,
                start_time_sec=start_t,
                end_time_sec=end_t,
            )
        )

    def set_demand_profile(self, profile: Any) -> None:
        if callable(profile):
            self.demand_fn = profile
        elif hasattr(profile, "get_rate"):
            self.demand_fn = profile.get_rate
        else:
            self.demand_fn = None

    def set_signal_controller(self, controller: Any) -> None:
        self.controller = controller

    def _get_effective_capacity(self) -> np.ndarray:
        q_eff = self.q_nom.copy()
        for w in self.capacity_windows:
            if w.start_time_sec <= self.time_sec < w.end_time_sec:
                for cid in w.cell_ids:
                    if 0 <= cid < self.n_cells:
                        q_eff[cid] = self.q_nom[cid] * w.factor
        return q_eff

    def step(self, signal_actions: dict[int, int] | None = None) -> SimulationState:
        if signal_actions:
            for jid, phase in signal_actions.items():
                self.signal_phases[jid] = phase
        elif self.controller is not None and hasattr(self.controller, "act"):
            actions = self.controller.act(self.time_sec, self.density)
            if actions:
                for jid, phase in actions.items():
                    self.signal_phases[jid] = phase

        q_eff = self._get_effective_capacity()

        # Sending flow: S_i = min(vf * k_i, Q_eff_i) * lanes_i
        sending = np.minimum(self.vf * self.density, q_eff) * self.lanes

        # Receiving flow: R_i = min(Q_eff_i, w * (kj - k_i)) * lanes_i
        space_avail = np.maximum(self.kj - self.density, 0.0)
        receiving = np.minimum(q_eff, self.w * space_avail) * self.lanes

        # Transfer flows between cells
        n_transfers = self.n_cells
        in_transfer = np.zeros(n_transfers, dtype=np.float64)
        out_transfer = np.zeros(n_transfers, dtype=np.float64)

        has_downstream = ~self.sink_mask
        src_indices = np.where(has_downstream)[0]
        dst_indices = self.downstream_cells[src_indices]

        # Compute signal mask
        signal_mask = np.ones(len(src_indices), dtype=np.float64)
        for idx in range(len(src_indices)):
            src_cell = src_indices[idx]
            if src_cell in self.signal_controlled_cells:
                jid, allowed_phase = self.signal_controlled_cells[src_cell]
                current_phase = self.signal_phases.get(jid, 0)
                if current_phase != allowed_phase:
                    signal_mask[idx] = 0.0

        flow_potential = np.minimum(sending[src_indices], receiving[dst_indices])
        link_flow = flow_potential * signal_mask
        step_transfer = link_flow * self.dt

        np.add.at(out_transfer, src_indices, step_transfer)
        np.add.at(in_transfer, dst_indices, step_transfer)

        # Handle demand injection at sources
        current_demand_rate = 0.0
        if self.demand_fn is not None:
            current_demand_rate = self.demand_fn(self.time_sec)

        sources = np.where(self.source_mask)[0]
        for src in sources:
            self.entrance_buffer[src] += current_demand_rate * self.dt
            max_enter = receiving[src] * self.dt
            enter_count = min(self.entrance_buffer[src], max_enter)
            self.entrance_buffer[src] -= enter_count
            in_transfer[src] += enter_count
            self.total_entered += enter_count

        # Handle exits at sinks
        sinks = np.where(self.sink_mask)[0]
        for snk in sinks:
            exit_rate = np.minimum(sending[snk], q_eff[snk] * self.lanes[snk])
            exit_count = exit_rate * self.dt
            out_transfer[snk] += exit_count
            self.total_exited += exit_count

        # Update vehicle counts and densities
        self.vehicles += in_transfer - out_transfer
        np.maximum(self.vehicles, 0.0, out=self.vehicles)

        cell_capacity_veh = self.kj * self.length * self.lanes
        np.minimum(self.vehicles, cell_capacity_veh, out=self.vehicles)

        lane_length = self.length * self.lanes
        self.density = self.vehicles / lane_length

        # Compute per-cell flow and speed
        self.flow = out_transfer / (self.dt * self.lanes)
        self.speed = np.full(self.n_cells, self.config.free_flow_speed, dtype=np.float64)
        has_density = self.density > self.config.density_epsilon
        if np.any(has_density):
            raw_speed = self.flow[has_density] / self.density[has_density]
            self.speed[has_density] = np.minimum(raw_speed, self.vf[has_density])

        self.time_sec += self.dt
        self.step_count += 1

        return self.state()

    def state(self) -> SimulationState:
        return SimulationState(
            density=self.density.copy(),
            flow=self.flow.copy(),
            speed=self.speed.copy(),
            queue=self.vehicles.copy(),
            time_sec=self.time_sec,
            step_count=self.step_count,
            total_entered=self.total_entered,
            total_exited=self.total_exited,
            queued_demand=float(np.sum(self.entrance_buffer)),
        )

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            state=self.state(),
            signal_phases=copy.deepcopy(self.signal_phases),
            signal_timers=copy.deepcopy(self.signal_timers),
            capacity_windows=copy.deepcopy(self.capacity_windows),
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, snapshot: EngineSnapshot) -> None:
        self.density = snapshot.state.density.copy()
        self.vehicles = snapshot.state.queue.copy()
        self.flow = snapshot.state.flow.copy()
        self.speed = snapshot.state.speed.copy()
        self.time_sec = snapshot.state.time_sec
        self.step_count = snapshot.state.step_count
        self.total_entered = snapshot.state.total_entered
        self.total_exited = snapshot.state.total_exited
        self.entrance_buffer.fill(0.0)
        self.signal_phases = copy.deepcopy(snapshot.signal_phases)
        self.signal_timers = copy.deepcopy(snapshot.signal_timers)
        self.capacity_windows = copy.deepcopy(snapshot.capacity_windows)
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
