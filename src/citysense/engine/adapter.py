"""Abstract base interface for traffic simulation engine adapters."""

from abc import ABC, abstractmethod
from typing import Any

from citysense.engine.types import EngineSnapshot, SimulationState


class EngineAdapter(ABC):
    """Abstract interface gating all downstream traffic operations."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> SimulationState:
        """Reset simulation state to initial conditions."""

    @abstractmethod
    def step(self, signal_actions: dict[int, int] | None = None) -> SimulationState:
        """Advance the simulation by dt seconds with optional signal phase decisions."""

    @abstractmethod
    def state(self) -> SimulationState:
        """Return the current per-cell density, flow, and speed."""

    @abstractmethod
    def set_capacity_factor(
        self, cell_ids: list[int] | int, factor: float, time_window: tuple[float, float]
    ) -> None:
        """Apply dynamic capacity scaling factor to specified cells over a time window."""

    @abstractmethod
    def set_demand_profile(self, profile: Any) -> None:
        """Attach time-varying demand schedule."""

    @abstractmethod
    def set_signal_controller(self, controller: Any) -> None:
        """Attach traffic signal controller."""

    @abstractmethod
    def snapshot(self) -> EngineSnapshot:
        """Capture deep immutable snapshot for counterfactual branching."""

    @abstractmethod
    def restore(self, snapshot: EngineSnapshot) -> None:
        """Restore exact simulation state from a snapshot."""
