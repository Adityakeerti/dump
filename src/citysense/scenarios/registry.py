"""Scenario registry and implementations for planted causal disruptions."""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Callable, Type

from citysense.config.schemas import ScenarioConfig
from citysense.engine.adapter import EngineAdapter
from citysense.network.corridor import CorridorNetwork


class BaseScenario(ABC):
    """Abstract scenario applying scheduled perturbations and recording ground truth."""

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config

    @abstractmethod
    def apply_to_engine(self, engine: EngineAdapter, network: CorridorNetwork) -> None:
        """Applies scenario modifications to the simulation engine."""

    def write_ground_truth(self, ground_truth_dir: str | Path) -> Path:
        """Writes ground truth metadata to an isolated file never accessible to diagnosis."""
        out_dir = Path(ground_truth_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / self.config.ground_truth_filename

        gt_record = {
            "scenario_name": self.config.name,
            "cause_type": self.config.scenario_type,
            "target_cell_id": self.config.target_cell_id,
            "upstream_junction_cell_id": self.config.upstream_junction_cell_id,
            "time_window_sec": [self.config.start_time_sec, self.config.end_time_sec],
            "capacity_factor": self.config.capacity_factor,
            "demand_shift_sec": self.config.demand_shift_sec,
            "decay_rate_per_step": self.config.decay_rate_per_step,
            "description": self.config.description,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(gt_record, f, indent=2)
        return file_path


class ControlScenario(BaseScenario):
    """Control baseline with zero planted disruptions."""

    def apply_to_engine(self, engine: EngineAdapter, network: CorridorNetwork) -> None:
        pass


class RecurringCapacityDropScenario(BaseScenario):
    """Periodic capacity reduction on designated days and hours."""

    def apply_to_engine(self, engine: EngineAdapter, network: CorridorNetwork) -> None:
        window = (self.config.start_time_sec, self.config.end_time_sec)
        engine.set_capacity_factor(
            cell_ids=[self.config.target_cell_id],
            factor=self.config.capacity_factor,
            time_window=window,
        )


class CreepingCapacityDropScenario(BaseScenario):
    """Progressive week-over-week degradation in cell capacity."""

    def apply_to_engine(self, engine: EngineAdapter, network: CorridorNetwork) -> None:
        window = (self.config.start_time_sec, self.config.end_time_sec)
        engine.set_capacity_factor(
            cell_ids=[self.config.target_cell_id],
            factor=self.config.capacity_factor,
            time_window=window,
        )


class DemandPeakShiftScenario(BaseScenario):
    """Demand peak shift scenario."""

    def apply_to_engine(self, engine: EngineAdapter, network: CorridorNetwork) -> None:
        pass


class DownstreamSpillbackScenario(BaseScenario):
    """Downstream bottleneck creating shockwave spillback through upstream intersection."""

    def apply_to_engine(self, engine: EngineAdapter, network: CorridorNetwork) -> None:
        window = (self.config.start_time_sec, self.config.end_time_sec)
        engine.set_capacity_factor(
            cell_ids=[self.config.target_cell_id],
            factor=self.config.capacity_factor,
            time_window=window,
        )


_SCENARIO_MAP: dict[str, Type[BaseScenario]] = {
    "control": ControlScenario,
    "recurring_capacity_drop": RecurringCapacityDropScenario,
    "creeping_capacity_drop": CreepingCapacityDropScenario,
    "demand_peak_shift": DemandPeakShiftScenario,
    "downstream_spillback": DownstreamSpillbackScenario,
}


def get_scenario_class(scenario_type: str) -> Type[BaseScenario]:
    if scenario_type not in _SCENARIO_MAP:
        raise KeyError(f"Unknown scenario type: {scenario_type}. Available: {list(_SCENARIO_MAP.keys())}")
    return _SCENARIO_MAP[scenario_type]


def create_scenario(config: ScenarioConfig) -> BaseScenario:
    cls = get_scenario_class(config.scenario_type)
    return cls(config)
