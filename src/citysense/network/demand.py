"""Time-varying traffic demand generator with weekday/weekend patterns and peak surge profiles."""

from typing import Callable
from citysense.config.schemas import DemandConfig


class CorridorDemand:
    """Computes time-varying vehicular demand rate (veh/s) from DemandConfig."""

    def __init__(
        self,
        config: DemandConfig,
        is_weekend: bool = False,
        peak_shift_sec: float = 0.0,
    ) -> None:
        self.config = config
        self.is_weekend = is_weekend
        self.peak_shift_sec = peak_shift_sec
        self.day_factor = config.weekend_factor if is_weekend else config.weekday_factor

    def get_rate(self, time_sec: float) -> float:
        eff_time = time_sec - self.peak_shift_sec

        # Linear warmup phase
        if time_sec < self.config.warmup_duration_sec:
            # Clamp: a negative time (e.g. a caller shifting t backwards) must never yield negative demand.
            warmup_ratio = max(0.0, time_sec) / self.config.warmup_duration_sec
            return self.config.base_flow_veh_per_sec * self.day_factor * warmup_ratio

        # Peak hours window
        if self.config.peak_start_sec <= eff_time <= self.config.peak_end_sec:
            return self.config.peak_flow_veh_per_sec * self.day_factor

        # Standard base flow
        return self.config.base_flow_veh_per_sec * self.day_factor

    def as_callable(self) -> Callable[[float], float]:
        return self.get_rate
