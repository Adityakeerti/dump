"""Data models for detected congestion points and spatio-temporal jam clusters."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CongestionPoint:
    cell_id: int
    timestamp_sec: float
    speed_mps: float
    day: int = 0


@dataclass
class JamCluster:
    cluster_id: int
    cell_ids: list[int]
    timestamps_sec: list[float]
    tail_cell: int
    head_cell: int
    start_time_sec: float
    end_time_sec: float
    extent_m: float
    duration_sec: float
    mean_speed_mps: float
    recurring: bool = False
    matched_days: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cell_count": len(self.cell_ids),
            "tail_cell": self.tail_cell,
            "head_cell": self.head_cell,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "extent_m": self.extent_m,
            "duration_sec": self.duration_sec,
            "mean_speed_mps": self.mean_speed_mps,
            "recurring": self.recurring,
            "matched_days": self.matched_days,
        }
