"""Typed observation records representing sensor measurements."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ObservationRecord:
    cell_id: int
    timestamp_sec: float
    speed_mps: float
    flow_vps: float
    density_vpm: float
    source: str
    simulated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
