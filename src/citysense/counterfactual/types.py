"""Data models for counterfactual ladder evaluation and hypothesis ranking."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CounterfactualRungResult:
    rung_name: str
    mean_recovery_vhr: float
    ci_lower_vhr: float
    ci_upper_vhr: float
    replicate_recoveries: list[float]
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung_name": self.rung_name,
            "mean_recovery_vhr": round(self.mean_recovery_vhr, 2),
            "ci_lower_vhr": round(self.ci_lower_vhr, 2),
            "ci_upper_vhr": round(self.ci_upper_vhr, 2),
            "rank": self.rank,
        }


@dataclass
class CounterfactualLadderResult:
    scenario_name: str
    rungs: list[CounterfactualRungResult]
    top_ranked_rung: str
    baseline_total_vhr: float
    noise_floor_vhr: float
    explains_jam: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "top_ranked_rung": self.top_ranked_rung,
            "baseline_total_vhr": round(self.baseline_total_vhr, 2),
            "noise_floor_vhr": round(self.noise_floor_vhr, 2),
            "explains_jam": self.explains_jam,
            "rungs": [r.to_dict() for r in self.rungs],
        }
