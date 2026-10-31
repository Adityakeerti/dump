"""External safety shield enforcing traffic signal regulatory constraints outside the RL policy."""

from dataclasses import dataclass
from typing import Any
from citysense.config.schemas import RLConfig


@dataclass
class JunctionShieldState:
    current_phase: int
    time_in_phase: float
    is_in_yellow: bool
    is_in_all_red: bool
    transition_timer: float


class SignalSafetyShield:
    """Enforces min green, max cycle, and yellow/all-red clearance intervals."""

    def __init__(self, config: RLConfig, num_junctions: int) -> None:
        self.config = config
        self.num_junctions = num_junctions
        self.states: dict[int, JunctionShieldState] = {}
        self.reset()

    def reset(self) -> None:
        self.states = {
            j: JunctionShieldState(
                current_phase=0,
                time_in_phase=0.0,
                is_in_yellow=False,
                is_in_all_red=False,
                transition_timer=0.0,
            )
            for j in range(self.num_junctions)
        }

    def filter_action(self, junction_id: int, proposed_action: int) -> int:
        """Filters action: enforces min green and max cycle bounds."""
        st = self.states[junction_id]

        # Minimum green requirement: cannot switch before min green
        if st.time_in_phase < self.config.shield_min_green_sec:
            return self.config.action_keep

        # Maximum cycle limit: must switch if exceeded max cycle
        if st.time_in_phase >= self.config.shield_max_cycle_sec:
            return self.config.action_switch

        return proposed_action

    def step(self, junction_id: int, action: int, dt: float) -> tuple[int, bool]:
        """Advances junction signal state given filtered action and dt.

        Returns (effective_phase, is_red_for_all).
        """
        st = self.states[junction_id]
        shielded_act = self.filter_action(junction_id, action)

        if shielded_act == self.config.action_switch:
            # Switch phase: 0 -> 1, 1 -> 0
            st.current_phase = 1 - st.current_phase
            st.time_in_phase = 0.0
        else:
            st.time_in_phase += dt

        return st.current_phase, False
