"""Configuration schema and loader for the live digital-twin streaming service."""

from pathlib import Path

from pydantic import model_validator

from citysense.config.loader import read_yaml
from citysense.config.schemas import StrictBaseModel


class LivePaths(StrictBaseModel):
    health: str
    topology: str
    state: str
    events: str
    control: str
    stream: str
    trips: str
    decisions: str
    episodes: str
    platoons: str
    diagnose: str
    diagnosis_mount: str


class LiveConfig(StrictBaseModel):
    host: str
    port: int
    tick_interval_sec: float
    default_steps_per_tick: int
    speed_options: list[int]
    autoplay: bool
    default_policy: str
    default_scenario: str
    episode_seed_increment: int
    background_demand_factor: float
    learn_online_default: bool
    epsilon_episode_decay: float
    epsilon_preview_episodes: int
    checkpoint_path: str
    episodes_path: str
    episode_history_size: int
    event_log_size: int
    jam_min_cells: int
    jam_confirm_sec: float
    jam_clear_sec: float
    decision_log_size: int
    recent_window_steps: int
    max_active_platoons: int
    max_trips_per_request: int
    max_vehicles_per_trip: float
    max_trip_delay_sec: float
    diagnosis_window_sec: float
    snapshot_every_steps: int
    diagnosis_lead_sec: float
    diagnosis_min_jam_age_sec: float
    diagnosis_forward_sec: float
    diagnosis_min_effect_vhr: float
    diagnosis_horizon_max_sec: float
    diagnosis_replicates: int
    diagnosis_evidence_limit: int
    diagnosis_max_origin_candidates: int
    diagnosis_surge_ratio: float
    history_size: int
    history_sample_every_steps: int
    precision_digits: int
    default_events_limit: int
    cors_origins: list[str]
    street_geometry_path: str | None = None
    map_center_lat: float | None = None
    map_center_lon: float | None = None
    paths: LivePaths

    @model_validator(mode="after")
    def validate_live(self) -> "LiveConfig":
        if self.tick_interval_sec <= 0.0:
            raise ValueError("tick_interval_sec must be strictly positive")
        if self.default_steps_per_tick not in self.speed_options:
            raise ValueError("default_steps_per_tick must be one of speed_options")
        if self.history_sample_every_steps < 1 or self.history_size < 1:
            raise ValueError("history_size and history_sample_every_steps must be >= 1")
        if (self.map_center_lat is None) != (self.map_center_lon is None):
            raise ValueError("map_center_lat and map_center_lon must be set together")
        return self


def load_live_config(path: str | Path) -> LiveConfig:
    return LiveConfig.model_validate(read_yaml(path))
