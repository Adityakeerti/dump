"""Pydantic configuration schemas with strict validation, extra-key forbidding, and CFL checks."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base model that strictly rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictBaseModel):
    project_name: str
    version: str
    max_fix_attempts: int
    seed: int
    recon_file_path: str
    decisions_file_path: str
    tolerances_file_path: str
    ground_truth_dir: str
    reports_dir: str
    config_dir: str
    outputs_dir: str


class EngineConfig(StrictBaseModel):
    dt: float
    cell_length: float
    free_flow_speed: float
    wave_speed: float
    jam_density: float
    capacity: float
    density_epsilon: float
    flow_epsilon: float
    conservation_tolerance: float
    fd_r2_tolerance: float
    shockwave_speed_tolerance: float
    capacity_factor_identity: float
    built_in_scenario: str
    test_steps: int
    min_green: float
    max_green: float
    yellow: float
    all_red: float
    lost_time: float

    @field_validator("capacity", "jam_density", "wave_speed", "free_flow_speed", "dt", "cell_length")
    @classmethod
    def validate_strictly_positive(cls, v: float, info: Any) -> float:
        if v <= 0.0:
            raise ValueError(f"{info.field_name} must be strictly positive, got {v}")
        return v

    @model_validator(mode="after")
    def validate_cfl_condition(self) -> "EngineConfig":
        cfl_distance = self.free_flow_speed * self.dt
        if cfl_distance > self.cell_length:
            raise ValueError(
                f"CFL condition violated: vf * dt ({cfl_distance}) > cell_length ({self.cell_length})"
            )
        return self


class NetworkConfig(StrictBaseModel):
    corridor_name: str
    center_lat: float
    center_lon: float
    radius_m: float
    bbox_north: float
    bbox_south: float
    bbox_east: float
    bbox_west: float
    num_signalized_junctions: int
    num_cells_per_link: int
    lanes: int
    default_free_flow_speed: float
    default_capacity: float
    default_jam_density: float
    default_wave_speed: float
    signal_cycle_time: float
    green_fraction: float
    min_cell_length: float
    # Side-road legs per junction (1 = T-junction, 2 = four-way) and cells per leg (None = num_cells_per_link).
    cross_street_legs: int = 1
    cross_cells_per_leg: int | None = None

    @field_validator(
        "radius_m",
        "cross_street_legs",
        "num_signalized_junctions",
        "num_cells_per_link",
        "lanes",
        "default_capacity",
        "default_jam_density",
        "default_wave_speed",
        "default_free_flow_speed",
        "signal_cycle_time",
        "min_cell_length",
    )
    @classmethod
    def validate_network_positives(cls, v: float, info: Any) -> float:
        if v <= 0.0:
            raise ValueError(f"{info.field_name} must be strictly positive, got {v}")
        return v


class DemandConfig(StrictBaseModel):
    base_flow_veh_per_sec: float
    peak_flow_veh_per_sec: float
    peak_start_sec: float
    peak_end_sec: float
    peak_shift_sec: float
    simulation_duration_sec: float
    weekday_factor: float
    weekend_factor: float
    warmup_duration_sec: float
    cooldown_duration_sec: float
    cross_street_flow_factor: float

    @field_validator("base_flow_veh_per_sec", "peak_flow_veh_per_sec", "simulation_duration_sec")
    @classmethod
    def validate_demand_positives(cls, v: float, info: Any) -> float:
        if v <= 0.0:
            raise ValueError(f"{info.field_name} must be strictly positive, got {v}")
        return v


class ScenarioConfig(StrictBaseModel):
    name: str
    scenario_type: str
    description: str
    start_time_sec: float
    end_time_sec: float
    target_cell_id: int
    upstream_junction_cell_id: int
    capacity_factor: float
    demand_shift_sec: float
    decay_rate_per_step: float
    recurring_weekdays: list[int]
    recurring_hours: list[int]
    ground_truth_filename: str

    @field_validator("capacity_factor")
    @classmethod
    def validate_capacity_factor(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"capacity_factor must be between 0.0 and 1.0, got {v}")
        return v


class ObservationConfig(StrictBaseModel):
    coverage_fraction: float
    reporting_interval_sec: float
    noise_type: str
    noise_std: float
    flow_noise_rel: float = 0.0  # relative Gaussian noise on reported flow (loop detectors miscount too)
    dropout_fraction: float
    min_reported_speed_mps: float
    max_reported_speed_mps: float
    simulated_flag: bool
    timestamp_format: str
    speed_column_name: str
    flow_column_name: str
    cell_column_name: str
    time_column_name: str
    source_column_name: str
    simulated_column_name: str
    density_column_name: str
    default_simulated_source: str
    csv_output_filename: str

    @field_validator("coverage_fraction", "dropout_fraction")
    @classmethod
    def validate_fractions(cls, v: float, info: Any) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0, got {v}")
        return v


class DiagnosisConfig(StrictBaseModel):
    congestion_speed_factor: float
    dbscan_eps: float
    dbscan_min_samples: int
    spatial_scale_m: float
    temporal_scale_s: float
    wave_speed_theoretical_mps: float
    wave_speed_tolerance_mps: float
    capacity_fit_min_samples: int
    change_point_penalty: float
    change_point_min_size: int
    confidence_min: float
    confidence_max: float
    hypothesis_weights: dict[str, float]
    calibration_learning_rate: float
    counterfactual_replicates: int
    counterfactual_perturbation_std: float
    counterfactual_noise_floor_vhr: float
    responsible_party_capacity: str
    responsible_party_demand: str
    responsible_party_signal: str
    action_template_capacity: str
    action_template_demand: str
    action_template_signal: str
    action_template_control: str
    default_status: str
    simulation_tag: str
    recurrent_jam_day_threshold: int
    single_day_jam_day_count: int
    evaluation_k_seeds: int
    creeping_delay_allowed_steps: int
    change_point_min_drop_ratio: float
    output_dir_key: str
    ladder_ranking_order: list[str]
    top_ranked_name_capacity: str
    top_ranked_name_signal: str
    top_ranked_name_demand: str
    top_ranked_name_control: str
    cause_name_recurring: str
    cause_name_creeping: str
    cause_name_demand_shift: str
    cause_name_spillback: str
    cause_name_control: str
    evidence_min_count: int
    ladder_min_recovery_diff: float
    spillback_recovery_margin: float
    min_control_max_recovery: float
    calibration_max_step_bound: float
    initial_calibration_weight: float
    min_calibrated_weight: float
    max_calibrated_weight: float
    outcome_target_coverage_rate: float
    ladder_noise_threshold: float
    ladder_spillback_diff_threshold: float
    report_html_filename: str
    heatmap_png_filename: str
    ladder_png_filename: str
    final_results_filename: str
    summary_title: str
    figure_dpi: int
    figure_width_heatmap: int
    figure_height_heatmap: int
    figure_width_ladder: int
    figure_height_ladder: int
    origin_marker_size: int
    jam_contour_levels: int
    barchart_alpha: float
    barchart_capsize: float
    zorder_overlay: int
    bottleneck_search_radius_cells: int
    simulated_banner_text: str
    status_ok: str
    status_fail: str
    status_pending: str
    status_running: str
    status_completed: str
    status_error: str


class RLConfig(StrictBaseModel):
    seed: int
    algorithm: str
    learning_rate: float
    gamma: float
    batch_size: int
    buffer_capacity: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    total_training_steps: int
    eval_episodes: int
    queue_norm_factor: float
    reward_queue_weight: float
    reward_worst_approach_weight: float
    shield_min_green_sec: float
    shield_max_cycle_sec: float
    shield_yellow_sec: float
    shield_all_red_sec: float
    hidden_dim: int
    checkpoint_filename: str
    baseline_algorithms: list[str]
    action_keep: int
    action_switch: int
    evaluation_steps: int
    smoke_train_steps: int
    num_phases_per_junction: int
    state_feature_dim_per_approach: int
    state_extra_dim: int
    checkpoint_metadata_format: str
    action_space_size: int


class APIConfig(StrictBaseModel):
    host: str
    port: int
    timeout_seconds: float
    max_concurrent_jobs: int
    job_retention_seconds: float
    status_poll_interval_seconds: float
    api_title: str
    api_version: str
    api_description: str
    job_queue_poll_sleep_sec: float
    concurrent_rehearsal_batch_size: int
    default_query_limit: int
    error_code_not_found: int
    error_code_bad_request: int
    error_code_conflict: int
    rehearsal_endpoint_path: str
    jobs_endpoint_path: str
    situation_endpoint_path: str
    health_endpoint_path: str


class CitySenseConfig(StrictBaseModel):
    project: ProjectConfig
    engine: EngineConfig
    network: NetworkConfig
    demand: DemandConfig
    scenarios: dict[str, ScenarioConfig]
    observation: ObservationConfig
    diagnosis: DiagnosisConfig
    rl: RLConfig
    api: APIConfig

    @model_validator(mode="after")
    def validate_cross_file_consistency(self) -> "CitySenseConfig":
        # Check cross-file CFL condition between engine dt and network min_cell_length
        cfl_net_dist = self.network.default_free_flow_speed * self.engine.dt
        if cfl_net_dist > self.network.min_cell_length:
            raise ValueError(
                f"Cross-file CFL violation: network vf * engine dt ({cfl_net_dist}) > min_cell_length ({self.network.min_cell_length})"
            )
        return self
