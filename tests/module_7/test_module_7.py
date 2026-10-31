"""Gate tests for Module 7: Capacity Fitting and Change Detection."""

from pathlib import Path
import numpy as np
import pytest
import yaml

from citysense.config.loader import (
    load_demand_config,
    load_diagnosis_config,
    load_engine_config,
    load_network_config,
    load_observation_config,
    load_scenario_config,
)
from citysense.diagnosis.capacity import CapacityEstimator
from citysense.engine.ctm import NativeCTMEngine
from citysense.network.corridor import build_corridor_network
from citysense.network.demand import CorridorDemand
from citysense.observation.generator import ObservationGenerator
from citysense.scenarios.registry import create_scenario


CONFIG_DIR = Path("config")
ENGINE_PATH = CONFIG_DIR / "engine.yaml"
NETWORK_PATH = CONFIG_DIR / "network.yaml"
DEMAND_PATH = CONFIG_DIR / "demand.yaml"
OBSERVATION_PATH = CONFIG_DIR / "observation.yaml"
DIAGNOSIS_PATH = CONFIG_DIR / "diagnosis.yaml"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"
TOLERANCES_PATH = Path("tests/tolerances.yaml")


def load_tolerances() -> dict[str, float]:
    with open(TOLERANCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v["value"] for k, v in data.items()}


def test_recovered_capacity_ratio_matches_planted():
    """Verify recovered capacity ratio matches planted ratio within tolerance."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "recurring_capacity_drop.yaml")
    tols = load_tolerances()

    net = build_corridor_network(net_cfg, seed=42)
    demand = CorridorDemand(dem_cfg)
    engine = NativeCTMEngine(
        config=eng_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine.set_demand_profile(demand.as_callable())
    scenario = create_scenario(sc_cfg)
    scenario.apply_to_engine(engine, net)

    gen = ObservationGenerator(obs_cfg, n_cells=net.n_cells, seed=42)
    # Ensure target cell is monitored
    if sc_cfg.target_cell_id not in gen.monitored_cells:
        gen.monitored_cells.append(sc_cfg.target_cell_id)

    estimator = CapacityEstimator(eng_cfg, diag_cfg)

    all_obs = []
    for _ in range(350):
        s = engine.step()
        all_obs.extend(gen.observe_step(s, eng_cfg.dt))

    fit_window = (sc_cfg.start_time_sec, sc_cfg.end_time_sec)
    res = estimator.fit_capacity(sc_cfg.target_cell_id, all_obs, fit_window)

    ratio_error = abs(res.capacity_ratio - sc_cfg.capacity_factor)
    assert ratio_error <= tols["capacity_ratio_tolerance"], (
        f"Capacity ratio error {ratio_error:.3f} exceeds tolerance {tols['capacity_ratio_tolerance']}"
    )
    assert res.significant_drop_detected is True
    assert res.confidence_interval[0] <= res.capacity_ratio <= res.confidence_interval[1]


def test_creeping_capacity_change_point_detected():
    """Verify creeping scenario change points are detected within allowed delay."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "creeping_capacity_drop.yaml")

    net = build_corridor_network(net_cfg, seed=42)
    demand = CorridorDemand(dem_cfg)
    engine = NativeCTMEngine(
        config=eng_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine.set_demand_profile(demand.as_callable())
    scenario = create_scenario(sc_cfg)
    scenario.apply_to_engine(engine, net)

    gen = ObservationGenerator(obs_cfg, n_cells=net.n_cells, seed=42)
    if sc_cfg.target_cell_id not in gen.monitored_cells:
        gen.monitored_cells.append(sc_cfg.target_cell_id)

    estimator = CapacityEstimator(eng_cfg, diag_cfg)

    all_obs = []
    # Step past disruption onset
    steps = 450
    for _ in range(steps):
        s = engine.step()
        all_obs.extend(gen.observe_step(s, eng_cfg.dt))

    target_obs = [r for r in all_obs if r.cell_id == sc_cfg.target_cell_id]
    change_pts = estimator.detect_change_points(
        target_obs, eng_cfg.capacity, warmup_sec=dem_cfg.warmup_duration_sec
    )

    assert len(change_pts) > 0, "No change points detected for creeping capacity drop"

    first_cp = change_pts[0]
    delay = first_cp - sc_cfg.start_time_sec
    max_allowed_delay = diag_cfg.creeping_delay_allowed_steps * eng_cfg.dt
    assert delay <= max_allowed_delay, (
        f"Change point delay {delay:.1f}s exceeded allowed limit {max_allowed_delay:.1f}s"
    )


def test_control_scenario_produces_no_significant_drop():
    """Verify control scenario produces no significant drop."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "control.yaml")

    net = build_corridor_network(net_cfg, seed=42)
    demand = CorridorDemand(dem_cfg)
    engine = NativeCTMEngine(
        config=eng_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine.set_demand_profile(demand.as_callable())
    scenario = create_scenario(sc_cfg)
    scenario.apply_to_engine(engine, net)

    gen = ObservationGenerator(obs_cfg, n_cells=net.n_cells, seed=42)
    estimator = CapacityEstimator(eng_cfg, diag_cfg)

    all_obs = []
    for _ in range(350):
        s = engine.step()
        all_obs.extend(gen.observe_step(s, eng_cfg.dt))

    # Test an arterial cell during peak window (300 to 350)
    test_cell = net.arterial_cells[len(net.arterial_cells) // 2]
    res = estimator.fit_capacity(test_cell, all_obs, (300.0, 350.0))

    assert res.significant_drop_detected is False
    assert res.capacity_ratio >= 0.85
