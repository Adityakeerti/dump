"""Gate tests for Module 6: Shockwave Back-Tracing."""

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
from citysense.diagnosis.clustering import JamDetector
from citysense.diagnosis.shockwave import ShockwaveTracer
from citysense.diagnosis.types import JamCluster
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


def test_origin_cell_error_across_k_seeds():
    """Verify origin cell error is within tolerance across N seeded planted scenarios."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "recurring_capacity_drop.yaml")
    tols = load_tolerances()

    seeds = [42, 101, 777][: diag_cfg.evaluation_k_seeds]
    target_cell = sc_cfg.target_cell_id

    for s in seeds:
        net = build_corridor_network(net_cfg, seed=s)
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

        gen = ObservationGenerator(obs_cfg, n_cells=net.n_cells, seed=s)
        detector = JamDetector(diag_cfg)
        tracer = ShockwaveTracer(eng_cfg, diag_cfg)

        all_obs = []
        for _ in range(350):
            state = engine.step()
            all_obs.extend(gen.observe_step(state, eng_cfg.dt))

        ff_speeds = {i: float(net.vf[i]) for i in range(net.n_cells)}
        congested_pts = detector.identify_congested_points(all_obs, ff_speeds)
        clusters = detector.cluster_jams(congested_pts, net_cfg.min_cell_length)

        assert len(clusters) > 0, f"No jam detected for seed {s}"
        primary_cluster = max(clusters, key=lambda c: len(c.cell_ids))

        origin_res = tracer.trace_origin(primary_cluster, all_obs, net_cfg.min_cell_length)
        error = abs(origin_res.origin_cell - target_cell)

        assert error <= tols["origin_cell_tolerance"], (
            f"Seed {s}: Origin cell error {error} exceeds tolerance {tols['origin_cell_tolerance']}"
        )


def test_downstream_spillback_origin_disambiguation():
    """Verify that in spillback scenario, traced origin is downstream bottleneck and NOT upstream junction."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "downstream_spillback.yaml")
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
    detector = JamDetector(diag_cfg)
    tracer = ShockwaveTracer(eng_cfg, diag_cfg)

    all_obs = []
    for _ in range(400):
        state = engine.step()
        all_obs.extend(gen.observe_step(state, eng_cfg.dt))

    ff_speeds = {i: float(net.vf[i]) for i in range(net.n_cells)}
    congested_pts = detector.identify_congested_points(all_obs, ff_speeds)
    clusters = detector.cluster_jams(congested_pts, net_cfg.min_cell_length)
    assert len(clusters) > 0

    primary_cluster = max(clusters, key=lambda c: len(c.cell_ids))
    origin_res = tracer.trace_origin(primary_cluster, all_obs, net_cfg.min_cell_length)

    downstream_bottleneck = sc_cfg.target_cell_id
    upstream_junction = sc_cfg.upstream_junction_cell_id

    # The traced origin must match the downstream bottleneck, NOT the upstream junction
    origin_error = abs(origin_res.origin_cell - downstream_bottleneck)
    assert origin_error <= tols["origin_cell_tolerance"], (
        f"Traced origin {origin_res.origin_cell} differs from downstream bottleneck {downstream_bottleneck}"
    )
    assert origin_res.origin_cell != upstream_junction, (
        f"Traced origin erroneously pinned to upstream junction {upstream_junction}"
    )


def test_estimated_wave_speed_matches_theoretical():
    """Verify estimated backward wave speed matches theoretical w within tolerance."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    tracer = ShockwaveTracer(eng_cfg, diag_cfg)

    # Synthetic jam cluster and observation data
    cluster = JamCluster(
        cluster_id=0,
        cell_ids=[10, 11, 12],
        timestamps_sec=[100.0, 105.0],
        tail_cell=10,
        head_cell=12,
        start_time_sec=100.0,
        end_time_sec=105.0,
        extent_m=45.0,
        duration_sec=5.0,
        mean_speed_mps=3.0,
    )
    w_est = tracer.estimate_wave_speed(cluster, [])
    diff = abs(w_est - eng_cfg.wave_speed)
    assert diff <= diag_cfg.wave_speed_tolerance_mps
