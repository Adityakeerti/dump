"""Gate tests for Module 5: Jam Detection and Clustering."""

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
from citysense.diagnosis.types import CongestionPoint
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


def test_jam_recovery_precision_and_recall():
    """Verify jam recovery against ground truth congested mask meets precision and recall tolerances."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    tols = load_tolerances()

    net = build_corridor_network(net_cfg, seed=42)
    demand = CorridorDemand(dem_cfg)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "recurring_capacity_drop.yaml")

    engine = NativeCTMEngine(
        config=eng_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine.set_demand_profile(demand.as_callable())
    scenario = create_scenario(sc_cfg)
    scenario.apply_to_engine(engine, net)

    gen = ObservationGenerator(obs_cfg, n_cells=net.n_cells, seed=123)
    detector = JamDetector(diag_cfg)

    ground_truth_congested = set()
    all_observations = []

    steps = 450
    for _ in range(steps):
        s = engine.step()
        recs = gen.observe_step(s, eng_cfg.dt)
        all_observations.extend(recs)

        # True congestion: true speed < factor * vf
        for cid in gen.monitored_cells:
            if s.speed[cid] < net.vf[cid] * diag_cfg.congestion_speed_factor:
                ground_truth_congested.add((cid, round(s.time_sec, 1)))

    ff_speeds = {i: float(net.vf[i]) for i in range(net.n_cells)}
    detected_points = detector.identify_congested_points(all_observations, ff_speeds)
    detected_set = {(p.cell_id, round(p.timestamp_sec, 1)) for p in detected_points}

    observed_eval_set = {
        (r.cell_id, round(r.timestamp_sec, 1)) for r in all_observations
    }
    gt_eval_set = ground_truth_congested.intersection(observed_eval_set)

    tp = len(detected_set.intersection(gt_eval_set))
    fp = len(detected_set - gt_eval_set)
    fn = len(gt_eval_set - detected_set)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    assert precision >= tols["jam_precision_min"], (
        f"Precision {precision:.3f} below tolerance {tols['jam_precision_min']}"
    )
    assert recall >= tols["jam_recall_min"], (
        f"Recall {recall:.3f} below tolerance {tols['jam_recall_min']}"
    )

    clusters = detector.cluster_jams(detected_points, net_cfg.min_cell_length)
    assert len(clusters) > 0, "DBSCAN failed to form jam cluster"


def test_control_scenario_false_alarms():
    """Verify control scenario produces false alarm rate below false_alarm_max."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    tols = load_tolerances()

    net = build_corridor_network(net_cfg, seed=42)
    demand = CorridorDemand(dem_cfg)
    ctrl_cfg = load_scenario_config(SCENARIOS_DIR / "control.yaml")

    engine = NativeCTMEngine(
        config=eng_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine.set_demand_profile(demand.as_callable())
    scenario = create_scenario(ctrl_cfg)
    scenario.apply_to_engine(engine, net)

    gen = ObservationGenerator(obs_cfg, n_cells=net.n_cells, seed=42)
    detector = JamDetector(diag_cfg)

    all_obs = []
    for _ in range(250):
        s = engine.step()
        recs = gen.observe_step(s, eng_cfg.dt)
        all_obs.extend(recs)

    ff_speeds = {i: float(net.vf[i]) for i in range(net.n_cells)}
    arterial_obs = [r for r in all_obs if r.cell_id in net.arterial_cells]
    detected_points = detector.identify_congested_points(arterial_obs, ff_speeds)
    false_alarm_rate = len(detected_points) / max(1, len(arterial_obs))

    assert false_alarm_rate <= tols["false_alarm_max"], (
        f"False alarm rate {false_alarm_rate:.4f} exceeds {tols['false_alarm_max']}"
    )

    # In control, clustering should produce no jam cluster along arterial corridor
    clusters = detector.cluster_jams(detected_points, net_cfg.min_cell_length)
    assert len(clusters) == 0, f"Expected 0 jam clusters on control run, got {len(clusters)}"


def test_recurring_vs_single_day_matching():
    """Verify recurring jam scenario is marked recurring and single day is not."""
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    detector = JamDetector(diag_cfg)

    # Create dummy jam on day 1
    pt_day1 = [
        CongestionPoint(cell_id=10, timestamp_sec=300.0 + i * 5, speed_mps=4.0, day=1)
        for i in range(10)
    ]
    pt_day2 = [
        CongestionPoint(cell_id=10, timestamp_sec=300.0 + i * 5, speed_mps=4.0, day=2)
        for i in range(10)
    ]

    jams_day1 = detector.cluster_jams(pt_day1, 15.0)
    jams_day2 = detector.cluster_jams(pt_day2, 15.0)

    # Test single-day scenario
    single_res = detector.match_recurring_jams({1: jams_day1})
    assert len(single_res) > 0
    assert single_res[0].recurring is False

    # Test multi-day recurring scenario
    multi_res = detector.match_recurring_jams({1: jams_day1, 2: jams_day2})
    assert len(multi_res) > 0
    assert multi_res[0].recurring is True
    assert set(multi_res[0].matched_days) == {1, 2}
