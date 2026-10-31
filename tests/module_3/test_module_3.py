"""Gate tests for Module 3: Network and Scenario Builder."""

import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from citysense.config.loader import (
    load_citysense_config,
    load_demand_config,
    load_engine_config,
    load_network_config,
    load_project_config,
    load_scenario_config,
)
from citysense.engine.ctm import NativeCTMEngine
from citysense.network.corridor import build_corridor_network
from citysense.network.demand import CorridorDemand
from citysense.scenarios.registry import create_scenario


CONFIG_DIR = Path("config")
PROJECT_PATH = CONFIG_DIR / "project.yaml"
ENGINE_PATH = CONFIG_DIR / "engine.yaml"
NETWORK_PATH = CONFIG_DIR / "network.yaml"
DEMAND_PATH = CONFIG_DIR / "demand.yaml"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"
OBSERVATION_PATH = CONFIG_DIR / "observation.yaml"
DIAGNOSIS_PATH = CONFIG_DIR / "diagnosis.yaml"
RL_PATH = CONFIG_DIR / "rl.yaml"
API_PATH = CONFIG_DIR / "api.yaml"
TOLERANCES_PATH = Path("tests/tolerances.yaml")


def load_tolerances() -> dict[str, float]:
    with open(TOLERANCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v["value"] for k, v in data.items()}


def test_network_seed_determinism():
    """Verify that same seed gives identical network."""
    net_cfg = load_network_config(NETWORK_PATH)
    seed = 42

    net1 = build_corridor_network(net_cfg, seed=seed)
    net2 = build_corridor_network(net_cfg, seed=seed)

    np.testing.assert_array_equal(net1.upstream_cells, net2.upstream_cells)
    np.testing.assert_array_equal(net1.downstream_cells, net2.downstream_cells)
    np.testing.assert_array_equal(net1.lanes, net2.lanes)
    np.testing.assert_array_equal(net1.vf, net2.vf)
    np.testing.assert_array_equal(net1.q_nom, net2.q_nom)
    assert net1.metadata.num_cells == net2.metadata.num_cells
    assert net1.metadata.assumptions == net2.metadata.assumptions


def test_network_connectivity_and_junction_count():
    """Verify network is connected and has the requested junction count."""
    net_cfg = load_network_config(NETWORK_PATH)
    seed = 42
    net = build_corridor_network(net_cfg, seed=seed)

    assert len(net.junction_cells) == net_cfg.num_signalized_junctions
    assert net.metadata.num_signalized_junctions == net_cfg.num_signalized_junctions

    # Verify arterial cells form a continuous connected path
    arterial = net.arterial_cells
    for i in range(len(arterial) - 1):
        cell_curr = arterial[i]
        cell_next = arterial[i + 1]
        assert net.downstream_cells[cell_curr] == cell_next
        assert net.upstream_cells[cell_next] == cell_curr


def test_ground_truth_matches_config(tmp_path: Path):
    """Verify ground truth file matches config and is written to specified folder."""
    proj_cfg = load_project_config(PROJECT_PATH)
    sc_path = SCENARIOS_DIR / "recurring_capacity_drop.yaml"
    sc_cfg = load_scenario_config(sc_path)

    scenario = create_scenario(sc_cfg)
    gt_file = scenario.write_ground_truth(tmp_path)

    assert gt_file.exists()
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    assert gt_data["scenario_name"] == sc_cfg.name
    assert gt_data["cause_type"] == sc_cfg.scenario_type
    assert gt_data["target_cell_id"] == sc_cfg.target_cell_id
    assert gt_data["capacity_factor"] == sc_cfg.capacity_factor
    assert gt_data["time_window_sec"] == [sc_cfg.start_time_sec, sc_cfg.end_time_sec]


def test_control_scenario_has_no_capacity_edits():
    """Verify control scenario applies no capacity factor perturbations."""
    engine_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    sc_path = SCENARIOS_DIR / "control.yaml"
    sc_cfg = load_scenario_config(sc_path)

    net = build_corridor_network(net_cfg, seed=42)
    engine = NativeCTMEngine(
        config=engine_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )

    scenario = create_scenario(sc_cfg)
    scenario.apply_to_engine(engine, net)

    assert len(engine.capacity_windows) == 0


def test_capacity_drop_flow_reduction():
    """Verify capacity drop scenario produces measurably lower flow than control."""
    engine_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    tols = load_tolerances()

    net = build_corridor_network(net_cfg, seed=42)
    demand = CorridorDemand(dem_cfg)

    # Setup Control Engine
    engine_control = NativeCTMEngine(
        config=engine_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine_control.set_demand_profile(demand.as_callable())
    ctrl_scenario = create_scenario(load_scenario_config(SCENARIOS_DIR / "control.yaml"))
    ctrl_scenario.apply_to_engine(engine_control, net)

    # Setup Disrupted Engine
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "recurring_capacity_drop.yaml")
    engine_disrupt = NativeCTMEngine(
        config=engine_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine_disrupt.set_demand_profile(demand.as_callable())
    disrupt_scenario = create_scenario(sc_cfg)
    disrupt_scenario.apply_to_engine(engine_disrupt, net)

    # Run simulations past start_time_sec into disruption
    target_cell = sc_cfg.target_cell_id
    steps = 350
    for _ in range(steps):
        s_ctrl = engine_control.step()
        s_disrupt = engine_disrupt.step()

    ctrl_flow = s_ctrl.flow[target_cell]
    disrupt_flow = s_disrupt.flow[target_cell]

    flow_drop = (ctrl_flow - disrupt_flow) / max(ctrl_flow, 1e-6)
    threshold = tols["capacity_drop_flow_reduction"]
    assert flow_drop >= threshold, (
        f"Flow drop {flow_drop:.3f} did not meet required threshold {threshold}"
    )


def test_ground_truth_isolation_rule():
    """Enforce Hard Rule 7: diagnosis module must never import ground truth modules or files."""
    diag_dir = Path("src/citysense/diagnosis")
    if not diag_dir.exists():
        return

    for py_file in diag_dir.glob("**/*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            code = f.read()
        assert "ground_truth" not in code, f"Diagnosis file {py_file} references ground_truth!"
