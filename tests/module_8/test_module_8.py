"""Gate tests for Module 8: Counterfactual Ladder."""

from pathlib import Path
import numpy as np
import pytest

from citysense.config.loader import (
    load_demand_config,
    load_diagnosis_config,
    load_engine_config,
    load_network_config,
    load_scenario_config,
)
from citysense.counterfactual.ladder import CounterfactualLadder
from citysense.engine.ctm import NativeCTMEngine
from citysense.network.corridor import build_corridor_network
from citysense.network.demand import CorridorDemand
from citysense.scenarios.registry import create_scenario


CONFIG_DIR = Path("config")
ENGINE_PATH = CONFIG_DIR / "engine.yaml"
NETWORK_PATH = CONFIG_DIR / "network.yaml"
DEMAND_PATH = CONFIG_DIR / "demand.yaml"
DIAGNOSIS_PATH = CONFIG_DIR / "diagnosis.yaml"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"


def test_planted_capacity_drop_ranks_restore_capacity_first():
    """Verify on planted capacity drop, the restore-capacity rung ranks first."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "recurring_capacity_drop.yaml")

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

    # Step past onset of disruption into active congestion queue
    steps = int(sc_cfg.start_time_sec / eng_cfg.dt) + 50
    for _ in range(steps):
        engine.step()

    snapshot = engine.snapshot()
    ladder = CounterfactualLadder(eng_cfg, diag_cfg)

    res = ladder.evaluate(
        engine=engine,
        network=net,
        base_demand=demand,
        suspect_cell=sc_cfg.target_cell_id,
        snapshot=snapshot,
        horizon_steps=80,
        scenario_name=sc_cfg.name,
        seed=42,
    )

    assert res.explains_jam is True
    assert res.top_ranked_rung == diag_cfg.top_ranked_name_capacity
    assert res.rungs[0].rung_name == diag_cfg.top_ranked_name_capacity
    assert res.rungs[0].mean_recovery_vhr > res.rungs[1].mean_recovery_vhr


def test_spillback_scenario_signal_recovers_less_than_downstream_capacity():
    """Verify signal rung recovers significantly less than downstream capacity rung in spillback."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "downstream_spillback.yaml")

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

    steps = int(sc_cfg.start_time_sec / eng_cfg.dt) + 50
    for _ in range(steps):
        engine.step()

    snapshot = engine.snapshot()
    ladder = CounterfactualLadder(eng_cfg, diag_cfg)

    res = ladder.evaluate(
        engine=engine,
        network=net,
        base_demand=demand,
        suspect_cell=sc_cfg.target_cell_id,
        snapshot=snapshot,
        horizon_steps=80,
        scenario_name=sc_cfg.name,
        seed=42,
    )

    cap_rec = next(
        r.mean_recovery_vhr
        for r in res.rungs
        if r.rung_name == diag_cfg.top_ranked_name_capacity
    )
    sig_rec = next(
        r.mean_recovery_vhr
        for r in res.rungs
        if r.rung_name == diag_cfg.top_ranked_name_signal
    )

    margin = diag_cfg.ladder_min_recovery_diff
    assert cap_rec - sig_rec >= margin, (
        f"Downstream capacity recovery ({cap_rec:.3f}) did not exceed signal recovery ({sig_rec:.3f}) by margin {margin}"
    )


def test_control_scenario_recovery_below_noise_floor():
    """Verify control scenario shows no rung with recovery above noise floor."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
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

    for _ in range(50):
        engine.step()

    snapshot = engine.snapshot()
    ladder = CounterfactualLadder(eng_cfg, diag_cfg)

    res = ladder.evaluate(
        engine=engine,
        network=net,
        base_demand=demand,
        suspect_cell=net.arterial_cells[len(net.arterial_cells) // 2],
        snapshot=snapshot,
        horizon_steps=50,
        scenario_name=sc_cfg.name,
        seed=42,
    )

    assert res.explains_jam is False
    assert res.top_ranked_rung == diag_cfg.top_ranked_name_control
    for r in res.rungs:
        assert r.mean_recovery_vhr <= diag_cfg.counterfactual_noise_floor_vhr


def test_ladder_determinism_across_reruns_with_same_seed():
    """Verify counterfactual ladder results are identical across reruns with the same seed."""
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    diag_cfg = load_diagnosis_config(DIAGNOSIS_PATH)
    sc_cfg = load_scenario_config(SCENARIOS_DIR / "recurring_capacity_drop.yaml")

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

    for _ in range(100):
        engine.step()

    snapshot = engine.snapshot()
    ladder = CounterfactualLadder(eng_cfg, diag_cfg)

    res1 = ladder.evaluate(
        engine=engine,
        network=net,
        base_demand=demand,
        suspect_cell=sc_cfg.target_cell_id,
        snapshot=snapshot,
        horizon_steps=40,
        scenario_name=sc_cfg.name,
        seed=123,
    )
    res2 = ladder.evaluate(
        engine=engine,
        network=net,
        base_demand=demand,
        suspect_cell=sc_cfg.target_cell_id,
        snapshot=snapshot,
        horizon_steps=40,
        scenario_name=sc_cfg.name,
        seed=123,
    )

    assert res1.top_ranked_rung == res2.top_ranked_rung
    for r1, r2 in zip(res1.rungs, res2.rungs):
        assert r1.rung_name == r2.rung_name
        assert r1.mean_recovery_vhr == pytest.approx(r2.mean_recovery_vhr, abs=1e-12)
        assert r1.ci_lower_vhr == pytest.approx(r2.ci_lower_vhr, abs=1e-12)
        assert r1.ci_upper_vhr == pytest.approx(r2.ci_upper_vhr, abs=1e-12)
