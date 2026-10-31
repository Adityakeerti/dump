"""Gate tests for Module 2: Engine Adapter and CTM Implementation."""

from pathlib import Path
import numpy as np
import pytest
import yaml

from citysense.config.loader import load_engine_config
from citysense.config.schemas import EngineConfig
from citysense.engine.ctm import NativeCTMEngine


ENGINE_CFG_PATH = Path("config/engine.yaml")
TOLERANCES_PATH = Path("tests/tolerances.yaml")


def load_tolerances() -> dict[str, float]:
    with open(TOLERANCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v["value"] for k, v in data.items()}


def create_linear_engine(config: EngineConfig, n_cells: int = 20) -> NativeCTMEngine:
    upstream = np.arange(-1, n_cells - 1, dtype=np.int64)
    downstream = np.arange(1, n_cells + 1, dtype=np.int64)
    downstream[-1] = -1
    return NativeCTMEngine(
        config=config,
        n_cells=n_cells,
        upstream_cells=upstream,
        downstream_cells=downstream,
    )


def test_fundamental_diagram_sweep():
    """Verify that steady-state flow vs density matches triangular FD with R2 > tolerance."""
    config = load_engine_config(ENGINE_CFG_PATH)
    tols = load_tolerances()

    vf = config.free_flow_speed
    w = config.wave_speed
    kj = config.jam_density
    q_cap = config.capacity
    kc = q_cap / vf

    n_cells = 12
    observed_densities = []
    observed_flows = []
    theoretical_flows = []

    # Sweep demand rates for free-flow branch (0.05 to 0.45)
    free_flow_demands = np.linspace(0.05, q_cap * 0.95, 8)
    for dem in free_flow_demands:
        engine = create_linear_engine(config, n_cells=n_cells)
        engine.set_demand_profile(lambda t, d=dem: d)
        for _ in range(40):
            state = engine.step()
        mid_cell = n_cells // 2
        k_obs = state.density[mid_cell]
        q_obs = state.flow[mid_cell]
        observed_densities.append(k_obs)
        observed_flows.append(q_obs)
        # Theoretical free-flow
        q_theo = min(vf * k_obs, q_cap)
        theoretical_flows.append(q_theo)

    # Congested branch: high demand with downstream bottleneck to force congestion
    bottleneck_factors = [0.8, 0.6, 0.4, 0.25]
    for factor in bottleneck_factors:
        engine = create_linear_engine(config, n_cells=n_cells)
        engine.set_demand_profile(lambda t: q_cap * 0.9)
        engine.set_capacity_factor(n_cells - 1, factor, (0.0, 1000.0))
        for _ in range(60):
            state = engine.step()
        mid_cell = n_cells // 2
        k_obs = state.density[mid_cell]
        q_obs = state.flow[mid_cell]
        observed_densities.append(k_obs)
        observed_flows.append(q_obs)
        # Theoretical congested flow
        if k_obs > kc:
            q_theo = min(q_cap, w * (kj - k_obs))
        else:
            q_theo = vf * k_obs
        theoretical_flows.append(q_theo)

    obs_arr = np.array(observed_flows)
    theo_arr = np.array(theoretical_flows)
    ss_res = np.sum((obs_arr - theo_arr) ** 2)
    ss_tot = np.sum((obs_arr - np.mean(obs_arr)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)

    assert r2 >= tols["fd_r2"], f"FD R2 {r2:.4f} below tolerance {tols['fd_r2']}"


def test_conservation_of_vehicles():
    """Verify vehicles_in + initial = vehicles_out + remaining within tolerance."""
    config = load_engine_config(ENGINE_CFG_PATH)
    tols = load_tolerances()

    n_cells = 15
    engine = create_linear_engine(config, n_cells=n_cells)

    # Apply fluctuating demand profile
    engine.set_demand_profile(lambda t: 0.35 + 0.15 * np.sin(t * 0.1))

    initial_vehicles = float(np.sum(engine.vehicles))
    steps = 45
    for _ in range(steps):
        state = engine.step()

    total_entered = state.total_entered
    total_exited = state.total_exited
    remaining_vehicles = float(np.sum(engine.vehicles)) + state.queued_demand

    balance_error = abs((total_entered + initial_vehicles) - (total_exited + remaining_vehicles))
    assert balance_error <= tols["conservation"], (
        f"Conservation balance error {balance_error} exceeds tolerance {tols['conservation']}"
    )


def test_shockwave_speed():
    """Verify backward wave speed w_shock = (q2 - q1)/(k2 - k1) within tolerance."""
    config = load_engine_config(ENGINE_CFG_PATH)
    tols = load_tolerances()

    n_cells = 60
    engine = create_linear_engine(config, n_cells=n_cells)

    q1 = 0.40
    engine.set_demand_profile(lambda t: q1)

    # Establish initial steady state
    for _ in range(50):
        engine.step()

    k1 = q1 / config.free_flow_speed

    # Plant capacity drop at downstream cell
    bottleneck_cell = n_cells - 5
    factor = 0.4
    engine.set_capacity_factor(bottleneck_cell, factor, (engine.time_sec, 2000.0))
    q2 = config.capacity * factor
    k2 = config.jam_density - (q2 / config.wave_speed)

    # Theoretical shockwave speed (negative = propagating backward/upstream)
    w_theoretical = (q2 - q1) / (k2 - k1)
    speed_theoretical = abs(w_theoretical)

    k_threshold = (k1 + k2) / 2.0
    front_positions = []
    times = []

    for _ in range(80):
        s = engine.step()
        dens = s.density[:bottleneck_cell]
        crossings = np.where(dens >= k_threshold)[0]
        if len(crossings) > 0:
            c = crossings[0]
            if c > 0 and dens[c] != dens[c - 1]:
                frac = (k_threshold - dens[c - 1]) / (dens[c] - dens[c - 1])
                pos = (c - 1 + frac) * config.cell_length
            else:
                pos = c * config.cell_length
            front_positions.append(pos)
            times.append(s.time_sec)

    assert len(front_positions) >= 10, "Shockwave did not propagate across enough cells"

    slope, _ = np.polyfit(times, front_positions, 1)
    measured_speed = abs(slope)

    rel_error = abs(measured_speed - speed_theoretical) / speed_theoretical
    assert rel_error <= tols["shockwave_speed_relative_error"], (
        f"Measured speed {measured_speed:.3f} vs theoretical {speed_theoretical:.3f}, rel error {rel_error:.4f} > {tols['shockwave_speed_relative_error']}"
    )


def test_snapshot_restore_identical_future():
    """Verify snapshot/restore reproduces an identical future."""
    config = load_engine_config(ENGINE_CFG_PATH)
    engine = create_linear_engine(config, n_cells=15)
    engine.set_demand_profile(lambda t: 0.3)

    # Run for 15 steps
    for _ in range(15):
        engine.step()

    # Capture snapshot
    snap = engine.snapshot()

    # Continue for 15 steps (run A)
    run_a_densities = []
    for _ in range(15):
        s = engine.step()
        run_a_densities.append(s.density.copy())

    # Restore snapshot and continue for 15 steps (run B)
    engine.restore(snap)
    run_b_densities = []
    for _ in range(15):
        s = engine.step()
        run_b_densities.append(s.density.copy())

    for idx in range(len(run_a_densities)):
        np.testing.assert_allclose(run_a_densities[idx], run_b_densities[idx], atol=1e-12)


def test_capacity_factor_one_changes_nothing():
    """Verify capacity factor of 1.0 does not alter simulation trajectory."""
    config = load_engine_config(ENGINE_CFG_PATH)

    engine_baseline = create_linear_engine(config, n_cells=15)
    engine_factor1 = create_linear_engine(config, n_cells=15)

    demand = lambda t: 0.35
    engine_baseline.set_demand_profile(demand)
    engine_factor1.set_demand_profile(demand)

    engine_factor1.set_capacity_factor([5, 6], 1.0, (0.0, 500.0))

    for _ in range(25):
        s_base = engine_baseline.step()
        s_fact = engine_factor1.step()
        np.testing.assert_allclose(s_base.density, s_fact.density, atol=1e-12)
        np.testing.assert_allclose(s_base.flow, s_fact.flow, atol=1e-12)
