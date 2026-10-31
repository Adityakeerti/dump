import os
import yaml
import pytest


def test_environment_imports_cleanly():
    """Verify all allowed third-party libraries can be imported cleanly."""
    import numpy as np
    import scipy
    import sklearn
    import networkx as nx
    import pydantic
    import yaml
    import pytest
    import torch
    import gymnasium
    import fastapi
    import uvicorn
    import httpx
    import osmnx
    import matplotlib
    import lightsim

    assert np is not None
    assert scipy is not None
    assert sklearn is not None
    assert nx is not None
    assert pydantic is not None
    assert yaml is not None
    assert pytest is not None
    assert torch is not None
    assert gymnasium is not None
    assert fastapi is not None
    assert uvicorn is not None
    assert httpx is not None
    assert osmnx is not None
    assert matplotlib is not None
    assert lightsim is not None


def test_recon_file_exists_and_contains_required_sections():
    """Verify docs/engine_recon.md exists and contains required sections."""
    with open("config/project.yaml", "r", encoding="utf-8") as f:
        proj_cfg = yaml.safe_load(f)

    recon_path = proj_cfg["recon_file_path"]
    assert os.path.exists(recon_path), f"Missing recon file at {recon_path}"

    with open(recon_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_topics = [
        "Public Interface",
        "OpenStreetMap",
        "Demand Generation",
        "Signal",
        "Density and Flow",
        "Runtime Capacity",
    ]
    for topic in required_topics:
        assert topic in content, f"Recon file missing topic: {topic}"


def test_builtin_scenario_runs_from_config():
    """Verify a built-in lightsim scenario runs for a number of steps configured in YAML."""
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        engine_cfg = yaml.safe_load(f)

    scenario_name = engine_cfg["built_in_scenario"]
    steps = int(engine_cfg["test_steps"])
    dt = float(engine_cfg["dt"])

    from lightsim.benchmarks.scenarios import get_scenario
    from lightsim.core.engine import SimulationEngine

    factory = get_scenario(scenario_name)
    network, demand_profiles = factory()

    engine = SimulationEngine(network=network, dt=dt, demand_profiles=demand_profiles)
    state = engine.reset()

    for _ in range(steps):
        state = engine.step()

    assert state.step_count == steps
    assert state.density.shape[0] > 0
