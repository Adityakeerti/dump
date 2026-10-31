"""Gate tests for Module 1: Configuration System."""

import copy
from pathlib import Path
import pytest
from pydantic import ValidationError

from citysense.config.loader import (
    compute_config_hash,
    load_citysense_config,
    load_engine_config,
    read_yaml,
)
from citysense.config.schemas import CitySenseConfig, EngineConfig
from tests.test_no_hardcoding import test_no_hardcoding_in_src


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


def test_valid_configs_load():
    """Verify all project YAML configs load and validate cleanly."""
    cfg, cfg_hash = load_citysense_config(
        project_path=PROJECT_PATH,
        engine_path=ENGINE_PATH,
        network_path=NETWORK_PATH,
        demand_path=DEMAND_PATH,
        scenarios_dir=SCENARIOS_DIR,
        observation_path=OBSERVATION_PATH,
        diagnosis_path=DIAGNOSIS_PATH,
        rl_path=RL_PATH,
        api_path=API_PATH,
    )
    assert isinstance(cfg, CitySenseConfig)
    assert isinstance(cfg_hash, str)
    assert len(cfg_hash) == 64
    assert len(cfg.scenarios) >= 5
    assert "control" in cfg.scenarios
    assert "recurring_capacity_drop" in cfg.scenarios


def test_missing_key_raises_error():
    """Verify that omitting a required key raises a ValidationError."""
    raw = read_yaml(ENGINE_PATH)
    del raw["dt"]
    with pytest.raises(ValidationError) as excinfo:
        EngineConfig.model_validate(raw)
    assert "dt" in str(excinfo.value)


def test_negative_capacity_raises_error():
    """Verify that negative capacity raises a ValidationError."""
    raw = read_yaml(ENGINE_PATH)
    raw["capacity"] = -0.5
    with pytest.raises(ValidationError) as excinfo:
        EngineConfig.model_validate(raw)
    assert "capacity" in str(excinfo.value)


def test_cfl_violation_raises_error():
    """Verify that CFL violation (vf * dt > cell_length) raises a ValidationError."""
    raw = read_yaml(ENGINE_PATH)
    # Cell length 10 is less than vf (13.89) * dt (1.0) = 13.89
    raw["cell_length"] = 10.0
    with pytest.raises(ValidationError) as excinfo:
        EngineConfig.model_validate(raw)
    assert "CFL condition violated" in str(excinfo.value)


def test_unknown_key_raises_error():
    """Verify that unknown/extra keys are rejected."""
    raw = read_yaml(ENGINE_PATH)
    raw["unexpected_extra_parameter"] = 123.45
    with pytest.raises(ValidationError) as excinfo:
        EngineConfig.model_validate(raw)
    assert "extra_forbidden" in str(excinfo.value)


def test_config_hash_stability_and_sensitivity():
    """Verify config hash is deterministic and sensitive to value changes."""
    cfg1, hash1 = load_citysense_config(
        project_path=PROJECT_PATH,
        engine_path=ENGINE_PATH,
        network_path=NETWORK_PATH,
        demand_path=DEMAND_PATH,
        scenarios_dir=SCENARIOS_DIR,
        observation_path=OBSERVATION_PATH,
        diagnosis_path=DIAGNOSIS_PATH,
        rl_path=RL_PATH,
        api_path=API_PATH,
    )
    cfg2, hash2 = load_citysense_config(
        project_path=PROJECT_PATH,
        engine_path=ENGINE_PATH,
        network_path=NETWORK_PATH,
        demand_path=DEMAND_PATH,
        scenarios_dir=SCENARIOS_DIR,
        observation_path=OBSERVATION_PATH,
        diagnosis_path=DIAGNOSIS_PATH,
        rl_path=RL_PATH,
        api_path=API_PATH,
    )
    assert hash1 == hash2

    # Mutate a single value and verify hash changes
    mutated_engine = cfg1.engine.model_copy(update={"cell_length": 25.0})
    mutated_cfg = cfg1.model_copy(update={"engine": mutated_engine})
    mutated_hash = compute_config_hash(mutated_cfg)
    assert mutated_hash != hash1


def test_no_hardcoding_active_check():
    """Verify the no-hardcoding scanner executes and passes on current codebase."""
    test_no_hardcoding_in_src()
