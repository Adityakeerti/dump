"""Configuration loader with YAML parsing, Pydantic validation, and SHA-256 hash computation."""

import hashlib
import json
from pathlib import Path
from typing import Any
import yaml

from citysense.config.schemas import (
    APIConfig,
    CitySenseConfig,
    DemandConfig,
    DiagnosisConfig,
    EngineConfig,
    NetworkConfig,
    ObservationConfig,
    ProjectConfig,
    RLConfig,
    ScenarioConfig,
)


def read_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_project_config(path: str | Path) -> ProjectConfig:
    return ProjectConfig.model_validate(read_yaml(path))


def load_engine_config(path: str | Path) -> EngineConfig:
    return EngineConfig.model_validate(read_yaml(path))


def load_network_config(path: str | Path) -> NetworkConfig:
    return NetworkConfig.model_validate(read_yaml(path))


def load_demand_config(path: str | Path) -> DemandConfig:
    return DemandConfig.model_validate(read_yaml(path))


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    return ScenarioConfig.model_validate(read_yaml(path))


def load_observation_config(path: str | Path) -> ObservationConfig:
    return ObservationConfig.model_validate(read_yaml(path))


def load_diagnosis_config(path: str | Path) -> DiagnosisConfig:
    return DiagnosisConfig.model_validate(read_yaml(path))


def load_rl_config(path: str | Path) -> RLConfig:
    return RLConfig.model_validate(read_yaml(path))


def load_api_config(path: str | Path) -> APIConfig:
    return APIConfig.model_validate(read_yaml(path))


def load_all_scenarios(scenarios_dir: str | Path) -> dict[str, ScenarioConfig]:
    p = Path(scenarios_dir)
    scenarios: dict[str, ScenarioConfig] = {}
    for item in sorted(p.iterdir()):
        if item.is_file() and (item.suffix.lower() == ".yaml" or item.suffix.lower() == ".yml"):
            sc = load_scenario_config(item)
            scenarios[sc.name] = sc
    return scenarios


def compute_config_hash(config: CitySenseConfig) -> str:
    """Computes a stable SHA-256 hash of the canonical serialized config dictionary."""
    dumped = config.model_dump()
    canonical_json = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_citysense_config(
    project_path: str | Path,
    engine_path: str | Path,
    network_path: str | Path,
    demand_path: str | Path,
    scenarios_dir: str | Path,
    observation_path: str | Path,
    diagnosis_path: str | Path,
    rl_path: str | Path,
    api_path: str | Path,
) -> tuple[CitySenseConfig, str]:
    project = load_project_config(project_path)
    engine = load_engine_config(engine_path)
    network = load_network_config(network_path)
    demand = load_demand_config(demand_path)
    scenarios = load_all_scenarios(scenarios_dir)
    observation = load_observation_config(observation_path)
    diagnosis = load_diagnosis_config(diagnosis_path)
    rl = load_rl_config(rl_path)
    api = load_api_config(api_path)

    full_cfg = CitySenseConfig(
        project=project,
        engine=engine,
        network=network,
        demand=demand,
        scenarios=scenarios,
        observation=observation,
        diagnosis=diagnosis,
        rl=rl,
        api=api,
    )
    cfg_hash = compute_config_hash(full_cfg)
    return full_cfg, cfg_hash
