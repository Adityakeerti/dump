"""Gate tests for Module 4: Observation Layer."""

from pathlib import Path
import numpy as np
import pytest
import yaml

from citysense.config.loader import load_engine_config, load_observation_config
from citysense.engine.ctm import NativeCTMEngine
from citysense.observation.generator import ObservationGenerator
from citysense.observation.loader import load_speed_series
from citysense.observation.types import ObservationRecord


CONFIG_DIR = Path("config")
ENGINE_PATH = CONFIG_DIR / "engine.yaml"
OBSERVATION_PATH = CONFIG_DIR / "observation.yaml"
TOLERANCES_PATH = Path("tests/tolerances.yaml")


def load_tolerances() -> dict[str, float]:
    with open(TOLERANCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v["value"] for k, v in data.items()}


def test_dropout_fraction_matches_config():
    """Verify empirical dropout fraction matches config within tolerance."""
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    tols = load_tolerances()

    n_cells = 50
    gen = ObservationGenerator(obs_cfg, n_cells=n_cells, seed=123)

    # Count dropouts over many simulation steps
    num_steps = 200
    total_opportunities = 0
    actual_records = 0

    eng_cfg = load_engine_config(ENGINE_PATH)
    upstream = np.arange(-1, n_cells - 1, dtype=np.int64)
    downstream = np.arange(1, n_cells + 1, dtype=np.int64)
    downstream[-1] = -1
    engine = NativeCTMEngine(eng_cfg, n_cells, upstream, downstream)

    for _ in range(num_steps):
        s = engine.step()
        # Only evaluate on reporting steps
        if s.step_count % int(round(obs_cfg.reporting_interval_sec / eng_cfg.dt)) == 0:
            total_opportunities += len(gen.monitored_cells)
            recs = gen.observe_step(s, eng_cfg.dt)
            actual_records += len(recs)

    observed_dropout = (total_opportunities - actual_records) / total_opportunities
    error = abs(observed_dropout - obs_cfg.dropout_fraction)
    assert error <= tols["dropout_tolerance"], (
        f"Dropout error {error:.4f} exceeds tolerance {tols['dropout_tolerance']}"
    )


def test_noise_statistics_match_config():
    """Verify sample standard deviation of added sensor noise matches config within tolerance."""
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    tols = load_tolerances()

    n_cells = 30
    gen = ObservationGenerator(obs_cfg, n_cells=n_cells, seed=42)

    eng_cfg = load_engine_config(ENGINE_PATH)
    upstream = np.arange(-1, n_cells - 1, dtype=np.int64)
    downstream = np.arange(1, n_cells + 1, dtype=np.int64)
    downstream[-1] = -1
    engine = NativeCTMEngine(eng_cfg, n_cells, upstream, downstream)
    engine.set_demand_profile(lambda t: 0.3)

    # Zero-out dropout for pure noise assessment
    gen.config = obs_cfg.model_copy(update={"dropout_fraction": 0.0})

    # Warm up engine
    for _ in range(30):
        engine.step()

    errors = []
    for _ in range(100):
        s = engine.step()
        recs = gen.observe_step(s, eng_cfg.dt)
        for r in recs:
            true_spd = float(s.speed[r.cell_id])
            errors.append(r.speed_mps - true_spd)

    sample_std = float(np.std(errors))
    std_diff = abs(sample_std - obs_cfg.noise_std)
    assert std_diff <= tols["noise_std_tolerance"], (
        f"Noise std diff {std_diff:.4f} exceeds tolerance {tols['noise_std_tolerance']}"
    )


def test_loader_exact_round_trip(tmp_path: Path):
    """Verify loader maps generated CSV round trip exactly."""
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    eng_cfg = load_engine_config(ENGINE_PATH)

    n_cells = 20
    gen = ObservationGenerator(obs_cfg, n_cells=n_cells, seed=99)
    upstream = np.arange(-1, n_cells - 1, dtype=np.int64)
    downstream = np.arange(1, n_cells + 1, dtype=np.int64)
    downstream[-1] = -1
    engine = NativeCTMEngine(eng_cfg, n_cells, upstream, downstream)

    for _ in range(30):
        s = engine.step()
        gen.observe_step(s, eng_cfg.dt)

    assert len(gen.records) > 0

    csv_path = tmp_path / "test_observations.csv"
    gen.save_to_csv(csv_path)

    loaded = load_speed_series(csv_path, obs_cfg)
    assert len(loaded) == len(gen.records)

    for orig, loaded_r in zip(gen.records, loaded):
        assert orig.cell_id == loaded_r.cell_id
        assert pytest.approx(orig.timestamp_sec) == loaded_r.timestamp_sec
        assert pytest.approx(orig.speed_mps) == loaded_r.speed_mps
        assert pytest.approx(orig.flow_vps) == loaded_r.flow_vps
        assert pytest.approx(orig.density_vpm) == loaded_r.density_vpm
        assert orig.source == loaded_r.source
        assert orig.simulated == loaded_r.simulated


def test_simulated_flag_always_set():
    """Verify the simulated flag is always set for simulated data."""
    obs_cfg = load_observation_config(OBSERVATION_PATH)
    eng_cfg = load_engine_config(ENGINE_PATH)

    gen = ObservationGenerator(obs_cfg, n_cells=10, seed=5)
    upstream = np.arange(-1, 9, dtype=np.int64)
    downstream = np.arange(1, 11, dtype=np.int64)
    downstream[-1] = -1
    engine = NativeCTMEngine(eng_cfg, 10, upstream, downstream)

    for _ in range(15):
        s = engine.step()
        recs = gen.observe_step(s, eng_cfg.dt)
        for r in recs:
            assert r.simulated is True
            assert r.source == obs_cfg.default_simulated_source


def test_observations_never_contain_ground_truth_fields():
    """Verify observations schema and output dictionaries never contain ground-truth fields."""
    forbidden_terms = ["cause", "planted", "ground_truth", "bottleneck_factor", "shift"]
    rec = ObservationRecord(
        cell_id=1,
        timestamp_sec=10.0,
        speed_mps=12.0,
        flow_vps=0.4,
        density_vpm=0.03,
        source="SIMULATED",
        simulated=True,
    )
    rec_dict = rec.to_dict()
    for key in rec_dict:
        for term in forbidden_terms:
            assert term not in key.lower(), f"Observation record leaked ground-truth field: {key}"
