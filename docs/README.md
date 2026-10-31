# CitySense Traffic: Macroscopic Traffic Digital Twin

> **[SIMULATED TWIN EVIDENCE]**
> All traffic simulations, vehicle-hours recovered metrics, jam detections, and policy benchmarks are derived from a macroscopic Cell Transmission Model (CTM) digital twin. RL is employed as an adaptive signal control benchmark within the counterfactual ladder, not as an oracle root-cause classifier.

---

## 1. Overview
**CitySense Traffic** is an autonomous macroscopic traffic twin designed to determine the **root cause** of recurring or unexpected urban arterial traffic jams, evaluate counterfactual interventions (capacity restoration, demand shifting, signal control), and rank causal hypotheses deterministically.

A jam observed on one road link is frequently a symptom of an underlying disruption elsewhere in the corridor:
1. **Downstream Spillback:** A downstream bottleneck causes queues to propagate backward, blocking upstream intersections.
2. **Recurring Capacity Drop:** Physical lane reductions (e.g. delivery double-parking, loading zones) recurring on fixed weekdays/hours.
3. **Creeping Capacity Drop:** Progressive physical degradation of corridor capacity over successive periods.
4. **Demand Peak Shift:** Spatial-temporal surges in traffic demand overwhelming nominal capacity.

The twin diagnoses these causes deterministically using traffic physics (Rankine-Hugoniot shockwave back-tracing, CUSUM capacity fitting, and a counterfactual simulation ladder), completely isolating ground truth from diagnosis logic.

---

## 2. Architecture & Pipeline

```
[Corridor Network & Demand (CTM)]
               │
               ▼
   [Sensor Observation Layer] (Coverage, Dropout, Gaussian Noise)
               │
               ▼
    [Jam Detection & DBSCAN] (Spatio-Temporal Clustering)
               │
               ▼
   [Shockwave Back-Tracing] (Rankine-Hugoniot Discharge Front Localization)
               │
               ▼
   [Effective Capacity Fit] (Triangular FD Fitting & CUSUM Change-Points)
               │
               ▼
   [Counterfactual Ladder] (Capacity Restore vs Demand Shift vs RL Signal)
               │
               ▼
    [Situation Record Entity] (Deterministic Ranking & Evidence Trail)
        │                 │
        ▼                 ▼
[Outcome Tracker]   [Visual Reports & FastAPI Rehearsal Service]
```

---

## 3. Setup & Installation

### Requirements
- Linux OS
- Python 3.11+
- `uv` package manager (recommended) or virtual environment

### Installation via uv
```bash
# Clone repository
cd "CITY SENSE RL"

# Create virtualenv and install dependencies
uv sync
```

### Key Dependencies
- `numpy`, `scipy`, `scikit-learn`, `networkx`: Core numerics, clustering, and graph topology.
- `torch`, `gymnasium`: Reinforcement learning (Double DQN) signal control and multi-junction environments.
- `fastapi`, `uvicorn`, `httpx`: Asynchronous REST API rehearsal service.
- `pydantic`, `pyyaml`: Strict configuration schema validation.
- `matplotlib`: Publication-quality heatmaps and ranking charts.
- `lightsim`: Macroscopic simulator reconnaissance package.

---

## 4. Configuration Guide

All physical parameters, time steps, network coordinates, scenario profiles, thresholds, and tolerances are managed entirely in validated YAML configurations:

- `config/project.yaml`: Project metadata, master random seed, fix attempt budgets, outputs directory.
- `config/engine.yaml`: Daganzo CTM parameters (free-flow speed $v_f$, wave speed $w$, nominal capacity $q_{\text{nom}}$, jam density $k_j$, discrete time step $\Delta t$ conforming to the CFL condition $\Delta t \le L / v_f$).
- `config/network.yaml`: Corridor geometry, number of signalized junctions, arterial cells per link, lane allocations, cell lengths.
- `config/demand.yaml`: Base flow rate, peak surge flow rate, peak duration windows, weekday/weekend scaling factors.
- `config/scenarios/*.yaml`: Scenario definitions specifying planted disruptions, capacity factors, and temporal activation windows.
- `config/observation.yaml`: Sensor coverage fraction, measurement reporting intervals, Gaussian noise standard deviation, Bernoulli dropout fraction.
- `config/diagnosis.yaml`: Jam speed threshold fraction, DBSCAN spatio-temporal $\epsilon$ and sample counts, CUSUM change-point parameters, counterfactual replicate counts.
- `config/rl.yaml`: Double DQN network hyperparameters, replay buffer capacity, exploration decay, and external safety shield timing constraints.
- `config/api.yaml`: Asynchronous rehearsal service endpoints, concurrency worker limits, and timeout thresholds.

---

## 5. How to Add a New Scenario in YAML

To register a new traffic disruption scenario:
1. Create a new YAML configuration in `config/scenarios/<scenario_name>.yaml`:
```yaml
name: "incident_blockage"
scenario_type: "recurring_capacity_drop"
description: "Temporary mid-block single lane obstruction"
start_time_sec: 300.0
end_time_sec: 420.0
target_cell_id: 12
upstream_junction_cell_id: 10
capacity_factor: 0.40
demand_shift_sec: 0.0
decay_rate_per_step: 0.0
recurring_weekdays: [1, 2, 3, 4, 5]
recurring_hours: [8, 9]
ground_truth_filename: "incident_blockage_ground_truth.json"
```
2. The scenario will automatically be discovered, validated against `ScenarioConfig`, and registered by `load_citysense_config()`.
3. If introducing a brand-new physical disruption mechanism, implement a subclass of `BaseScenario` in `src/citysense/scenarios/registry.py`.

---

## 6. How to Run Tests

### Running Individual Module Tests
Each module features independent gate tests with theoretical expectations computed from config:
```bash
# Module 0: Recon & Environment
uv run pytest tests/module_0/test_module_0.py

# Module 1: Configuration System & CFL Validation
uv run pytest tests/module_1/test_module_1.py

# Module 2: Native Vectorized CTM Engine (Conservation, Shockwave Speed, Snapshots)
uv run pytest tests/module_2/test_module_2.py

# Module 3: Arterial Corridor & Scenario Registry
uv run pytest tests/module_3/test_module_3.py

# Module 4: Sensor Observations, Noise & Dropout
uv run pytest tests/module_4/test_module_4.py

# Module 5: Spatio-Temporal DBSCAN Jam Detection
uv run pytest tests/module_5/test_module_5.py

# Module 6: Shockwave Back-Tracing & Bottleneck Origin
uv run pytest tests/module_6/test_module_6.py

# Module 7: Capacity Estimation & CUSUM Change-Points
uv run pytest tests/module_7/test_module_7.py

# Module 8: Counterfactual Simulation Ladder (Vehicle-Hours Recovered)
uv run pytest tests/module_8/test_module_8.py

# Module 9: External Signal Safety Shield & Double DQN Control
uv run pytest tests/module_9/test_module_9.py

# Module 10: Civic Situation Records & Deterministic Ranking
uv run pytest tests/module_10/test_module_10.py

# Module 11: FastAPI Rehearsal Service Round-Trip
uv run pytest tests/module_11/test_module_11.py

# Module 12: Outcome Tracker & Bounded Calibration Table
uv run pytest tests/module_12/test_module_12.py

# Module 13: Static Space-Time Visuals & Standalone HTML Report
uv run pytest tests/module_13/test_module_13.py

# Module 14: End-to-End Evaluation Suite & Final Results
uv run pytest tests/module_14/test_module_14.py
```

### Running the AST Zero-Hardcoding Enforcement Test
Enforces Hard Rule 2 across all files in `src/`:
```bash
uv run pytest tests/test_no_hardcoding.py
```

### Running the Full Regression Suite
```bash
uv run pytest
```
