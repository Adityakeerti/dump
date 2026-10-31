# Module 2 Report: Engine Adapter and CTM Implementation

## What Was Built
- Data structures in `src/citysense/engine/types.py`:
  - `SimulationState`: captures per-cell density, flow, speed, queues, step counts, vehicle totals, and queued demand.
  - `CapacityFactorWindow`: holds scheduled dynamic capacity scaling parameters over time intervals.
  - `EngineSnapshot`: encapsulates full deep engine state including random generator state.
- Abstract interface in `src/citysense/engine/adapter.py`:
  - `EngineAdapter` defining `reset`, `step`, `state`, `set_capacity_factor`, `set_demand_profile`, `set_signal_controller`, `snapshot`, and `restore`.
- Vectorized native CTM implementation in `src/citysense/engine/ctm.py`:
  - `NativeCTMEngine` strictly implementing Daganzo's Cell Transmission Model.
  - Supports arbitrary cell lengths, lane configurations, signal masks, and dynamic capacity degradation windows.
  - Implements exact numerical vehicle conservation across links and entrance buffers.
  - Zero hardcoded numeric or path literals (verified by AST scan).
- Gate tests in `tests/module_2/test_module_2.py`:
  - `test_fundamental_diagram_sweep`: Sweeps demand across free-flow and congested branches; R-squared exceeds tolerance.
  - `test_conservation_of_vehicles`: Vehicles in + initial = vehicles out + remaining within floating-point tolerance.
  - `test_shockwave_speed`: Verified backward wave speed against theoretical Rankine-Hugoniot condition $w = (q_2 - q_1) / (k_2 - k_1)$ with under 0.1% relative error.
  - `test_snapshot_restore_identical_future`: Bit-identical reproduction of simulation trajectories across snapshot and restore.
  - `test_capacity_factor_one_changes_nothing`: Verified identity capacity factor reproduces baseline identically.

## Test Results
- `tests/module_0/test_module_0.py`: 3 passed.
- `tests/module_1/test_module_1.py`: 8 passed.
- `tests/module_2/test_module_2.py`: 5 passed.
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 17 passed in 2.15s.

## Deviations from Spec
None.

## Open Risks
- Shockwave speed tracking relies on continuous sub-cell threshold interpolation when cells are coarse.
