# Module 3 Report: Network and Scenario Builder

## What Was Built
- Corridor topology generator in `src/citysense/network/corridor.py`:
  - `build_corridor_network`: builds connected arterial corridors with multi-phase signalized intersections.
  - Generates full cell coordinate mappings, upstream/downstream connection arrays, lane counts, and parameter ranges sampled deterministically from seed.
  - Labels sampled parameters as assumptions in `CorridorMetadata`.
- Demand generator in `src/citysense/network/demand.py`:
  - `CorridorDemand`: generates time-varying flow rates accounting for warmup, peak surge periods, weekday/weekend factors, and temporal shifts.
- Planted-cause scenario registry in `src/citysense/scenarios/registry.py`:
  - `ControlScenario`: baseline with no perturbations.
  - `RecurringCapacityDropScenario`: periodic constriction on set days and hours.
  - `CreepingCapacityDropScenario`: progressive degradation of effective capacity.
  - `DemandPeakShiftScenario`: temporal shift of peak demand.
  - `DownstreamSpillbackScenario`: downstream bottleneck creating shockwave spillback.
  - Ground truth writer isolating planted cause metadata in external directory without diagnosis leakage (Hard Rule 7).
- Module 3 gate tests in `tests/module_3/test_module_3.py`:
  - Seed determinism test.
  - Connectivity and junction count test.
  - Ground truth schema matching test.
  - Control scenario verification (no capacity edits).
  - Capacity drop flow reduction test (drop exceeds tolerance threshold).
  - Ground truth isolation rule test.

## Test Results
- `tests/module_0/test_module_0.py`: 3 passed.
- `tests/module_1/test_module_1.py`: 8 passed.
- `tests/module_2/test_module_2.py`: 5 passed.
- `tests/module_3/test_module_3.py`: 6 passed.
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 23 passed in 2.18s.

## Deviations from Spec
None.

## Open Risks
None.
