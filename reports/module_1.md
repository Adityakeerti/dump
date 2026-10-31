# Module 1 Report: Configuration System

## What Was Built
- Named unit conversions in `src/citysense/config/units.py` strictly complying with Hard Rule 2.
- AST scanner in `tests/test_no_hardcoding.py` scanning `src/` to enforce zero disallowed numeric literals (outside {-1, 0, 1, 2}) and zero hardcoded paths or URLs.
- Complete set of Pydantic configuration schemas in `src/citysense/config/schemas.py`:
  - `ProjectConfig`
  - `EngineConfig` (with field validation and CFL stability constraint checking)
  - `NetworkConfig`
  - `DemandConfig`
  - `ScenarioConfig`
  - `ObservationConfig`
  - `DiagnosisConfig`
  - `RLConfig`
  - `APIConfig`
  - `CitySenseConfig` (with cross-file CFL consistency validation)
  - All models configured with `model_config = ConfigDict(extra="forbid")` to reject unknown keys.
- YAML configuration files:
  - `config/project.yaml`
  - `config/engine.yaml`
  - `config/network.yaml`
  - `config/demand.yaml`
  - `config/scenarios/control.yaml`
  - `config/scenarios/recurring_capacity_drop.yaml`
  - `config/scenarios/creeping_capacity_drop.yaml`
  - `config/scenarios/demand_peak_shift.yaml`
  - `config/scenarios/downstream_spillback.yaml`
  - `config/observation.yaml`
  - `config/diagnosis.yaml`
  - `config/rl.yaml`
  - `config/api.yaml`
- Configuration loader and canonical SHA-256 hash generator in `src/citysense/config/loader.py`.
- Module 1 gate tests in `tests/module_1/test_module_1.py`.

## Test Results
- `tests/module_0/test_module_0.py`: 3 passed.
- `tests/module_1/test_module_1.py`: 8 passed:
  - `test_valid_configs_load`: PASSED
  - `test_missing_key_raises_error`: PASSED
  - `test_negative_capacity_raises_error`: PASSED
  - `test_cfl_violation_raises_error`: PASSED
  - `test_unknown_key_raises_error`: PASSED
  - `test_config_hash_stability_and_sensitivity`: PASSED
  - `test_no_hardcoding_active_check`: PASSED
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 12 passed in 1.90s.

## Deviations from Spec
None.

## Open Risks
- Subsequent modules must ensure zero literal constants or paths are written in `src/` to maintain the AST invariant.
