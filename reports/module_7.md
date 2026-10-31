# Module 7 Report: Capacity Fitting and Change Detection

## What Was Built
- Data models and fitting algorithms in `src/citysense/diagnosis/capacity.py`:
  - `CapacityFitResult`: encapsulates estimated discharge capacity, capacity ratio against nominal, confidence intervals, identified change points, and significant drop flag.
  - `CapacityEstimator`:
    - Estimates effective capacity $Q_{\text{eff}}$ by fitting sustained bottleneck discharge flows.
    - Computes confidence intervals via normal standard error approximation.
    - Implements CUSUM-based change-point detection to identify sudden capacity drops as well as progressive, creeping degradation.
- Module 7 gate tests in `tests/module_7/test_module_7.py`:
  - `test_recovered_capacity_ratio_matches_planted`: verified recovered capacity ratio matches planted ratio (0.4) within tolerance.
  - `test_creeping_capacity_change_point_detected`: verified change point is detected within allowed delay limit for creeping degradation.
  - `test_control_scenario_produces_no_significant_drop`: verified control scenario produces no false capacity drop during normal traffic flow.

## Test Results
- `tests/module_0/test_module_0.py`: 3 passed.
- `tests/module_1/test_module_1.py`: 8 passed.
- `tests/module_2/test_module_2.py`: 5 passed.
- `tests/module_3/test_module_3.py`: 6 passed.
- `tests/module_4/test_module_4.py`: 5 passed.
- `tests/module_5/test_module_5.py`: 3 passed.
- `tests/module_6/test_module_6.py`: 3 passed.
- `tests/module_7/test_module_7.py`: 3 passed.
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 37 passed in 2.57s.

## Deviations from Spec
None.

## Open Risks
None.
