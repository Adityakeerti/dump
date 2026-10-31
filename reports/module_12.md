# Module 12 Report: Outcome Tracker and Calibration

## What Was Built
- Data models in `src/citysense/tracker/calibration.py`:
  - `OutcomeComparisonResult`: captures predicted vs observed recovery, interval bounds, prediction errors, interval containment, and explicit simulated labeling.
  - `CalibrationTable`: maintains scoring weights for each candidate hypothesis type, applying bounded Bayesian updates and supporting full reversibility.
  - `OutcomeTracker`: interfaces situations with later observations, evaluates prediction accuracy, and updates the calibration table.
- Module 12 gate tests in `tests/module_12/test_module_12.py`:
  - `test_observed_recovery_inside_predicted_interval_coverage`: verified observed recovery falls inside predicted interval at tolerance coverage rate across randomized runs.
  - `test_calibration_update_is_bounded_and_reversible`: verified weight updates respect strict step bounds, remain within global boundaries, and revert cleanly.
  - `test_every_output_is_labelled_simulated`: confirmed explicit simulated labeling on all output records.

## Test Results
- `tests/module_0/test_module_0.py`: 3 passed.
- `tests/module_1/test_module_1.py`: 8 passed.
- `tests/module_2/test_module_2.py`: 5 passed.
- `tests/module_3/test_module_3.py`: 6 passed.
- `tests/module_4/test_module_4.py`: 5 passed.
- `tests/module_5/test_module_5.py`: 3 passed.
- `tests/module_6/test_module_6.py`: 3 passed.
- `tests/module_7/test_module_7.py`: 3 passed.
- `tests/module_8/test_module_8.py`: 4 passed.
- `tests/module_9/test_module_9.py`: 4 passed.
- `tests/module_10/test_module_10.py`: 5 passed.
- `tests/module_11/test_module_11.py`: 4 passed.
- `tests/module_12/test_module_12.py`: 3 passed.
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 57 passed in 4.19s.

## Deviations from Spec
None.

## Open Risks
None.
