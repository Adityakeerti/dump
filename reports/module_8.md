# Module 8 Report: Counterfactual Ladder

## What Was Built
- Data models and ladder logic in `src/citysense/counterfactual/types.py` and `src/citysense/counterfactual/ladder.py`:
  - `CounterfactualRungResult`: captures recovered vehicle-hours (mean, 95% confidence interval, replicate trials, and rank).
  - `CounterfactualLadderResult`: records top-ranked causal explanation, noise floor, baseline total vehicle-hours, and scenario breakdown.
  - `CounterfactualLadder`:
    - Evaluates 3 counterfactual intervention rungs:
      1. `restore_capacity`: lifts physical bottleneck restriction at the suspect cell ($Q_{\text{eff}} \to Q_{\text{nom}}$).
      2. `demand_peak_shift`: retimes demand surge timing by `peak_shift_sec`.
      3. `rl_signal_control`: executes adaptive coordination along the arterial corridor.
    - Evaluates across $K$ demand perturbation replicates ($K = \text{counterfactual\_replicates}$, $\sigma = \text{counterfactual\_perturbation\_std}$).
    - Calculates vehicle-hours recovered from congestion delay relative to observed baseline.
    - Deterministically ranks hypotheses based on recovered vehicle-hours.
- Module 8 gate tests in `tests/module_8/test_module_8.py`:
  - `test_planted_capacity_drop_ranks_restore_capacity_first`: verified that on a planted capacity drop, the restore-capacity rung ranks #1.
  - `test_spillback_scenario_signal_recovers_less_than_downstream_capacity`: verified that in a spillback scenario, signal retiming recovers significantly less than clearing the downstream bottleneck.
  - `test_control_scenario_recovery_below_noise_floor`: verified that in control, all rungs remain below the noise floor (`explains_jam == False`).
  - `test_ladder_determinism_across_reruns_with_same_seed`: verified bit-identical reproducibility across seeded reruns.

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
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 41 passed in 3.30s.

## Deviations from Spec
None.

## Open Risks
None.
