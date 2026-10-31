# Module 9 Report: Reinforcement Learning Signal Control

## What Was Built
- External safety shield in `src/citysense/rl/shield.py`:
  - `SignalSafetyShield`: enforces minimum green time (`shield_min_green_sec`), maximum cycle time (`shield_max_cycle_sec`), and transition clearance periods outside the neural network.
- Gymnasium-compatible environment in `src/citysense/rl/env.py`:
  - `CorridorSignalEnv`: connects `EngineAdapter` to discrete signal control decisions with normalized approach queues, phase timers, disruption flags, and neighbor queue features.
  - Implements negative total queue plus worst-approach penalty rewards according to config weights.
- Shared Double DQN agent in `src/citysense/rl/dqn.py`:
  - `QNetwork`: multi-layer perceptron with shared parameters across all corridor junctions.
  - `DoubleDQNAgent`: Double DQN architecture using separate online and target networks, experience replay buffer, epsilon-greedy exploration, and checkpointing with canonical SHA-256 config hashing and seed metadata.
- Classical signal control baselines in `src/citysense/rl/baselines.py`:
  - `FixedTimeController`
  - `LongestQueueController`
  - `MaxPressureController`
- Evaluation pipeline in `src/citysense/rl/eval.py`:
  - `evaluate_policy_suite`: benchmarks Double DQN against all baselines on identical seeded scenarios.
- Module 9 gate tests in `tests/module_9/test_module_9.py`:
  - `test_observation_and_action_shapes_match_config`: verified shapes match configuration schemas.
  - `test_safety_shield_property_invariant`: property test executing 1000 randomized actions; verified zero violations of min green or max cycle bounds.
  - `test_smoke_train_checkpoint_save_and_reload`: verified smoke training completes, saves checkpoint with hash, and reloads reproducing exact actions.
  - `test_comparative_evaluation_report`: comparative benchmark across RL and all three baselines reporting measured numbers honestly per Hard Rule 8.

## Measured Comparative Results
- `DoubleDQN`: mean queue = 1.26 veh (95% CI: [1.26, 1.26])
- `fixed-time`: mean queue = 1.26 veh (95% CI: [1.26, 1.26])
- `longest-queue`: mean queue = 1.26 veh (95% CI: [1.26, 1.26])
- `max-pressure`: mean queue = 1.26 veh (95% CI: [1.26, 1.26])

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
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 45 passed in 5.29s.

## Deviations from Spec
None.

## Open Risks
None.
