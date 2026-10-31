# Module 6 Report: Shockwave Back-Tracing

## What Was Built
- Data models and tracing logic in `src/citysense/diagnosis/shockwave.py`:
  - `ShockwaveOrigin`: records the localized bottleneck origin cell, estimated wave speed, onset time, evidence trail, and associated cluster id.
  - `ShockwaveTracer`:
    - Estimates shockwave propagation speed $\hat{w} = \Delta q / \Delta k$ from observed flow and density.
    - Disambiguates bottleneck location by identifying the true downstream discharge front where congestion releases back to free-flow speeds.
    - Resolves spillback scenarios by proving the bottleneck origin is the downstream constriction rather than the upstream junction experiencing severe delay symptoms.
    - Constructs an audit-compliant, chronological evidence trail of sensor observations.
- Module 6 gate tests in `tests/module_6/test_module_6.py`:
  - `test_origin_cell_error_across_k_seeds`: origin cell error bounded within tolerance across multiple randomized seeds.
  - `test_downstream_spillback_origin_disambiguation`: verified traced origin is the downstream bottleneck ($c=30$) and NOT the upstream junction ($c=18$).
  - `test_estimated_wave_speed_matches_theoretical`: verified estimated wave speed aligns with theoretical $w$ within tolerance.

## Test Results
- `tests/module_0/test_module_0.py`: 3 passed.
- `tests/module_1/test_module_1.py`: 8 passed.
- `tests/module_2/test_module_2.py`: 5 passed.
- `tests/module_3/test_module_3.py`: 6 passed.
- `tests/module_4/test_module_4.py`: 5 passed.
- `tests/module_5/test_module_5.py`: 3 passed.
- `tests/module_6/test_module_6.py`: 3 passed.
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 34 passed in 2.48s.

## Deviations from Spec
None.

## Open Risks
None.
