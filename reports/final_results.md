# Final Evaluation Results: CitySense Traffic Twin

> [SIMULATED TWIN EVIDENCE]
> All figures, tables, and metrics reported below are derived from macroscopic Cell Transmission Model (CTM) simulations.
> RL is evaluated as an adaptive signal control benchmark within the counterfactual ladder, not as an oracle root-cause classifier.

## 1. Executive Summary
- **Simulated Corridor:** 10 signalized junctions (2 side-road leg(s) each), 176 arterial cells of 15 m, 2.64 km arterial length.
- **Config Hash:** `7f44447f5feab506b1f61379ed719d0ba05c5e9213e8a28eee84388c2c0945ae`
- **Scenario Suite:** 5 registered scenarios evaluated across 3 seeded replicates.
- **Root-Cause Identification:** 100.0% correct top-ranked cause rate across all scenarios and seeds.
- **Shockwave Origin Localisation Error:** 1.11 cells mean error across planted bottleneck scenarios.

---

## 2. Benchmark Metrics by Planted Scenario

| Scenario | Detection Recall | Detection Precision | Origin Error | Capacity Error | Top-Ranked Cause Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| control | 1.000 | 1.000 | N / A | 0.000 | 100.0% |
| creeping_capacity_drop | 1.000 | 1.000 | 1.00 cells | 0.025 | 100.0% |
| demand_peak_shift | 0.000 | 0.000 | N / A | N / A | 100.0% |
| downstream_spillback | 0.983 | 1.000 | 1.33 cells | 0.018 | 100.0% |
| recurring_capacity_drop | 0.964 | 1.000 | 1.00 cells | 0.027 | 100.0% |

---

## 3. RL vs Classical Baseline Signal Controllers

Evaluated on the arterial corridor with randomised demand and capacity disruptions:

| Controller Policy | Mean Queue (veh) | 95% Confidence Interval |
| :--- | :--- | :--- |
| DoubleDQN | 170.96 | [170.96, 170.96] |
| fixed-time | 234.08 | [234.08, 234.08] |
| longest-queue | 165.57 | [165.57, 165.57] |
| max-pressure | 163.13 | [163.13, 163.13] |

### Observations on Signal Control:
- Signal controllers (Double DQN, Fixed-Time, Longest-Queue, Max-Pressure) manage phase timing at junctions.
- In bottleneck scenarios caused by physical lane capacity cuts (e.g. downstream blockages or recurring mid-block drops), signal control alone is insufficient to clear the jam; capacity restoration provides superior vehicle-hours recovered (VHR).

---

## 4. Failure Modes, Edge Cases, and Limitations

1. **Sensor Coverage and Dropout:**
   - With cell coverage below 50% or sensor dropout above 30%, sparse shockwave front observations can introduce +/- 1 cell jitter in wave speed estimation.
2. **Cross-Street Demand Spillover:**
   - Under severe arterial spillback, cross-street green allocation cannot prevent gridlock once downstream arterial storage is exhausted.
3. **Discrete Cell Granularity:**
   - Origin back-tracing is bounded by the discrete cell length (20m); sub-cell bottleneck shifts within a single cell are indistinguishable macroscopically.

---

## 5. Verification & Determinism
- Full suite executed under seed `42`.
- All randomness strictly derived from NumPy / PyTorch Generators.
