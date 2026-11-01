# Live root-cause pipeline benchmark

> [SIMULATED TWIN EVIDENCE] CTM digital twin, background demand x live factor plus seeded synthetic agent trips.

Runs: cases ['control', 'creeping_capacity_drop', 'demand_peak_shift', 'demand_surge', 'downstream_spillback', 'recurring_capacity_drop'], policies ['fixed-time', 'longest-queue', 'DoubleDQN'], seeds [0, 1, 2]. Wall time 641 s (re-rendered). Minimum jam age 180 s; minimum effect 0.25 veh-h.

## 1. Maturity gate

Diagnosis requested the moment a jam was confirmed: refused as immature in 100% (50/50) of runs.

## 2. By case (fixed pipeline, all mature attempts)

| Case | Diagnosed | Origin error (cells) mean / max | Correct | Correct when conclusive | Verdicts (conclusive/tied/inconclusive) | Top hypotheses | No mature jam |
|---|---|---|---|---|---|---|---|
| recurring_capacity_drop | 26 | 0.00 / 0 | 88% (23/26) | 88% (21/24) | 24/2/0 | recurring_capacity_drop 23, signal_coordination_fault 3 | 1 |
| creeping_capacity_drop | 24 | 1.79 / 11 | 62% (15/24) | 59% (13/22) | 22/2/0 | recurring_capacity_drop 15, signal_coordination_fault 8, demand_peak_shift 1 | 3 |
| downstream_spillback | 26 | 1.00 / 1 | 96% (25/26) | 100% (24/24) | 24/2/0 | recurring_capacity_drop 25, signal_coordination_fault 1 | 1 |
| control | 7 | n/a | 100% (7/7) | 100% (7/7) | 7/0/0 | signal_coordination_fault 4, demand_peak_shift 3 | 12 |
| demand_surge | 27 | n/a | 100% (27/27) | 100% (25/25) | 25/2/0 | demand_peak_shift 21, signal_coordination_fault 6 | 0 |
| demand_peak_shift | 8 | n/a | 100% (8/8) | 100% (6/6) | 6/1/1 | signal_coordination_fault 5, demand_peak_shift 3 | 7 |

Secondary (capacity cases): capacity is a *material* cause (its relief is at least 50% of the top rung's) in recurring_capacity_drop 100% (26/26), creeping_capacity_drop 83% (20/24), downstream_spillback 100% (26/26). Origin chosen from a non-front candidate (chained bottleneck) in 1% (1/118) of diagnoses.

## 3. By jam age at diagnosis (capacity cases)

| Attempt | n | Correct | Conclusive | Mean top recovery (veh-h) | Replay started before jam |
|---|---|---|---|---|---|
| maturity | 27 | 81% (22/27) | 85% (23/27) | 8.33 | 100% (27/27) |
| +300s | 27 | 85% (23/27) | 96% (26/27) | 12.44 | 100% (27/27) |
| +600s | 22 | 82% (18/22) | 95% (21/22) | 17.12 | 95% (21/22) |

## 4. By running signal policy (all cases)

| Policy | n | Correct | Best alternative signal policy (most frequent) |
|---|---|---|---|
| fixed-time | 41 | 71% (29/41) | longest-queue (33) |
| longest-queue | 36 | 97% (35/36) | DoubleDQN (35) |
| DoubleDQN | 41 | 100% (41/41) | longest-queue (32) |

## 5. Demand-rung fix: legacy vs fixed (maturity and +300 s attempts)

| Pipeline | n | Correct (capacity cases) | Correct (negative controls) | Top = demand_peak_shift |
|---|---|---|---|---|
| legacy (pre-fix) | 85 | 81% (44/54) | 100% (31/31) | 27% (23/85) |
| fixed | 85 | 83% (45/54) | 100% (31/31) | 25% (21/85) |

Legacy demand rung vs fixed on identical runs: higher by mean 0.82 veh-h (max 8.89 veh-h); top hypothesis changed in 2/85 paired diagnoses. The inflation comes from negative demand while the replay covers t < 420 s; mature-jam replays here start after most of that window, so rankings rarely flip. Young-jam diagnoses (now refused) were the ones affected.

## 6. Can it name demand as the cause? (demand_surge, strict)

Top hypothesis = demand (demand_peak_shift rung) in 78% (21/27); signal_coordination_fault in 22% (6/27); capacity in 0% (0/27). A trip surge was detected and thinned in 96% (26/27) of surge diagnoses (and in 100% (15/15) of control diagnoses, where agent rounds vary).

## 7. Capacity-drop flag on negative controls

`significant_drop_detected` was true in 2% (1/42) of diagnoses where no capacity was cut (congested flow read as lost capacity).

## 8. Ablation: this round's fixes off vs on (same runs, pinned DQN)

Off = origin from the tracer's front only and no trip-surge thinning (the previous pipeline).

| Case | Pipeline | n | Correct | Origin error mean / max | Capacity material | Top = demand |
|---|---|---|---|---|---|---|
| recurring_capacity_drop | off | 26 | 88% (23/26) | 0.00 / 0 | 100% (26/26) | 0% (0/26) |
| recurring_capacity_drop | on | 26 | 88% (23/26) | 0.00 / 0 | 100% (26/26) | 0% (0/26) |
| creeping_capacity_drop | off | 24 | 62% (15/24) | 2.17 / 11 | 79% (19/24) | 4% (1/24) |
| creeping_capacity_drop | on | 24 | 62% (15/24) | 1.79 / 11 | 83% (20/24) | 4% (1/24) |
| downstream_spillback | off | 26 | 96% (25/26) | 1.00 / 1 | 100% (26/26) | 0% (0/26) |
| downstream_spillback | on | 26 | 96% (25/26) | 1.00 / 1 | 100% (26/26) | 0% (0/26) |
| control | off | 7 | 100% (7/7) | n/a | 0% (0/7) | 14% (1/7) |
| control | on | 7 | 100% (7/7) | n/a | 0% (0/7) | 43% (3/7) |
| demand_surge | off | 27 | 100% (27/27) | n/a | 4% (1/27) | 15% (4/27) |
| demand_surge | on | 27 | 100% (27/27) | n/a | 0% (0/27) | 78% (21/27) |

DQN: pinned checkpoint `outputs/benchmark/dqn_pinned.pt` (copied once from the live twin) so runs are comparable.

Runtime per diagnosis: mean 18.1 s, max 34.3 s.
