"""Benchmark the live root-cause pipeline (citysense.live.diagnose) across scenarios, policies, seeds and jam ages.

    .venv/bin/python scripts/benchmark_live_diagnosis.py            # ~3 min on 12 processes
    .venv/bin/python scripts/benchmark_live_diagnosis.py --quick    # 1 seed, 2 policies

Cases
  * capacity cases (planted capacity cut): recurring_capacity_drop, creeping_capacity_drop, downstream_spillback.
    Correct = top hypothesis is the capacity rung; origin error = |origin - planted cell|.
  * negative controls (no capacity cut): control, and demand_surge = control + an extra gateway surge of trips.
    (The repo's demand_peak_shift scenario is a no-op in the engine, so it is reported but behaves like control.)
    Correct = the pipeline does NOT conclude capacity: top rung is not capacity, or the verdict is not "conclusive".
Every case adds seeded synthetic trips that mimic the population agents' volumes (no LLM, reproducible).
Diagnosis attempts per run: at confirmation (must be refused as immature), at maturity, +300 s and +600 s later.
"legacy" mode reproduces the pre-fix demand rung (whole curve shifted, negative demand early) for comparison.
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import shutil
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

CAPACITY_CASES = ["recurring_capacity_drop", "creeping_capacity_drop", "downstream_spillback"]
NEGATIVE_CASES = ["control", "demand_surge", "demand_peak_shift"]
# Synthetic population like city_sim's generator: 24 clusters, lognormal sizes totalling ~24000 people,
# each attached to a random entry (sources are drawn per seed in run_job).
N_CLUSTERS, TOTAL_POP, SIZE_SIGMA = 24, 24000, 0.9
SHARES = [0.01, 0.025, 0.04]
CAR_RATIO = 0.45
ROUND_SEC = 180
SURGE = {"start": 400, "end": 900, "every": 24, "vehicles": 12.0, "source": 0}  # +0.5 veh/s at the gateway


def load():
    from citysense.config.loader import load_citysense_config
    from citysense.live import load_live_config
    c = ROOT / "config"
    cfg, h = load_citysense_config(c / "project.yaml", c / "engine.yaml", c / "network.yaml", c / "demand.yaml",
                                   c / "scenarios", c / "observation.yaml", c / "diagnosis.yaml", c / "rl.yaml", c / "api.yaml")
    return cfg, h, load_live_config(c / "live.yaml")


class LegacyShift:
    """Pre-fix demand rung: base.get_rate(t - shift) without the warm-up clamp (negative demand for t < shift)."""

    def __init__(self, bg):
        self.bg = bg

    def get_rate(self, t):
        d = self.bg.demand
        c = d.config
        t2 = t - c.peak_shift_sec
        if t2 < c.warmup_duration_sec:
            r = c.base_flow_veh_per_sec * d.day_factor * (t2 / c.warmup_duration_sec)
        elif c.peak_start_sec <= t2 <= c.peak_end_sec:
            r = c.peak_flow_veh_per_sec * d.day_factor
        else:
            r = c.base_flow_veh_per_sec * d.day_factor
        return r * self.bg.factor


def make_clusters(rng, n_sources):
    w = rng.lognormal(0.0, SIZE_SIGMA, N_CLUSTERS)
    pops = np.maximum(40, np.round(TOTAL_POP * w / w.sum())).astype(int)
    return [(int(rng.integers(n_sources)), int(p)) for p in pops]


def synthetic_trips(rng, t_now, case, clusters):
    trips = []
    for i, (src, pop) in enumerate(clusters):
        veh = pop * SHARES[rng.integers(len(SHARES))] * CAR_RATIO
        n = max(1, int(veh // 12) + 1)
        for _ in range(n):
            trips.append({"cluster_id": f"c{i}", "cluster_name": f"cluster {i}", "intent": "commute", "plan_id": f"b{t_now:.0f}",
                          "source_id": src, "vehicles": round(veh / n, 2), "depart_in_sec": float(rng.uniform(0, 240))})
    return trips


def run_job(job):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from citysense.live import LiveTwin
    from citysense.live.diagnose import diagnose
    from citysense.live.twin import ScaledDemand
    from citysense.observation.generator import ObservationGenerator

    case, policy, seed, mode, ckpt = job
    legacy = mode == "legacy"
    cfg, h, live = load()
    tmp = tempfile.mkdtemp(prefix="cs-bench-")
    shutil.copy(ckpt, Path(tmp) / "dqn.pt") if ckpt else None
    update = {"checkpoint_path": str(Path(tmp) / "dqn.pt"), "episodes_path": str(Path(tmp) / "ep.json"), "autoplay": False}
    if mode == "ablated":  # previous pipeline: tracer front only, no surge thinning
        update.update({"diagnosis_max_origin_candidates": 1, "diagnosis_surge_ratio": float("inf")})
    live = live.model_copy(update=update)
    # Workers are reused across jobs: always set (and later restore) the demand rung explicitly.
    original_retimed = ScaledDemand.__dict__.get("_original_retimed") or ScaledDemand.retimed
    ScaledDemand._original_retimed = original_retimed
    ScaledDemand.retimed = (lambda self: LegacyShift(self)) if legacy else original_retimed
    twin = LiveTwin(cfg, h, live)
    twin.seed = cfg.project.seed + seed
    for run in twin.runs.values():
        run.sensors = ObservationGenerator(cfg.observation, twin.net.n_cells, twin.seed)
    scenario = "control" if case == "demand_surge" else case
    twin.control(None, policy, scenario, None, False)
    rng = np.random.default_rng(1000 + seed)
    clusters = make_clusters(rng, len(twin.sources))
    planted = cfg.scenarios[scenario].target_cell_id if case in CAPACITY_CASES else None
    duration = cfg.demand.simulation_duration_sec
    min_age = live.diagnosis_min_jam_age_sec

    rows, attempts_done = [], set()
    next_round, next_surge, maturity_t = 0.0, SURGE["start"], None
    while True:
        t = twin.state()["sim_time"]
        if t >= duration - 20:
            break
        if t >= next_round:
            twin.add_trips(synthetic_trips(rng, t, case, clusters))
            next_round += ROUND_SEC
        if case == "demand_surge":
            while next_surge <= min(t, SURGE["end"]):
                twin.add_trips([{"cluster_id": "surge", "cluster_name": "surge", "intent": "commute", "plan_id": "surge",
                                 "source_id": SURGE["source"], "vehicles": SURGE["vehicles"], "depart_in_sec": 0.0}])
                next_surge += SURGE["every"]
        twin.advance(5)
        s = twin.state()
        pts = s["congestion_points"]
        ages = {p["head_cell"]: s["sim_time"] - p["since_sec"] for p in pts}

        def attempt(label, pick):
            res = diagnose(twin, pick["head_cell"], twin.prepare_diagnosis())
            row = {"case": case, "policy": policy, "seed": seed, "legacy": legacy, "mode": mode, "attempt": label, "t": s["sim_time"],
                   "cell": pick["head_cell"], "found": res.get("found"), "immature": res.get("immature", False)}
            if res.get("found"):
                cf = res["counterfactual"]
                top = res["situation"]["top_hypothesis"]["hypothesis_type"]
                is_cap = top == cfg.diagnosis.cause_name_recurring
                row.update({
                    "origin": res["origin"]["cell"],
                    "origin_error": abs(res["origin"]["cell"] - planted) if planted is not None else None,
                    "top": top, "verdict": cf["verdict"]["status"], "age": res["jam"]["age_sec"],
                    "snapshot_before_jam": cf["snapshot_sec"] <= res["jam"]["start_time_sec"],
                    "rungs": {x["rung_name"]: x["mean_recovery_vhr"] for x in cf["rungs"]},
                    "best_signal_alternative": cf["best_signal_alternative"],
                    "capacity_drop_detected": res["capacity_at_origin"]["significant_drop_detected"],
                    "traced_front": res["origin"].get("traced_front_cell"),
                    "origin_from_candidates": res["origin"]["cell"] != res["origin"].get("traced_front_cell"),
                    "best_demand_intervention": cf.get("best_demand_intervention"),
                    "surge_thinned": bool(cf.get("surge_sources_thinned")),
                    # secondary: capacity is a material cause = its rung achieves >= 50% of the top rung's relief
                    "capacity_material": cf["rungs"] and next((x["mean_recovery_vhr"] for x in cf["rungs"] if x["rung_name"] == cfg.diagnosis.top_ranked_name_capacity), 0.0) >= 0.5 * max(x["mean_recovery_vhr"] for x in cf["rungs"]),
                    "correct": is_cap if planted is not None else not (is_cap and cf["verdict"]["status"] == "conclusive"),
                    "runtime": res["runtime_sec"],
                })
            rows.append(row)

        if pts and "confirmation" not in attempts_done and not legacy:
            attempts_done.add("confirmation")
            attempt("confirmation", max(pts, key=lambda p: p["head_cell"]))
        mature = [p for p in pts if ages[p["head_cell"]] >= min_age]
        if mature and maturity_t is None:
            maturity_t = s["sim_time"]
        for label, offset in (("maturity", 0), ("+300s", 300), ("+600s", 600)):
            if legacy and label == "+600s":
                continue
            if maturity_t is not None and label not in attempts_done and s["sim_time"] >= maturity_t + offset:
                attempts_done.add(label)
                if mature:
                    attempt(label, min(mature, key=lambda p: p["min_speed_ratio"]))
                else:
                    rows.append({"case": case, "policy": policy, "seed": seed, "legacy": legacy, "mode": mode, "attempt": label,
                                 "t": s["sim_time"], "found": None, "no_jam": True})
    if maturity_t is None:
        rows.append({"case": case, "policy": policy, "seed": seed, "legacy": legacy, "mode": mode, "attempt": "maturity", "found": None, "no_jam": True})
    ScaledDemand.retimed = original_retimed
    shutil.rmtree(tmp, ignore_errors=True)
    return rows


def pct(x, n):
    return f"{100 * x / n:.0f}% ({x}/{n})" if n else "–"


def report(rows, args, wall):
    full = [r for r in rows if r.get("mode", "full") == "full"]
    abl = [r for r in rows if r.get("mode") == "ablated" and r.get("found") and r["attempt"] != "confirmation"]
    diag = [r for r in full if r.get("found")]
    lines = ["# Live root-cause pipeline benchmark", "",
             "> [SIMULATED TWIN EVIDENCE] CTM digital twin, background demand x live factor plus seeded synthetic agent trips.", "",
             f"Runs: cases {sorted(set(r['case'] for r in rows))}, policies {args.policies}, seeds {args.seeds}. "
             f"Wall time {wall:.0f} s. Minimum jam age {args.min_age:.0f} s; minimum effect {args.min_effect} veh-h.", ""]
    conf = [r for r in full if r["attempt"] == "confirmation"]
    lines += ["## 1. Maturity gate", "", f"Diagnosis requested the moment a jam was confirmed: refused as immature in "
              f"{pct(sum(1 for r in conf if r.get('immature')), len(conf))} of runs.", ""]

    lines += ["## 2. By case (fixed pipeline, all mature attempts)", "",
              "| Case | Diagnosed | Origin error (cells) mean / max | Correct | Correct when conclusive | Verdicts (conclusive/tied/inconclusive) | Top hypotheses | No mature jam |",
              "|---|---|---|---|---|---|---|---|"]
    for case in CAPACITY_CASES + NEGATIVE_CASES:
        rs = [r for r in diag if r["case"] == case and r["attempt"] != "confirmation"]
        nj = sum(1 for r in full if r["case"] == case and r.get("no_jam"))
        if not rs:
            lines.append(f"| {case} | 0 | – | – | – | – | – | {nj} |")
            continue
        errs = [r["origin_error"] for r in rs if r["origin_error"] is not None]
        cc = [r for r in rs if r["verdict"] == "conclusive"]
        tops = {}
        for r in rs:
            tops[r["top"]] = tops.get(r["top"], 0) + 1
        v = [sum(1 for r in rs if r["verdict"] == k) for k in ("conclusive", "tied", "inconclusive")]
        lines.append(f"| {case} | {len(rs)} | {f'{st.mean(errs):.2f} / {max(errs)}' if errs else 'n/a'} | "
                     f"{pct(sum(r['correct'] for r in rs), len(rs))} | {pct(sum(r['correct'] for r in cc), len(cc))} | "
                     f"{v[0]}/{v[1]}/{v[2]} | {', '.join(f'{k} {n}' for k, n in sorted(tops.items(), key=lambda kv: -kv[1]))} | {nj} |")

    lines += ["", "Secondary (capacity cases): capacity is a *material* cause (its relief is at least 50% of the top rung's) in "
              + ", ".join(f"{c} {pct(sum(r['capacity_material'] for r in diag if r['case'] == c and r['attempt'] != 'confirmation'), sum(1 for r in diag if r['case'] == c and r['attempt'] != 'confirmation'))}" for c in CAPACITY_CASES)
              + ". Origin chosen from a non-front candidate (chained bottleneck) in "
              + pct(sum(r["origin_from_candidates"] for r in diag if r["attempt"] != "confirmation"), sum(1 for r in diag if r["attempt"] != "confirmation")) + " of diagnoses."]
    lines += ["", "## 3. By jam age at diagnosis (capacity cases)", "", "| Attempt | n | Correct | Conclusive | Mean top recovery (veh-h) | Replay started before jam |", "|---|---|---|---|---|---|"]
    for label in ("maturity", "+300s", "+600s"):
        rs = [r for r in diag if r["attempt"] == label and r["case"] in CAPACITY_CASES]
        if rs:
            lines.append(f"| {label} | {len(rs)} | {pct(sum(r['correct'] for r in rs), len(rs))} | {pct(sum(r['verdict'] == 'conclusive' for r in rs), len(rs))} | "
                         f"{st.mean(max(r['rungs'].values()) for r in rs):.2f} | {pct(sum(r['snapshot_before_jam'] for r in rs), len(rs))} |")

    lines += ["", "## 4. By running signal policy (all cases)", "", "| Policy | n | Correct | Best alternative signal policy (most frequent) |", "|---|---|---|---|"]
    for p in args.policies:
        rs = [r for r in diag if r["policy"] == p and r["attempt"] != "confirmation"]
        if rs:
            alts = {}
            for r in rs:
                alts[r["best_signal_alternative"]] = alts.get(r["best_signal_alternative"], 0) + 1
            lines.append(f"| {p} | {len(rs)} | {pct(sum(r['correct'] for r in rs), len(rs))} | {max(alts, key=alts.get)} ({alts[max(alts, key=alts.get)]}) |")

    leg = [r for r in rows if r.get("found") and r["legacy"]]
    fix = [r for r in diag if r["attempt"] in ("maturity", "+300s")]
    if leg:
        lines += ["", "## 5. Demand-rung fix: legacy vs fixed (maturity and +300 s attempts)", "",
                  "| Pipeline | n | Correct (capacity cases) | Correct (negative controls) | Top = demand_peak_shift |", "|---|---|---|---|---|"]
        for name, rs in (("legacy (pre-fix)", leg), ("fixed", fix)):
            capr = [r for r in rs if r["case"] in CAPACITY_CASES]
            negr = [r for r in rs if r["case"] in NEGATIVE_CASES]
            lines.append(f"| {name} | {len(rs)} | {pct(sum(r['correct'] for r in capr), len(capr))} | {pct(sum(r['correct'] for r in negr), len(negr))} | "
                         f"{pct(sum(r['top'] == 'demand_peak_shift' for r in rs), len(rs))} |")

    if leg:
        key = lambda r: (r["case"], r["policy"], r["seed"], r["attempt"])
        fixed_by = {key(r): r for r in fix}
        pairs = [(fixed_by[key(r)], r) for r in leg if key(r) in fixed_by]
        # Absolute difference: the fixed rung is often ~0, so a ratio is meaningless.
        infl = [l["rungs"]["demand_peak_shift"] - f["rungs"]["demand_peak_shift"] for f, l in pairs]
        flips = sum(1 for f, l in pairs if f["top"] != l["top"])
        if infl:
            lines += ["", f"Legacy demand rung vs fixed on identical runs: higher by mean {st.mean(infl):.2f} veh-h "
                      f"(max {max(infl):.2f} veh-h); top hypothesis changed in {flips}/{len(pairs)} paired diagnoses. "
                      "The inflation comes from negative demand while the replay covers t < 420 s; mature-jam replays here start "
                      "after most of that window, so rankings rarely flip. Young-jam diagnoses (now refused) were the ones affected."]

    surge = [r for r in diag if r["case"] == "demand_surge" and r["attempt"] != "confirmation"]
    if surge:
        others = [r for r in diag if r["case"] in ("control", "demand_peak_shift") and r["attempt"] != "confirmation"]
        lines += ["", "## 6. Can it name demand as the cause? (demand_surge, strict)", "",
                  f"Top hypothesis = demand (demand_peak_shift rung) in {pct(sum(r['top'] == 'demand_peak_shift' for r in surge), len(surge))}; "
                  f"signal_coordination_fault in {pct(sum(r['top'] == 'signal_coordination_fault' for r in surge), len(surge))}; "
                  f"capacity in {pct(sum(r['top'] == 'recurring_capacity_drop' for r in surge), len(surge))}. "
                  f"A trip surge was detected and thinned in {pct(sum(r['surge_thinned'] for r in surge), len(surge))} of surge diagnoses "
                  f"(and in {pct(sum(r['surge_thinned'] for r in others), len(others))} of control diagnoses, where agent rounds vary)."]

    neg = [r for r in diag if r["case"] in NEGATIVE_CASES and r["attempt"] != "confirmation"]
    if neg:
        lines += ["", "## 7. Capacity-drop flag on negative controls", "",
                  f"`significant_drop_detected` was true in {pct(sum(r['capacity_drop_detected'] for r in neg), len(neg))} of diagnoses where no capacity was cut "
                  "(congested flow read as lost capacity)."]
    if abl:
        lines += ["", "## 8. Ablation: this round's fixes off vs on (same runs, pinned DQN)", "",
                  "Off = origin from the tracer's front only and no trip-surge thinning (the previous pipeline).", "",
                  "| Case | Pipeline | n | Correct | Origin error mean / max | Capacity material | Top = demand |", "|---|---|---|---|---|---|---|"]
        for case in CAPACITY_CASES + ["control", "demand_surge"]:
            for name, rs in (("off", [r for r in abl if r["case"] == case]),
                             ("on", [r for r in diag if r["case"] == case and r["attempt"] != "confirmation"])):
                if not rs:
                    continue
                errs = [r["origin_error"] for r in rs if r["origin_error"] is not None]
                lines.append(f"| {case} | {name} | {len(rs)} | {pct(sum(r['correct'] for r in rs), len(rs))} | "
                             f"{f'{st.mean(errs):.2f} / {max(errs)}' if errs else 'n/a'} | {pct(sum(r['capacity_material'] for r in rs), len(rs))} | "
                             f"{pct(sum(r['top'] == 'demand_peak_shift' for r in rs), len(rs))} |")

    lines += ["", f"DQN: pinned checkpoint `outputs/benchmark/dqn_pinned.pt` (copied once from the live twin) so runs are comparable."]
    lines += ["", f"Runtime per diagnosis: mean {st.mean(r['runtime'] for r in diag):.1f} s, max {max(r['runtime'] for r in diag):.1f} s." if diag else ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="re-render the report from outputs/benchmark/live_diagnosis.json")
    ap.add_argument("--processes", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()
    cfg, h, live = load()
    args.policies = ["fixed-time", "longest-queue", cfg.rl.algorithm] if not args.quick else ["fixed-time", "longest-queue"]
    args.seeds = [0, 1, 2] if not args.quick else [0]
    args.min_age, args.min_effect = live.diagnosis_min_jam_age_sec, live.diagnosis_min_effect_vhr
    # Pin the DQN: the live twin keeps training, so copy its checkpoint once and reuse it for every run.
    pinned = ROOT / "outputs" / "benchmark" / "dqn_pinned.pt"
    if not pinned.is_file() and (ROOT / live.checkpoint_path).is_file():
        pinned.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / live.checkpoint_path, pinned)
    ckpt = str(pinned) if pinned.is_file() else None
    if args.render_only:
        rows = json.loads((ROOT / "outputs" / "benchmark" / "live_diagnosis.json").read_text())
        md = report(rows, args, 0.0)
        (ROOT / "reports" / "live_diagnosis_benchmark.md").write_text(md)
        print(md)
        return
    jobs = [(c, p, s, mode, ckpt) for mode in ("full", "legacy") for c in CAPACITY_CASES + NEGATIVE_CASES
            for p in args.policies for s in args.seeds]
    jobs += [(c, p, s, "ablated", ckpt) for c in CAPACITY_CASES + ["control", "demand_surge"] for p in args.policies for s in args.seeds]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.processes) as pool:
        rows = [r for rs in pool.imap_unordered(run_job, jobs) for r in rs]
    wall = time.time() - t0
    out = ROOT / "outputs" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_diagnosis.json").write_text(json.dumps(rows, indent=1))
    md = report(rows, args, wall)
    (ROOT / "reports" / "live_diagnosis_benchmark.md").write_text(md)
    print(md)
    print(f"DQN checkpoint: {'trained live checkpoint' if ckpt else 'none (untrained)'}")


if __name__ == "__main__":
    main()
