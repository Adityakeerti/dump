"""Module 15: live twin service (lockstep policy runs, online DQN, car-cluster tracers, API)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from citysense.config.loader import load_citysense_config
from citysense.live import LiveTwin, create_live_app, load_live_config

C = Path("config")


@pytest.fixture()
def configs(tmp_path):
    cfg, cfg_hash = load_citysense_config(
        C / "project.yaml", C / "engine.yaml", C / "network.yaml", C / "demand.yaml", C / "scenarios",
        C / "observation.yaml", C / "diagnosis.yaml", C / "rl.yaml", C / "api.yaml",
    )
    live = load_live_config(C / "live.yaml").model_copy(
        update={
            "checkpoint_path": str(tmp_path / "ck.pt"),
            "episodes_path": str(tmp_path / "episodes.json"),
            "autoplay": False,
        }
    )
    return cfg, cfg_hash, live


def trip(source_id=0, vehicles=10.0, delay=0.0):
    return {
        "cluster_id": "c1", "cluster_name": "Test cluster", "intent": "commute", "plan_id": "p1",
        "source_id": source_id, "vehicles": vehicles, "depart_in_sec": delay,
    }


def test_signals_change_outcomes(configs):
    """With signal cells wired into the engine, policies must produce different queues."""
    twin = LiveTwin(*configs)
    twin.advance(600)
    means = {n: k["mean_system_vehicles"] for n, k in twin.state()["kpis"].items()}
    assert len(set(means.values())) > 1, means


def test_runs_share_demand_and_trips(configs):
    twin = LiveTwin(*configs)
    twin.add_trips([trip(vehicles=20.0)])
    twin.advance(5)
    entered = {n: r.engine.total_entered for n, r in twin.runs.items()}
    assert len({round(v, 6) for v in entered.values()}) == 1


def test_platoon_completes_and_links_decisions(configs):
    twin = LiveTwin(*configs)
    twin.add_trips([trip(vehicles=5.0)])
    twin.advance(400)
    done = twin.finished_platoons(10)
    assert done, "arterial platoon should exit within 400 s"
    p = done[0]
    assert len(p["passes"]) == twin.num_junctions
    assert p["travel_sec"] > 0 and p["delay_sec"] >= 0


def test_decisions_are_explained(configs):
    twin = LiveTwin(*configs)
    twin.advance(120)
    recs = twin.decisions(twin.cfg.rl.algorithm, 0, 50)
    assert recs
    for r in recs:
        assert r["mode"] in {"explore", "exploit"}
        assert r["text"]
        assert r["attribution"] and {"feature", "contribution"} <= set(r["attribution"][0])


def test_episode_rollover_records_metrics_and_decays_epsilon(configs):
    twin = LiveTwin(*configs)
    eps0 = twin.epsilon
    twin.advance(int(twin.cfg.demand.simulation_duration_sec / twin.cfg.engine.dt))
    assert len(twin.episodes) == 1
    assert set(twin.episodes[0]["policies"]) == set(twin.policies)
    assert twin.epsilon < eps0
    assert Path(configs[2].checkpoint_path).is_file()


def test_trip_validation(configs):
    twin = LiveTwin(*configs)
    with pytest.raises(ValueError):
        twin.add_trips([trip(source_id=99)])
    with pytest.raises(ValueError):
        twin.add_trips([trip(vehicles=configs[2].max_vehicles_per_trip + 1)])


def test_api_endpoints(configs):
    app = create_live_app(*configs)
    p = configs[2].paths
    with TestClient(app) as client:
        topo = client.get(p.topology).json()
        assert topo["simulated"] is True and topo["geometry"]["sources"]
        assert client.post(p.control, json={"action": "step"}).status_code == 200
        assert client.post(p.control, json={"policy": "nope"}).status_code == 400
        res = client.post(p.trips, json={"trips": [trip()]})
        assert res.status_code == 200 and res.json()["trip_ids"]
        with client.websocket_connect(p.stream) as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and "kpis" in hello["state"]


def test_street_geometry_follows_cached_osm_file(configs):
    """Cells are laid along the real arterial; junctions sit on the cached real intersections."""
    import json
    live = configs[2]
    path = Path(live.street_geometry_path or "")
    if not path.is_file():
        pytest.skip("no cached street geometry")
    streets = json.loads(path.read_text())
    twin = LiveTwin(*configs)
    geo = twin.geometry
    assert geo["schematic"] is False
    first = geo["cells"][0]["coords"][0]
    assert first == pytest.approx(streets["arterial"][0], abs=1e-6)
    for j, jn in enumerate(geo["junctions"]):
        assert [jn["lat"], jn["lon"]] == pytest.approx(streets["junction_locations"][j], abs=1e-6)
        approach = geo["cells"][jn["arterial_approach"]]["coords"][-1]
        assert approach == pytest.approx([jn["lat"], jn["lon"]], abs=1e-4)
        # every side-road leg ends at the junction's stop line
        for c in jn["cross_approaches"]:
            assert geo["cells"][c]["coords"][-1] == pytest.approx([jn["lat"], jn["lon"]], abs=1e-4)
    legs_per_junction = twin.cfg.network.cross_street_legs
    assert len(geo["sources"]) == 1 + len(geo["junctions"]) * legs_per_junction


def test_schematic_fallback_without_street_file(configs, tmp_path):
    cfg, cfg_hash, live = configs
    twin = LiveTwin(cfg, cfg_hash, live.model_copy(update={"street_geometry_path": str(tmp_path / "missing.json")}))
    assert twin.geometry["schematic"] is True and len(twin.geometry["cells"]) == twin.net.n_cells


def _run_until_mature(twin, extra_sec):
    """Advance until the first confirmed jam, then extra_sec more; returns congestion points that are at least
    `extra_sec` old (several jams of different ages coexist on the long corridor)."""
    while not twin.state()["congestion_points"]:
        twin.advance(20)
        assert twin.state()["sim_time"] < twin.cfg.demand.simulation_duration_sec, "no jam formed"
    twin.advance(int(extra_sec))
    while True:
        st = twin.state()
        mature = [p for p in st["congestion_points"] if st["sim_time"] - p["since_sec"] >= extra_sec]
        if mature or extra_sec == 0:
            return mature or st["congestion_points"]
        assert st["sim_time"] < twin.cfg.demand.simulation_duration_sec, "no mature jam"
        twin.advance(20)


def _near(points, cell):
    return min(points, key=lambda p: abs(p["head_cell"] - cell))["head_cell"]


def test_young_jams_are_not_traced(configs):
    from citysense.live.diagnose import diagnose
    twin = LiveTwin(*configs)
    twin.control(None, "fixed-time", "downstream_spillback", None, False)
    points = _run_until_mature(twin, 0)  # diagnose the moment the jam is confirmed
    res = diagnose(twin, max(p["head_cell"] for p in points), twin.prepare_diagnosis())
    assert res["found"] is False and res["immature"] is True and res["retry_after_sec"] > 0


def test_live_diagnosis_localises_planted_bottleneck(configs):
    """Mature jam: origin near the planted cell, capacity drop, complete ladder with a verdict, correct top rung."""
    from citysense.live.diagnose import diagnose
    twin = LiveTwin(*configs)
    twin.control(None, "fixed-time", "downstream_spillback", None, False)
    planted = twin.cfg.scenarios["downstream_spillback"].target_cell_id
    points = _run_until_mature(twin, twin.live.diagnosis_min_jam_age_sec + 120)
    res = diagnose(twin, _near(points, planted), twin.prepare_diagnosis())
    assert res["found"], res.get("reason")
    assert abs(res["origin"]["cell"] - planted) <= twin.cfg.diagnosis.bottleneck_search_radius_cells
    assert res["capacity_at_origin"]["significant_drop_detected"]
    cf = res["counterfactual"]
    assert {r["rung_name"] for r in cf["rungs"]} == {twin.cfg.diagnosis.top_ranked_name_capacity, twin.cfg.diagnosis.top_ranked_name_demand, twin.cfg.diagnosis.top_ranked_name_signal}
    assert {a["policy"] for a in cf["signal_alternatives"]} == set(twin.policies) - {"fixed-time"}
    assert cf["verdict"]["status"] in {"conclusive", "tied", "inconclusive"}
    assert res["validation"]["origin_error_cells"] <= 1
    assert res["validation"]["top_matches"], cf["rungs"]
    assert all(e["timestamp_sec"] >= res["jam"]["start_time_sec"] for e in res["origin"]["evidence_trail"])


def test_retimed_demand_is_never_negative(configs):
    twin = LiveTwin(*configs)
    retimed = twin.background.retimed()
    assert all(retimed.get_rate(float(t)) >= 0.0 for t in range(-600, 2400, 30))


def test_diagnosis_without_jam_reports_not_found(configs):
    from citysense.live.diagnose import diagnose
    twin = LiveTwin(*configs)
    twin.control(None, None, "control", None, False)
    twin.advance(60)
    res = diagnose(twin, 3, twin.prepare_diagnosis())
    assert res["found"] is False and res["reason"]


def test_candidate_search_recovers_bottleneck_when_tracer_names_downstream_queue(configs, monkeypatch):
    """Chained bottlenecks: if the tracer names a downstream queue (here forced to the next stop line), the
    counterfactual candidate search must move the origin back to the planted bottleneck."""
    from citysense.diagnosis.shockwave import ShockwaveTracer
    from citysense.live.diagnose import diagnose
    twin = LiveTwin(*configs)
    twin.control(None, "fixed-time", "recurring_capacity_drop", None, False)
    planted = twin.cfg.scenarios["recurring_capacity_drop"].target_cell_id
    points = _run_until_mature(twin, twin.live.diagnosis_min_jam_age_sec + 60)
    stop_lines = sorted(jc["arterial_approach"] for jc in twin.net.junction_cells.values())
    wrong = next(c for c in stop_lines if c > planted + twin.cfg.diagnosis.bottleneck_search_radius_cells)
    original = ShockwaveTracer.trace_origin

    def chained(self, cluster, obs, cell_len):
        res = original(self, cluster, obs, cell_len)
        res.origin_cell = wrong
        return res

    monkeypatch.setattr(ShockwaveTracer, "trace_origin", chained)
    cell = _near(points, planted)
    res = diagnose(twin, cell, twin.prepare_diagnosis())
    assert res["found"], res.get("reason")
    assert res["origin"]["traced_front_cell"] == wrong
    assert abs(res["origin"]["cell"] - planted) <= twin.cfg.diagnosis.bottleneck_search_radius_cells, res["origin"]["candidates"]
    # Localisation is what the search fixes. Ranking at a cut that sits on a signal stop line can still favour
    # the signal rung (relief, not blame), so require capacity to be a material cause rather than the top one.
    rungs = {r["rung_name"]: r["mean_recovery_vhr"] for r in res["counterfactual"]["rungs"]}
    assert rungs[twin.cfg.diagnosis.top_ranked_name_capacity] >= 0.5 * max(rungs.values()), rungs
    wrong_cand = next(c for c in res["origin"]["candidates"] if c["cell"] == wrong)
    assert wrong_cand["mean_recovery_vhr"] < rungs[twin.cfg.diagnosis.top_ranked_name_capacity]
