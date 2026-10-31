"""FastAPI service providing asynchronous counterfactual rehearsal jobs."""

import time
import uuid
from typing import Any
from fastapi import BackgroundTasks, FastAPI, HTTPException

from citysense.api.models import (
    JobStatusResponse,
    RehearsalRequest,
    SituationResponse,
)
from citysense.config.schemas import CitySenseConfig
from citysense.counterfactual.ladder import CounterfactualLadder
from citysense.diagnosis.capacity import CapacityEstimator
from citysense.diagnosis.clustering import JamDetector
from citysense.diagnosis.shockwave import ShockwaveTracer
from citysense.engine.ctm import NativeCTMEngine
from citysense.network.corridor import build_corridor_network
from citysense.network.demand import CorridorDemand
from citysense.observation.generator import ObservationGenerator
from citysense.scenarios.registry import create_scenario
from citysense.situations.builder import build_situation


def run_rehearsal_pipeline(
    req: RehearsalRequest,
    cfg: CitySenseConfig,
    config_hash: str,
) -> dict[str, Any]:
    """Executes the full simulation, observation, clustering, tracing, and ladder pipeline."""
    # 1. Network & Scenario
    net = build_corridor_network(cfg.network, seed=req.seed)
    demand = CorridorDemand(cfg.demand)

    if req.scenario_name not in cfg.scenarios:
        raise ValueError(f"Unknown scenario {req.scenario_name}")
    sc_cfg = cfg.scenarios[req.scenario_name]
    scenario = create_scenario(sc_cfg)

    # 2. Engine
    engine = NativeCTMEngine(
        config=cfg.engine,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    engine.set_demand_profile(demand.as_callable())
    scenario.apply_to_engine(engine, net)

    # 3. Step engine up to end_time_sec
    gen = ObservationGenerator(cfg.observation, n_cells=net.n_cells, seed=req.seed)
    total_steps = int(req.end_time_sec / cfg.engine.dt)

    all_obs = []
    snapshot = None
    snapshot_step = int(req.start_time_sec / cfg.engine.dt)

    for step in range(total_steps):
        if step == snapshot_step:
            snapshot = engine.snapshot()
        state = engine.step()
        all_obs.extend(gen.observe_step(state, cfg.engine.dt))

    if snapshot is None:
        snapshot = engine.snapshot()

    # 4. Jam detection
    detector = JamDetector(cfg.diagnosis)
    ff_speeds = {i: float(net.vf[i]) for i in range(net.n_cells)}
    pts = detector.identify_congested_points(all_obs, ff_speeds)
    clusters = detector.cluster_jams(pts, cfg.network.min_cell_length)

    # 5. Shockwave tracing
    tracer = ShockwaveTracer(cfg.engine, cfg.diagnosis)
    origin = None
    if clusters:
        primary = max(clusters, key=lambda c: c.head_cell)
        origin = tracer.trace_origin(primary, all_obs, cfg.network.min_cell_length)

    # 6. Capacity fitting
    estimator = CapacityEstimator(cfg.engine, cfg.diagnosis)
    target_c = origin.origin_cell if origin else net.arterial_cells[0]
    fit_window = (req.start_time_sec, req.end_time_sec)
    cap_res = estimator.fit_capacity(target_c, all_obs, fit_window)

    # 7. Counterfactual ladder
    ladder = CounterfactualLadder(cfg.engine, cfg.diagnosis)
    horizon = int((req.end_time_sec - req.start_time_sec) / cfg.engine.dt)
    horizon = max(1, horizon)
    ladder_res = ladder.evaluate(
        engine=engine,
        network=net,
        base_demand=demand,
        suspect_cell=target_c,
        snapshot=snapshot,
        horizon_steps=horizon,
        scenario_name=req.scenario_name,
        seed=req.seed,
    )

    # 8. Build Situation
    situation_id = f"SIT-{req.seed}-{int(req.start_time_sec)}"
    situation = build_situation(
        situation_id=situation_id,
        corridor_name=req.corridor_name,
        observations=all_obs,
        jam_clusters=clusters,
        shockwave_origin=origin,
        capacity_fit=cap_res,
        ladder_result=ladder_res,
        diag_cfg=cfg.diagnosis,
        config_hash=config_hash,
    )
    return situation.to_dict()


def create_app(config: CitySenseConfig, config_hash: str) -> FastAPI:
    """Creates the FastAPI application for async rehearsal jobs."""
    api_cfg = config.api
    app = FastAPI(title=api_cfg.api_title, version=api_cfg.api_version)

    jobs: dict[str, dict[str, Any]] = {}

    def _execute_job(job_id: str, request: RehearsalRequest) -> None:
        jobs[job_id]["status"] = config.diagnosis.status_running
        try:
            sit_dict = run_rehearsal_pipeline(request, config, config_hash)
            jobs[job_id]["status"] = config.diagnosis.status_completed
            jobs[job_id]["result"] = sit_dict
            jobs[job_id]["audit"].append(
                {
                    "action": "rehearsal_completed",
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            jobs[job_id]["status"] = config.diagnosis.status_error
            jobs[job_id]["error"] = str(e)

    @app.post(api_cfg.rehearsal_endpoint_path, response_model=JobStatusResponse)
    def submit_rehearsal(
        request: RehearsalRequest, background_tasks: BackgroundTasks
    ) -> JobStatusResponse:
        # Check concurrency limits
        active_jobs = sum(
            1
            for j in jobs.values()
            if j["status"] in (config.diagnosis.status_pending, config.diagnosis.status_running)
        )
        if active_jobs >= api_cfg.max_concurrent_jobs:
            raise HTTPException(
                status_code=api_cfg.error_code_conflict,
                detail="Maximum concurrent rehearsal jobs reached",
            )

        if request.scenario_name not in config.scenarios:
            raise HTTPException(
                status_code=api_cfg.error_code_bad_request,
                detail=f"Unknown scenario {request.scenario_name}",
            )

        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job_id": job_id,
            "status": config.diagnosis.status_pending,
            "created_at_sec": time.time(),
            "request": request,
            "result": None,
            "error": None,
            "approval_status": "PENDING_APPROVAL",
            "audit": [
                {
                    "action": "job_submitted",
                    "timestamp": time.time(),
                }
            ],
        }

        background_tasks.add_task(_execute_job, job_id, request)
        return JobStatusResponse(
            job_id=job_id,
            status=config.diagnosis.status_pending,
            created_at_sec=jobs[job_id]["created_at_sec"],
        )

    @app.get(api_cfg.jobs_endpoint_path, response_model=JobStatusResponse)
    def poll_job_status(job_id: str) -> JobStatusResponse:
        if job_id not in jobs:
            raise HTTPException(
                status_code=api_cfg.error_code_not_found,
                detail=f"Job {job_id} not found",
            )
        j = jobs[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=j["status"],
            created_at_sec=j["created_at_sec"],
            error=j["error"],
        )

    @app.get(api_cfg.situation_endpoint_path, response_model=SituationResponse)
    def fetch_situation_result(job_id: str) -> SituationResponse:
        if job_id not in jobs:
            raise HTTPException(
                status_code=api_cfg.error_code_not_found,
                detail=f"Job {job_id} not found",
            )
        j = jobs[job_id]
        if j["status"] != config.diagnosis.status_completed:
            raise HTTPException(
                status_code=api_cfg.error_code_bad_request,
                detail=f"Job {job_id} is not completed (current status: {j['status']})",
            )
        return SituationResponse(
            job_id=job_id,
            situation=j["result"],
            approval_status=j["approval_status"],
            audit_entries=j["audit"],
        )

    @app.get(api_cfg.health_endpoint_path)
    def health_check() -> dict[str, str]:
        return {"status": config.diagnosis.status_ok}

    return app
