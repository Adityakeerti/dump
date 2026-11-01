"""FastAPI app for the live twin: REST snapshots, control, agent trip intake and a WebSocket stream."""

import asyncio
import contextlib
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from citysense.api.service import create_app
from citysense.config.schemas import CitySenseConfig
from citysense.live.config import LiveConfig
from citysense.live.diagnose import diagnose
from citysense.live.twin import STATUS_RUNNING, LiveTwin


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControlRequest(StrictModel):
    action: Literal["play", "pause", "step", "reset"] | None = None
    policy: str | None = None
    scenario: str | None = None
    steps_per_tick: int | None = None
    learning: bool | None = None


class TripRequest(StrictModel):
    cluster_id: str = Field(min_length=1)
    cluster_name: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    source_id: int
    vehicles: float
    depart_in_sec: float


class DiagnoseRequest(StrictModel):
    cell: int


class TripsRequest(StrictModel):
    trips: list[TripRequest]


def create_live_app(cfg: CitySenseConfig, cfg_hash: str, live: LiveConfig) -> FastAPI:
    twin = LiveTwin(cfg, cfg_hash, live)
    paths = live.paths
    clients: set[WebSocket] = set()

    async def loop() -> None:
        last_event = twin.events[-1]["id"] if twin.events else 0
        last_decision = 0
        last_finish = 0
        while True:
            if twin.status == STATUS_RUNNING:
                twin.advance(twin.steps_per_tick)
            if clients:
                new_events = [e for e in twin.events if e["id"] > last_event]
                new_decisions = twin.decisions(None, last_decision, live.default_events_limit)
                run_hist = twin.runs[twin.active_policy].history
                new_finished = [p for p in twin.runs[twin.active_policy].finished if p["seq"] > last_finish]
                if new_finished:
                    last_finish = max(p["seq"] for p in new_finished)
                msg = {
                    "type": "tick",
                    "state": twin.state(),
                    "events": new_events,
                    "decisions": new_decisions,
                    "history_row": run_hist[-1] if run_hist else None,
                    "finished": new_finished,
                }
                if new_decisions:
                    last_decision = new_decisions[-1]["id"]
                for ws in list(clients):
                    try:
                        await ws.send_json(msg)
                    except Exception:
                        clients.discard(ws)
            if twin.events:
                last_event = twin.events[-1]["id"]
            await asyncio.sleep(live.tick_interval_sec)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(loop())
        yield
        task.cancel()

    app = FastAPI(title="CitySense Live Twin", version=cfg.api.api_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=live.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.twin = twin

    @app.get(paths.health)
    async def health() -> dict[str, Any]:
        return {"status": cfg.diagnosis.status_ok, "twin": twin.status, "episode": twin.episode}

    @app.get(paths.topology)
    async def topology() -> dict[str, Any]:
        return twin.topology()

    @app.get(paths.state)
    async def state() -> dict[str, Any]:
        return twin.state()

    @app.get(paths.events)
    async def events(after: int = 0, limit: int = live.default_events_limit) -> dict[str, Any]:
        items = [e for e in twin.events if e["id"] > after][-max(1, limit):]
        return {"events": items}

    @app.get(paths.decisions)
    async def decisions(
        policy: str | None = None, after: int = 0, limit: int = live.default_events_limit
    ) -> dict[str, Any]:
        if policy is not None and policy not in twin.runs:
            raise HTTPException(status_code=cfg.api.error_code_bad_request, detail=f"Unknown policy {policy}")
        return {"decisions": twin.decisions(policy, after, max(1, limit))}

    @app.get(paths.episodes)
    async def episodes() -> dict[str, Any]:
        return {"episodes": list(twin.episodes), "learning": twin.learning_view()}

    @app.get(paths.platoons)
    async def platoons(limit: int = live.default_events_limit) -> dict[str, Any]:
        return {
            "active": twin.state()["platoons"],
            "finished": twin.finished_platoons(max(1, limit)),
            "episode": twin.episode,
        }

    @app.post(paths.control)
    async def control(req: ControlRequest) -> dict[str, Any]:
        try:
            twin.control(req.action, req.policy, req.scenario, req.steps_per_tick, req.learning)
        except ValueError as exc:
            raise HTTPException(status_code=cfg.api.error_code_bad_request, detail=str(exc)) from exc
        return twin.state()

    @app.post(paths.trips)
    async def trips(req: TripsRequest) -> dict[str, Any]:
        try:
            ids = twin.add_trips([t.model_dump() for t in req.trips])
        except ValueError as exc:
            raise HTTPException(status_code=cfg.api.error_code_bad_request, detail=str(exc)) from exc
        return {"trip_ids": ids, "sim_time": twin.state()["sim_time"], "episode": twin.episode}

    diagnose_lock = asyncio.Lock()

    @app.post(paths.diagnose)
    async def diagnose_jam(req: DiagnoseRequest) -> dict[str, Any]:
        if not 0 <= req.cell < twin.net.n_cells:
            raise HTTPException(status_code=cfg.api.error_code_bad_request, detail=f"Unknown cell {req.cell}")
        async with diagnose_lock:  # one counterfactual ladder at a time (CPU-bound)
            prepared = twin.prepare_diagnosis()
            result = await asyncio.to_thread(diagnose, twin, req.cell, prepared)
        if result.get("found"):
            top = result["situation"]["top_hypothesis"] or {}
            twin.emit(
                "diagnosis",
                "root_cause_traced",
                f"Root cause traced for cell {req.cell}: origin cell {result['origin']['cell']}, "
                f"top hypothesis {top.get('hypothesis_type')}",
                origin_cell=result["origin"]["cell"],
            )
        return result

    @app.websocket(paths.stream)
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_json(
            {
                "type": "hello",
                "topology": twin.topology(),
                "state": twin.state(),
                "events": list(twin.events)[-live.default_events_limit:],
                "decisions": twin.decisions(None, 0, live.default_events_limit),
                "history": twin.history(),
                "episodes": list(twin.episodes),
                "finished_platoons": twin.finished_platoons(live.default_events_limit),
            }
        )
        clients.add(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive; control goes through REST
        except WebSocketDisconnect:
            clients.discard(ws)

    app.mount(paths.diagnosis_mount, create_app(cfg, cfg_hash))
    return app
