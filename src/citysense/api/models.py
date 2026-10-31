"""Pydantic API request and response data models for asynchronous rehearsal service."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class StrictAPIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RehearsalRequest(StrictAPIModel):
    corridor_name: str
    scenario_name: str
    start_time_sec: float
    end_time_sec: float
    seed: int


class JobStatusResponse(StrictAPIModel):
    job_id: str
    status: str
    created_at_sec: float
    error: str | None = None


class SituationResponse(StrictAPIModel):
    job_id: str
    situation: dict[str, Any]
    approval_status: str
    audit_entries: list[dict[str, Any]]
