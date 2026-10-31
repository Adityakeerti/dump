"""Gate tests for Module 11: FastAPI Rehearsal Service."""

from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from citysense.api.service import create_app
from citysense.config.loader import load_citysense_config


CONFIG_DIR = Path("config")
PROJECT_PATH = CONFIG_DIR / "project.yaml"
ENGINE_PATH = CONFIG_DIR / "engine.yaml"
NETWORK_PATH = CONFIG_DIR / "network.yaml"
DEMAND_PATH = CONFIG_DIR / "demand.yaml"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"
OBSERVATION_PATH = CONFIG_DIR / "observation.yaml"
DIAGNOSIS_PATH = CONFIG_DIR / "diagnosis.yaml"
RL_PATH = CONFIG_DIR / "rl.yaml"
API_PATH = CONFIG_DIR / "api.yaml"


@pytest.fixture(scope="module")
def api_client():
    cfg, cfg_hash = load_citysense_config(
        project_path=PROJECT_PATH,
        engine_path=ENGINE_PATH,
        network_path=NETWORK_PATH,
        demand_path=DEMAND_PATH,
        scenarios_dir=SCENARIOS_DIR,
        observation_path=OBSERVATION_PATH,
        diagnosis_path=DIAGNOSIS_PATH,
        rl_path=RL_PATH,
        api_path=API_PATH,
    )
    app = create_app(cfg, cfg_hash)
    return TestClient(app), cfg


def test_submit_poll_and_fetch_round_trip(api_client):
    """Verify submit, poll, and fetch round trip via FastAPI TestClient."""
    client, cfg = api_client

    payload = {
        "corridor_name": "Metropolitan Corridor",
        "scenario_name": "recurring_capacity_drop",
        "start_time_sec": 300.0,
        "end_time_sec": 340.0,
        "seed": 42,
    }

    # Submit
    res = client.post(cfg.api.rehearsal_endpoint_path, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    job_id = data["job_id"]

    # In TestClient, BackgroundTasks execute during request-response cycle
    # Poll status
    poll_url = cfg.api.jobs_endpoint_path.format(job_id=job_id)
    poll_res = client.get(poll_url)
    assert poll_res.status_code == 200
    poll_data = poll_res.json()
    assert poll_data["status"] == cfg.diagnosis.status_completed

    # Fetch result
    fetch_url = cfg.api.situation_endpoint_path.format(job_id=job_id)
    fetch_res = client.get(fetch_url)
    assert fetch_res.status_code == 200
    fetch_data = fetch_res.json()

    assert fetch_data["job_id"] == job_id
    assert "situation" in fetch_data
    assert "approval_status" in fetch_data
    assert "audit_entries" in fetch_data
    assert fetch_data["situation"]["corridor_name"] == "Metropolitan Corridor"


def test_invalid_requests_return_typed_errors(api_client):
    """Verify invalid requests return typed errors."""
    client, cfg = api_client

    # Invalid scenario name
    bad_payload = {
        "corridor_name": "Metropolitan Corridor",
        "scenario_name": "non_existent_scenario",
        "start_time_sec": 100.0,
        "end_time_sec": 200.0,
        "seed": 42,
    }
    res = client.post(cfg.api.rehearsal_endpoint_path, json=bad_payload)
    assert res.status_code == cfg.api.error_code_bad_request

    # Non-existent job
    missing_url = cfg.api.jobs_endpoint_path.format(job_id="non-existent-uuid")
    res_missing = client.get(missing_url)
    assert res_missing.status_code == cfg.api.error_code_not_found


def test_identical_request_and_seed_return_identical_result(api_client):
    """Verify identical request and seed return identical result."""
    client, cfg = api_client

    payload = {
        "corridor_name": "Metropolitan Corridor",
        "scenario_name": "recurring_capacity_drop",
        "start_time_sec": 300.0,
        "end_time_sec": 330.0,
        "seed": 77,
    }

    res1 = client.post(cfg.api.rehearsal_endpoint_path, json=payload)
    job_id1 = res1.json()["job_id"]
    sit1 = client.get(cfg.api.situation_endpoint_path.format(job_id=job_id1)).json()["situation"]

    res2 = client.post(cfg.api.rehearsal_endpoint_path, json=payload)
    job_id2 = res2.json()["job_id"]
    sit2 = client.get(cfg.api.situation_endpoint_path.format(job_id=job_id2)).json()["situation"]

    assert sit1["ranked_hypotheses"] == sit2["ranked_hypotheses"]
    assert sit1["top_hypothesis"] == sit2["top_hypothesis"]


def test_concurrent_jobs_do_not_mix_results(api_client):
    """Verify concurrent jobs do not mix results."""
    client, cfg = api_client

    payload_a = {
        "corridor_name": "Corridor A",
        "scenario_name": "recurring_capacity_drop",
        "start_time_sec": 300.0,
        "end_time_sec": 320.0,
        "seed": 10,
    }
    payload_b = {
        "corridor_name": "Corridor B",
        "scenario_name": "control",
        "start_time_sec": 100.0,
        "end_time_sec": 120.0,
        "seed": 20,
    }

    res_a = client.post(cfg.api.rehearsal_endpoint_path, json=payload_a)
    res_b = client.post(cfg.api.rehearsal_endpoint_path, json=payload_b)

    job_a = res_a.json()["job_id"]
    job_b = res_b.json()["job_id"]

    data_a = client.get(cfg.api.situation_endpoint_path.format(job_id=job_a)).json()
    data_b = client.get(cfg.api.situation_endpoint_path.format(job_id=job_b)).json()

    assert data_a["job_id"] == job_a
    assert data_b["job_id"] == job_b
    assert data_a["situation"]["corridor_name"] == "Corridor A"
    assert data_b["situation"]["corridor_name"] == "Corridor B"
    assert data_a["situation"]["situation_id"] != data_b["situation"]["situation_id"]
