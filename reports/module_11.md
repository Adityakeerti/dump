# Module 11 Report: FastAPI Rehearsal Service

## What Was Built
- Data transfer schemas in `src/citysense/api/models.py`:
  - `RehearsalRequest`: validates corridor name, scenario name, time window, and seed. Extra fields strictly forbidden.
  - `JobStatusResponse`: reports job identifier, state (PENDING, RUNNING, COMPLETED, ERROR), timestamp, and error descriptions.
  - `SituationResponse`: packages resulting Situation entity, civic approval status, and audit log entries.
- FastAPI service and execution manager in `src/citysense/api/service.py`:
  - `create_app`: builds application instance driven entirely by `APIConfig` schema.
  - Asynchronous rehearsal execution lifecycle via background workers.
  - Enforces concurrent job limits (`max_concurrent_jobs`) and returns typed HTTP error responses (`error_code_bad_request`, `error_code_not_found`, `error_code_conflict`).
  - Stores governance metadata fields (`approval_status`, `audit_entries`) without external dependencies.
- Module 11 gate tests in `tests/module_11/test_module_11.py`:
  - `test_submit_poll_and_fetch_round_trip`: verified full round trip using FastAPI TestClient.
  - `test_invalid_requests_return_typed_errors`: verified unknown scenario and invalid job ID raise appropriate 400 and 404 responses.
  - `test_identical_request_and_seed_return_identical_result`: verified deterministic repeatability.
  - `test_concurrent_jobs_do_not_mix_results`: verified multiple concurrent jobs remain isolated and cannot cross-contaminate.

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
- `tests/module_10/test_module_10.py`: 5 passed.
- `tests/module_11/test_module_11.py`: 4 passed.
- `tests/test_no_hardcoding.py`: 1 passed.
- Total suite: 54 passed in 4.38s.

## Deviations from Spec
None.

## Open Risks
None.
