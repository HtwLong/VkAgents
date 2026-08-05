import json

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from cvmodellearning.jobs import error_persistence
from cvmodellearning.jobs.job_manager import JobManager


def test_route_persists_http_error_for_body_job_id(monkeypatch, tmp_path):
    monkeypatch.setattr(error_persistence, "run_dir", lambda job_id: tmp_path / job_id)

    router = APIRouter(route_class=error_persistence.ErrorPersistingRoute)

    @router.post("/fail")
    async def fail(payload: dict):
        (tmp_path / payload["job_id"]).mkdir(parents=True, exist_ok=True)
        raise HTTPException(status_code=422, detail={"message": "invalid input"})

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/fail", json={"job_id": "job-1"}
    )

    assert response.status_code == 422
    errors = json.loads((tmp_path / "job-1" / "errors.json").read_text())
    assert errors[0]["job_id"] == "job-1"
    assert errors[0]["endpoint"] == "/fail"
    assert errors[0]["status_code"] == 422
    assert errors[0]["error"] == {"message": "invalid input"}


def test_route_persists_unhandled_error_for_query_job_id(monkeypatch, tmp_path):
    def fake_run_dir(job_id):
        path = tmp_path / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(error_persistence, "run_dir", fake_run_dir)
    router = APIRouter(route_class=error_persistence.ErrorPersistingRoute)

    @router.get("/fail")
    async def fail(job_id: str):
        raise RuntimeError("unexpected failure")

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app, raise_server_exceptions=False).get(
        "/fail", params={"job_id": "job-2"}
    )

    assert response.status_code == 500
    errors = json.loads((tmp_path / "job-2" / "errors.json").read_text())
    assert errors[0]["error_type"] == "RuntimeError"
    assert "unexpected failure" in errors[0]["error"]


def test_job_manager_keeps_background_error():
    manager = JobManager()
    manager._jobs = {}
    manager.create_job("job-3")
    manager.update_job_status("job-3", "error", error="training failed")

    assert manager.get_job("job-3") == {
        "status": "error",
        "error": "training failed",
    }
