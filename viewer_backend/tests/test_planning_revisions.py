import importlib
import json
import sys

from viewer_backend.schemas import RevisionChange, RevisionPlan


def _plan():
    return RevisionPlan(
        required_text="Use batch size 4",
        summary="Reduce the batch size",
        restart_from="choose-hyperparameters",
        changes=[RevisionChange(
            id="batch", target_step="choose-hyperparameters",
            field="hpo_config.batch_size", operation="set", value=4,
            strength="required", summary="Use batch size four",
        )],
    )


def test_revision_activation_archives_downstream_and_verifies(tmp_path, monkeypatch):
    from test_api import make_client
    client = make_client(tmp_path, monkeypatch)
    directory = tmp_path / "revision" / "artifacts" / "planning"
    directory.mkdir(parents=True)
    context = {"task": "detection", "classes": ["person"], "hpo_config": {"batch_size": 16}}
    (directory / "STATE_04_DATASET_SELECTION.json").write_text(json.dumps(context))
    (directory / "STATE_05_HYPERPARAMETERS.json").write_text(json.dumps(context))
    (directory / "RESULT_HYPERPARAMETERS.json").write_text("{}")
    response = client.post("/api/v1/planning/activate-revision", json={
        "job_id": "revision", "context": context, "plan": _plan().model_dump(mode="json"),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["archived_files"]) == {"STATE_05_HYPERPARAMETERS.json", "RESULT_HYPERPARAMETERS.json"}
    assert (directory / "STATE_04_DATASET_SELECTION.json").is_file()
    revised = body["context"]
    revised["hpo_config"] = {"batch_size": 4}
    verification = client.post("/api/v1/planning/verify-revision", json={"context": revised})
    assert verification.status_code == 200
    assert verification.json()["satisfied"] is True


def test_revision_rejected_after_execution_artifacts_exist(tmp_path, monkeypatch):
    from test_api import make_client
    client = make_client(tmp_path, monkeypatch)
    run = tmp_path / "revision-executed"
    directory = run / "artifacts" / "planning"
    directory.mkdir(parents=True)
    (directory / "STATE_04_DATASET_SELECTION.json").write_text("{}")
    (run / "artifacts" / "download_report.json").write_text("{}")
    response = client.post("/api/v1/planning/activate-revision", json={
        "job_id": "revision-executed", "context": {"task": "detection"},
        "plan": _plan().model_dump(mode="json"),
    })
    assert response.status_code == 409


def test_revision_plan_endpoint_returns_validated_atomic_changes(tmp_path, monkeypatch):
    from test_api import make_client
    client = make_client(tmp_path, monkeypatch)
    planning = sys.modules["viewer_backend.routers.planning"]
    schemas = importlib.import_module("viewer_backend.schemas")

    async def fake_structured_call(**kwargs):
        assert kwargs["response_model"] is schemas.RevisionPlan
        return schemas.RevisionPlan.model_validate(_plan().model_dump(mode="json"))

    monkeypatch.setattr(planning, "structured_call", fake_structured_call)
    response = client.post("/api/v1/planning/plan-revision", json={
        "job_id": "revision-plan",
        "context": {"task": "detection", "hpo_config": {"batch_size": 16}},
        "required_changes": "Use batch size 4",
        "preferences": "",
        "requested_target": "automatic",
    })
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["restart_from"] == "choose-hyperparameters"
    assert client.get("/api/v1/capabilities").json()["planning_revisions"] is True


def test_historical_revision_fork_copies_only_predecessors_and_writes_lineage(tmp_path, monkeypatch):
    from test_api import make_client
    client = make_client(tmp_path, monkeypatch)
    parent = tmp_path / "historical"
    directory = parent / "artifacts" / "planning"
    directory.mkdir(parents=True)
    context = {"task": "detection", "classes": ["person"], "user_query": "Detect people", "hpo_config": {"batch_size": 16}}
    for name in (
        "STATE_01_INTERPRETATION.json", "STATE_02_DATA_CHECK.json",
        "STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json",
        "STATE_05_HYPERPARAMETERS.json",
    ):
        (directory / name).write_text(json.dumps(context))
    (directory / "DATA_CHECK_QUERY.sparql").write_text("SELECT * WHERE {}")
    (directory / "STATE_04_PREPROCESSING.json").write_text("{}")
    (directory / "RESULT_HYPERPARAMETERS.json").write_text("{}")
    (parent / "artifacts" / "post_training_assessment.json").write_text(json.dumps({"assessment_id": "assessment-1"}))

    response = client.post("/api/v1/planning/fork-revision", json={
        "parent_job_id": "historical", "assessment_id": "assessment-1",
        "plan": _plan().model_dump(mode="json"),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    child = tmp_path / body["job_id"]
    assert body["reused_steps"] == ["task-interpretation", "check-data", "model-selection", "dataset-selection"]
    assert (child / "artifacts" / "planning" / "STATE_04_DATASET_SELECTION.json").is_file()
    assert not (child / "artifacts" / "planning" / "RESULT_HYPERPARAMETERS.json").exists()
    assert json.loads((child / "lineage.json").read_text())["parent_job_id"] == "historical"
    assert (directory / "RESULT_HYPERPARAMETERS.json").is_file()
    assert client.get("/api/v1/capabilities").json()["assessment_revision"] is True
