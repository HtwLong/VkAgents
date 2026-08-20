from __future__ import annotations

import importlib
import json
import sys

from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("VIEWER_RUNS_DIR", str(tmp_path))
    for name in list(sys.modules):
        if name == "viewer_backend.settings" or name.startswith("viewer_backend.routers") or name in {
            "viewer_backend.api", "viewer_backend.store", "viewer_backend.llm"
        }:
            sys.modules.pop(name)
    api = importlib.import_module("viewer_backend.api")
    return TestClient(api.app)


def seed_run(root, job_id="example"):
    planning = root / job_id / "artifacts" / "planning"
    planning.mkdir(parents=True)
    state = {
        "user_query": "Classify cats and dogs",
        "task": "classification",
        "classes": ["cat", "dog"],
        "hpo_config": {"model_name": "resnet50"},
    }
    (planning / "STATE_05_HYPERPARAMETERS.json").write_text(json.dumps(state))
    (planning / "RESULT_HYPERPARAMETERS.json").write_text(json.dumps(state["hpo_config"]))
    artifacts = root / job_id / "artifacts"
    (artifacts / "evaluation_report.json").write_text(json.dumps({
        "job_id": job_id,
        "task": "classification",
        "metrics": {"accuracy": 0.9},
    }))


def test_health_and_capabilities_do_not_import_torch(tmp_path, monkeypatch):
    sys.modules.pop("torch", None)
    client = make_client(tmp_path, monkeypatch)
    assert "torch" not in sys.modules
    assert client.get("/health").json() == {"status": "ok", "mode": "viewer", "pytorch": False}
    assert client.get("/api/v1/capabilities").json()["training"] is False
    assert "torch" not in sys.modules


def test_all_heavy_operations_are_forbidden(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    cases = [
        ("post", "/api/v1/download-data"),
        ("post", "/api/v1/prepare-data"),
        ("post", "/api/v1/train/start"),
        ("post", "/api/v1/evaluate"),
        ("post", "/api/v1/load-model"),
        ("post", "/api/v1/infer"),
    ]
    for method, path in cases:
        response = getattr(client, method)(path)
        assert response.status_code == 403, (path, response.text)
        assert "No execution" in response.json()["detail"]


def test_historical_run_and_report_can_be_read(tmp_path, monkeypatch):
    seed_run(tmp_path)
    client = make_client(tmp_path, monkeypatch)
    snapshot = client.get("/api/v1/runs/example")
    assert snapshot.status_code == 200
    assert snapshot.json()["evaluation_report"]["metrics"]["accuracy"] == 0.9
    assert snapshot.json()["artifacts"]
    report = client.get("/api/v1/evaluate/example/report")
    assert report.status_code == 200
    assert report.json()["task"] == "classification"


def test_checkpoint_downloads_are_not_exposed(tmp_path, monkeypatch):
    seed_run(tmp_path)
    checkpoint = tmp_path / "example" / "artifacts" / "best_model.pt"
    checkpoint.write_bytes(b"not a real model")
    client = make_client(tmp_path, monkeypatch)
    manifest = client.get("/artifacts/example/manifest").json()
    assert all(item["filename"] != "best_model.pt" for item in manifest["artifacts"])


def test_graphrag_planning_routes_ground_and_validate_decisions(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    planning = sys.modules["viewer_backend.routers.planning"]
    schemas = importlib.import_module("viewer_backend.schemas")

    async def fake_structured_call(**kwargs):
        model = kwargs["response_model"]
        if model is schemas.ModelPlan:
            return model(
                model_id="yolo11n",
                display_name="YOLO11n",
                family="YOLO11",
                rationale="Small ontology candidate suitable for the latency constraint.",
            )
        if model is schemas.DatasetPlan:
            return model(
                sources=[{
                    "dataset_name": "BDD100K detection train",
                    "classes": ["traffic light"],
                    "rationale": "Ontology-aligned street-scene source.",
                }],
                rationale="Use domain-aligned evidence without claiming availability.",
            )
        if model is schemas.HyperparameterPlan:
            return model(
                model_name="yolo11n",
                epochs=100,
                batch_size=16,
                learning_rate=0.01,
                optimizer="auto",
                image_size=640,
                rationale="Values remain within the selected ontology recipe.",
            )
        raise AssertionError(model)

    monkeypatch.setattr(planning, "structured_call", fake_structured_call)
    context = {
        "user_query": "Detect traffic lights in urban street scenes with low latency.",
        "task": "detection",
        "classes": ["traffic light"],
        "application_domain": "urban street scenes",
        "performance_requirements": {"latency_category": "Low"},
        "deployment_constraints": {"max_runtime_memory_mb": 6144},
    }
    body = {"job_id": "graph-plan", "context": context, "use_graphrag": True}
    model_response = client.post("/api/v1/planning/select-model", json=body)
    assert model_response.status_code == 200, model_response.text
    context = model_response.json()["context"]
    assert context["selected_model_info"]["id"] == "yolo11n"
    assert context["model_selection_graph_context"]["candidate_models"]

    body["context"] = context
    dataset_response = client.post("/api/v1/planning/select-datasets", json=body)
    assert dataset_response.status_code == 200, dataset_response.text
    context = dataset_response.json()["context"]
    assert context["selected_data"][0]["dataset_name"] == "bdd_100k_det_train"
    assert context["dataset_selection_graph_context"]["matched_domains"]

    body["context"] = context
    hpo_response = client.post("/api/v1/planning/choose-hyperparameters", json=body)
    assert hpo_response.status_code == 200, hpo_response.text
    context = hpo_response.json()["context"]
    assert context["hpo_config"]["ontology_recipe_id"] == "ultralytics_yolo_detection_finetune_balanced"
    assert context["hyperparameter_graph_context"]["candidate_recipes"]


def test_ontology_recipe_bounds_reject_invalid_values():
    ontology = importlib.import_module("viewer_backend.graphrag.ontology")
    store = ontology.OntologyStore()
    recipe = store.recipe_context({
        "task": "detection",
        "selected_model_info": {"family": "YOLO11"},
    })["candidate_recipes"][0]
    assert store.validate_hyperparameters({"learning_rate": 2.0}, recipe) == [
        "learning_rate=2 is above ontology maximum 0.01"
    ]


def test_all_llm_response_models_have_strict_openai_schemas():
    from openai.lib._pydantic import to_strict_json_schema

    schemas = importlib.import_module("viewer_backend.schemas")

    def assert_strict(node, path="$"):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, path
            for key, value in node.items():
                assert_strict(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                assert_strict(value, f"{path}[{index}]")

    for model in (
        schemas.CompletenessDecision,
        schemas.TaskInterpretation,
        schemas.ModelPlan,
        schemas.DatasetPlan,
        schemas.HyperparameterPlan,
        schemas.AssessmentDraft,
    ):
        assert_strict(to_strict_json_schema(model))
