import json

import pytest
from fastapi import HTTPException

from routers import runs
from cvmodellearning.schemas.post_training_assessment import PostTrainingAssessment
from cvmodellearning.schemas.revision import RevisionChange, RevisionPlan


def test_run_snapshot_groups_persisted_outputs(monkeypatch, tmp_path):
    run = tmp_path / "job-123"
    planning = run / "artifacts" / "planning"
    planning.mkdir(parents=True)
    state = {
        "task": "classification",
        "hpo_config": {"model_name": "resnet50"},
        "model_selection_decision_evidence": {"decision_type": "model_selection"},
        "selected_data": [{"class_name": "cat", "sources": [{"dataset_name": "cats"}]}],
        "step_history": ["Data Selection Rationale: Use the persisted cat dataset."],
    }
    (planning / "STATE_05_HYPERPARAMETERS.json").write_text(json.dumps(state))
    (planning / "RESULT_HYPERPARAMETERS.json").write_text(json.dumps(state["hpo_config"]))
    usage = {
        "schema_version": 1,
        "job_id": "job-123",
        "scope": "planning",
        "currency": "USD",
        "totals": {"requests": 2, "input_tokens": 100, "output_tokens": 20,
                   "total_tokens": 120, "calculated_cost_usd": "0.00001300"},
    }
    (planning / "planning_llm_usage.json").write_text(json.dumps(usage))

    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("job-123")

    assert snapshot["job_id"] == "job-123"
    assert snapshot["status"] == "waiting"
    assert snapshot["chosen_parameters"] == state["hpo_config"]
    assert snapshot["steps"]["choose-hyperparameters"]["status"] == "done"
    assert snapshot["steps"]["choose-hyperparameters"]["outputs"][0].startswith(
        "Hyperparameter planning output:\n"
    )
    assert "files" not in snapshot["steps"]["choose-hyperparameters"]
    assert snapshot["decision_evidence"]["model-selection"] == state[
        "model_selection_decision_evidence"
    ]
    assert snapshot["decision_evidence"]["dataset-selection"]["rationale"] == (
        "Use the persisted cat dataset."
    )
    assert snapshot["steps"]["choose-hyperparameters"]["duration_ms"] is None
    assert snapshot["post_training_assessment"] is None
    assert snapshot["assessment_eligibility"]["eligible"] is False
    assert snapshot["planning_llm_usage"] == usage


def test_historical_completed_run_exposes_saved_assessment(monkeypatch, tmp_path):
    run = tmp_path / "historical"
    planning = run / "artifacts" / "planning"
    planning.mkdir(parents=True)
    state = {"user_query": "Detect cars", "task": "detection", "classes": ["car"]}
    (planning / "STATE_04_DATASET_SELECTION.json").write_text(json.dumps(state))
    (run / "artifacts" / "evaluation_report.json").write_text(json.dumps({
        "job_id": "historical", "task": "detection", "model": {"name": "yolov8_n"},
        "classes": ["car"], "metrics": {"mAP": 0.4}, "per_class": [],
        "confusion_matrix": [], "training_history": [], "dataset": {"splits": {}},
        "configuration": {},
    }))
    assessment = {
        "assessment_id": "assessment-1", "job_id": "historical",
        "created_at": "2026-01-01T00:00:00Z", "verdict": "partially_satisfied",
        "summary": "More accuracy is needed.", "requirements": [],
        "recommended_plan": None, "limitations": [],
    }
    (run / "artifacts" / "post_training_assessment.json").write_text(json.dumps(assessment))
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("historical")

    assert snapshot["post_training_assessment"] == assessment
    assert snapshot["assessment_eligibility"]["eligible"] is True
    assert snapshot["assessment_eligibility"]["can_create_revision"] is True


def test_assessment_recommendation_is_limited_to_one_supported_area():
    model_change = RevisionChange(
        id="model", target_step="model-selection", field="model_name",
        operation="set", value="faster_rcnn_r50", strength="required",
        summary="Use Faster R-CNN.",
    )
    hpo_change = RevisionChange(
        id="epochs", target_step="choose-hyperparameters", field="hpo_config.num_epochs",
        operation="set", value=10, strength="required", summary="Use ten epochs.",
    )
    assessment = PostTrainingAssessment(
        verdict="not_satisfied", summary="The target was missed.",
        recommended_plan=RevisionPlan(
            required_text="Change the model and epochs.", summary="Too broad.",
            restart_from="model-selection", changes=[model_change, hpo_change],
        ),
    )

    with pytest.raises(HTTPException) as exc:
        runs._validate_assessment_recommendation(
            assessment, {"hpo_config": {"num_epochs": 5}}
        )

    assert exc.value.status_code == 502
    assert "exactly one area" in str(exc.value.detail)


def test_run_snapshot_uses_saved_config_when_planning_state_is_stale(monkeypatch, tmp_path):
    run = tmp_path / "adjusted-run"
    planning = run / "artifacts" / "planning"
    planning.mkdir(parents=True)
    (planning / "STATE_05_HYPERPARAMETERS.json").write_text(
        json.dumps({"hpo_config": {"amp": True, "multi_scale": 0.25}})
    )
    adjusted = {"amp": False, "multi_scale": 0.0}
    (planning / "RESULT_HYPERPARAMETERS.json").write_text(json.dumps(adjusted))
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("adjusted-run")

    assert snapshot["chosen_parameters"] == adjusted


def test_step_timing_is_persisted_and_restored(monkeypatch, tmp_path):
    run = tmp_path / "timed-run"
    run.mkdir()
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    saved = runs.save_step_timing(
        "timed-run",
        "task-interpretation",
        runs.StepTimingUpdate(duration_ms=1234, status="done"),
    )
    snapshot = runs.run_snapshot("timed-run")

    assert saved == {
        "step_id": "task-interpretation",
        "duration_ms": 1234,
        "status": "done",
    }
    assert snapshot["steps"]["task-interpretation"]["duration_ms"] == 1234


def test_step_timing_rejects_unknown_step(monkeypatch, tmp_path):
    (tmp_path / "timed-run").mkdir()
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    with pytest.raises(HTTPException) as exc:
        runs.save_step_timing(
            "timed-run",
            "unknown-step",
            runs.StepTimingUpdate(duration_ms=1, status="done"),
        )

    assert exc.value.status_code == 400


def test_partial_download_artifacts_are_not_treated_as_complete(monkeypatch, tmp_path):
    run = tmp_path / "partial-download"
    (run / "artifacts").mkdir(parents=True)
    (run / "data").mkdir()
    (run / "artifacts" / "download_report.json").write_text(
        json.dumps({"complete": False, "sources": []})
    )
    (run / "data" / "dataset_manifest.json").write_text(
        json.dumps({"samples": [{"image_path": "one.jpg"}]})
    )
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("partial-download")

    assert snapshot["steps"]["download-data"]["status"] == "pending"


def test_preparation_requires_summary_and_all_split_files(monkeypatch, tmp_path):
    run = tmp_path / "partial-preparation"
    data = run / "data"
    data.mkdir(parents=True)
    (data / "preparation_summary.json").write_text(json.dumps({"task": "classification"}))
    (data / "train_labels.csv").write_text("image_filename,labels\na.jpg,cat\n")
    (data / "val_labels.csv").write_text("image_filename,labels\nb.jpg,cat\n")
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("partial-preparation")

    assert snapshot["steps"]["prepare-data"]["status"] == "pending"


def test_clear_run_errors_archives_previous_failures(monkeypatch, tmp_path):
    run = tmp_path / "failed-run"
    run.mkdir()
    (run / "errors.json").write_text(json.dumps([{"error": "failure"}]))
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    result = runs.clear_run_errors("failed-run")

    assert result["status"] == "ok"
    assert not (run / "errors.json").exists()
    assert json.loads((run / "errors.previous.json").read_text())[0]["error"] == "failure"


def test_completed_run_restores_all_steps_and_deliverables(monkeypatch, tmp_path):
    run = tmp_path / "completed-run"
    planning = run / "artifacts" / "planning"
    planning.mkdir(parents=True)
    (planning / "STATE_05_HYPERPARAMETERS.json").write_text(
        json.dumps({"hpo_config": {"model_name": "resnet50"}})
    )
    (planning / "RESULT_HYPERPARAMETERS.json").write_text("{}")
    (run / "artifacts" / "best_model.pth").write_bytes(b"checkpoint")
    (run / "artifacts" / "evaluation_report.json").write_text(
        json.dumps({"job_id": "completed-run", "metrics": {"accuracy": 0.9}})
    )
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("completed-run")

    assert snapshot["status"] == "done"
    assert snapshot["evaluation_report"]["metrics"]["accuracy"] == 0.9
    assert {artifact["id"] for artifact in snapshot["artifacts"]} == {
        "model",
        "hyperparameters",
    }

def test_legacy_detection_report_is_loaded_but_evaluation_can_be_backfilled(monkeypatch, tmp_path):
    run = tmp_path / "legacy-detection"
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "evaluation_report.json").write_text(json.dumps({
        "job_id": "legacy-detection", "task": "detection", "classes": ["car"],
        "metrics": {"map": 0.2},
    }))
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("legacy-detection")

    assert snapshot["evaluation_report"]["per_class"] == []
    assert snapshot["evaluation_report"]["schema_version"] == 1
    assert snapshot["steps"]["running-evaluation"]["status"] == "pending"


def test_completed_legacy_detection_run_does_not_hide_backfill_action(monkeypatch, tmp_path):
    run = tmp_path / "legacy-complete"
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "best_model.pt").write_bytes(b"checkpoint")
    (run / "artifacts" / "evaluation_report.json").write_text(json.dumps({
        "job_id": "legacy-complete", "task": "detection", "classes": ["car"],
        "metrics": {"map": 0.2},
    }))
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)

    snapshot = runs.run_snapshot("legacy-complete")

    assert snapshot["status"] != "done"
    assert snapshot["steps"]["running-evaluation"]["status"] == "pending"
    assert snapshot["steps"]["preparing-results"]["status"] == "pending"


@pytest.mark.parametrize("job_id", ["", "../secret", "nested/run", "bad id"])
def test_run_snapshot_rejects_invalid_job_ids(monkeypatch, tmp_path, job_id):
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)
    with pytest.raises(HTTPException) as exc:
        runs.run_snapshot(job_id)
    assert exc.value.status_code == 400
