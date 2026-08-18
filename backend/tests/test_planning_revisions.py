import json

from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.schemas.revision import (
    RevisionChange,
    RevisionPlan,
    earliest_revision_step,
    explicit_required_model_id,
    explicit_required_model_reference,
    hpo_override_values,
)
from routers import planning
from cvmodellearning import paths


def change(change_id, step, field, value, strength="required", operation="set"):
    return RevisionChange(
        id=change_id,
        target_step=step,
        field=field,
        operation=operation,
        value=value,
        strength=strength,
        summary=f"Change {field}",
    )


def plan(*changes):
    return RevisionPlan(
        required_text="Apply the requested changes.",
        summary="Test revision",
        restart_from=earliest_revision_step(list(changes)),
        changes=list(changes),
    )


def test_earliest_revision_step_routes_mixed_request_to_model_selection():
    changes = [
        change("hpo", "choose-hyperparameters", "hpo_config.batch_size", 16),
        change("model", "model-selection", "model_name", "yolov11m"),
    ]

    assert earliest_revision_step(changes) == "model-selection"


def test_hpo_overrides_include_only_required_set_operations():
    revision = plan(
        change("required", "choose-hyperparameters", "hpo_config.batch_size", 16),
        change(
            "preferred", "choose-hyperparameters", "hpo_config.scheduler_name",
            "cosine", strength="preferred",
        ),
    )
    context = {"revision": {"active": revision.model_dump(mode="json")}}

    assert hpo_override_values(context) == {"batch_size": 16}


def test_initial_required_dinov2_lora_is_normalized_to_executable_constraints():
    context = {
        "task": "classification",
        "user_query": "Please use the dinov2 vits14 and LoRA.",
        "model_requirements": [{
            "name": "DiNoV2 ViT-S14 with LoRA",
            "backbone": "ViT-S14",
            "requirement_strength": "required",
            "training_mode": "lora",
        }],
    }

    assert explicit_required_model_id(context) == "dinov2_vits14"
    assert explicit_required_model_reference(context) == "DiNoV2 ViT-S14 with LoRA"
    assert hpo_override_values(context) == {
        "training_mode": "lora",
        "model_weights": "default",
    }


def test_required_model_recovers_exact_identifier_from_original_query():
    context = {
        "task": "classification",
        "user_query": "Please use the dinov2 vits14 and LoRA.",
        "model_requirements": [{
            "name": "DINOv2 ViT-14",
            "requirement_strength": "required",
            "training_mode": "lora",
        }],
    }

    assert explicit_required_model_id(context) == "dinov2_vits14"


def test_preferred_initial_model_is_not_a_hard_override():
    context = {
        "task": "classification",
        "user_query": "I would prefer DINOv2 ViT-S/14 with LoRA if possible.",
        "model_requirements": [{
            "name": "DINOv2 ViT-S/14",
            "requirement_strength": "preferred",
            "training_mode": "lora",
        }],
    }

    assert explicit_required_model_id(context) is None
    assert hpo_override_values(context) == {}


def test_activate_revision_restores_predecessor_and_archives_downstream(monkeypatch, tmp_path):
    planning_dir = tmp_path / "job" / "artifacts" / "planning"
    planning_dir.mkdir(parents=True)
    predecessor = PipelineState(
        task="detection",
        classes=["car"],
        user_query="Detect cars",
    )
    (planning_dir / "STATE_02_DATA_CHECK.json").write_text(
        predecessor.model_dump_json(), encoding="utf-8"
    )
    for filename in planning.DOWNSTREAM_PLANNING_FILES["model-selection"]:
        (planning_dir / filename).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(planning, "planning_artifacts_dir", lambda _job_id: planning_dir)

    revision = plan(change("model", "model-selection", "model_name", "yolov11m"))
    response = planning.activate_revision(planning.ActivateRevisionRequest(
        context=predecessor.model_dump(mode="json"),
        plan=revision,
        job_id="job",
    ))

    restored = response["context"]
    assert restored["task"] == "detection"
    assert restored["revision"]["active"]["changes"][0]["value"] == "yolov11m"
    assert (planning_dir / "STATE_ACTIVE_REVISION.json").is_file()
    archive = planning_dir / "revisions" / response["revision_id"]
    assert (archive / "STATE_03_MODEL_SELECTION.json").is_file()
    assert not (planning_dir / "STATE_03_MODEL_SELECTION.json").exists()


def test_verify_revision_blocks_unsatisfied_required_change():
    revision = plan(
        change("batch", "choose-hyperparameters", "hpo_config.batch_size", 16),
        change(
            "schedule", "choose-hyperparameters", "hpo_config.scheduler_name",
            "cosine", strength="preferred",
        ),
    )
    state = PipelineState(
        task="classification",
        hpo_config={"batch_size": 8, "scheduler_name": "cosine"},
        revision={"active": revision.model_dump(mode="json")},
    )

    result = planning.verify_revision(
        planning.VerifyRevisionRequest(context=state.model_dump(mode="json"))
    )

    assert result["satisfied"] is False
    assert result["checks"][0]["satisfied"] is False
    assert result["checks"][1]["satisfied"] is True


def test_fork_revision_creates_child_and_canonicalizes_faster_rcnn(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    parent_planning = parent / "artifacts" / "planning"
    parent_planning.mkdir(parents=True)
    state = PipelineState(
        task="detection", classes=["car"], user_query="Detect cars accurately"
    )
    for filename in (
        "STATE_01_INTERPRETATION.json", "STATE_02_DATA_CHECK.json",
        "STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json",
        "STATE_05_HYPERPARAMETERS.json",
    ):
        (parent_planning / filename).write_text(state.model_dump_json(), encoding="utf-8")
    (parent / "artifacts" / "post_training_assessment.json").write_text(json.dumps({
        "assessment_id": "assessment-1",
    }), encoding="utf-8")
    monkeypatch.setattr(planning, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)

    revision = plan(change(
        "model", "model-selection", "model_name",
        "faster-rcnn_r50_fpn_1x_coco",
    ))
    response = planning.fork_revision(planning.ForkRevisionRequest(
        parent_job_id="parent", assessment_id="assessment-1", plan=revision,
    ))

    child = tmp_path / response["job_id"]
    assert child.is_dir()
    assert response["restart_from"] == "model-selection"
    assert response["plan"]["changes"][0]["value"] == "faster_rcnn_r50"
    assert response["reused_steps"] == ["task-interpretation", "check-data"]
    assert (child / "artifacts" / "planning" / "STATE_02_DATA_CHECK.json").is_file()
    assert not (child / "artifacts" / "planning" / "STATE_03_MODEL_SELECTION.json").exists()
    assert json.loads((child / "lineage.json").read_text())["parent_job_id"] == "parent"
