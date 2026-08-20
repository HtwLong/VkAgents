import sys
import json
from pathlib import Path

from viewer_backend.planning_contracts import (
    ClassificationHPOConfig,
    DetectionHPOConfig,
    PipelineState,
    VQAHPOConfig,
)


def test_pipeline_state_preserves_original_planning_sections():
    expected = {
        "available_data", "selected_data", "performance_requirements",
        "constraint_strengths", "robustness_requirements", "deployment_constraints",
        "available_hardware", "model_requirements", "training_hardware",
        "model_selection_graph_context", "dataset_selection_graph_context",
        "hyperparameter_graph_context", "selected_model_info", "hpo_config",
        "hpo_decision", "model_selection_decision_evidence",
        "dataset_selection_decision_evidence", "hyperparameter_decision_evidence",
        "dataset_profile", "data_plan_constraints", "revision",
    }
    assert expected <= PipelineState.model_fields.keys()


def test_hpo_contracts_are_not_reduced_to_basic_fields():
    assert len(ClassificationHPOConfig.model_fields) == 54
    assert len(DetectionHPOConfig.model_fields) == 70
    assert len(VQAHPOConfig.model_fields) == 28
    assert {"scheduler_name", "criterion_name", "llm_field_rationales"} <= ClassificationHPOConfig.model_fields.keys()
    assert {"loss_box", "lambda_dfl", "mosaic", "nms_iou_threshold"} <= DetectionHPOConfig.model_fields.keys()
    assert {"max_seq_length", "use_lora", "lora_r", "precision"} <= VQAHPOConfig.model_fields.keys()


def test_planning_contracts_do_not_import_execution_frameworks():
    forbidden = {"torch", "torchvision", "ultralytics"}
    assert forbidden.isdisjoint(sys.modules)


def test_sanitized_original_planning_documents_validate():
    fixtures = Path(__file__).parent / "fixtures" / "planning"
    state = PipelineState.model_validate_json((fixtures / "detection_state.json").read_text())
    hpo = DetectionHPOConfig.model_validate_json((fixtures / "detection_hpo.json").read_text())
    assert state.available_data[0].sources[0].count == 64115
    assert state.hyperparameter_graph_context["candidate_recipes"]
    assert hpo.model_name == "yolo11n"
    assert hpo.model_dump()["nms_iou_threshold"] == 0.7


def test_pipeline_state_round_trip_does_not_drop_planning_sections():
    fixture = Path(__file__).parent / "fixtures" / "planning" / "detection_state.json"
    source = json.loads(fixture.read_text())
    round_tripped = PipelineState.model_validate(source).model_dump(mode="json")
    for field in source:
        assert field in round_tripped
