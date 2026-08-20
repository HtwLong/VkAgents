import sys

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
