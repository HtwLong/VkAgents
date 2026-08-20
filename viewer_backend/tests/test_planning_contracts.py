import sys
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from viewer_backend.planning_contracts import (
    ClassificationHPOConfig,
    DetectionHPOConfig,
    PipelineState,
    VQAHPOConfig,
)


ROOT = Path(__file__).resolve().parents[2]


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


def test_classification_contract_uses_original_defaults_literals_and_bounds():
    fields = ClassificationHPOConfig.model_fields
    assert fields["batch_size"].default == 32
    assert fields["weight_decay"].default == 0.0
    assert fields["training_mode"].default == "fine_tune_pretrained"
    schema = ClassificationHPOConfig.model_json_schema()
    assert schema["properties"]["model_name"]["enum"] == [
        "resnet50", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
        "densenet121", "convnext_tiny", "clip_vit_b16", "dinov2_vits14",
        "dinov2_vitb14", "vit_b_16", "swin_v2_t", "swin_v2_s",
    ]
    assert schema["properties"]["image_size"]["maximum"] == 4096


def test_vqa_contract_uses_original_defaults_and_cross_field_validation():
    fields = VQAHPOConfig.model_fields
    assert fields["batch_size"].default == 2
    assert fields["optimizer_name"].default == "adamw"
    assert fields["learning_rate"].default == 2e-5
    fixture = json.loads((Path(__file__).parent / "fixtures" / "planning" / "detection_hpo.json").read_text())
    base = {name: value for name, value in fixture.items() if name in {
        "classes", "selected_data", "num_epochs", "patience", "rationale"
    }}
    open_ended = VQAHPOConfig.model_validate({
        **base,
        "classes": [],
        "model_name": "Qwen3-VL-2B-Instruct",
    })
    assert open_ended.classes == []
    with pytest.raises(ValidationError, match="sum to 1.0"):
        VQAHPOConfig.model_validate({
            **base,
            "model_name": "Qwen3-VL-2B-Instruct",
            "train_data_ratio": 0.8,
            "val_data_ratio": 0.2,
            "test_data_ratio": 0.2,
        })
    with pytest.raises(ValidationError, match="suspiciously high"):
        VQAHPOConfig.model_validate({
            **base,
            "model_name": "Qwen3-VL-2B-Instruct",
            "learning_rate": 0.01,
        })


def test_detection_contract_uses_original_defaults_literals_and_aliases():
    fields = DetectionHPOConfig.model_fields
    assert fields["batch_size"].default == 16
    assert fields["optimizer_name"].default == "adamw"
    assert fields["scheduler_name"].default == "linear"
    assert fields["warmup_epochs"].default == 3.0
    assert fields["amp"].default is True
    assert fields["workers"].default == 8
    assert fields["max_size"].default == 1333
    schema = DetectionHPOConfig.model_json_schema()
    assert schema["properties"]["input_size"]["maximum"] == 4096
    assert schema["properties"]["model_name"]["enum"][-4:] == [
        "retinanet_r50", "faster_rcnn_r50", "ssd300", "rtdetr_hgnetv2_l",
    ]
    fixture = json.loads((Path(__file__).parent / "fixtures" / "planning" / "detection_hpo.json").read_text())
    config = DetectionHPOConfig.model_validate({
        **fixture,
        "data_plan_constraints": {
            "minimum_unique_images": 120,
            "preferred_unique_images": 240,
        },
        "scheduler_gamma": 0,
    })
    assert config.data_plan_constraints.minimum_unique_pool_images == 120
    assert config.data_plan_constraints.preferred_unique_pool_images == 240
    assert config.scheduler_gamma == 0.1


def test_planning_contracts_do_not_import_execution_frameworks():
    forbidden = {"torch", "torchvision", "ultralytics"}
    assert forbidden.isdisjoint(sys.modules)


def test_sanitized_original_planning_documents_validate():
    fixtures = Path(__file__).parent / "fixtures" / "planning"
    state = PipelineState.model_validate_json((fixtures / "detection_state.json").read_text())
    hpo = DetectionHPOConfig.model_validate_json((fixtures / "detection_hpo.json").read_text())
    assert state.available_data[0].sources[0].count == 64115
    assert state.hyperparameter_graph_context["candidate_recipes"]
    assert hpo.model_name == "yolov11_n"
    assert hpo.model_dump()["nms_iou_threshold"] == 0.7


def test_pipeline_state_round_trip_does_not_drop_planning_sections():
    fixture = Path(__file__).parent / "fixtures" / "planning" / "detection_state.json"
    source = json.loads(fixture.read_text())
    round_tripped = PipelineState.model_validate(source).model_dump(mode="json")
    for field in source:
        assert field in round_tripped


def test_interpretation_contract_matches_original_fields_requiredness_and_types():
    # These original planning modules are deliberately safe to import: they contain
    # Pydantic contracts only and do not import torch or training implementations.
    sys.path.insert(0, str(ROOT / "backend" / "src"))
    from cvmodellearning.schemas.interpretation_schema import (  # noqa: PLC0415
        DeploymentConstraints as OriginalDeploymentConstraints,
        HardwareSpecModel as OriginalHardwareSpec,
        InterpretationRequirements as OriginalInterpretationRequirements,
        PerformanceSpecModel as OriginalPerformanceSpec,
        PipelineState as OriginalPipelineState,
    )
    from viewer_backend.planning_contracts import (
        DeploymentConstraints,
        HardwareSpec,
        InterpretationRequirements,
        PerformanceSpec,
    )

    pairs = (
        (InterpretationRequirements, OriginalInterpretationRequirements),
        (PipelineState, OriginalPipelineState),
        (HardwareSpec, OriginalHardwareSpec),
        (PerformanceSpec, OriginalPerformanceSpec),
        (DeploymentConstraints, OriginalDeploymentConstraints),
    )
    for viewer, original in pairs:
        assert viewer.model_fields.keys() == original.model_fields.keys()
        assert {
            name: field.is_required() for name, field in viewer.model_fields.items()
        } == {
            name: field.is_required() for name, field in original.model_fields.items()
        }
