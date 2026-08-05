from cvmodellearning.policies.hyperparameter_policy_registry import (
    build_hyperparameter_policy_context,
    normalize_policy_rationales,
    policy_fields,
    policy_ids_by_field,
    validate_policy_rationales,
)
from cvmodellearning.graphrag.decision_evidence import (
    build_hyperparameter_decision_evidence,
)
from cvmodellearning.schemas.classification_hpo import LLMFieldRationale
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile


def _state(task: str = "classification") -> PipelineState:
    return PipelineState(
        task=task,
        application_domain="general",
        classes=["cat", "dog"],
        selected_data=[
            {
                "class_name": "cat",
                "sources": [{"dataset_name": "source", "count": 100}],
            },
            {
                "class_name": "dog",
                "sources": [{"dataset_name": "source", "count": 400}],
            },
        ],
        available_hardware={
            "hardware_category": "ConsumerCPU",
            "ram_gb": 4.0,
        },
        deployment_constraints={
            "max_runtime_memory_mb": 500.0,
            "max_cpu_latency_ms": 200.0,
        },
        training_hardware=get_training_hardware_profile("macbook_air_m4_16gb"),
    )


def test_registry_derives_profile_and_task_specific_policies():
    context = build_hyperparameter_policy_context(_state())

    assert context["profile"]["data"]["split_counts"]["train"] == 500
    assert context["profile"]["data"]["class_imbalance_ratio"] == 4.0
    assert "hpo.common.effective_batch.v1" in context["policy_ids"]
    assert "hpo.classification.freeze_by_data.v1" in context["policy_ids"]
    assert "freeze_backbone_epochs" in policy_fields(context)
    assert "mosaic" not in policy_fields(context)


def test_registry_profile_separates_training_and_deployment_hardware():
    profile = build_hyperparameter_policy_context(_state())["profile"]

    assert profile["training_hardware"]["profile_id"] == "macbook_air_m4_16gb"
    assert profile["deployment_hardware"]["hardware_category"] == "ConsumerCPU"
    assert profile["deployment_hardware"]["ram_gb"] == 4.0
    assert profile["deployment_constraints"]["max_runtime_memory_mb"] == 500.0
    assert profile["deployment_constraints"]["max_cpu_latency_ms"] == 200.0
    assert "hardware" not in profile


def test_detection_registry_only_exposes_detection_policies():
    context = build_hyperparameter_policy_context(_state("detection"))

    assert "hpo.detection.augmentation_by_data.v1" in context["policy_ids"]
    assert "mosaic" in policy_fields(context)
    assert "optimizer_name" in policy_fields(context)
    assert "learning_rate" in policy_fields(context)
    assert "freeze_backbone_epochs" not in policy_fields(context)
    freeze_policy = next(
        item
        for item in context["applicable_policies"]
        if item["id"] == "hpo.detection.transfer_depth.v1"
    )
    assert "freeze=10" in freeze_policy["guidance"]
    assert "not a number of epochs" in freeze_policy["guidance"]


def test_policy_rationale_validation_rejects_missing_and_unknown_ids():
    context = build_hyperparameter_policy_context(_state())
    missing = LLMFieldRationale(field="patience", reason="Fits the schedule")
    unknown = LLMFieldRationale(
        field="patience",
        reason="Fits the schedule",
        applied_policy_ids=["hpo.unknown.v1"],
    )
    valid = LLMFieldRationale(
        field="patience",
        reason="Fits the schedule",
        applied_policy_ids=["hpo.common.schedule_data_size.v1"],
    )

    assert validate_policy_rationales([missing], {"patience"}, context)
    assert validate_policy_rationales([unknown], {"patience"}, context)
    assert validate_policy_rationales([valid], {"patience"}, context) == []


def test_policy_rationale_validation_rejects_policy_for_another_field():
    context = build_hyperparameter_policy_context(_state())
    wrong_field_policy = LLMFieldRationale(
        field="patience",
        reason="Incorrectly cites the batch policy",
        applied_policy_ids=["hpo.common.effective_batch.v1"],
    )

    errors = validate_policy_rationales(
        [wrong_field_policy],
        {"patience"},
        context,
    )

    assert errors == [
        "patience cites policies not registered for that field: "
        "['hpo.common.effective_batch.v1']"
    ]


def test_policy_ids_are_indexed_by_guided_field():
    context = build_hyperparameter_policy_context(_state())

    by_field = policy_ids_by_field(context)

    assert by_field["label_smoothing"] == [
        "hpo.classification.regularization.v1"
    ]
    assert "hpo.common.effective_batch.v1" not in by_field["patience"]


def test_policy_rationales_are_normalized_to_exact_field_mapping():
    context = build_hyperparameter_policy_context(_state())
    rationales = [
        LLMFieldRationale(
            field="batch_size",
            reason="Largest hardware-safe batch.",
            applied_policy_ids=[
                "hpo.common.effective_batch.v1",
                "hpo.common.schedule_data_size.v1",
            ],
        ),
        LLMFieldRationale(
            field="patience",
            reason="Fits the training duration.",
        ),
        LLMFieldRationale(
            field="model_name",
            reason="Owned by model selection.",
            applied_policy_ids=["hpo.classification.freeze_by_data.v1"],
        ),
    ]

    normalized = normalize_policy_rationales(rationales, context)
    by_field = {item.field: item.applied_policy_ids for item in normalized}

    assert by_field == {
        "batch_size": ["hpo.common.effective_batch.v1"],
        "patience": ["hpo.common.schedule_data_size.v1"],
        "model_name": [],
    }
    assert validate_policy_rationales(
        normalized,
        {"batch_size", "patience"},
        context,
    ) == []
    assert rationales[0].applied_policy_ids == [
        "hpo.common.effective_batch.v1",
        "hpo.common.schedule_data_size.v1",
    ]


def test_decision_evidence_contains_only_policies_used_by_saved_fields():
    policy_context = build_hyperparameter_policy_context(_state())
    evidence = build_hyperparameter_decision_evidence(
        {"patience": 5},
        "Policy-guided patience.",
        {"hyperparameter_policy_context": policy_context},
        field_provenance={
            "patience": {
                "source": "llm_completion",
                "source_id": "missing_recipe_field",
                "reason": "Fits the schedule.",
                "support_type": "llm_judgment",
                "evidence_ids": [],
                "applied_policy_ids": ["hpo.common.schedule_data_size.v1"],
            },
            "seed": {
                "source": "schema_default",
                "source_id": "schema",
                "reason": "Default.",
                "support_type": "schema_default",
                "evidence_ids": [],
            },
        },
    )

    used = evidence["policy_guidance"]["used_policies"]
    assert [policy["id"] for policy in used] == ["hpo.common.schedule_data_size.v1"]
    assert used[0]["influenced_fields"] == ["patience"]
    assert evidence["policy_guidance"]["profile"]["data"]["split_counts"]["train"] == 500
