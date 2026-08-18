import math
import asyncio
from cvmodellearning.agents.hyperparameter_agents import (
    _hpo_model_call,
    apply_owned_pipeline_fields,
    changed_configuration_fields,
    complete_required_field_rationales,
    evaluator_active_configuration,
    evaluator_runtime_guidance,
    discard_inactive_schema_findings,
    demote_quality_heuristic_findings,
    merge_authorized_repair_fields,
    reconcile_optimizer_explanations,
    hpo_advisory_findings,
    validate_hpo_cross_field_configuration,
)
from pydantic import BaseModel
from cvmodellearning.schemas.classification_hpo_completion import complete_classification_config
from cvmodellearning.schemas.classification_hpo import ClassificationConfigDraft
from cvmodellearning.schemas.detection_hpo import DetectionConfigDraft
from pydantic import ValidationError
import pytest
from cvmodellearning.models.detection_capabilities import detection_prompt_constraints
from cvmodellearning.schemas.decision_schema import HpoDecision, HpoFinding


class _ScheduleProposal(BaseModel):
    model_name: str = "yolov11_s"
    num_epochs: int
    patience: int
    warmup_epochs: float
    mosaic: float
    close_mosaic: int
    input_size: int = 640
    batch_size: int = 4
    translate: float = 0.0
    scale: float = 0.0


def test_hpo_model_call_retries_one_timeout_without_replaying_other_work():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError
        return "ok"

    result = asyncio.run(_hpo_model_call(
        operation, phase="optimizer", round_idx=1,
    ))

    assert result == "ok"
    assert calls == 2


def test_detection_prompt_constraints_expose_yolo_runtime_capabilities():
    constraints = detection_prompt_constraints("yolov11_s")

    assert constraints["supported_weights"] == ("coco", "default")
    assert constraints["supported_training_modes"] == ("full_finetune",)
    assert constraints["lora_supported"] is False


def test_quality_heuristic_evaluator_finding_is_non_blocking():
    decision = HpoDecision(
        accept=False,
        reason="Prefer the longer recipe schedule.",
        findings=[HpoFinding(
            field="num_epochs", severity="safety_warning",
            reason="60 is below the 100 epoch reference.",
        )],
    )

    normalized, fields = demote_quality_heuristic_findings(decision)

    assert fields == ["num_epochs"]
    assert normalized.findings[0].severity == "preference"


def _schedule_context(query: str = "Detect small traffic signs") -> dict:
    return {
        "task": "detection",
        "user_query": query,
        "hyperparameter_graph_context": {
            "reference_configuration": {"num_epochs": 100, "patience": 20}
        },
    }


def test_cross_field_validation_rejects_run_schedule_12_3_10_2():
    proposal = _ScheduleProposal(
        num_epochs=12,
        patience=2,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
    )

    violations = validate_hpo_cross_field_configuration(
        proposal,
        _schedule_context(),
    )
    messages = " ".join(item["message"] for item in violations)

    assert "more than 20%" in messages
    assert "mosaic is active for only 2" in messages
    assert "patience=2 is too short" in messages


def test_cross_field_validation_allows_coherent_adaptation():
    proposal = _ScheduleProposal(
        num_epochs=80,
        patience=15,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
    )

    assert validate_hpo_cross_field_configuration(
        proposal,
        _schedule_context(),
    ) == []


def test_explicit_user_time_limit_allows_large_epoch_reduction_but_not_bad_schedule():
    proposal = _ScheduleProposal(
        num_epochs=12,
        patience=4,
        warmup_epochs=2,
        mosaic=0.0,
        close_mosaic=0,
    )

    assert validate_hpo_cross_field_configuration(
        proposal,
        _schedule_context("Quick experiment with a maximum epochs limit of 12"),
    ) == []


def test_small_object_low_vram_policy_is_advisory_for_executable_baseline():
    proposal = _ScheduleProposal(
        num_epochs=80,
        patience=20,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
    )
    context = _schedule_context("Detect small objects far away")
    context["robustness_requirements"] = {"object_scale": ["small"]}
    context["training_hardware"] = {"training_memory_budget_gb": 5}

    assert validate_hpo_cross_field_configuration(proposal, context) == []
    messages = " ".join(item["reason"] for item in hpo_advisory_findings(proposal, context))

    assert "bounded geometric variation" in messages
    assert "input_size=768" in messages


def test_large_recipe_epoch_reduction_is_advisory_not_blocking():
    proposal = _ScheduleProposal(
        num_epochs=30, patience=5, warmup_epochs=3, mosaic=1.0,
        close_mosaic=10, translate=0.05, scale=0.25,
    )
    context = _schedule_context("Detect traffic participants")

    assert validate_hpo_cross_field_configuration(proposal, context) == []
    findings = hpo_advisory_findings(proposal, context)
    assert any(item["field"] == "num_epochs" and item["severity"] == "preference" for item in findings)


def test_small_object_low_vram_cross_field_policy_accepts_bounded_profile():
    proposal = _ScheduleProposal(
        num_epochs=80,
        patience=20,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
        input_size=768,
        batch_size=2,
        translate=0.05,
        scale=0.25,
    )
    context = _schedule_context("Detect small objects far away")
    context["robustness_requirements"] = {"object_scale": ["small"]}
    context["training_hardware"] = {"training_memory_budget_gb": 5}

    assert validate_hpo_cross_field_configuration(proposal, context) == []


def _make_minimal_draft(train=0.9, val=0.1, test=0.0001):
    # Build a minimal valid draft for ClassificationConfigDraft
    data = {
        "classes": ["car", "truck"],
        "selected_data": [
            {"class_name": "car", "sources": [{"dataset_name": "ds1", "count": 10}]},
            {"class_name": "truck", "sources": [{"dataset_name": "ds1", "count": 10}]},
        ],
        "train_data_ratio": train,
        "val_data_ratio": val,
        "test_data_ratio": test,
        "num_epochs": 5,
        "patience": 1,
        "batch_size": 4,
        "image_size": 224,
        "precision": "fp32",
        "scheduler_name": "none",
        "warmup_epochs": 0,
        "warmup_start_factor": 0.01,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 0.0,
        "freeze_backbone_epochs": 0,
        "head_learning_rate_multiplier": 1.0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "random_erasing": 0.0,
        "auto_augment_policy": "none",
        "random_resized_crop_scale_min": 0.6,
        "horizontal_flip_probability": 0.5,
        "use_model_ema": False,
        "model_ema_decay": 0.999,
        "model_ema_steps": 1,
        "repeated_augmentation_repetitions": 1,
        "use_activation_checkpointing": False,
        "track_metric": "val_acc",
        "model_name": "efficientnet_b4",
        "model_weights": "default",
        "training_mode": "fine_tune_pretrained",
        "training_recipe_id": "",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "optimizer_name": "sgd",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "eps": 1e-08,
        "beta1": 0.9,
        "beta2": 0.999,
        "nesterov": False,
        "momentum": 0.9,
        "alpha": 0.99,
        "centered": False,
        "criterion_name": "cross_entropy",
        "label_smoothing": 0.0,
        "pos_weight": 1.0,
        "rationale": "test rationale",
        "llm_field_rationales": [],
    }
    return ClassificationConfigDraft.model_validate(data)


def test_split_ratios_are_derived_from_dataset_plan():
    draft = _make_minimal_draft()
    data = draft.model_dump()
    state = {
        "classes": ["car", "truck"],
        "selected_data": data.get("selected_data", []),
        "dataset_profile": {
            "planned_counts": {"train": 16, "validation": 2, "test": 2},
        },
    }
    model_name = "efficientnet_b4"
    config, adjustments = complete_classification_config(draft, state, model_name)
    s = config["train_data_ratio"] + config["val_data_ratio"] + config["test_data_ratio"]
    assert math.isclose(s, 1.0, rel_tol=1e-12)
    assert config["train_data_ratio"] == 0.8
    assert config["val_data_ratio"] == 0.1
    assert config["test_data_ratio"] == 0.1
    assert {item["field"] for item in adjustments} >= {
        "train_data_ratio",
        "test_data_ratio",
    }


def test_owned_pipeline_fields_override_hpo_proposal():
    draft = _make_minimal_draft()
    context = {
        "classes": ["bus"],
        "selected_data": [{
            "class_name": "bus",
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [
                    {"split": "train", "count": 8, "assignment_type": "official_split"},
                    {"split": "validation", "count": 1, "assignment_type": "derived_from_train"},
                    {"split": "test", "count": 1, "assignment_type": "derived_from_train"},
                ],
            }],
        }],
        "dataset_profile": {
            "planned_counts": {"train": 8, "validation": 1, "test": 1},
        },
    }

    owned = apply_owned_pipeline_fields(draft, context)

    assert owned.classes == ["bus"]
    assert owned.selected_data[0].class_name == "bus"
    assert owned.train_data_ratio == 0.8
    assert owned.val_data_ratio == 0.1
    assert owned.test_data_ratio == 0.1


def test_required_field_rationales_returns_updated_proposal_when_entries_are_added():
    draft = _make_minimal_draft()

    completed, added = complete_required_field_rationales(
        draft,
        {"patience", "learning_rate"},
        graph_context={
            "reference_configuration": {"patience": 5},
        },
    )

    assert completed is not None
    assert added == ["learning_rate", "patience"]
    rationales = {item.field: item.reason for item in completed.llm_field_rationales}
    assert "passed the executable schema constraints" in rationales["learning_rate"]
    assert "GraphRAG-grounded configuration" in rationales["patience"]
    assert "Deterministic provenance was completed for: learning_rate, patience." in completed.rationale


def test_required_field_rationales_returns_unchanged_tuple_for_empty_or_existing_fields():
    draft = _make_minimal_draft()
    unchanged, added = complete_required_field_rationales(draft, set())
    assert unchanged is draft
    assert added == []

    completed, _ = complete_required_field_rationales(draft, {"patience"})
    unchanged, added = complete_required_field_rationales(completed, {"patience"})
    assert unchanged is completed
    assert added == []


def test_repair_merge_keeps_only_evaluator_authorized_value_changes():
    previous = _make_minimal_draft()
    repaired = previous.model_copy(update={
        "image_size": 64,
        "min_learning_rate": 1e-6,
        "rationale": "Adjusted image size.",
    })

    merged = merge_authorized_repair_fields(previous, repaired, {"image_size"})

    assert merged.image_size == 64
    assert merged.min_learning_rate == previous.min_learning_rate
    assert merged.rationale == "Adjusted image size."
    assert changed_configuration_fields(previous, merged) == {"image_size"}


def test_evaluator_sees_adjustable_hpo_fields_but_not_pipeline_owned_context():
    draft = _make_minimal_draft()

    active = evaluator_active_configuration(draft, "classification")

    assert "image_size" in active
    assert "learning_rate" in active
    assert "classes" not in active
    assert "selected_data" not in active
    assert "model_name" not in active
    assert "training_recipe_id" not in active


def test_pipeline_owned_evaluator_blocker_is_discarded_not_repaired():
    draft = _make_minimal_draft()
    active = set(evaluator_active_configuration(draft, "classification"))
    decision = HpoDecision(
        accept=False,
        reason="Dataset plan should change.",
        findings=[HpoFinding(
            field="selected_data",
            severity="safety_warning",
            reason="Change the already validated dataset plan.",
        )],
    )

    normalized, discarded = discard_inactive_schema_findings(decision, active)

    assert discarded == ["selected_data"]
    assert normalized.findings == []


def test_repair_explanation_cannot_claim_pipeline_owned_field_changed():
    previous = _make_minimal_draft()
    repaired_data = previous.model_dump(mode="json")
    repaired_data.update({
        "image_size": 64,
        "rationale": "Changed image size and selected_data.",
        "llm_field_rationales": [
            {"field": "image_size", "reason": "Reduced memory usage."},
            {"field": "selected_data", "reason": "Changed the dataset plan."},
        ],
    })
    repaired = type(previous).model_validate(repaired_data)

    reconciled = reconcile_optimizer_explanations(previous, repaired)

    assert {item.field for item in reconciled.llm_field_rationales} == {"image_size"}
    assert "selected_data" not in reconciled.rationale
    assert "image_size" in reconciled.rationale


def test_evaluator_receives_authoritative_classification_runtime_facts():
    guidance = evaluator_runtime_guidance(
        "classification",
        {
            "model_name": "mobilenet_v3_small",
            "native_image_size": 224,
            "configurable_image_size": True,
        },
    )

    assert '"configurable_image_size": true' in guidance
    assert "converts source images to RGB" in guidance
    assert "resizes them to the configured image_size" in guidance
    assert "does not by itself make a supported image_size incompatible" in guidance


def _minimal_detection_draft(**updates):
    data = {
        "task_type": "detection",
        "classes": ["car"],
        "selected_data": [{
            "class_name": "car",
            "sources": [{
                "dataset_name": "ds",
                "allocations": [
                    {"split": "train", "count": 8, "assignment_type": "official_split"},
                    {"split": "validation", "count": 1, "assignment_type": "official_split"},
                    {"split": "test", "count": 1, "assignment_type": "official_split"},
                ],
            }],
        }],
        "num_epochs": 10,
        "patience": 2,
        "model_name": "yolov10_n",
        "rationale": "Safe detector configuration.",
    }
    data.update(updates)
    return DetectionConfigDraft.model_validate(data)


def test_inactive_detection_scheduler_gamma_is_normalized_before_validation():
    draft = _minimal_detection_draft(
        scheduler_name="linear",
        scheduler_gamma=0.0,
    )

    assert draft.scheduler_gamma == 0.1


def test_active_multistep_scheduler_gamma_remains_strictly_validated():
    with pytest.raises(ValidationError, match="greater than 0"):
        _minimal_detection_draft(
            scheduler_name="multistep",
            scheduler_gamma=0.0,
        )


def test_detection_draft_normalizes_only_backend_inactive_fields():
    draft = _minimal_detection_draft(
        optimizer_name="auto",
        learning_rate=0.2,
        momentum=0.0,
        beta1=0.0,
        positive_fraction=0.0,
        scheduler_gamma=0.0,
    )

    assert draft.beta1 == 0.9
    assert draft.positive_fraction == 0.25
    assert draft.scheduler_gamma == 0.1
    assert draft.learning_rate == 0.2
    assert draft.momentum == 0.0


def test_prune_inactive_fields():
    draft = _make_minimal_draft()
    data = draft.model_dump()
    state = {"classes": ["car", "truck"], "selected_data": data.get("selected_data", [])}
    model_name = "efficientnet_b4"
    config, adjustments = complete_classification_config(draft, state, model_name)

    assert "model_ema_decay" not in config
    assert "model_ema_steps" not in config
    assert "lora_rank" not in config
    assert "lora_alpha" not in config
    assert "lora_dropout" not in config
    assert "min_learning_rate" not in config
    assert "eps" not in config
    assert "beta1" not in config
    assert "beta2" not in config
    assert "alpha" not in config
    assert "centered" not in config
    assert "label_smoothing" in config
    assert "pos_weight" not in config
