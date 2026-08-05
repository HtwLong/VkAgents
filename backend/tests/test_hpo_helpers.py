import math
from cvmodellearning.agents.hyperparameter_agents import (
    _ensure_llm_rationales_on_dict,
    apply_owned_pipeline_fields,
    changed_configuration_fields,
    evaluator_runtime_guidance,
    merge_authorized_repair_fields,
)
from cvmodellearning.schemas.classification_hpo_completion import complete_classification_config
from cvmodellearning.schemas.classification_hpo import ClassificationConfigDraft
from cvmodellearning.policies.hyperparameter_policy_registry import (
    build_hyperparameter_policy_context,
)
from cvmodellearning.schemas.interpretation_schema import PipelineState


def test_ensure_llm_rationales_on_dict():
    cfg = {"rationale": "base rationale", "llm_field_rationales": []}
    missing = {"model_ema_decay", "warmup_start_factor"}
    out = _ensure_llm_rationales_on_dict(cfg, missing)
    assert "llm_field_rationales" in out
    fields = {r["field"] for r in out["llm_field_rationales"]}
    assert "model_ema_decay" in fields and "warmup_start_factor" in fields
    assert "Added fallback rationales for" in out["rationale"] or out["rationale"].startswith("base rationale")


def test_fallback_rationale_cites_only_policies_registered_for_its_field():
    context = build_hyperparameter_policy_context(
        PipelineState(task="classification", classes=["cat", "dog"])
    )
    cfg = {"rationale": "base", "llm_field_rationales": []}

    out = _ensure_llm_rationales_on_dict(
        cfg,
        {"label_smoothing", "image_size"},
        context,
    )
    by_field = {item["field"]: item for item in out["llm_field_rationales"]}

    assert by_field["label_smoothing"]["applied_policy_ids"] == [
        "hpo.classification.regularization.v1"
    ]
    assert by_field["image_size"]["applied_policy_ids"] == []


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
