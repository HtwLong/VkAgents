import asyncio
import csv
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from cvmodellearning.agents.hyperparameter_agents import (
    changed_configuration_fields,
)
from cvmodellearning.graphrag.hyperparameter_context import (
    build_field_provenance,
    build_hyperparameter_context,
    format_hyperparameter_context,
    get_hyperparameter_graph,
    materialize_base_recipe_config,
    validate_executable_recipe_config,
    validate_graph_grounded_config,
)
from cvmodellearning.graphrag.model_selection_context import build_model_selection_context
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel, LLMFieldRationale
from cvmodellearning.schemas.classification_model_requirements import ModelSpecModel
from cvmodellearning.schemas.decision_schema import HpoDecision
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile
from cvmodellearning.models.registry import model_ids
from cvmodellearning.models.classification_model_utils import make_model


SELECTED_DATA = [
    {
        "class_name": "cat",
        "sources": [{"dataset_name": "coco", "count": 250}],
    },
    {
        "class_name": "dog",
        "sources": [{"dataset_name": "coco", "count": 250}],
    },
]


def _state() -> PipelineState:
    return PipelineState(
        task="classification",
        application_domain="general everyday objects",
        classes=["cat", "dog"],
        selected_data=SELECTED_DATA,
        performance_requirements={
            "primary_metric": "accuracy",
            "accuracy_category": "High",
            "latency_category": "Low",
        },
        available_hardware={"hardware_category": "ConsumerGPU", "vram_gb": 8},
        training_hardware=get_training_hardware_profile("macbook_air_m4_16gb"),
        selected_model_info={
            "model": [
                {
                    "model_architecture": "resnet50",
                    "architecture_family": "resnet",
                }
            ]
        },
    )


def _valid_config_from_retrieved_recipe(context) -> ClassificationConfigModel:
    """Build the complete candidate from the deterministic graph recommendation."""
    return ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=SELECTED_DATA,
        train_data_ratio=0.8,
        val_data_ratio=0.1,
        test_data_ratio=0.1,
        rationale=f"Grounded in recipe {context['base_recipe']['id']}",
        **context["reference_configuration"],
    )


def test_retrieves_resnet_finetuning_recipe_and_builds_valid_config():
    context = build_hyperparameter_context(_state())

    assert context["selected_model"]["id"] == "resnet50"
    assert context["base_recipe"]["id"] == "torchvision_resnet50_imagenet_pretrained_custom_finetune"
    assert context["recipe_details"]
    assert context["evidence_sources"]
    assert "Base recipe" in format_hyperparameter_context(context)

    config = _valid_config_from_retrieved_recipe(context)
    assert config.model_name == "resnet50"
    assert config.learning_rate == 0.001
    assert config.batch_size == 4
    assert config.image_size == 224
    assert config.runtime_config()["optimizer"]["name"] == "sgd"
    validate_executable_recipe_config(config.model_dump())


def test_recipe_provenance_rejects_wrong_model_or_initialization():
    context = build_hyperparameter_context(_state())
    config = _valid_config_from_retrieved_recipe(context).model_dump()

    incompatible_model = {**config, "model_name": "mobilenet_v2"}
    with pytest.raises(ValueError, match="not compatible"):
        validate_executable_recipe_config(incompatible_model)

    incompatible_initialization = {**config, "model_weights": "none"}
    with pytest.raises(ValueError, match="requires model_weights='default'"):
        validate_executable_recipe_config(incompatible_initialization)


def test_repair_diff_reports_only_changed_configuration_fields():
    context = build_hyperparameter_context(_state())
    original = _valid_config_from_retrieved_recipe(context)
    allowed_repair = original.model_copy(update={"batch_size": 2, "rationale": "Repaired batch size"})
    accidental_rewrite = original.model_copy(update={"batch_size": 2, "learning_rate": 0.01})

    assert changed_configuration_fields(original, allowed_repair) == {"batch_size"}
    assert changed_configuration_fields(original, accidental_rewrite) == {"batch_size", "learning_rate"}


def test_choose_hyperparameters_passes_materialized_baseline_and_saves_valid_recipe(monkeypatch):
    import routers.planning as planning

    state = _state()
    candidate = _valid_config_from_retrieved_recipe(build_hyperparameter_context(state))
    saved = []

    async def fake_generate(json_data, job_id):
        received = json.loads(json_data)
        baseline = received["hyperparameter_graph_context"]["reference_configuration"]
        assert baseline["training_recipe_id"] == candidate.training_recipe_id
        assert baseline["learning_rate"] == candidate.learning_rate
        return candidate, HpoDecision(accept=True, reason="valid", findings=[])

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: saved.append("checkpoint"))
    monkeypatch.setattr(planning, "save_hpo_result", lambda *_args: saved.append("hpo"))

    result = asyncio.run(
        planning.choose_hyperparameters(
            planning.StateRequest(context=state.model_dump(), job_id="graph-valid")
        )
    )

    assert result["context"]["hpo_config"]["optimizer"]["params"]["learning_rate"] == 0.001
    assert "field_provenance" not in result["context"]["hpo_config"]
    provenance = result["decision_evidence"]["field_provenance"]
    assert provenance["learning_rate"]["source"] == "recipe"
    assert provenance["classes"]["source"] == "pipeline_state"
    evidence = result["decision_evidence"]
    assert evidence == result["context"]["hyperparameter_decision_evidence"]
    assert evidence["decision_type"] == "hyperparameter_selection"
    assert evidence["grounded"] is True
    assert evidence["evidence_backed"] is True
    assert evidence["retrieved_facts"]
    assert evidence["evidence_sources"]
    resolved_sources = {source["id"] for source in evidence["evidence_sources"]}
    assert all(
        source_id in resolved_sources
        for fact in evidence["retrieved_facts"]
        for source_id in fact["evidence_ids"]
    )
    assert saved == ["checkpoint", "hpo"]


def test_choose_hyperparameters_can_run_without_graphrag(monkeypatch):
    import routers.planning as planning

    state = _state()
    candidate = _valid_config_from_retrieved_recipe(build_hyperparameter_context(state))
    received = {}

    async def fake_generate(json_data, job_id):
        received.update(json.loads(json_data))
        return candidate, HpoDecision(accept=True, reason="valid", findings=[])

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)
    monkeypatch.setattr(
        planning,
        "build_hyperparameter_context",
        lambda *_args: pytest.fail("GraphRAG context should not be built"),
    )
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    monkeypatch.setattr(planning, "save_hpo_result", lambda *_args: None)

    result = asyncio.run(
        planning.choose_hyperparameters(
            planning.StateRequest(
                context=state.model_dump(),
                job_id="graph-disabled",
                use_graphrag=False,
            )
        )
    )

    assert received["use_graphrag"] is False
    assert received["hyperparameter_graph_context"] is None
    assert result["context"]["hyperparameter_graph_context"] is None
    assert "field_provenance" not in result["context"]["hpo_config"]
    assert result["decision_evidence"]["grounded"] is False
    assert result["decision_evidence"]["evidence_backed"] is False


def test_choose_hyperparameters_reports_invalid_generation_as_422(monkeypatch):
    import routers.planning as planning

    async def fake_generate(_json_data, job_id):
        del job_id
        decision = HpoDecision(
            accept=False,
            reason="Generation attempts were exhausted.",
            findings=[],
        )
        decision._diagnostics = [{"phase": "schema_validation", "field": "batch_size"}]
        return None, decision

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)

    with pytest.raises(planning.HTTPException) as exc_info:
        asyncio.run(
            planning.choose_hyperparameters(
                planning.StateRequest(
                    context=_state().model_dump(),
                    job_id="invalid-generation",
                    use_graphrag=False,
                )
            )
        )

    assert exc_info.value.status_code == 422
    assert "did not produce a valid candidate" in exc_info.value.detail["message"]
    assert exc_info.value.detail["diagnostics"] == [
        {"phase": "schema_validation", "field": "batch_size"}
    ]


def test_state_request_enables_graphrag_by_default():
    import routers.planning as planning

    request = planning.StateRequest(context={}, job_id="default-enabled")

    assert request.use_graphrag is True


def test_select_model_can_run_without_graphrag(monkeypatch):
    import routers.planning as planning

    monkeypatch.setattr(
        planning,
        "build_model_selection_context",
        lambda *_args: pytest.fail("GraphRAG context should not be built"),
    )
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)

    result = asyncio.run(
        planning.select_model(
            planning.StateRequest(
                context={},
                job_id="model-graph-disabled",
                use_graphrag=False,
            )
        )
    )

    assert result["context"]["use_graphrag"] is False
    assert result["context"]["model_selection_graph_context"] is None


def test_vit_uses_executable_custom_recipe_and_exposes_llm_completion_fields():
    state = _state().model_copy(update={
        "selected_model_info": {
            "model": [{"model_architecture": "vit_b_16", "architecture_family": "vit"}]
        }
    })

    context = build_hyperparameter_context(state)

    assert context["base_recipe"]["id"] == (
        "torchvision_vit_b16_imagenet_pretrained_custom_finetune"
    )
    assert context["reference_configuration"]["image_size"] == 224
    assert context["reference_configuration"]["optimizer_name"] == "adamw"
    assert context["reference_configuration"]["scheduler_name"] == "cosine"
    assert context["reference_configuration"]["gradient_accumulation_steps"] == 1
    assert context["fields_requiring_llm_completion"] == ["patience", "track_metric"]
    assert context["critical_materialization_errors"] == []
    assert "swag_vit_b16_imagenet1k_e2e_finetune" in (
        context["excluded_non_executable_recipe_ids"]
    )


@pytest.mark.parametrize(
    ("training_mode", "freeze_backbone_epochs"),
    (
        ("fine_tune_pretrained", 0),
        ("staged_fine_tune", 3),
        ("head_only", None),
    ),
)
def test_vit_recipe_supports_all_pretrained_finetuning_modes(
    training_mode,
    freeze_backbone_epochs,
):
    state = _state().model_copy(update={
        "selected_model_info": {
            "model": [{"model_architecture": "vit_b_16", "architecture_family": "vit"}]
        }
    })
    context = build_hyperparameter_context(state)
    candidate = ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=SELECTED_DATA,
        patience=5,
        track_metric="val_acc",
        rationale="Executable ViT pretrained fine-tuning mode.",
        **context["reference_configuration"],
    )
    config = {
        **candidate.model_dump(mode="json"),
        "training_mode": training_mode,
        "freeze_backbone_epochs": (
            candidate.num_epochs
            if training_mode == "head_only"
            else freeze_backbone_epochs
        ),
    }

    parsed = ClassificationConfigModel.model_validate(config)
    validate_executable_recipe_config(parsed.model_dump(mode="json"))


def test_unknown_recipe_fields_are_llm_completion_not_materialization_errors():
    context = {
        "selected_model": {"id": "vit_b_16"},
        "base_recipe": {
            "id": "example_partial_vit_recipe",
            "task_id": "image_classification",
            "training_mode": "FineTunePretrained",
            "pretrained": "True",
            "optimizer": "unknown",
            "scheduler": "unknown",
            "image_size_default": "224",
        },
        "recipe_details": [],
        "recipe_parameters": [],
    }

    materialized = materialize_base_recipe_config(context)

    assert materialized["image_size"] == 224
    assert "optimizer_name" not in materialized
    assert "scheduler_name" not in materialized
    assert context["fields_requiring_llm_completion"] == [
        "optimizer_name",
        "scheduler_name",
    ]
    assert context["critical_materialization_errors"] == []


def test_choose_hyperparameters_saves_vit_llm_completion_with_provenance(monkeypatch):
    import routers.planning as planning

    state = _state().model_copy(update={
        "selected_model_info": {
            "model": [{"model_architecture": "vit_b_16", "architecture_family": "vit"}]
        }
    })
    context = build_hyperparameter_context(state)
    candidate = ClassificationConfigModel.model_validate({
        "classes": ["cat", "dog"],
        "selected_data": SELECTED_DATA,
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "patience": 5,
        "track_metric": "val_acc",
        "rationale": "Completed the recipe using conservative validation controls.",
        **context["reference_configuration"],
            "llm_field_rationales": [
                {"field": "patience", "reason": "Five epochs permits stable early stopping."},
                {"field": "track_metric", "reason": "Validation accuracy matches the requested metric."},
                {"field": "batch_size", "reason": "Fits the training hardware budget."},
                {"field": "precision", "reason": "Matches accelerator support."},
                {"field": "head_learning_rate_multiplier", "reason": "Uses a conservative head rate."},
                {"field": "use_model_ema", "reason": "Disabled for this compact run."},
            ],
    })

    async def fake_generate(*_args, **_kwargs):
        return candidate, HpoDecision(accept=True, reason="valid", findings=[])

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    monkeypatch.setattr(planning, "save_hpo_result", lambda *_args: None)

    result = asyncio.run(planning.choose_hyperparameters(
        planning.StateRequest(context=state.model_dump(), job_id="vit-completion-provenance")
    ))
    config = result["context"]["hpo_config"]

    assert config["model_name"] == "vit_b_16"
    assert config["image_size"] == 224
    assert "field_provenance" not in config
    provenance = result["decision_evidence"]["field_provenance"]
    assert provenance["optimizer_name"]["source"] == "recipe"
    assert provenance["patience"]["source"] == "llm_completion"
    assert provenance["track_metric"]["reason"] == (
        "Validation accuracy matches the requested metric."
    )
    assert "rationale" not in config
    assert "LLM-completed or adjusted fields" in result["decision_evidence"]["rationale"]


def test_field_provenance_distinguishes_recipe_defaults_completion_and_repair():
    state = _state().model_copy(update={
        "selected_model_info": {
            "model": [{"model_architecture": "vit_b_16", "architecture_family": "vit"}]
        }
    })
    context = build_hyperparameter_context(state)
    candidate = ClassificationConfigModel.model_validate({
        "classes": ["cat", "dog"],
        "selected_data": SELECTED_DATA,
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "patience": 5,
        "track_metric": "val_acc",
        "rationale": "Grounded ViT configuration with bounded LLM completion.",
        **context["reference_configuration"],
        "learning_rate": 5e-5,
        "llm_field_rationales": [
            {"field": "track_metric", "reason": "Accuracy is the requested primary metric."},
            {"field": "learning_rate", "reason": "Reduced after the evaluator identified instability."},
        ],
    })

    provenance = build_field_provenance(
        candidate.model_dump(mode="json"),
        context,
        llm_adjusted_fields={"learning_rate"},
    )

    assert provenance["model_name"]["source"] == "selected_model"
    assert provenance["optimizer_name"]["source"] == "recipe"
    assert provenance["classes"]["source"] == "pipeline_state"
    assert provenance["beta1"]["source"] == "schema_default"
    assert provenance["track_metric"]["source"] == "llm_completion"
    assert provenance["learning_rate"]["source"] == "llm_adjustment"
    assert provenance["learning_rate"]["source_id"] == "evaluator_authorized_repair"
    assert provenance["learning_rate"]["reason"] == (
        "Reduced after the evaluator identified instability."
    )
    assert provenance["learning_rate"]["support_type"] == "llm_judgment"
    assert provenance["learning_rate"]["evidence_ids"] == []
    assert provenance["optimizer_name"]["support_type"] == "direct_evidence"
    assert provenance["optimizer_name"]["evidence_ids"]
    assert provenance["classes"]["support_type"] == "user_constraint"


def test_choose_hyperparameters_allows_llm_change_to_reference_recipe(monkeypatch):
    import routers.planning as planning

    state = _state()
    candidate = _valid_config_from_retrieved_recipe(
        build_hyperparameter_context(state)
    ).model_copy(update={"optimizer_name": "rmsprop"})
    saved = []

    async def fake_generate(*_args, **_kwargs):
        return candidate, HpoDecision(accept=True, reason="incorrectly accepted", findings=[])

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: saved.append("checkpoint"))
    monkeypatch.setattr(planning, "save_hpo_result", lambda *_args: saved.append("hpo"))

    result = asyncio.run(
        planning.choose_hyperparameters(
            planning.StateRequest(context=state.model_dump(), job_id="graph-reference-change")
        )
    )

    assert result["context"]["hpo_config"]["optimizer"]["name"] == "rmsprop"
    assert saved == ["checkpoint", "hpo"]


def test_choose_hyperparameters_allows_only_evaluator_authorized_baseline_repair(monkeypatch):
    import routers.planning as planning

    state = _state()
    candidate = _valid_config_from_retrieved_recipe(
        build_hyperparameter_context(state)
    ).model_copy(update={
        "learning_rate": 0.002,
        "llm_field_rationales": [
            LLMFieldRationale(
                field="learning_rate",
                reason="Reduced to the evaluator-authorized stable value.",
            )
        ],
    })
    decision = HpoDecision(accept=True, reason="safe repair", findings=[])
    decision._authorized_repair_fields = {"learning_rate"}

    async def fake_generate(*_args, **_kwargs):
        return candidate, decision

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    monkeypatch.setattr(planning, "save_hpo_result", lambda *_args: None)

    result = asyncio.run(
        planning.choose_hyperparameters(
            planning.StateRequest(context=state.model_dump(), job_id="graph-authorized-repair")
        )
    )

    assert result["context"]["hpo_config"]["optimizer"]["params"]["learning_rate"] == 0.002
    assert result["decision_evidence"]["field_provenance"]["learning_rate"]["source"] == (
        "llm_adjustment"
    )
    assert "rationale" not in result["context"]["hpo_config"]
    assert "learning_rate" in result["decision_evidence"]["rationale"]


def test_choose_hyperparameters_omits_inactive_explanations_from_evidence(monkeypatch):
    import routers.planning as planning

    state = _state()
    candidate = _valid_config_from_retrieved_recipe(
        build_hyperparameter_context(state)
    ).model_copy(update={
        "llm_field_rationales": [
            LLMFieldRationale(
                field="lora_rank",
                reason="Inactive LoRA explanation must not reach user evidence.",
            )
        ],
    })

    async def fake_generate(*_args, **_kwargs):
        return candidate, HpoDecision(accept=True, reason="valid", findings=[])

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    monkeypatch.setattr(planning, "save_hpo_result", lambda *_args: None)

    result = asyncio.run(planning.choose_hyperparameters(
        planning.StateRequest(context=state.model_dump(), job_id="inactive-explanation")
    ))

    evidence = result["decision_evidence"]
    assert "Inactive LoRA explanation" not in evidence["rationale"]
    assert "lora_rank" not in evidence["field_provenance"]


def test_choose_hyperparameters_rejects_invalid_recipe_before_saving(monkeypatch):
    import routers.planning as planning

    state = _state()
    valid = _valid_config_from_retrieved_recipe(build_hyperparameter_context(state))
    invalid = valid.model_copy(
        update={
            "training_recipe_id": "torchvision_swin_v2_t_imagenet_v1_adapted_custom_finetune"
        }
    )
    saved = []

    async def fake_generate(*_args, **_kwargs):
        return invalid, HpoDecision(accept=True, reason="incorrectly accepted", findings=[])

    monkeypatch.setattr(planning, "generate_and_evaluate_hpo", fake_generate)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: saved.append("checkpoint"))
    monkeypatch.setattr(planning, "save_hpo_result", lambda *_args: saved.append("hpo"))

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            planning.choose_hyperparameters(
                planning.StateRequest(context=state.model_dump(), job_id="graph-invalid")
            )
        )

    assert error.value.status_code == 422
    assert "failed final recipe validation" in error.value.detail["message"]
    assert saved == []


def test_mobilenet_v2_retrieves_v2_pretrained_finetuning_recipe():
    state = _state().model_copy(
        update={
            "selected_model_info": {
                "model": [{"model_architecture": "mobilenet_v2", "architecture_family": "mobilenet"}]
            }
        }
    )

    context = build_hyperparameter_context(state)

    assert context["selected_model"]["id"] == "mobilenet_v2"
    assert context["base_recipe"]["id"] == "torchvision_mobilenetv2_imagenet_v2_custom_finetune"
    assert float(context["base_recipe"]["learning_rate_default"]) == 0.001
    assert context["base_recipe"]["learning_rate_min"] == ""
    assert context["base_recipe"]["learning_rate_max"] == ""
    assert context["base_recipe"]["patience_default"] == ""
    assert context["base_recipe"]["precision"] == "unknown"
    assert context["base_recipe"]["recipe_name"].startswith("Adapted MobileNet V2")
    assert context["recipe_details"][0]["classifier_head_field"] == "model.classifier[1]"
    assert any(rule["id"] == "rule_mobilenetv2_prefer_v2_weights" for rule in context["applicable_rules"])

    graph = get_hyperparameter_graph()
    assert "torchvision_mobilenetv2_imagenet_qnnpack_qat" in graph
    assert any(
        target == "bench_mobilenetv2_qnnpack_imagenet_top1"
        and edge.get("relation") == "has_reference_benchmark_result"
        for _, target, edge in graph.out_edges(
            "torchvision_mobilenetv2_imagenet_qnnpack_qat", data=True
        )
    )


def test_densenet121_retrieves_adapted_full_finetuning_recipe():
    state = _state().model_copy(
        update={
            "selected_model_info": {
                "model": [{"model_architecture": "densenet121", "architecture_family": "densenet"}]
            }
        }
    )

    context = build_hyperparameter_context(state)

    assert context["selected_model"]["id"] == "densenet121"
    assert context["base_recipe"]["id"] == "densenet121_imagenet_v1_adapted_custom_finetune"
    assert context["base_recipe"]["recipe_name"].startswith("Adapted DenseNet-121")
    assert context["base_recipe"]["learning_rate_min"] == ""
    assert context["base_recipe"]["learning_rate_default"] == "0.001"
    assert context["base_recipe"]["precision"] == "unknown"
    assert context["recipe_details"][0]["classifier_head_field"] == "model.classifier"
    assert context["base_recipe"]["freeze_default"] == "False"
    assert context["recipe_details"][0]["feature_extraction_supported"] == "true"
    assert context["reference_configuration"]["scheduler_step_size"] == 7
    assert context["reference_configuration"]["scheduler_gamma"] == 0.1

    graph = get_hyperparameter_graph()
    assert not any(
        edge.get("relation") == "has_reference_benchmark_result"
        for _, _, edge in graph.out_edges("densenet121_luatorch_imagenet_v1_training", data=True)
    )


def test_convnext_tiny_retrieves_finetuning_recipe_and_exact_pretraining_benchmark():
    state = _state().model_copy(
        update={
            "selected_model_info": {
                "model": [{"model_architecture": "convnext_tiny", "architecture_family": "convnext"}]
            }
        }
    )

    context = build_hyperparameter_context(state)

    assert context["selected_model"]["id"] == "convnext_tiny"
    assert context["base_recipe"]["id"] == (
        "torchvision_convnext_tiny_imagenet_v1_adapted_custom_finetune"
    )
    assert context["base_recipe"]["training_mode"] == "FineTunePretrained"
    assert context["base_recipe"]["precision"] == "fp32"
    assert context["base_recipe"]["scheduler"] == "StepLR"
    assert context["base_recipe"]["warmup_epochs_default"] == "0"
    assert context["base_recipe"]["patience_default"] == "0"
    assert context["recipe_details"][0]["classifier_head_field"] == "model.classifier[2]"
    assert context["recipe_details"][0]["feature_extraction_supported"] == "true"
    assert context["materialization_warnings"] == []

    graph = get_hyperparameter_graph()
    assert any(
        target == "bench_convnext_tiny_imagenet_top1"
        and edge.get("relation") == "has_reference_benchmark_result"
        for _, target, edge in graph.out_edges(
            "torchvision_convnext_tiny_imagenet_v1_training", data=True
        )
    )


def test_convnext_tiny_model_factory_replaces_actual_classifier_layer():
    model, weights = make_model("convnext_tiny", "none", 2)

    assert weights is None
    assert model.classifier[2].out_features == 2


def test_vgg16_is_not_in_executable_classification_registry():
    assert "vgg16" not in model_ids("classification")
    assert "mobilenet_v2" in model_ids("classification")


def test_swin_v2_tiny_and_small_do_not_retrieve_base_only_recipe():
    for model_id in ("swin_v2_t", "swin_v2_s"):
        state = _state().model_copy(
            update={
                "selected_model_info": {
                    "model": [{"model_architecture": model_id, "architecture_family": "swin_v2"}]
                }
            }
        )

        context = build_hyperparameter_context(state)

        assert context["selected_model"]["id"] == model_id
        assert context["base_recipe"]["id"] == (
            f"torchvision_{model_id}_imagenet_v1_adapted_custom_finetune"
        )
        assert context["base_recipe"]["training_mode"] == "FineTunePretrained"
        expected_lr = 0.00005 if model_id == "swin_v2_t" else 0.00003
        expected_batch_size = 8 if model_id == "swin_v2_t" else 4
        assert context["reference_configuration"]["model_name"] == model_id
        assert context["reference_configuration"]["learning_rate"] == expected_lr
        assert context["reference_configuration"]["batch_size"] == expected_batch_size
        assert context["reference_configuration"]["image_size"] == 256
        assert context["reference_configuration"]["training_mode"] == "fine_tune_pretrained"
        assert context["reference_configuration"]["model_weights"] == "default"
        assert context["allowed_adjustment_fields"] == [
            "batch_size",
            "freeze_backbone_epochs",
            "gradient_accumulation_steps",
            "head_learning_rate_multiplier",
                "lora_alpha",
                "lora_dropout",
                "lora_rank",
                "precision",
                "training_mode",
            "use_activation_checkpointing",
            "use_model_ema",
        ]
        assert any(
            rule["executable_adjustments"].get("training_mode") == "lora"
            for rule in context["matched_adjustment_rules"]
        )
        low_vram_rule = next(
            rule
            for rule in context["matched_adjustment_rules"]
            if rule["id"] == "rule_swin_v2_low_vram_checkpoint_accumulation"
        )
        assert low_vram_rule["executable_adjustments"] == {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "use_activation_checkpointing": True,
        }

        parameter_names = {parameter["param_name"] for parameter in context["recipe_parameters"]}
        assert {
            "min_learning_rate",
            "head_learning_rate_multiplier",
            "gradient_clip_norm",
            "warmup_start_factor",
        }.issubset(parameter_names)

        formatted = format_hyperparameter_context(context)
        assert "optimizer_name: adamw" in formatted
        assert "training_mode: fine_tune_pretrained" in formatted
        assert "model_weights: default" in formatted
        assert f"training_recipe_id: torchvision_{model_id}_imagenet_v1_adapted_custom_finetune" in formatted
        assert "scheduler_name: cosine" in formatted
        assert "precision: mixed" in formatted
        assert "gradient_accumulation_steps" in formatted
        assert "random_erasing" in formatted
        assert "use_activation_checkpointing" in formatted
        assert "use_checkpoint" not in formatted
        assert "random_erase=" not in formatted


def test_swin_low_vram_rule_has_an_exact_threshold_and_required_values():
    low_vram_state = _state().model_copy(
        update={
            "training_hardware": _state().training_hardware.model_copy(
                update={"training_memory_budget_gb": 8}
            ),
            "selected_model_info": {
                "model": [{"model_architecture": "swin_v2_t", "architecture_family": "swin_v2"}]
            },
        }
    )
    low_vram = build_hyperparameter_context(low_vram_state)
    low_vram_rule = next(
        rule for rule in low_vram["matched_adjustment_rules"]
        if rule["id"] == "rule_swin_v2_low_vram_checkpoint_accumulation"
    )
    assert low_vram_rule["executable_adjustments"]["batch_size"] == 1

    roomy_state = low_vram_state.model_copy(
        update={
            "training_hardware": get_training_hardware_profile("rtx6000_48gb")
        }
    )
    roomy = build_hyperparameter_context(roomy_state)
    assert {rule["id"] for rule in roomy["matched_adjustment_rules"]} == {
        "rule_swin_v2_small_dataset_staged_finetune"
    }
    roomy_rule = roomy["matched_adjustment_rules"][0]
    assert roomy_rule["executable_adjustments"] == {
        "training_mode": "staged_fine_tune",
        "freeze_backbone_epochs": 3,
    }


def test_swin_low_vram_rule_requires_gpu_hardware():
    state = PipelineState.model_validate(
        {
            **_state().model_dump(),
            "training_hardware": _state().training_hardware.model_copy(update={
                "accelerator": "cpu",
                "hardware_category": "ConsumerCPU",
                "gpu_count": 0,
                "training_memory_budget_gb": 1,
            }),
            "selected_model_info": {
                "model": [{"model_architecture": "swin_v2_t", "architecture_family": "swin_v2"}]
            },
        }
    )

    context = build_hyperparameter_context(state)
    rule_ids = {rule["id"] for rule in context["matched_adjustment_rules"]}

    assert "rule_swin_v2_low_vram_checkpoint_accumulation" not in rule_ids


def test_low_vram_yolo_context_disables_multi_scale_training():
    state = PipelineState(
        task="detection",
        application_domain="dense urban street scenes",
        classes=["traffic light", "traffic sign"],
        selected_data=[{
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train",
                    "count": 4000,
                    "assignment_type": "official_split",
                }],
            }],
        } for class_name in ("traffic light", "traffic sign")],
        training_hardware=get_training_hardware_profile(
            "rtx2060_6gb_ryzen5600x_16gb"
        ),
        selected_model_info={
            "model": [{"model_architecture": "yolov10_n"}],
        },
    )

    context = build_hyperparameter_context(state)

    assert context["training_hardware_adjustments"]["multi_scale"] == 0.0
    assert (
        context["training_hardware_adjustment_provenance"]["multi_scale"]
        == "rtx2060_6gb_ryzen5600x_16gb"
    )


def test_low_vram_small_object_yolo_context_uses_bounded_high_resolution_profile():
    state = PipelineState(
        task="detection",
        user_query="Detect small and far away traffic lights",
        classes=["traffic light"],
        robustness_requirements={"object_scale": ["small"]},
        selected_data=[{
            "class_name": "traffic light",
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train", "count": 100,
                    "assignment_type": "official_split",
                }],
            }],
        }],
        training_hardware=get_training_hardware_profile(
            "rtx2060_6gb_ryzen5600x_16gb"
        ),
        selected_model_info={"model": [{"model_architecture": "yolov11_s"}]},
    )

    context = build_hyperparameter_context(state)
    adjustments = context["training_hardware_adjustments"]

    assert adjustments | {
        "input_size": 768,
        "batch_size": 2,
        "translate": 0.05,
        "scale": 0.25,
        "fliplr": 0.5,
        "multi_scale": 0.0,
    } == adjustments
    assert context["hardware_safe_resolution_candidates"]
    assert all(
        item["requires_measured_runtime_preflight"]
        for item in context["hardware_safe_resolution_candidates"]
    )


@pytest.mark.parametrize(
    (
        "model_architecture",
        "profile_id",
        "expected_rule_id",
        "expected_adjustments",
        "expected_input_size",
    ),
    [
        (
            "yolov12_x",
            "rtx2060_6gb_ryzen5600x_16gb",
            "rule_yolo_low_vram_batch",
            {"batch_size": 4, "learning_rate": 0.0025},
            640,
        ),
        (
            "yolov12_x",
            "rtx6000_48gb",
            "rule_yolo_high_memory_batch_lr",
            {"batch_size": 16, "learning_rate": 0.01},
            640,
        ),
        (
            "faster-rcnn_r50_fpn_1x_coco",
            "rtx2060_6gb_ryzen5600x_16gb",
            "rule_fasterrcnn_low_memory_batch_lr",
            {"batch_size": 1, "learning_rate": 0.00125},
            800,
        ),
        (
            "faster-rcnn_r50_fpn_1x_coco",
            "rtx6000_48gb",
            "rule_fasterrcnn_high_memory_batch_lr",
            {"batch_size": 8, "learning_rate": 0.01},
            800,
        ),
        (
            "retinanet_r50_fpn_1x_coco",
            "rtx2060_6gb_ryzen5600x_16gb",
            "rule_retinanet_low_memory_batch_lr",
            {"batch_size": 1, "learning_rate": 0.000625},
            800,
        ),
        (
            "retinanet_r50_fpn_1x_coco",
            "rtx6000_48gb",
            "rule_retinanet_high_memory_batch_lr",
            {"batch_size": 8, "learning_rate": 0.005},
            800,
        ),
        (
            "rtdetr_hgnetv2_l",
            "rtx2060_6gb_ryzen5600x_16gb",
            "rule_rtdetr_l_low_memory_batch",
            {"batch_size": 1, "learning_rate": 0.001, "warmup_epochs": 3.0},
            640,
        ),
        (
            "rtdetr_hgnetv2_l",
            "rtx6000_48gb",
            "rule_rtdetr_l_high_memory_batch",
            {"batch_size": 16, "learning_rate": 0.001, "warmup_epochs": 3.0},
            640,
        ),
        (
            "ssd300_coco",
            "rtx2060_6gb_ryzen5600x_16gb",
            "rule_ssd300_low_memory_batch",
            {"batch_size": 1, "learning_rate": 0.002},
            300,
        ),
        (
            "ssd300_coco",
            "rtx6000_48gb",
            "rule_ssd300_high_memory_batch",
            {"batch_size": 8, "learning_rate": 0.002},
            300,
        ),
    ],
)
def test_detection_hardware_recommendations_are_retrieved_for_exact_model(
    model_architecture,
    profile_id,
    expected_rule_id,
    expected_adjustments,
    expected_input_size,
):
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_data=[{
            "class_name": "car",
            "sources": [{"dataset_name": "demo", "count": 100}],
        }],
        training_hardware=get_training_hardware_profile(profile_id),
        selected_model_info={"model": [{"model_architecture": model_architecture}]},
    )

    context = build_hyperparameter_context(state)
    matched = {
        rule["id"]: rule for rule in context["matched_adjustment_rules"]
    }

    assert expected_rule_id in matched
    assert matched[expected_rule_id]["executable_adjustments"] == expected_adjustments
    assert set(expected_adjustments) <= set(context["allowed_adjustment_fields"])
    assert context["hardware_safe_resolution_candidates"]
    assert any(
        candidate["batch_size"] == expected_adjustments["batch_size"]
        and candidate["input_size"] == expected_input_size
        for candidate in context["hardware_safe_resolution_candidates"]
    )
    formatted = format_hyperparameter_context(context)
    assert "Matched evidence-backed recommendations" in formatted
    assert f"CONSIDER {expected_adjustments}" in formatted


def test_detection_hardware_recommendations_do_not_cross_model_boundaries():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_data=[{
            "class_name": "car",
            "sources": [{"dataset_name": "demo", "count": 100}],
        }],
        training_hardware=get_training_hardware_profile("rtx6000_48gb"),
        selected_model_info={
            "model": [{"model_architecture": "faster-rcnn_r50_fpn_1x_coco"}]
        },
    )

    context = build_hyperparameter_context(state)
    applicable_ids = {rule["id"] for rule in context["applicable_rules"]}
    matched_ids = {rule["id"] for rule in context["matched_adjustment_rules"]}

    assert "rule_fasterrcnn_high_memory_batch_lr" in matched_ids
    assert "rule_retinanet_high_memory_batch_lr" not in applicable_ids
    assert "rule_rtdetr_l_high_memory_batch" not in applicable_ids
    assert "rule_ssd300_high_memory_batch" not in applicable_ids


def test_inference_rtx2060_does_not_limit_rtx6000_yolov12x_training_candidates():
    state = PipelineState(
        task="detection",
        user_query="Detect small distant traffic lights for inference on an RTX 2060.",
        classes=["traffic light"],
        robustness_requirements={"object_scale": ["small"]},
        available_hardware={
            "hardware_category": "ConsumerGPU",
            "gpu_type": "NVIDIA GeForce RTX 2060",
            "gpu_count": 1,
            "vram_gb": 6,
        },
        training_hardware=get_training_hardware_profile("rtx6000_48gb"),
        selected_data=[{
            "class_name": "traffic light",
            "sources": [{"dataset_name": "bdd_100k_det_train", "count": 100}],
        }],
        selected_model_info={"model": [{"model_architecture": "yolov12_x"}]},
    )

    context = build_hyperparameter_context(state)
    matched_ids = {rule["id"] for rule in context["matched_adjustment_rules"]}
    candidates = context["hardware_safe_resolution_candidates"]

    assert "rule_yolo_low_vram_batch" not in matched_ids
    assert "rule_yolo_high_memory_batch_lr" in matched_ids
    high_memory_rule = next(
        rule
        for rule in context["matched_adjustment_rules"]
        if rule["id"] == "rule_yolo_high_memory_batch_lr"
    )
    assert high_memory_rule["executable_adjustments"] == {
        "batch_size": 16,
        "learning_rate": 0.01,
    }
    assert context["training_hardware_adjustments"] == {"workers": 8}
    assert context["hardware_role_context"]["training_hardware_authority"][
        "profile_id"
    ] == "rtx6000_48gb"
    assert context["hardware_role_context"]["deployment_hardware_not_for_training_memory"][
        "gpu_type"
    ] == "NVIDIA GeForce RTX 2060"
    assert "batch_size" not in context["small_object_training_policy"]
    assert any(
        candidate["input_size"] == 768 and candidate["batch_size"] == 16
        for candidate in candidates
    )
    assert any(
        candidate["input_size"] == 1280
        for candidate in candidates
    )
    assert "deployment_constraints describe inference/deployment" in (
        context["hardware_role_context"]["instruction"]
    )


@pytest.mark.parametrize(
    ("images_per_class", "expected_mode", "expected_freeze"),
    [
        (50, "head_only", 30),
        (51, "staged_fine_tune", 3),
        (500, "staged_fine_tune", 3),
        (501, "fine_tune_pretrained", 0),
    ],
)
def test_swin_dataset_size_rules_materialize_executable_training_modes(
    images_per_class, expected_mode, expected_freeze
):
    selected_data = [
        {
            "class_name": name,
            "sources": [{"dataset_name": "example", "count": images_per_class}],
        }
        for name in ("cat", "dog")
    ]
    state = PipelineState.model_validate(
        {
            **_state().model_dump(),
            "selected_data": selected_data,
            "training_hardware": get_training_hardware_profile("rtx6000_48gb"),
            "selected_model_info": {
                "model": [{"model_architecture": "swin_v2_t", "architecture_family": "swin_v2"}]
            },
        }
    )

    context = build_hyperparameter_context(state)

    adjustments = {
        field: value for rule in context["matched_adjustment_rules"]
        for field, value in rule["executable_adjustments"].items()
    }
    effective = {**context["reference_configuration"], **adjustments}
    assert effective["training_mode"] == expected_mode
    assert effective["freeze_backbone_epochs"] == expected_freeze


def test_swin_head_only_rule_suppresses_incompatible_checkpointing_rule():
    state = PipelineState.model_validate(
        {
            **_state().model_dump(),
            "selected_data": [
                {
                    "class_name": name,
                    "sources": [{"dataset_name": "example", "count": 50}],
                }
                for name in ("cat", "dog")
            ],
            "training_hardware": get_training_hardware_profile("macbook_air_m4_16gb"),
            "selected_model_info": {
                "model": [{"model_architecture": "swin_v2_t", "architecture_family": "swin_v2"}]
            },
        }
    )

    context = build_hyperparameter_context(state)
    rule_ids = {rule["id"] for rule in context["matched_adjustment_rules"]}

    assert "rule_swin_v2_very_small_dataset_head_only" in rule_ids
    assert "rule_swin_v2_low_vram_checkpoint_accumulation" not in rule_ids
    assert any(
        rule["executable_adjustments"].get("training_mode") == "head_only"
        for rule in context["matched_adjustment_rules"]
    )
    validate_graph_grounded_config(context["reference_configuration"], context)


def test_model_selection_context_contains_neutral_executable_shortlist():
    context = build_model_selection_context(PipelineState(task="classification"), top_k=20)
    candidates = context["candidate_models"]
    candidate_ids = [candidate["model"]["id"] for candidate in candidates]

    assert set(candidate_ids) == set(model_ids("classification"))
    assert candidate_ids == sorted(candidate_ids)
    assert all("rank" not in candidate for candidate in candidates)
    assert all(candidate["shortlist_roles"] for candidate in candidates)
    assert all(candidate["criterion_assessments"] for candidate in candidates)
    assert "mobilenet_v3_small" in candidate_ids
    assert "clip_vit_b16" in candidate_ids


def test_recipe_provenance_rejects_hyperparameters_outside_declared_bounds():
    config = {
        "training_recipe_id": "torchvision_swin_v2_t_imagenet_v1_adapted_custom_finetune",
        "model_name": "swin_v2_t",
        "training_mode": "fine_tune_pretrained",
        "model_weights": "default",
        "learning_rate": 0.001,
        "batch_size": 8,
        "num_epochs": 30,
        "weight_decay": 0.01,
        "image_size": 256,
    }

    with pytest.raises(ValueError, match="learning_rate <= 0.0001"):
        validate_executable_recipe_config(config)


def test_swin_v2_base_is_absent_from_planning_graph_and_execution():
    assert "swin_v2_b" not in model_ids("classification")
    assert "swin_v2_b" not in get_hyperparameter_graph()
    assert "swin_v2_b" not in ClassificationConfigModel.model_json_schema()["$defs"]["ClassificationModelId"]["enum"]

    with pytest.raises(ValueError):
        ModelSpecModel.model_validate(
            {"model_architecture": "swin_v2_b", "description": "unsupported variant"}
        )
    with pytest.raises(ValueError, match="Unsupported model: swin_v2_b"):
        make_model("swin_v2_b", "none", 2)


def test_swin_finetune_recipe_allows_staged_and_head_only_execution_modes():
    base = {
        "training_recipe_id": "torchvision_swin_v2_t_imagenet_v1_adapted_custom_finetune",
        "model_name": "swin_v2_t",
        "model_weights": "default",
        "learning_rate": 0.00005,
        "batch_size": 8,
        "num_epochs": 30,
        "weight_decay": 0.01,
        "image_size": 256,
    }

    for mode in ("fine_tune_pretrained", "staged_fine_tune", "head_only"):
        validate_executable_recipe_config({**base, "training_mode": mode})


def test_added_classification_csv_rows_have_valid_shape_and_references():
    nodes = Path(__file__).parents[1] / "ontology_data" / "nodes"

    tables = {}
    for path in nodes.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert all(len(row) == len(rows[0]) for row in rows[1:]), path.name
        tables[path.name] = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    evidence_ids = {row["id"] for row in tables["evidence_sources.csv"]}
    recipe_ids = {row["id"] for row in tables["training_recipes.csv"]}
    hardware_ids = {row["id"] for row in tables["hardware_profiles.csv"]}

    audited_rows = [
        row
        for rows in tables.values()
        for row in rows
        if any(
            token in row.get("id", "").lower()
            for token in (
                "mobilenetv2",
                "mobilenet_v2",
                "densenet121",
                "convnext_tiny",
                "swinv2",
                "swin_v2",
            )
        )
    ]
    assert audited_rows
    for row in audited_rows:
        for evidence_id in filter(None, row.get("evidence_ids", "").split("|")):
            assert evidence_id in evidence_ids, (row["id"], evidence_id)
        if row.get("recipe_id"):
            assert row["recipe_id"] in recipe_ids, row["id"]
        if row.get("training_recipe_id"):
            assert row["training_recipe_id"] in recipe_ids, row["id"]
        if row.get("hardware_profile_id"):
            assert row["hardware_profile_id"] in hardware_ids, row["id"]

    benchmarks = {row["id"]: row for row in tables["model_benchmark_results.csv"]}
    assert benchmarks["bench_mobilenetv2_v1_imagenet_top1"]["training_recipe_id"] == (
        "torchvision_mobilenetv2_imagenet_v1_training"
    )
    assert benchmarks["bench_mobilenetv2_v2_imagenet_top1"]["training_recipe_id"] == (
        "torchvision_mobilenetv2_imagenet_v2_training"
    )
    assert benchmarks["bench_mobilenetv2_qnnpack_imagenet_top1"]["training_recipe_id"] == (
        "torchvision_mobilenetv2_imagenet_qnnpack_qat"
    )
    assert benchmarks["bench_densenet121_imagenet_top1"]["training_recipe_id"] == ""
    assert benchmarks["bench_densenet121_imagenet_top1"]["hardware_profile_id"] == ""
    assert benchmarks["bench_convnext_tiny_imagenet_top1"]["training_recipe_id"] == (
        "torchvision_convnext_tiny_imagenet_v1_training"
    )
    assert benchmarks["bench_convnext_tiny_imagenet_top1"]["hardware_profile_id"] == ""
