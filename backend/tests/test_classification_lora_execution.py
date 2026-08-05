from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import pytest
from pydantic import ValidationError

from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    validate_graph_grounded_config,
)
from cvmodellearning.models.classification_capabilities import classification_prompt_constraints
from cvmodellearning.models.classification_lora import (
    CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
    LORA_CHECKPOINT_FORMAT,
    apply_classification_lora,
    classification_lora_metadata,
    classification_lora_state_dict,
    load_classification_lora_state_dict,
)
from cvmodellearning.models.registry import (
    LORA_CLASSIFICATION_MODEL_IDS,
    MODEL_REGISTRY,
)
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile
from cvmodellearning.pipelines import classification_pipe
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState


LORA_MODELS = (
    "swin_v2_t",
    "swin_v2_s",
    "clip_vit_b16",
    "dinov2_vits14",
    "dinov2_vitb14",
    "vit_b_16",
)


def _config(model_name: str, **overrides) -> ClassificationConfigModel:
    values = {
        "classes": ["cat", "dog"],
        "selected_data": [
            {
                "class_name": class_name,
                "sources": [{"dataset_name": "demo", "count": 2000}],
            }
            for class_name in ("cat", "dog")
        ],
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "num_epochs": 1,
        "patience": 0,
        "batch_size": 2,
        "image_size": 256 if model_name.startswith("swin_v2_") else 224,
        "track_metric": "val_loss",
        "model_name": model_name,
        "model_weights": "default",
        "training_mode": "lora",
        "optimizer_name": "adamw",
        "learning_rate": 1e-4,
        "criterion_name": "cross_entropy",
        "freeze_backbone_epochs": 0,
        "head_learning_rate_multiplier": 1.0,
        "use_model_ema": False,
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "rationale": "Use a registered parameter-efficient adapter.",
    }
    values.update(overrides)
    return ClassificationConfigModel.model_validate(values)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(self.qkv(inputs))


class _AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(4, 4)
        self.attn = _Attention()
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.head(self.attn(inputs))


class _Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.head(inputs)


class _VitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(), nn.Linear(8, 4))
        self.heads = _Heads()

    def forward(self, inputs):
        return self.heads(self.mlp(inputs))


def _tiny_model(model_name: str):
    return _VitClassifier() if model_name == "vit_b_16" else _AttentionClassifier()


def _adapter_checkpoint(model, model_name: str, config: dict) -> dict:
    return {
        "checkpoint_format": LORA_CHECKPOINT_FORMAT,
        "checkpoint_format_version": CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
        "adapter_metadata": classification_lora_metadata(model_name, config),
        "model_state_dict": classification_lora_state_dict(model),
    }


def test_registry_and_agent_constraints_expose_exact_lora_capability():
    registry_ids = {
        model.id
        for model in MODEL_REGISTRY
        if model.task == "classification" and model.lora_supported
    }

    assert registry_ids == set(LORA_MODELS) == set(LORA_CLASSIFICATION_MODEL_IDS)
    for model_name in LORA_MODELS:
        assert classification_prompt_constraints(model_name)["supports_lora"] is True
    assert classification_prompt_constraints("resnet50")["supports_lora"] is False


@pytest.mark.parametrize("model_name", LORA_MODELS)
def test_lora_schema_accepts_only_registered_transformers(model_name):
    assert _config(model_name).training_mode == "lora"

    with pytest.raises(ValidationError, match="model_weights='default'"):
        _config(model_name, model_weights="none")

    with pytest.raises(ValidationError, match="freeze_backbone_epochs must be 0"):
        _config(model_name, freeze_backbone_epochs=1)


def test_lora_schema_rejects_unregistered_classifier():
    with pytest.raises(ValidationError, match="not executable"):
        _config("resnet50")


@pytest.mark.parametrize("model_name", LORA_MODELS)
def test_real_peft_adapter_trains_and_round_trips(model_name):
    torch.manual_seed(7)
    base = _tiny_model(model_name)
    model = apply_classification_lora(base, model_name, _config(model_name).model_dump())
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all("lora_" in name or "modules_to_save" in name for name in trainable)
    if model_name != "vit_b_16":
        assert not any("patch_embed.proj.lora_" in name for name in model.state_dict())
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) < sum(
        parameter.numel() for parameter in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    inputs = torch.randn(3, 4)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    state = classification_lora_state_dict(model)

    assert state
    assert all("lora_" in key or key.endswith(("head.weight", "head.bias")) for key in state)
    torch.manual_seed(7)
    restored = apply_classification_lora(
        _tiny_model(model_name),
        model_name,
        _config(model_name).model_dump(),
    )
    load_classification_lora_state_dict(restored, state)
    model.eval()
    restored.eval()
    assert torch.allclose(model(inputs), restored(inputs))


def test_pipeline_restores_adapter_checkpoint_for_inference(monkeypatch):
    model_name = "clip_vit_b16"
    config = _config(model_name).model_dump()
    torch.manual_seed(11)
    trained = apply_classification_lora(_tiny_model(model_name), model_name, config)
    checkpoint = _adapter_checkpoint(trained, model_name, config)

    def fake_make_model(_name, which_weights, num_classes):
        assert which_weights == "default"
        assert num_classes == 2
        torch.manual_seed(11)
        return _tiny_model(model_name), object()

    monkeypatch.setattr(classification_pipe, "make_model", fake_make_model)
    restored = classification_pipe.ClassificationPipeline._restore_checkpoint_model(
        model_name,
        2,
        config,
        checkpoint,
    )

    inputs = torch.randn(2, 4)
    trained.eval()
    restored.eval()
    assert torch.allclose(trained(inputs), restored(inputs))


@pytest.mark.parametrize("missing_fragment", ["lora_A", "head.weight"])
def test_lora_loader_rejects_incomplete_adapter_or_head(missing_fragment):
    model_name = "clip_vit_b16"
    config = _config(model_name).model_dump()
    model = apply_classification_lora(_tiny_model(model_name), model_name, config)
    state = classification_lora_state_dict(model)
    state.pop(next(key for key in state if missing_fragment in key))

    with pytest.raises(ValueError, match="Invalid LoRA checkpoint state: missing="):
        load_classification_lora_state_dict(model, state)


def test_lora_loader_rejects_unexpected_state_key():
    model_name = "clip_vit_b16"
    config = _config(model_name).model_dump()
    model = apply_classification_lora(_tiny_model(model_name), model_name, config)
    state = classification_lora_state_dict(model)
    state["unexpected.weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match="unexpected=\\['unexpected.weight'\\]"):
        load_classification_lora_state_dict(model, state)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda checkpoint: checkpoint.update(checkpoint_format="full_model"), "requires checkpoint_format"),
        (lambda checkpoint: checkpoint.update(checkpoint_format_version=2), "Unsupported LoRA checkpoint"),
        (
            lambda checkpoint: checkpoint["adapter_metadata"]["lora_config"].update(rank=99),
            "metadata does not match",
        ),
        (
            lambda checkpoint: checkpoint["adapter_metadata"].update(base_weights_id="wrong:weights"),
            "metadata does not match",
        ),
    ],
)
def test_pipeline_rejects_incompatible_lora_checkpoint_before_model_construction(
    monkeypatch,
    mutation,
    message,
):
    model_name = "clip_vit_b16"
    config = _config(model_name).model_dump()
    trained = apply_classification_lora(_tiny_model(model_name), model_name, config)
    checkpoint = deepcopy(_adapter_checkpoint(trained, model_name, config))
    mutation(checkpoint)
    monkeypatch.setattr(
        classification_pipe,
        "make_model",
        lambda *_args, **_kwargs: pytest.fail("invalid metadata must fail before loading weights"),
    )

    with pytest.raises(ValueError, match=message):
        classification_pipe.ClassificationPipeline._restore_checkpoint_model(
            model_name,
            2,
            config,
            checkpoint,
        )


def test_pipeline_rejects_unknown_version_for_new_full_model_checkpoint(monkeypatch):
    monkeypatch.setattr(
        classification_pipe,
        "make_model",
        lambda *_args, **_kwargs: pytest.fail("invalid format version must fail first"),
    )

    with pytest.raises(ValueError, match="Unsupported full-model checkpoint format version"):
        classification_pipe.ClassificationPipeline._restore_checkpoint_model(
            "resnet50",
            2,
            {"training_mode": "fine_tune_pretrained"},
            {"checkpoint_format": "full_model", "checkpoint_format_version": 2},
        )


@pytest.mark.parametrize("model_name", LORA_MODELS)
def test_low_vram_graphrag_materializes_executable_lora(model_name):
    context = build_hyperparameter_context(PipelineState(
        task="classification",
        classes=["cat", "dog"],
        selected_data=[
            {
                "class_name": class_name,
                "sources": [{"dataset_name": "demo", "count": 2000}],
            }
            for class_name in ("cat", "dog")
        ],
        training_hardware=get_training_hardware_profile("macbook_air_m4_16gb"),
        selected_model_info={"model": [{"model_architecture": model_name}]},
    ))

    assert context["required_adjustments"]["training_mode"] == "lora"
    assert context["required_adjustments"]["lora_rank"] == 8
    assert context["required_adjustments"]["lora_alpha"] == 16
    assert context["required_adjustments"]["lora_dropout"] == 0.05
    assert context["adjustment_rule_provenance"]["training_mode"] == (
        "rule_transformer_classifier_low_vram_use_lora"
    )
    candidate = _config(
        model_name,
        **{
            key: value
            for key, value in context["recommended_configuration"].items()
            if key != "model_name"
        },
    ).model_dump(mode="json")
    validate_graph_grounded_config(candidate, context)
