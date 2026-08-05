"""Minimal PEFT LoRA adapter support for registered transformer classifiers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from cvmodellearning.models.registry import (
    CLASSIFICATION_LORA_TARGET_MODULES,
    CLASSIFIER_HEAD_PATHS,
)
from cvmodellearning.models.classification_model_utils import get_model_weights_id


LORA_CHECKPOINT_FORMAT = "peft_lora_adapter"
CLASSIFICATION_CHECKPOINT_FORMAT_VERSION = 1


def apply_classification_lora(
    model: nn.Module,
    model_name: str,
    config: Mapping[str, Any],
) -> nn.Module:
    """Freeze the base model and train registered LoRA projections plus the new head."""
    try:
        target_modules = CLASSIFICATION_LORA_TARGET_MODULES[model_name]
        classifier_head = CLASSIFIER_HEAD_PATHS[model_name]
    except KeyError as exc:
        raise ValueError(f"LoRA is not executable for classification model '{model_name}'.") from exc

    lora_config = LoraConfig(
        r=int(config.get("lora_rank", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        modules_to_save=[classifier_head],
        bias="none",
    )
    return get_peft_model(model, lora_config)


def classification_lora_state_dict(model: nn.Module) -> dict[str, object]:
    """Return only adapter weights and the fully trained classifier head."""
    return dict(get_peft_model_state_dict(model))


def classification_lora_metadata(
    model_name: str,
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Describe the exact pretrained base and adapter layout needed for loading."""
    weights_policy = str(config.get("model_weights", "default"))
    if weights_policy != "default":
        raise ValueError("LoRA checkpoints require model_weights='default'.")
    return {
        "model_name": model_name,
        "base_weights_id": get_model_weights_id(model_name, weights_policy),
        "lora_config": {
            "rank": int(config.get("lora_rank", 8)),
            "alpha": int(config.get("lora_alpha", 16)),
            "dropout": float(config.get("lora_dropout", 0.05)),
            "target_modules": CLASSIFICATION_LORA_TARGET_MODULES[model_name],
        },
    }


def validate_classification_lora_metadata(
    model_name: str,
    config: Mapping[str, Any],
    metadata: object,
) -> None:
    """Reject adapters that cannot reproduce the registered training setup."""
    expected = classification_lora_metadata(model_name, config)
    if metadata != expected:
        raise ValueError(
            "LoRA checkpoint metadata does not match the requested model/configuration: "
            f"expected={expected!r}, received={metadata!r}."
        )


def load_classification_lora_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, object],
) -> None:
    """Load a saved adapter/head checkpoint into an identically wrapped base model."""
    expected_keys = set(classification_lora_state_dict(model))
    received_keys = set(state_dict)
    missing_keys = sorted(expected_keys - received_keys)
    unexpected_keys = sorted(received_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Invalid LoRA checkpoint state: "
            f"missing={missing_keys}, unexpected={unexpected_keys}."
        )
    incompatible = set_peft_model_state_dict(model, dict(state_dict))
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected LoRA checkpoint keys: {sorted(incompatible.unexpected_keys)}"
        )
