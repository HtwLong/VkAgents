from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from cvmodellearning.models.registry import (
    FREEZABLE_CLASSIFICATION_MODEL_IDS,
    HEAD_LR_MULTIPLIER_MODEL_IDS,
    LORA_CLASSIFICATION_MODEL_IDS,
    resolve_model_id,
)
from cvmodellearning.preprocessing.transformations import CLASSIFICATION_TRANSFORM_PROFILES


@dataclass(frozen=True)
class ClassificationCapabilities:
    native_image_size: int
    configurable_image_size: bool
    supports_freezing: bool
    supports_head_lr_multiplier: bool
    supports_activation_checkpointing: bool
    supports_lora: bool
    minimum_image_size: int | None = None
    image_size_multiple: int | None = None


def selected_classification_model_id(
    selected_model_info: Mapping[str, Any] | None,
) -> str | None:
    """Resolve the model selected by the preceding planning step."""
    selected = selected_model_info or {}
    models = selected.get("model") or []
    if isinstance(models, Mapping):
        models = [models]

    candidates: list[Any] = []
    if models:
        model = models[0]
        if isinstance(model, Mapping):
            candidates.extend(
                model.get(key) for key in ("model_architecture", "model_name", "name", "id")
            )
    candidates.extend(selected.get(key) for key in ("model_id", "model_name"))

    for candidate in candidates:
        if candidate:
            resolved = resolve_model_id("classification", str(candidate))
            if resolved:
                return resolved
    return None


def classification_capabilities(model_name: str) -> ClassificationCapabilities:
    """Return constraints imposed by the registered executable model path."""
    try:
        profile = CLASSIFICATION_TRANSFORM_PROFILES[model_name]
    except KeyError as exc:
        raise ValueError(f"No execution capabilities registered for {model_name}.") from exc

    is_swin_v2 = model_name.startswith("swin_v2_")
    return ClassificationCapabilities(
        native_image_size=profile.native_crop_size,
        configurable_image_size=profile.configurable_image_size,
        supports_freezing=model_name in FREEZABLE_CLASSIFICATION_MODEL_IDS,
        supports_head_lr_multiplier=model_name in HEAD_LR_MULTIPLIER_MODEL_IDS,
        supports_activation_checkpointing=is_swin_v2,
        supports_lora=model_name in LORA_CLASSIFICATION_MODEL_IDS,
        minimum_image_size=256 if is_swin_v2 else None,
        image_size_multiple=32 if is_swin_v2 else None,
    )


def classification_prompt_constraints(model_name: str) -> dict[str, Any]:
    """Serialize only constraints useful to hyperparameter generation."""
    capabilities = classification_capabilities(model_name)
    constraints = {"model_name": model_name, **asdict(capabilities)}
    if not capabilities.configurable_image_size:
        constraints["required_image_size"] = capabilities.native_image_size
    return {key: value for key, value in constraints.items() if value is not None}
