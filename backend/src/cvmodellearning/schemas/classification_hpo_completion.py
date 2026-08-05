from __future__ import annotations

from typing import Any, MutableMapping

from cvmodellearning.models.classification_capabilities import classification_capabilities
from cvmodellearning.schemas.classification_hpo import ClassificationConfigDraft
from cvmodellearning.schemas.dataset_assignment import planned_split_ratios


def _prune_inactive_classification_fields(config: MutableMapping[str, Any]) -> None:
    """Remove inactive fields no longer needed by the selected runtime path."""
    if not config.get("use_model_ema", False):
        config.pop("model_ema_decay", None)
        config.pop("model_ema_steps", None)

    if config.get("training_mode") != "lora":
        config.pop("lora_rank", None)
        config.pop("lora_alpha", None)
        config.pop("lora_dropout", None)

    if config.get("scheduler_name") != "cosine":
        config.pop("min_learning_rate", None)

    optimizer_name = config.get("optimizer_name")
    if optimizer_name != "adamw":
        config.pop("eps", None)
        config.pop("beta1", None)
        config.pop("beta2", None)
    if optimizer_name != "sgd":
        config.pop("momentum", None)
        config.pop("nesterov", None)
    if optimizer_name != "rmsprop":
        config.pop("alpha", None)
        config.pop("centered", None)

    criterion_name = config.get("criterion_name")
    if criterion_name != "cross_entropy":
        config.pop("label_smoothing", None)
    if criterion_name != "bce_with_logit":
        config.pop("pos_weight", None)


def normalize_inactive_classification_fields(
    config: MutableMapping[str, Any],
    field_sources: MutableMapping[str, dict[str, str]] | None = None,
) -> None:
    """Set schema-safe sentinels for fields ignored by the selected runtime path."""
    sources = field_sources if field_sources is not None else {}
    scheduler_name = config.get("scheduler_name")
    if scheduler_name in {"none", "step"}:
        config["min_learning_rate"] = 0.0
        sources["min_learning_rate"] = {
            "source": "system_policy",
            "source_id": f"inactive_for_{scheduler_name}_scheduler",
        }

    inactive_defaults = {
        "adamw": {"nesterov": False, "momentum": 0.0, "alpha": 0.99, "centered": False},
        "sgd": {"eps": 1e-8, "beta1": 0.9, "beta2": 0.999, "alpha": 0.99, "centered": False},
        "rmsprop": {"beta1": 0.9, "beta2": 0.999, "nesterov": False},
    }
    optimizer_name = str(config.get("optimizer_name", ""))
    for field_name, value in inactive_defaults.get(optimizer_name, {}).items():
        if field_name not in config:
            continue
        config[field_name] = value
        sources[field_name] = {
            "source": "system_policy",
            "source_id": f"inactive_for_{optimizer_name}_optimizer",
        }


def complete_classification_config(
    draft: ClassificationConfigDraft,
    state: dict[str, Any],
    model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply mechanical runtime constraints before strict cross-field validation."""
    config = draft.model_dump(mode="json")
    capabilities = classification_capabilities(model_name)
    adjustments: list[dict[str, Any]] = []

    def apply(field: str, value: Any, reason: str) -> None:
        previous = config.get(field)
        if previous == value:
            return
        config[field] = value
        adjustments.append({
            "field": field,
            "previous": previous,
            "applied": value,
            "reason": reason,
        })

    for field, ratio in (planned_split_ratios(state) or {}).items():
        apply(
            field,
            ratio,
            "Derived from the dataset assignment plan; dataset splitting is owned by dataset selection.",
        )

    apply("model_name", model_name, "Model selection is owned by the preceding planning step.")
    apply("classes", list(state.get("classes") or []), "Classes are owned by the interpreted pipeline state.")
    apply(
        "selected_data",
        list(state.get("selected_data") or []),
        "Dataset selection is owned by the preceding planning step.",
    )
    apply("criterion_name", "cross_entropy", "The runtime emits one integer class target per image.")
    apply("pos_weight", 1.0, "pos_weight is inactive for cross-entropy.")

    if not capabilities.configurable_image_size:
        apply(
            "image_size",
            capabilities.native_image_size,
            f"{model_name} has a fixed registered input size.",
        )
    if not capabilities.supports_activation_checkpointing:
        apply(
            "use_activation_checkpointing",
            False,
            f"Activation checkpointing is not implemented for {model_name}.",
        )
    if not capabilities.supports_head_lr_multiplier:
        apply(
            "head_learning_rate_multiplier",
            1.0,
            f"A separate head learning rate is not implemented for {model_name}.",
        )

    if config.get("training_mode") != "lora":
        apply("lora_rank", 8, "LoRA rank is inactive outside LoRA training.")
        apply("lora_alpha", 16, "LoRA alpha is inactive outside LoRA training.")
        apply("lora_dropout", 0.05, "LoRA dropout is inactive outside LoRA training.")

    if config.get("scheduler_name") == "cosine":
        learning_rate = float(config["learning_rate"])
        min_learning_rate = float(config.get("min_learning_rate", 0.0))
        if not 0.0 <= min_learning_rate < learning_rate:
            apply(
                "min_learning_rate",
                min(1e-6, learning_rate * 0.1),
                "Cosine decay requires a non-negative floor below the base learning rate.",
            )

    before_inactive = dict(config)
    normalize_inactive_classification_fields(config)
    for field, value in config.items():
        if before_inactive.get(field) != value:
            adjustments.append({
                "field": field,
                "previous": before_inactive.get(field),
                "applied": value,
                "reason": "The field is inactive for the selected optimizer or scheduler.",
            })

    _prune_inactive_classification_fields(config)

    if adjustments:
        summary = "; ".join(
            f"{item['field']}={item['applied']!r} ({item['reason']})" for item in adjustments
        )
        config["rationale"] = f"{config['rationale']} Executable constraints applied: {summary}"

    return config, adjustments
