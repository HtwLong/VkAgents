from typing import Any, Dict, Iterable, Mapping


"""
HPO schemas intentionally keep flat fields for LLM structured output stability.
These helpers convert that validation shape into the saved runtime shape and
provide a flat compatibility view for training code that still expects it.
"""

OPTIMIZER_FIELD_NAMES = {
    "optimizer_name",
    "learning_rate",
    "weight_decay",
    "eps",
    "beta1",
    "beta2",
    "nesterov",
    "momentum",
    "alpha",
    "centered",
}

CRITERION_FIELD_NAMES = {
    "criterion_name",
    "label_smoothing",
    "pos_weight",
}


def build_runtime_hpo_config(
    raw_config: Mapping[str, Any],
    optimizer_param_fields: Mapping[str, Iterable[str]],
    criterion_param_fields: Mapping[str, Iterable[str]] | None = None,
) -> Dict[str, Any]:
    config = dict(raw_config)

    optimizer_name = config.get("optimizer_name")
    if optimizer_name not in optimizer_param_fields:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    optimizer_params = {
        field_name: config[field_name]
        for field_name in optimizer_param_fields[optimizer_name]
        if field_name in config
    }

    for field_name in OPTIMIZER_FIELD_NAMES:
        config.pop(field_name, None)

    config["optimizer"] = {
        "name": optimizer_name,
        "params": optimizer_params,
    }

    if criterion_param_fields is not None:
        criterion_name = config.get("criterion_name")
        if criterion_name not in criterion_param_fields:
            raise ValueError(f"Unsupported criterion: {criterion_name}")

        criterion_params = {
            field_name: config[field_name]
            for field_name in criterion_param_fields[criterion_name]
            if field_name in config
        }

        for field_name in CRITERION_FIELD_NAMES:
            config.pop(field_name, None)

        config["criterion"] = {
            "name": criterion_name,
            "params": criterion_params,
        }

    return config


def training_compatible_hpo_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a flat view for training code while accepting the saved runtime shape."""
    normalized = dict(config)

    # Planning metadata is useful to users but is not an executable field.
    normalized.pop("field_provenance", None)

    optimizer = normalized.get("optimizer")
    if isinstance(optimizer, Mapping):
        optimizer_name = optimizer.get("name")
        optimizer_params = optimizer.get("params") or {}
        normalized["optimizer_name"] = optimizer_name
        if isinstance(optimizer_params, Mapping):
            normalized.update(optimizer_params)
        normalized.pop("optimizer", None)

    criterion = normalized.get("criterion")
    if isinstance(criterion, Mapping):
        criterion_name = criterion.get("name")
        criterion_params = criterion.get("params") or {}
        normalized["criterion_name"] = criterion_name
        if isinstance(criterion_params, Mapping):
            normalized.update(criterion_params)
        normalized.pop("criterion", None)

    return normalized


def optimizer_compatible_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return training_compatible_hpo_config(config)
