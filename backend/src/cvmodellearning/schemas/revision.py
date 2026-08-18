from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator


RevisionTarget = Literal[
    "task-interpretation",
    "model-selection",
    "dataset-selection",
    "choose-hyperparameters",
]
ChangeStrength = Literal["required", "preferred"]
ChangeOperation = Literal["set", "include", "exclude", "prefer", "avoid"]
RevisionValue: TypeAlias = (
    str | int | float | bool | list[str]
    | dict[str, str | int | float | bool | list[str]] | None
)


class RevisionChange(BaseModel):
    id: str = Field(min_length=1)
    target_step: RevisionTarget
    field: str = Field(min_length=1)
    operation: ChangeOperation = "set"
    value: RevisionValue = None
    strength: ChangeStrength
    summary: str = Field(min_length=1)


class RevisionPlan(BaseModel):
    required_text: str = ""
    preferred_text: str = ""
    summary: str = Field(min_length=1)
    restart_from: RevisionTarget
    changes: list[RevisionChange] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_content(self):
        if not self.required_text.strip() and not self.preferred_text.strip():
            raise ValueError("At least one required change or preference is required.")
        if not self.changes:
            raise ValueError("The request did not produce any actionable changes.")
        return self


class RevisionState(BaseModel):
    active: RevisionPlan | None = None
    history: list[RevisionPlan] = Field(default_factory=list)


STEP_ORDER: tuple[RevisionTarget, ...] = (
    "task-interpretation",
    "model-selection",
    "dataset-selection",
    "choose-hyperparameters",
)


def earliest_revision_step(changes: list[RevisionChange]) -> RevisionTarget:
    return min((change.target_step for change in changes), key=STEP_ORDER.index)


def changes_for(
    state: Any,
    step: RevisionTarget,
    *,
    strength: ChangeStrength | None = None,
) -> list[RevisionChange]:
    revision = getattr(state, "revision", None)
    plan = revision.active if revision else None
    if not plan:
        return []
    return [
        change
        for change in plan.changes
        if change.target_step == step and (strength is None or change.strength == strength)
    ]


def hpo_override_values(context: dict[str, Any]) -> dict[str, Any]:
    overrides = initial_hpo_override_values(context)
    revision = (context.get("revision") or {}).get("active") or {}
    overrides.update({
        change["field"].removeprefix("hpo_config."): change.get("value")
        for change in revision.get("changes", [])
        if change.get("target_step") == "choose-hyperparameters"
        and change.get("strength") == "required"
        and change.get("operation") == "set"
    })
    return overrides


def _state_value(context: Any, field: str, default: Any = None) -> Any:
    if isinstance(context, dict):
        return context.get(field, default)
    return getattr(context, field, default)


def _requirement_value(requirement: Any, field: str, default: Any = None) -> Any:
    if isinstance(requirement, dict):
        return requirement.get(field, default)
    return getattr(requirement, field, default)


def explicit_required_model_id(context: Any) -> str | None:
    """Resolve a user-required initial model to its canonical executable ID."""
    from cvmodellearning.models.registry import (
        canonical_model_id,
        enabled_models,
        resolve_model_id,
    )

    task = _state_value(context, "task")
    if task not in {"classification", "detection", "visual question answering"}:
        return None
    saw_required_model = False
    for requirement in _state_value(context, "model_requirements", None) or []:
        if _requirement_value(requirement, "requirement_strength", "required") != "required":
            continue
        saw_required_model = True
        values = [
            str(_requirement_value(requirement, field) or "")
            for field in ("name", "backbone")
        ]
        for value in values:
            resolved = resolve_model_id(task, value)
            if resolved:
                return resolved
        combined = canonical_model_id(" ".join(values))
        matches = []
        for model in enabled_models(task):
            references = [model.id, model.display_name, *model.aliases]
            if any(canonical_model_id(reference) in combined for reference in references):
                matches.append(model.id)
        if len(set(matches)) == 1:
            return matches[0]
    # Structured extraction can occasionally damage an explicit identifier
    # (for example, ``dinov2 vits14`` becoming ``DINOv2 ViT-14``). The original
    # user wording remains authoritative. Recover it only when exactly one
    # registered model reference is present, so this never guesses between
    # ambiguous variants.
    if not saw_required_model:
        return None
    query = canonical_model_id(str(_state_value(context, "user_query", "") or ""))
    query_matches = {
        model.id
        for model in enabled_models(task)
        if any(
            canonical_model_id(reference) in query
            for reference in (model.id, model.display_name, *model.aliases)
            if canonical_model_id(reference)
        )
    }
    if len(query_matches) == 1:
        return next(iter(query_matches))
    return None


def explicit_required_model_reference(context: Any) -> str | None:
    """Return the original required model text, including unsupported requests."""
    for requirement in _state_value(context, "model_requirements", None) or []:
        if _requirement_value(requirement, "requirement_strength", "required") != "required":
            continue
        return str(
            _requirement_value(requirement, "name")
            or _requirement_value(requirement, "backbone")
            or ""
        ).strip() or None
    return None


def initial_hpo_override_values(context: Any) -> dict[str, Any]:
    """Convert required initial training directives into deterministic HPO fields."""
    overrides: dict[str, Any] = {}
    query = str(_state_value(context, "user_query", "") or "").lower()
    for requirement in _state_value(context, "model_requirements", None) or []:
        if _requirement_value(requirement, "requirement_strength", "required") != "required":
            continue
        mode = _requirement_value(requirement, "training_mode")
        text = " ".join(
            str(_requirement_value(requirement, field) or "")
            for field in ("name", "description")
        ).lower()
        hyperparameters = _requirement_value(requirement, "hyperparameters", {}) or {}
        if isinstance(hyperparameters, dict):
            if mode is None:
                mode = hyperparameters.get("training_mode")
            if mode is None and "lora" in hyperparameters:
                mode = "lora"
        if mode is None and "lora" in text + " " + query:
            mode = "lora"
        if mode:
            overrides["training_mode"] = mode
        if mode == "lora":
            overrides["model_weights"] = "default"
            for source, target in (
                ("lora_rank", "lora_rank"),
                ("lora_alpha", "lora_alpha"),
                ("lora_dropout", "lora_dropout"),
            ):
                value = _requirement_value(requirement, source)
                if value is not None:
                    overrides[target] = value
    return overrides
