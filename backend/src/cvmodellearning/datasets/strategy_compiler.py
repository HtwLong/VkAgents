from __future__ import annotations

from dataclasses import dataclass

from cvmodellearning.schemas.data_strategy import (
    CompiledDataPlan,
    DataPlanConflict,
    DataStrategy,
    DatasetSourceDecision,
)
from cvmodellearning.schemas.interpretation_schema import (
    ClassDataSelection,
    DatasetSourceCount,
)
from cvmodellearning.datasets.registry import DatasetRole, resolve_dataset_info


_ROLE_ORDER = {"primary": 0, "training_supplement": 1, "external_evaluation": 2}


@dataclass(frozen=True)
class _ResolvedDecision:
    source: DatasetSourceDecision
    class_name: str
    role: str
    activation: str
    minimum_images_when_used: int

    @property
    def dataset_id(self) -> str:
        return self.source.dataset_id

    @property
    def priority(self) -> int:
        return self.source.priority


def build_strategy_split_permissions(
    strategy: DataStrategy,
    class_names: list[str],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]]]:
    """Translate per-class LLM usage decisions into split-builder permissions."""

    training_only: dict[str, set[str]] = {name: set() for name in class_names}
    allowed_splits: dict[tuple[str, str], set[str]] = {}
    for class_name in class_names:
        for decision in strategy.source_decisions:
            use = decision.use_for_class(class_name)
            if use["role"] == "external_evaluation":
                continue
            if use["role"] == "training_supplement":
                training_only[class_name].add(decision.dataset_id)
            allowed_splits[(class_name, decision.dataset_id)] = (
                {"train"}
                | ({"validation"} if use["allow_derived_validation"] else set())
                | ({"test"} if use["allow_derived_test"] else set())
            )
    return training_only, allowed_splits


def compile_data_strategy(
    strategy: DataStrategy,
    eligible_data: list[ClassDataSelection],
    *,
    supported_group_isolation_keys: set[str] | None = None,
) -> CompiledDataPlan:
    """Compile LLM decisions into exact feasible counts without inventing preferences."""
    eligible = {
        (item.class_name, source.dataset_name): source.count
        for item in eligible_data
        for source in item.sources
    }
    expected_classes = {item.class_name for item in eligible_data}
    objectives = {item.class_name: item for item in strategy.coverage_objectives}
    conflicts: list[DataPlanConflict] = []
    warnings: list[dict] = []
    selected: list[ClassDataSelection] = []
    source_roles: dict[str, dict[str, str]] = {}
    decisions_by_id: dict[str, int] = {}
    for decision in strategy.source_decisions:
        decisions_by_id[decision.dataset_id] = decisions_by_id.get(decision.dataset_id, 0) + 1
    for dataset_id, count in decisions_by_id.items():
        if count > 1:
            conflicts.append(DataPlanConflict(
                code="CONFLICTING_SOURCE_ROLES",
                dataset_id=dataset_id,
                reason=(
                    "A dataset must have one source decision; use class_use_overrides "
                    "when its purpose differs by class."
                ),
            ))
    unsupported_group_keys = (
        set(strategy.split_strategy.group_isolation_keys)
        - set(supported_group_isolation_keys or set())
    )
    for key in sorted(unsupported_group_keys):
        conflicts.append(DataPlanConflict(
            code="UNSUPPORTED_GROUP_ISOLATION_KEY",
            reason=(
                "The strategy requests group-aware splitting for metadata that is "
                "not available to the current candidate inventory."
            ),
            facts={"group_isolation_key": key},
            available_options=[
                {"option": "remove_group_isolation_key"},
                {"option": "provide_group_metadata"},
            ],
        ))

    for missing in sorted(expected_classes - set(objectives)):
        conflicts.append(DataPlanConflict(
            code="MISSING_CLASS_COVERAGE_OBJECTIVE",
            class_name=missing,
            reason="The strategy contains no coverage objective for a requested class.",
        ))
    for extra in sorted(set(objectives) - expected_classes):
        conflicts.append(DataPlanConflict(
            code="UNREQUESTED_CLASS_COVERAGE_OBJECTIVE",
            class_name=extra,
            reason="The strategy contains an objective for a class outside the inventory.",
        ))
    if objectives:
        largest_required_class_pool = max(
            item.minimum_positive_images for item in objectives.values()
        )
        if strategy.minimum_unique_pool_images < largest_required_class_pool:
            conflicts.append(DataPlanConflict(
                code="UNIQUE_IMAGE_MINIMUM_BELOW_CLASS_MINIMUM",
                reason=(
                    "A whole-pool unique-image minimum cannot be lower than the "
                    "largest mandatory positive-image objective for one class."
                ),
                facts={
                    "minimum_unique_pool_images": strategy.minimum_unique_pool_images,
                    "largest_class_minimum": largest_required_class_pool,
                },
            ))

    eligible_classes_by_source: dict[str, set[str]] = {}
    for class_name, dataset_id in eligible:
        eligible_classes_by_source.setdefault(dataset_id, set()).add(class_name)
    decisions_by_class: dict[str, list[_ResolvedDecision]] = {
        class_name: [] for class_name in expected_classes
    }
    for decision in strategy.source_decisions:
        inventory_classes = eligible_classes_by_source.get(decision.dataset_id, set())
        unsupported_focus = set(decision.focus_classes) - inventory_classes
        unsupported_overrides = {
            item.class_name for item in decision.class_use_overrides
        } - inventory_classes
        if unsupported_focus:
            warnings.append({
                "code": "UNSUPPORTED_SOURCE_FOCUS_IGNORED",
                "severity": "warning",
                "dataset_id": decision.dataset_id,
                "focus_classes": sorted(unsupported_focus),
                "reason": (
                    "The local inventory, not the LLM, determines class eligibility; "
                    "unsupported focus classes were ignored."
                ),
            })
        if unsupported_overrides:
            warnings.append({
                "code": "UNSUPPORTED_CLASS_USE_OVERRIDE_IGNORED",
                "severity": "warning",
                "dataset_id": decision.dataset_id,
                "classes": sorted(unsupported_overrides),
                "reason": "Overrides cannot grant class eligibility and were ignored.",
            })
        effective_classes = (
            inventory_classes & set(decision.focus_classes)
            if decision.focus_classes else inventory_classes
        )
        effective_classes |= {
            item.class_name for item in decision.class_use_overrides
        } & inventory_classes
        if not inventory_classes:
            conflicts.append(DataPlanConflict(
                code="SOURCE_NOT_IN_ELIGIBLE_INVENTORY",
                dataset_id=decision.dataset_id,
                reason="The chosen dataset is not available for any requested class.",
                available_options=[
                    {"dataset_id": source_id, "eligible_classes": sorted(classes)}
                    for source_id, classes in sorted(eligible_classes_by_source.items())
                ],
            ))
            continue
        for class_name in effective_classes:
            class_use = decision.use_for_class(class_name)
            role = str(class_use["role"])
            source_roles.setdefault(decision.dataset_id, {})[class_name] = role
            if (
                role != "external_evaluation"
                and (
                    (info := resolve_dataset_info(decision.dataset_id)) is None
                    or info.role != DatasetRole.TRAIN
                )
            ):
                conflicts.append(DataPlanConflict(
                    code="NON_TRAIN_SOURCE_SELECTED_FOR_TRAINING",
                    class_name=class_name,
                    dataset_id=decision.dataset_id,
                    reason=(
                        "Primary and training-supplement decisions must reference an "
                        "official training source. Official holdouts are chosen by split strategy."
                    ),
                ))
            elif role != "external_evaluation":
                decisions_by_class[class_name].append(_ResolvedDecision(
                    source=decision,
                    class_name=class_name,
                    role=role,
                    activation=str(class_use["activation"]),
                    minimum_images_when_used=int(class_use["minimum_images_when_used"]),
                ))

    for class_name in sorted(expected_classes):
        objective = objectives.get(class_name)
        if objective is None:
            continue
        decisions = sorted(
            decisions_by_class[class_name],
            key=lambda item: (_ROLE_ORDER[item.role], item.priority),
        )
        if not decisions:
            conflicts.append(DataPlanConflict(
                code="NO_TRAINING_SOURCE_FOR_CLASS",
                class_name=class_name,
                reason="No LLM-approved training source covers the requested class.",
            ))
            continue

        counts: dict[str, int] = {}
        total = 0
        for decision in decisions:
            available = eligible[(class_name, decision.dataset_id)]
            deficit_to_minimum = max(0, objective.minimum_positive_images - total)
            deficit_to_preferred = max(0, objective.preferred_positive_images - total)
            activate = (
                decision.activation == "required"
                or (
                    decision.activation == "if_needed_for_minimum"
                    and deficit_to_minimum > 0
                )
                or (
                    decision.activation == "if_needed_for_preferred"
                    and deficit_to_preferred >= max(1, decision.minimum_images_when_used)
                )
            )
            if decision.activation == "optional":
                activate = False
            if not activate:
                continue
            requested = max(
                decision.minimum_images_when_used,
                deficit_to_preferred,
            )
            count = min(available, requested, objective.maximum_positive_images - total)
            if count < decision.minimum_images_when_used:
                conflicts.append(DataPlanConflict(
                    code="SOURCE_MINIMUM_UNAVAILABLE",
                    class_name=class_name,
                    dataset_id=decision.dataset_id,
                    reason="The LLM-required source contribution cannot be materialized.",
                    facts={
                        "available": available,
                        "minimum_when_used": decision.minimum_images_when_used,
                    },
                    available_options=[
                        {"option": "lower_source_minimum", "maximum_available": available},
                        {"option": "disable_source"},
                    ],
                ))
                continue
            if count:
                counts[decision.dataset_id] = count
                total += count

        if total < objective.minimum_positive_images:
            conflicts.append(DataPlanConflict(
                code="CLASS_MINIMUM_COVERAGE_UNMET",
                class_name=class_name,
                reason="The compiled source decisions do not meet mandatory class coverage.",
                facts={
                    "compiled": total,
                    "minimum": objective.minimum_positive_images,
                    "preferred": objective.preferred_positive_images,
                },
                available_options=[
                    {
                        "option": "approve_eligible_source",
                        "dataset_id": dataset_id,
                        "available": available,
                    }
                    for (eligible_class, dataset_id), available in sorted(eligible.items())
                    if eligible_class == class_name
                    and dataset_id not in counts
                ] + [
                    {"option": "lower_class_minimum"},
                ],
            ))
        elif total < objective.preferred_positive_images:
            warnings.append({
                "code": "CLASS_PREFERRED_COVERAGE_UNMET",
                "severity": "warning",
                "class_name": class_name,
                "compiled": total,
                "preferred": objective.preferred_positive_images,
                "reason": "Mandatory coverage is satisfied but preferred coverage is not.",
            })
        if counts:
            selected.append(ClassDataSelection(
                class_name=class_name,
                sources=[
                    DatasetSourceCount(dataset_name=dataset_id, count=count)
                    for dataset_id, count in counts.items()
                ],
            ))

    return CompiledDataPlan(
        selected_data=selected,
        strategy=strategy,
        source_roles=source_roles,
        class_coverage={
            item.class_name: {
                "selected": sum(source.count for source in item.sources),
                "minimum": objectives[item.class_name].minimum_positive_images,
                "preferred": objectives[item.class_name].preferred_positive_images,
                "maximum": objectives[item.class_name].maximum_positive_images,
            }
            for item in selected if item.class_name in objectives
        },
        warnings=warnings,
        conflicts=conflicts,
    )
