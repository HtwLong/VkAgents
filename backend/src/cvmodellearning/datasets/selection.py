from collections import defaultdict
from typing import Iterable

from cvmodellearning.datasets.availability import is_dataset_downloadable
from cvmodellearning.datasets.registry import (
    DatasetRole,
    dataset_family,
    get_dataset_info,
    resolve_dataset_info,
)
from cvmodellearning.schemas.interpretation_schema import (
    ClassDataSelection,
    DatasetProfile,
    DatasetSourceCount,
)
from cvmodellearning.schemas.dataset_assignment import (
    AssignmentType,
    ClassDataAssignment,
    DatasetAssignmentValidationError,
    DatasetSourceAssignment,
    DatasetSplit,
    SplitAllocation,
    normalize_dataset_assignments,
    summarize_dataset_assignments,
    validate_dataset_assignments,
)


class DatasetSelectionValidationError(ValueError):
    def __init__(self, findings: list[dict[str, object]]):
        super().__init__("The proposed dataset selection is invalid.")
        self.findings = findings


DEFAULT_CLASSIFICATION_POOL_PER_CLASS = 1_500
MAX_CLASSIFICATION_POOL_PER_CLASS = 10_000
MAX_CLASSIFICATION_SELECTED_IMAGES = 50_000
DEFAULT_DETECTION_POOL_PER_CLASS = 2_000
MAX_DETECTION_POOL_PER_CLASS = 10_000
MAX_DETECTION_SELECTED_IMAGES = 50_000
DETECTION_SHARED_BACKBONE_MIN_COUNT = 50
DETECTION_SHARED_BACKBONE_MIN_SHARE = 0.25
DETECTION_SHARED_BACKBONE_TARGET_COUNT = 500


def _task_matches(requested_task: str, dataset_task: str) -> bool:
    """Return whether a dataset's native annotations support the requested task."""
    if requested_task == "visual question answering":
        return dataset_task in {"classification", "detection"}
    return dataset_task == requested_task


def build_default_dataset_selection(
    eligible_data: Iterable[ClassDataSelection],
    target_images_per_class: int = 1000,
    *,
    prefer_shared_training_family: bool = False,
) -> list[ClassDataSelection]:
    """Build a conservative valid selection when the preference model fails."""
    eligible = list(eligible_data)
    if prefer_shared_training_family:
        return _build_source_coherent_default_selection(
            eligible,
            target_images_per_class,
        )

    selections_by_class: dict[str, ClassDataSelection] = {}
    for item in eligible:
        if not item.sources or item.class_name in selections_by_class:
            continue
        source = next(
            (
                candidate
                for candidate in item.sources
                if (info := resolve_dataset_info(candidate.dataset_name)) is not None
                and info.role == DatasetRole.TRAIN
            ),
            None,
        )
        if source is None:
            continue
        selections_by_class[item.class_name] = ClassDataSelection(
            class_name=item.class_name,
            sources=[{
                "dataset_name": source.dataset_name,
                "count": min(source.count, target_images_per_class),
            }],
        )
    return list(selections_by_class.values())


def _build_source_coherent_default_selection(
    eligible_data: list[ClassDataSelection],
    target_images_per_class: int,
) -> list[ClassDataSelection]:
    """Select broad training families first so source identity cannot encode class."""

    candidates_by_family: dict[str, dict[str, DatasetSourceCount]] = defaultdict(dict)
    class_order = []
    for item in eligible_data:
        if item.class_name not in class_order:
            class_order.append(item.class_name)
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if info is None or info.role != DatasetRole.TRAIN:
                continue
            family_sources = candidates_by_family[dataset_family(source.dataset_name)]
            existing = family_sources.get(item.class_name)
            if existing is None or (source.count, source.dataset_name) > (
                existing.count,
                existing.dataset_name,
            ):
                family_sources[item.class_name] = source

    uncovered = set(class_order)
    chosen_by_class: dict[str, DatasetSourceCount] = {}
    while uncovered:
        ranked = sorted(
            candidates_by_family.items(),
            key=lambda pair: (
                -len(uncovered & set(pair[1])),
                -sum(
                    min(source.count, target_images_per_class)
                    for class_name, source in pair[1].items()
                    if class_name in uncovered
                ),
                pair[0],
            ),
        )
        if not ranked:
            break
        _, family_sources = ranked[0]
        covered = uncovered & set(family_sources)
        if not covered:
            break
        for class_name in covered:
            source = family_sources[class_name]
            chosen_by_class[class_name] = DatasetSourceCount(
                dataset_name=source.dataset_name,
                count=min(source.count, target_images_per_class),
            )
        uncovered -= covered

    return [
        ClassDataSelection(class_name=class_name, sources=[chosen_by_class[class_name]])
        for class_name in class_order
        if class_name in chosen_by_class
    ]


def validate_detection_source_coherence(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
) -> list[ClassDataSelection]:
    """Require an available common family to be a meaningful shared backbone."""

    selected = list(selected_data)
    eligible = list(eligible_data)
    requested_classes = {item.class_name for item in eligible}
    if len(requested_classes) < 2:
        return selected

    eligible_by_family: dict[str, dict[str, int]] = defaultdict(dict)
    for item in eligible:
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if info is not None and info.role == DatasetRole.TRAIN:
                family = dataset_family(source.dataset_name)
                eligible_by_family[family][item.class_name] = max(
                    source.count,
                    eligible_by_family[family].get(item.class_name, 0),
                )

    common_families = {
        family
        for family, counts in eligible_by_family.items()
        if requested_classes <= set(counts)
        and min(counts[class_name] for class_name in requested_classes)
        >= DETECTION_SHARED_BACKBONE_MIN_COUNT
    }
    if not common_families:
        return selected

    selected_totals: dict[str, int] = defaultdict(int)
    selected_by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in selected:
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if info is None or info.role != DatasetRole.TRAIN:
                continue
            selected_totals[item.class_name] += source.count
            selected_by_family[dataset_family(source.dataset_name)][item.class_name] += source.count

    for family in sorted(common_families):
        if all(
            selected_by_family[family][class_name] >= min(
                eligible_by_family[family][class_name],
                DETECTION_SHARED_BACKBONE_TARGET_COUNT,
                max(1, round(selected_totals[class_name] * DETECTION_SHARED_BACKBONE_MIN_SHARE)),
            )
            for class_name in requested_classes
        ):
            return selected

    raise DatasetSelectionValidationError([{
        "field": "sources",
        "classes": sorted(requested_classes),
        "common_families": sorted(common_families),
        "reason": (
            "A training family with sufficient data for every requested class is available, "
            "but no such family supplies a meaningful shared backbone across all classes."
        ),
    }])


def limit_selected_source_pools(
    selected_data: Iterable[ClassDataSelection],
    *,
    max_total_images: int,
    max_images_per_class: int | None = None,
) -> list[ClassDataSelection]:
    """Apply a balanced budget to complete per-class source allocations.

    Detection allocations can refer to the same image through multiple classes, so
    limiting their sum is a conservative upper bound on unique downloaded images.
    Training sources receive priority so limiting cannot remove the source required
    for downstream split planning. Official holdouts that do not fit are completed
    later by the split planner, which subtracts them from the same total pool.
    """

    selected = list(selected_data)
    capacities = []
    for item in selected:
        selected_count = sum(
            source.count
            for source in item.sources
            if resolve_dataset_info(source.dataset_name) is not None
        )
        capacities.append(min(
            selected_count,
            max_images_per_class if max_images_per_class is not None else selected_count,
        ))

    budgets = [0] * len(capacities)
    remaining = min(max_total_images, sum(capacities))
    active = {index for index, capacity in enumerate(capacities) if capacity > 0}
    while remaining and active:
        share = max(1, remaining // len(active))
        for index in sorted(active):
            take = min(share, capacities[index] - budgets[index], remaining)
            budgets[index] += take
            remaining -= take
            if not remaining:
                break
        active = {
            index for index in active if budgets[index] < capacities[index]
        }

    limited = []
    for item, class_budget in zip(selected, budgets):
        remaining_for_class = class_budget
        counts_by_index: dict[int, int] = {}
        source_order = sorted(
            range(len(item.sources)),
            key=lambda index: (
                0
                if (
                    (info := resolve_dataset_info(item.sources[index].dataset_name))
                    is not None
                    and info.role == DatasetRole.TRAIN
                )
                else 1,
                index,
            ),
        )
        for index in source_order:
            source = item.sources[index]
            count = min(source.count, remaining_for_class)
            if count:
                counts_by_index[index] = count
                remaining_for_class -= count
        sources = [
            DatasetSourceCount(
                dataset_name=source.dataset_name,
                count=counts_by_index[index],
            )
            for index, source in enumerate(item.sources)
            if index in counts_by_index
        ]
        limited.append(ClassDataSelection(class_name=item.class_name, sources=sources))
    return limited


def filter_training_candidates(
    available_data: Iterable[ClassDataSelection],
    task: str,
) -> list[ClassDataSelection]:
    candidates_by_class: dict[str, dict[str, DatasetSourceCount]] = {}
    for class_selection in available_data:
        class_sources = candidates_by_class.setdefault(class_selection.class_name, {})
        for source in class_selection.sources:
            info = resolve_dataset_info(source.dataset_name)
            task_matches = info and _task_matches(task, info.task)
            if (
                task_matches
                and info.role == DatasetRole.TRAIN
                and source.count > 0
                and is_dataset_downloadable(source.dataset_name)
            ):
                existing = class_sources.get(source.dataset_name)
                if existing is None or source.count > existing.count:
                    class_sources[source.dataset_name] = source
    return [
        ClassDataSelection(class_name=class_name, sources=list(sources.values()))
        for class_name, sources in candidates_by_class.items()
    ]


def filter_dataset_candidates(
    available_data: Iterable[ClassDataSelection],
    task: str,
) -> list[ClassDataSelection]:
    """Return compatible official train, validation, and test candidates."""

    candidates_by_class: dict[str, dict[str, DatasetSourceCount]] = {}
    for class_selection in available_data:
        class_sources = candidates_by_class.setdefault(class_selection.class_name, {})
        for source in class_selection.sources:
            info = resolve_dataset_info(source.dataset_name)
            task_matches = info and _task_matches(task, info.task)
            if (
                task_matches
                and info.role in {DatasetRole.TRAIN, DatasetRole.VALIDATION, DatasetRole.TEST}
                and source.count > 0
                and is_dataset_downloadable(source.dataset_name)
            ):
                existing = class_sources.get(source.dataset_name)
                if existing is None or source.count > existing.count:
                    class_sources[source.dataset_name] = source
    return [
        ClassDataSelection(class_name=class_name, sources=list(sources.values()))
        for class_name, sources in candidates_by_class.items()
    ]


def validate_dataset_selection(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
) -> list[ClassDataSelection]:
    selected = list(selected_data)
    eligible = list(eligible_data)
    findings: list[dict[str, object]] = []
    eligible_counts = {
        (item.class_name, source.dataset_name): source.count
        for item in eligible
        for source in item.sources
    }
    expected_classes = {item.class_name for item in eligible}
    seen_classes: set[str] = set()

    for item in selected:
        if item.class_name in seen_classes:
            findings.append({
                "field": "class_name",
                "class_name": item.class_name,
                "reason": "Class appears more than once.",
            })
        seen_classes.add(item.class_name)
        if item.class_name not in expected_classes:
            findings.append({
                "field": "class_name",
                "class_name": item.class_name,
                "reason": "Class was not requested.",
            })
        if not item.sources:
            findings.append({
                "field": "sources",
                "class_name": item.class_name,
                "reason": "At least one dataset source is required.",
            })

        seen_sources: set[str] = set()
        for source in item.sources:
            key = (item.class_name, source.dataset_name)
            info = resolve_dataset_info(source.dataset_name)
            if source.dataset_name in seen_sources:
                findings.append({
                    "field": "sources",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": "Dataset appears more than once for the class.",
                })
            seen_sources.add(source.dataset_name)
            if info is None:
                findings.append({
                    "field": "dataset_name",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": "Unknown VisionKG dataset identifier.",
                })
            elif info.role not in {DatasetRole.TRAIN, DatasetRole.VALIDATION, DatasetRole.TEST}:
                findings.append({
                    "field": "dataset_name",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": (
                        f"Dataset role '{info.role.value}' is not eligible for split planning."
                    ),
                })
            if key not in eligible_counts:
                findings.append({
                    "field": "dataset_name",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": "Dataset is not an eligible available source for this class.",
                })
            if source.count <= 0:
                findings.append({
                    "field": "count",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": "Selected count must be positive.",
                })
            elif key in eligible_counts and source.count > eligible_counts[key]:
                findings.append({
                    "field": "count",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "selected_count": source.count,
                    "available_count": eligible_counts[key],
                    "reason": "Selected count exceeds availability.",
                })

        selected_roles = {
            resolve_dataset_info(source.dataset_name).role
            for source in item.sources
            if resolve_dataset_info(source.dataset_name) is not None
        }
        if DatasetRole.TRAIN not in selected_roles:
            findings.append({
                "field": "sources",
                "class_name": item.class_name,
                "reason": "At least one official training source is required.",
            })

        training_families = {
            dataset_family(source.dataset_name)
            for source in item.sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role == DatasetRole.TRAIN
        }
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if (
                info is not None
                and info.role in {DatasetRole.VALIDATION, DatasetRole.TEST}
                and dataset_family(source.dataset_name) not in training_families
            ):
                findings.append({
                    "field": "dataset_name",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": (
                        "Primary validation/test data must belong to a selected "
                        "training dataset family. Use unrelated data only for a "
                        "separate external robustness evaluation."
                    ),
                })

    for missing_class in sorted(expected_classes - seen_classes):
        findings.append({
            "field": "class_name",
            "class_name": missing_class,
            "reason": "Requested class is missing from the selection.",
        })

    if findings:
        raise DatasetSelectionValidationError(findings)
    return selected


def complete_official_holdout_selection(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
) -> list[ClassDataSelection]:
    """Add same-family official holdouts; never introduce unrelated families."""

    eligible_by_class = {item.class_name: item for item in eligible_data}
    completed: list[ClassDataSelection] = []
    for item in selected_data:
        sources = list(item.sources)
        candidates = eligible_by_class.get(item.class_name)
        training_sources = [
            source
            for source in sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role == DatasetRole.TRAIN
        ]
        training_families = {
            dataset_family(source.dataset_name) for source in training_sources
        }

        # A partial official holdout is not representative of a multi-family
        # training pool.  In that case keep only the selected training sources;
        # build_dataset_assignments will derive both holdouts proportionally from
        # every source.  Official holdouts remain useful for single-family pools.
        if len(training_families) > 1:
            completed.append(ClassDataSelection(
                class_name=item.class_name,
                sources=training_sources,
            ))
            continue

        total_selected = sum(source.count for source in sources)
        holdout_target = recommend_holdout_count(total_selected)
        available = {
            source.dataset_name: source.count
            for source in candidates.sources
        } if candidates is not None else {}
        proposed_excess = sum(
            max(0, source.count - holdout_target)
            for source in sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role in {DatasetRole.VALIDATION, DatasetRole.TEST}
        )
        training_spare = sum(
            max(0, available.get(source.dataset_name, 0) - source.count)
            for source in sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role == DatasetRole.TRAIN
        )
        cap_official_holdouts = proposed_excess <= training_spare

        # Official split provenance is authoritative, but an agent-selected count is
        # not. Cap oversized holdouts and return their budget to compatible training
        # sources so an official validation set cannot accidentally starve training.
        excess_holdout = 0
        capped_sources = []
        for source in sources:
            info = resolve_dataset_info(source.dataset_name)
            count = source.count
            if (
                cap_official_holdouts
                and info is not None
                and info.role in {DatasetRole.VALIDATION, DatasetRole.TEST}
            ):
                capped_count = min(count, holdout_target)
                excess_holdout += count - capped_count
                count = capped_count
            capped_sources.append(DatasetSourceCount(
                dataset_name=source.dataset_name,
                count=count,
            ))
        sources = capped_sources

        if excess_holdout and candidates is not None:
            training_indexes = [
                index
                for index, source in enumerate(sources)
                if (info := resolve_dataset_info(source.dataset_name)) is not None
                and info.role == DatasetRole.TRAIN
            ]
            spare = [
                max(0, available.get(sources[index].dataset_name, 0) - sources[index].count)
                for index in training_indexes
            ]
            additions = _proportional_holdout_counts(
                spare,
                min(excess_holdout, sum(spare)),
            )
            for index, addition in zip(training_indexes, additions):
                source = sources[index]
                sources[index] = DatasetSourceCount(
                    dataset_name=source.dataset_name,
                    count=source.count + addition,
                )

        training_families = {
            dataset_family(source.dataset_name)
            for source in sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role == DatasetRole.TRAIN
        }
        roles = {
            info.role
            for source in sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
        }
        train_total = sum(
            source.count
            for source in sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role == DatasetRole.TRAIN
        )
        target = recommend_holdout_count(train_total)
        added_official_count = 0
        for role in (DatasetRole.VALIDATION, DatasetRole.TEST):
            if role in roles or candidates is None:
                continue
            official = next(
                (
                    source
                    for source in candidates.sources
                    if (info := resolve_dataset_info(source.dataset_name)) is not None
                    and info.role == role
                    and dataset_family(source.dataset_name) in training_families
                ),
                None,
            )
            if official is not None:
                official_count = min(official.count, target)
                sources.append(DatasetSourceCount(
                    dataset_name=official.dataset_name,
                    count=official_count,
                ))
                added_official_count += official_count
                roles.add(role)
        if added_official_count:
            training_indexes = [
                index
                for index, source in enumerate(sources)
                if (info := resolve_dataset_info(source.dataset_name)) is not None
                and info.role == DatasetRole.TRAIN
            ]
            training_counts = [sources[index].count for index in training_indexes]
            reductions = _proportional_holdout_counts(
                training_counts,
                min(added_official_count, sum(training_counts)),
            )
            for index, reduction in zip(training_indexes, reductions):
                source = sources[index]
                sources[index] = DatasetSourceCount(
                    dataset_name=source.dataset_name,
                    count=source.count - reduction,
                )
            sources = [source for source in sources if source.count > 0]
        completed.append(ClassDataSelection(class_name=item.class_name, sources=sources))
    return completed


def recommend_holdout_count(total: int) -> int:
    """Recommend one holdout size without starving the training pool."""

    if total < 100:
        return max(1, round(total * 0.15))
    if total < 500:
        return max(10, round(total * 0.15))
    if total < 1000:
        return max(75, round(total * 0.12))
    return min(500, max(100, round(total * 0.10)))


def _proportional_holdout_counts(source_counts: list[int], target: int) -> list[int]:
    """Distribute a holdout by source using deterministic largest remainders."""

    total = sum(source_counts)
    if target <= 0 or total <= 0:
        return [0] * len(source_counts)
    exact = [target * count / total for count in source_counts]
    allocated = [min(count, int(value)) for count, value in zip(source_counts, exact)]
    remaining = target - sum(allocated)
    order = sorted(
        range(len(source_counts)),
        key=lambda index: (exact[index] - int(exact[index]), source_counts[index], -index),
        reverse=True,
    )
    while remaining:
        progressed = False
        for index in order:
            if allocated[index] < source_counts[index]:
                allocated[index] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    return allocated


def build_dataset_assignments(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
) -> list[ClassDataAssignment]:
    """Create a complete, deterministic split plan from validated source choices."""

    selected = complete_official_holdout_selection(
        selected_data,
        eligible_data,
    )
    assignments: list[ClassDataAssignment] = []
    available_counts = {
        (item.class_name, source.dataset_name): source.count
        for item in eligible_data
        for source in item.sources
    }
    source_roles: dict[str, str] = {}

    for item in selected:
        train_sources = []
        official_sources = []
        present_roles = set()
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if info is None:
                continue
            source_roles[source.dataset_name] = info.role.value
            present_roles.add(info.role)
            if info.role == DatasetRole.TRAIN:
                train_sources.append(source)
            else:
                official_sources.append(DatasetSourceAssignment(
                    dataset_name=source.dataset_name,
                    allocations=[SplitAllocation(
                        split=DatasetSplit(info.role.value),
                        count=source.count,
                        assignment_type=AssignmentType.OFFICIAL_SPLIT,
                    )],
                ))

        train_total = sum(source.count for source in train_sources)
        missing_holdouts = [
            split
            for split, role in (
                (DatasetSplit.VALIDATION, DatasetRole.VALIDATION),
                (DatasetSplit.TEST, DatasetRole.TEST),
            )
            if role not in present_roles
        ]
        selected_pool_total = train_total + sum(
            allocation.count
            for source in official_sources
            for allocation in source.allocations
        )
        holdout_count = recommend_holdout_count(selected_pool_total)
        if train_total <= holdout_count * len(missing_holdouts):
            raise DatasetSelectionValidationError([{
                "class_name": item.class_name,
                "reason": "Not enough training images to create disjoint train/validation/test allocations.",
            }])

        remaining_by_source = [source.count for source in train_sources]
        derived_by_split: dict[DatasetSplit, list[int]] = {}
        for split in missing_holdouts:
            split_counts = _proportional_holdout_counts(remaining_by_source, holdout_count)
            derived_by_split[split] = split_counts
            remaining_by_source = [
                remaining - allocated
                for remaining, allocated in zip(remaining_by_source, split_counts)
            ]

        training_assignments = []
        for index, source in enumerate(train_sources):
            remaining = remaining_by_source[index]
            allocations = []
            for split in missing_holdouts:
                take = derived_by_split[split][index]
                if take:
                    allocations.append(SplitAllocation(
                        split=split,
                        count=take,
                        assignment_type=AssignmentType.DERIVED_FROM_TRAIN,
                    ))
            if remaining:
                allocations.insert(0, SplitAllocation(
                    split=DatasetSplit.TRAIN,
                    count=remaining,
                    assignment_type=AssignmentType.OFFICIAL_SPLIT,
                ))
            training_assignments.append(DatasetSourceAssignment(
                dataset_name=source.dataset_name,
                allocations=allocations,
            ))

        assignments.append(ClassDataAssignment(
            class_name=item.class_name,
            sources=training_assignments + official_sources,
        ))

    try:
        validated = validate_dataset_assignments(
            assignments,
            source_roles,
            available_counts=available_counts,
        )
    except DatasetAssignmentValidationError as exc:
        raise DatasetSelectionValidationError(exc.findings) from exc

    coverage_findings = validate_holdout_source_coverage(validated)
    if coverage_findings:
        raise DatasetSelectionValidationError(coverage_findings)
    return validated


def validate_holdout_source_coverage(
    assignments: Iterable[ClassDataAssignment],
) -> list[dict[str, object]]:
    """Require feasible major training families in both primary holdouts."""

    findings: list[dict[str, object]] = []
    for item in assignments:
        family_totals: dict[str, int] = defaultdict(int)
        split_totals: dict[str, int] = defaultdict(int)
        families_by_split: dict[str, set[str]] = defaultdict(set)

        for source in item.sources:
            family = dataset_family(source.dataset_name)
            for allocation in source.allocations:
                split = str(allocation.split)
                family_totals[family] += allocation.count
                split_totals[split] += allocation.count
                families_by_split[split].add(family)

        training_families = families_by_split[DatasetSplit.TRAIN.value]
        if len(training_families) <= 1:
            continue

        total = sum(family_totals[family] for family in training_families)
        for split in (DatasetSplit.VALIDATION.value, DatasetSplit.TEST.value):
            target = split_totals[split]
            required = {
                family
                for family in training_families
                if total and target * family_totals[family] / total >= 1.0
            }
            missing = required - families_by_split[split]
            if missing:
                findings.append({
                    "class_name": item.class_name,
                    "split": split,
                    "missing_families": sorted(missing),
                    "reason": (
                        "The holdout does not represent every training dataset "
                        "family large enough to receive a proportional sample."
                    ),
                })
    return findings


def build_split_construction_summary(
    assignments: Iterable[ClassDataAssignment],
) -> dict[str, object]:
    """Describe the final authoritative plan without relying on LLM prose."""

    values = list(assignments)
    classes = []
    total_counts = {split.value: 0 for split in DatasetSplit}
    for item in values:
        families_by_split: dict[str, set[str]] = defaultdict(set)
        class_counts = {split.value: 0 for split in DatasetSplit}
        sources = []
        for source in item.sources:
            family = dataset_family(source.dataset_name)
            source_counts = {split.value: 0 for split in DatasetSplit}
            for allocation in source.allocations:
                split = str(allocation.split)
                families_by_split[split].add(family)
                source_counts[split] += allocation.count
                class_counts[split] += allocation.count
                total_counts[split] += allocation.count
            sources.append({
                "dataset_name": source.dataset_name,
                "family": family,
                "counts": source_counts,
            })
        class_total = sum(class_counts.values())
        classes.append({
            "class_name": item.class_name,
            "total": class_total,
            "counts": class_counts,
            "ratios": {
                split: count / class_total if class_total else 0.0
                for split, count in class_counts.items()
            },
            "families_by_split": {
                split.value: sorted(families_by_split[split.value])
                for split in DatasetSplit
            },
            "sources": sources,
        })
    overall_total = sum(total_counts.values())
    return {
        "strategy": "source_stratified_primary_holdouts",
        "multi_family_policy": "derive_validation_and_test_from_all_training_sources",
        "single_family_policy": "prefer_compatible_official_holdouts_and_derive_missing_holdouts",
        "total": overall_total,
        "counts": total_counts,
        "ratios": {
            split: count / overall_total if overall_total else 0.0
            for split, count in total_counts.items()
        },
        "classes": classes,
    }


def build_dataset_profile(
    selected_data: Iterable[ClassDataSelection | ClassDataAssignment],
) -> DatasetProfile:
    selected_data = list(selected_data)
    per_class: dict[str, int] = defaultdict(int)
    dataset_ids: set[str] = set()
    domains: set[str] = set()
    primary_domains: set[str] = set()
    for item in selected_data:
        for source in item.sources:
            source_count = (
                source.count
                if hasattr(source, "count")
                else sum(allocation.count for allocation in source.allocations)
            )
            per_class[item.class_name] += source_count
            dataset_ids.add(source.dataset_name)
            info = resolve_dataset_info(source.dataset_name)
            if info:
                domains.update(info.domains)
                if info.domains:
                    primary_domains.add(info.domains[0])

    counts = list(per_class.values())
    minimum = min(counts) if counts else 0
    maximum = max(counts) if counts else 0
    assignments = (
        list(selected_data)
        if not selected_data or isinstance(selected_data[0], ClassDataAssignment)
        else normalize_dataset_assignments(selected_data)
    )
    planned_counts, official_counts, derived_counts = summarize_dataset_assignments(assignments)
    return DatasetProfile(
        total_selected_images=sum(counts),
        minimum_images_per_class=minimum,
        maximum_images_per_class=maximum,
        class_balance_ratio=(minimum / maximum if maximum else 0.0),
        number_of_sources=len(dataset_ids),
        domains=sorted(domains),
        multi_domain=len(primary_domains) > 1,
        planned_counts=planned_counts,
        official_counts=official_counts,
        derived_counts=derived_counts,
    )
