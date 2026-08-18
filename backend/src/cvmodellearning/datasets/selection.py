from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from cvmodellearning.datasets.availability import is_dataset_downloadable
from cvmodellearning.datasets.registry import (
    DatasetRole,
    dataset_family,
    get_dataset_info,
    resolve_dataset_info,
)
from cvmodellearning.policies.data_selection_policy import (
    DETECTION_DATA_SELECTION_POLICY,
    DetectionDataSelectionPolicy,
    matched_domain_tags,
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


def _dataset_reference_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def canonicalize_selected_dataset_ids(
    selected: Iterable[ClassDataSelection],
    eligible: Iterable[ClassDataSelection],
) -> tuple[list[ClassDataSelection], list[dict[str, str]]]:
    """Resolve unique eligible display names/aliases without weakening validation."""
    eligible_by_class = {
        item.class_name: {source.dataset_name for source in item.sources}
        for item in eligible
    }
    normalized: list[ClassDataSelection] = []
    adjustments: list[dict[str, str]] = []
    for item in selected:
        allowed_ids = eligible_by_class.get(item.class_name, set())
        sources = []
        for source in item.sources:
            requested = source.dataset_name
            canonical = requested
            if requested not in allowed_ids:
                requested_key = _dataset_reference_key(requested)
                matches = []
                for dataset_id in allowed_ids:
                    info = resolve_dataset_info(dataset_id)
                    references = [dataset_id]
                    if info is not None:
                        references.extend([info.display_name, *info.aliases])
                    if requested_key and requested_key in {
                        _dataset_reference_key(reference)
                        for reference in references if reference
                    }:
                        matches.append(dataset_id)
                if len(matches) == 1:
                    canonical = matches[0]
                    adjustments.append({
                        "class_name": item.class_name,
                        "provided_reference": requested,
                        "dataset_id": canonical,
                    })
            sources.append(source.model_copy(update={"dataset_name": canonical}))
        normalized.append(item.model_copy(update={"sources": sources}))
    return normalized, adjustments


def prune_ineligible_optional_sources(
    selected: Iterable[ClassDataSelection],
    eligible: Iterable[ClassDataSelection],
    *,
    minimum_images_per_class: int,
) -> tuple[list[ClassDataSelection] | None, list[dict[str, object]]]:
    """Remove only impossible sources when every class retains a sufficient pool.

    This intentionally cannot add a source, change a count, or rescue a missing
    class. It is a narrow boundary fallback for an otherwise usable LLM plan.
    """
    eligible_by_class = {
        item.class_name: {source.dataset_name for source in item.sources}
        for item in eligible
    }
    repaired: list[ClassDataSelection] = []
    removals: list[dict[str, object]] = []
    for item in selected:
        allowed = eligible_by_class.get(item.class_name)
        if allowed is None:
            return None, []
        retained = [source for source in item.sources if source.dataset_name in allowed]
        removed = [source for source in item.sources if source.dataset_name not in allowed]
        if not retained or sum(source.count for source in retained) < minimum_images_per_class:
            return None, []
        repaired.append(item.model_copy(update={"sources": retained}))
        removals.extend({
            "code": "INELIGIBLE_OPTIONAL_SOURCE_REMOVED",
            "class_name": item.class_name,
            "dataset_name": source.dataset_name,
            "count": source.count,
            "reason": (
                "The source was not eligible for this class and the remaining "
                "eligible pool already met the calculated minimum."
            ),
        } for source in removed)
    return (repaired, removals) if removals else (None, [])


DEFAULT_CLASSIFICATION_POOL_PER_CLASS = 1_500
MAX_CLASSIFICATION_POOL_PER_CLASS = 10_000
MAX_CLASSIFICATION_SELECTED_IMAGES = 50_000
DEFAULT_DETECTION_POOL_PER_CLASS = 1_000
MEDIUM_HIGH_DETECTION_POOL_PER_CLASS = 1_500
MAX_DETECTION_POOL_PER_CLASS = 10_000
MAX_DETECTION_SELECTED_IMAGES = 50_000
DETECTION_SHARED_BACKBONE_MIN_COUNT = 50
DETECTION_SHARED_BACKBONE_MIN_SHARE = 0.25
DETECTION_SHARED_BACKBONE_TARGET_COUNT = 500
MIN_HOLDOUT_IMAGES_PER_CLASS_SOURCE = 100
MIN_SOURCE_POOL_FOR_PRIMARY_HOLDOUT = 500
MIN_SOURCE_POOL_SHARE_FOR_PRIMARY_HOLDOUT = 0.10
MAX_PRIMARY_HOLDOUT_IMAGES = 500


@dataclass(frozen=True)
class DetectionSizingConfig:
    """Code-owned defaults for reproducible detection dataset sizing."""

    base_images_per_class: int = 1_000
    full_pretrained_coverage_adjustment: int = -250
    partial_pretrained_coverage_adjustment: int = 500
    no_pretrained_coverage_adjustment: int = 1_000
    moderate_domain_shift_adjustment: int = 500
    strong_domain_shift_adjustment: int = 1_000
    robustness_adjustment_per_dimension: int = 250
    robustness_adjustment_maximum: int = 500
    high_accuracy_adjustment: int = 500
    explicit_small_object_adjustment: int = 750
    observed_small_object_adjustment: int = 1_250
    normal_minimum: int = 750
    normal_maximum: int = 3_000
    evidence_backed_expansion_maximum: int = 5_000


@dataclass(frozen=True)
class DetectionSizingFacts:
    """Structured facts inferred before dataset sources are selected."""

    pretrained_class_coverage: str = "unknown"
    domain_shift: str = "unknown"
    accuracy_demand: str = "standard"
    robustness_dimensions: tuple[str, ...] = ()
    object_size_risk: str = "low"
    object_size_evidence: str = "unknown"
    small_object_fraction: float | None = None
    median_short_side_px_at_640: float | None = None
    previous_run_failed: bool = False


@dataclass(frozen=True)
class DetectionSizingRecommendation:
    target_images_per_class: int
    minimum_images_per_class: int
    maximum_images_per_class: int
    confidence: str
    facts: dict[str, object]
    adjustments: tuple[dict[str, object], ...]
    config: dict[str, object]
    expansion_requires_evidence: bool = True

    def as_context(self) -> dict[str, object]:
        return asdict(self)


COCO_DETECTION_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
})

COCO_PRETRAINED_DETECTOR_MARKERS = (
    "yolo", "rtdetr", "retinanet", "faster", "ssd300",
)

STRONG_DOMAIN_SHIFT_MARKERS = frozenset({
    "thermal", "infrared", "x-ray", "xray", "ultrasound", "mri", "ct scan",
    "microscopy", "satellite", "underwater", "sonar",
})


def infer_detection_sizing_facts(
    *,
    classes: Iterable[str],
    model_reference: str | None,
    application_domain: str | None,
    use_case_description: str | None,
    accuracy_category: str | None,
    other_constraints: Iterable[str] = (),
    robustness_dimensions: Iterable[str] = (),
    object_size_risk: str = "low",
    object_size_evidence: str = "unknown",
    small_object_fraction: float | None = None,
    median_short_side_px_at_640: float | None = None,
) -> DetectionSizingFacts:
    """Convert interpreted planning state into conservative sizing facts.

    These facts can later be enriched by GraphRAG or policy rules. This local
    inference path deliberately remains available when both are disabled.
    """

    requested = {str(value).strip().lower() for value in classes}
    model = (model_reference or "").lower()
    coco_pretrained = any(marker in model for marker in COCO_PRETRAINED_DETECTOR_MARKERS)
    covered = requested & COCO_DETECTION_CLASSES if coco_pretrained else set()
    if requested and covered == requested:
        pretrained_coverage = "full"
    elif covered:
        pretrained_coverage = "partial"
    else:
        pretrained_coverage = "unknown" if not model_reference else "none"

    domain_text = " ".join(filter(None, (application_domain, use_case_description))).lower()
    if any(marker in domain_text for marker in STRONG_DOMAIN_SHIFT_MARKERS):
        domain_shift = "strong"
    elif application_domain:
        domain_shift = "moderate"
    elif any(token in domain_text for token in ("photograph", "photo", "rgb", "camera")):
        domain_shift = "low"
    else:
        domain_shift = "unknown"

    normalized_accuracy = (accuracy_category or "").replace("_", "").replace("-", "").lower()
    constraints_text = " ".join(str(value).lower() for value in other_constraints)
    demanding_recall = "recall" in constraints_text and any(
        marker in constraints_text for marker in ("0.75", "75%", "0.8", "80%", "0.9", "90%")
    )
    accuracy_demand = (
        "high"
        if normalized_accuracy in {"mediumhigh", "high", "veryhigh"} or demanding_recall
        else "standard"
    )
    return DetectionSizingFacts(
        pretrained_class_coverage=pretrained_coverage,
        domain_shift=domain_shift,
        accuracy_demand=accuracy_demand,
        robustness_dimensions=tuple(sorted({
            str(value) for value in robustness_dimensions
            if str(value) not in {"object_scale", "small_object"}
        })),
        object_size_risk=str(object_size_risk or "low"),
        object_size_evidence=str(object_size_evidence or "unknown"),
        small_object_fraction=small_object_fraction,
        median_short_side_px_at_640=median_short_side_px_at_640,
    )


def _round_to_nearest_250(value: int) -> int:
    return max(250, int(round(value / 250)) * 250)


def determine_detection_dataset_size(
    facts: DetectionSizingFacts,
    config: DetectionSizingConfig = DetectionSizingConfig(),
) -> DetectionSizingRecommendation:
    """Calculate an auditable initial pool size from semantic sizing facts.

    The LLM and optional retrieval layers classify the facts. Arithmetic and
    safety bounds remain deterministic and available without either service.
    """

    target = config.base_images_per_class
    adjustments: list[dict[str, object]] = []

    def apply(rule: str, amount: int, reason: str) -> None:
        nonlocal target
        target += amount
        adjustments.append({"rule": rule, "value": amount, "reason": reason})

    coverage = facts.pretrained_class_coverage.lower()
    if coverage == "full":
        apply("full_pretrained_class_coverage", config.full_pretrained_coverage_adjustment,
              "The selected pretrained detector already covers every requested class.")
    elif coverage == "partial":
        apply("partial_pretrained_class_coverage", config.partial_pretrained_coverage_adjustment,
              "Some requested classes require adaptation of the pretrained detector.")
    elif coverage == "none":
        apply("no_pretrained_class_coverage", config.no_pretrained_coverage_adjustment,
              "The requested classes are not covered by the pretrained detector.")

    shift = facts.domain_shift.lower()
    if shift == "moderate" or shift == "unknown":
        apply("moderate_domain_shift", config.moderate_domain_shift_adjustment,
              "The deployment domain is moderately different or could not be verified.")
    elif shift == "strong":
        apply("strong_domain_shift", config.strong_domain_shift_adjustment,
              "The capture modality or deployment domain differs strongly from pretraining.")

    robustness_amount = min(
        len(set(facts.robustness_dimensions))
        * config.robustness_adjustment_per_dimension,
        config.robustness_adjustment_maximum,
    )
    if robustness_amount:
        apply("robustness_coverage", robustness_amount,
              "Requested robustness dimensions require stratified coverage, with a bounded volume increase.")

    size_risk = facts.object_size_risk.lower()
    size_evidence = facts.object_size_evidence.lower()
    if size_risk == "high" and size_evidence == "observed":
        apply(
            "observed_small_object_prevalence",
            config.observed_small_object_adjustment,
            "Measured box statistics show a high-prevalence small-object scenario.",
        )
    elif size_risk in {"medium", "high"}:
        apply(
            "explicit_small_object_requirement",
            config.explicit_small_object_adjustment,
            "The structured task requirements explicitly prioritize small objects.",
        )

    if facts.accuracy_demand.lower() in {"high", "very_high"}:
        apply("high_accuracy_demand", config.high_accuracy_adjustment,
              "The requested accuracy or per-class recall is demanding.")

    observed_expansion_evidence = size_risk == "high" and size_evidence == "observed"
    upper_bound = (
        config.evidence_backed_expansion_maximum
        if facts.previous_run_failed or observed_expansion_evidence
        else config.normal_maximum
    )
    target = min(max(_round_to_nearest_250(target), config.normal_minimum), upper_bound)
    confidence = "high"
    if coverage == "unknown" or shift == "unknown":
        confidence = "low" if coverage == shift == "unknown" else "medium"

    return DetectionSizingRecommendation(
        target_images_per_class=target,
        minimum_images_per_class=max(config.normal_minimum, target - 500),
        maximum_images_per_class=min(upper_bound, target + 500),
        confidence=confidence,
        facts=asdict(facts),
        adjustments=tuple(adjustments),
        config=asdict(config),
        expansion_requires_evidence=not (
            facts.previous_run_failed or observed_expansion_evidence
        ),
    )


def recommend_detection_pool_per_class(
    accuracy_category: str | None,
    requested_robustness_dimensions: Iterable[str] = (),
) -> int:
    """Backward-compatible sizing helper for callers without model/domain facts."""

    normalized = (accuracy_category or "").replace("_", "").replace("-", "").lower()
    return determine_detection_dataset_size(DetectionSizingFacts(
        domain_shift="low",
        accuracy_demand=("high" if normalized in {"mediumhigh", "high", "veryhigh"} else "standard"),
        robustness_dimensions=tuple(str(value) for value in requested_robustness_dimensions),
    )).target_images_per_class


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
    application_domain: str | None = None,
    detection_policy: DetectionDataSelectionPolicy = DETECTION_DATA_SELECTION_POLICY,
) -> list[ClassDataSelection]:
    """Build a conservative valid selection when the preference model fails."""
    eligible = list(eligible_data)
    if prefer_shared_training_family and application_domain:
        domain_aware = _build_domain_aware_detection_selection(
            eligible,
            target_images_per_class,
            application_domain,
            detection_policy,
        )
        if domain_aware:
            return domain_aware
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


def _build_domain_aware_detection_selection(
    eligible_data: list[ClassDataSelection],
    target_images_per_class: int,
    application_domain: str,
    policy: DetectionDataSelectionPolicy,
) -> list[ClassDataSelection]:
    """Prefer one sufficient shared target-domain family; generalize only to fill gaps.

    ``preferred_primary_domain_share`` is a lower-bound preference, not a quota that
    requires adding unrelated data. This also guarantees that the deterministic
    fallback cannot create a sub-threshold secondary source merely by applying an
    80/20 percentage to a small pool.
    """

    requested_classes = {item.class_name for item in eligible_data}
    aligned_by_family: dict[str, dict[str, DatasetSourceCount]] = defaultdict(dict)
    general_by_family: dict[str, dict[str, DatasetSourceCount]] = defaultdict(dict)
    for item in eligible_data:
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if info is None or info.role != DatasetRole.TRAIN or info.synthetic:
                continue
            target = (
                aligned_by_family
                if matched_domain_tags(info.domains, application_domain)
                else general_by_family
            )
            target[dataset_family(source.dataset_name)][item.class_name] = source
    shared_aligned = {
        family: sources
        for family, sources in aligned_by_family.items()
        if requested_classes <= set(sources)
    }
    shared_general = {
        family: sources
        for family, sources in general_by_family.items()
        if requested_classes <= set(sources)
    }
    primary_family = max(
        shared_aligned,
        key=lambda family: (
            sum(shared_aligned[family][name].count for name in requested_classes),
            family,
        ),
        default=None,
    )
    secondary_family = max(
        shared_general,
        key=lambda family: (
            sum(shared_general[family][name].count for name in requested_classes),
            family,
        ),
        default=None,
    )
    if len(requested_classes) > 1 and primary_family is None:
        return []

    result = []
    for item in eligible_data:
        candidates = [
            source
            for source in item.sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role == DatasetRole.TRAIN
            and not info.synthetic
        ]
        aligned = [
            source
            for source in candidates
            if matched_domain_tags(
                resolve_dataset_info(source.dataset_name).domains,
                application_domain,
            )
        ]
        if not aligned:
            continue
        primary = (
            shared_aligned[primary_family][item.class_name]
            if primary_family is not None
            else max(aligned, key=lambda source: (source.count, source.dataset_name))
        )
        secondary_candidates = [
            source
            for source in candidates
            if source.dataset_name != primary.dataset_name
            and dataset_family(source.dataset_name) != dataset_family(primary.dataset_name)
            and not matched_domain_tags(
                resolve_dataset_info(source.dataset_name).domains,
                application_domain,
            )
        ]
        secondary = (
            shared_general[secondary_family][item.class_name]
            if secondary_family is not None
            else max(
                secondary_candidates,
                key=lambda source: (source.count, source.dataset_name),
                default=None,
            )
        )
        primary_count = min(primary.count, target_images_per_class)
        remaining = target_images_per_class - primary_count
        selected_sources = [DatasetSourceCount(
            dataset_name=primary.dataset_name,
            count=primary_count,
        )]
        if secondary is not None and remaining:
            secondary_count = min(secondary.count, remaining)
            if secondary_count:
                selected_sources.append(DatasetSourceCount(
                    dataset_name=secondary.dataset_name,
                    count=secondary_count,
                ))
                remaining -= secondary_count
        result.append(ClassDataSelection(
            class_name=item.class_name,
            sources=selected_sources,
        ))
    return result


def validate_detection_domain_mix(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
    application_domain: str | None,
    policy: DetectionDataSelectionPolicy = DETECTION_DATA_SELECTION_POLICY,
) -> list[ClassDataSelection]:
    """Enforce target-domain priority only when local aligned data can support it."""

    selected = list(selected_data)
    findings = detection_domain_mix_findings(
        selected, eligible_data, application_domain, policy,
    )
    if findings:
        raise DatasetSelectionValidationError(findings)
    return selected


def detection_domain_mix_findings(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
    application_domain: str | None,
    policy: DetectionDataSelectionPolicy = DETECTION_DATA_SELECTION_POLICY,
) -> list[dict[str, object]]:
    """Describe domain-composition risks without deciding whether they block planning."""
    selected = list(selected_data)
    if not application_domain:
        return []
    eligible_by_class = {item.class_name: item for item in eligible_data}
    findings: list[dict[str, object]] = []
    for item in selected:
        total = sum(source.count for source in item.sources)
        if not total:
            continue
        eligible_aligned = [
            source
            for source in eligible_by_class.get(item.class_name, ClassDataSelection(
                class_name=item.class_name, sources=[]
            )).sources
            if (info := resolve_dataset_info(source.dataset_name)) is not None
            and info.role == DatasetRole.TRAIN
            and not info.synthetic
            and matched_domain_tags(info.domains, application_domain)
        ]
        required = round(total * policy.minimum_primary_domain_share)
        if sum(source.count for source in eligible_aligned) < required:
            continue
        aligned_count = 0
        generalization_sources = 0
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if info is not None and matched_domain_tags(info.domains, application_domain):
                aligned_count += source.count
            elif info is not None and info.role == DatasetRole.TRAIN:
                generalization_sources += 1
        share = aligned_count / total
        if share < policy.minimum_primary_domain_share:
            findings.append({
                "code": "INSUFFICIENT_PRIMARY_DOMAIN_SHARE",
                "severity": "warning",
                "field": "sources",
                "class_name": item.class_name,
                "selected_share": share,
                "minimum_share": policy.minimum_primary_domain_share,
                "reason": "Sufficient target-domain data is available but underrepresented.",
            })
        if generalization_sources > policy.maximum_generalization_sources:
            findings.append({
                "code": "TOO_MANY_GENERALIZATION_SOURCES",
                "severity": "warning",
                "field": "sources",
                "class_name": item.class_name,
                "selected_count": generalization_sources,
                "maximum_count": policy.maximum_generalization_sources,
                "reason": "Use one coherent generalization source at this pool size.",
            })
    return findings


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
            existing_info = (
                resolve_dataset_info(existing.dataset_name) if existing is not None else None
            )
            if existing is None or (
                not info.synthetic,
                source.count,
                source.dataset_name,
            ) > (
                not existing_info.synthetic if existing_info is not None else False,
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
    """Reject leakage-unsafe source combinations; quality is reported separately."""

    selected = list(selected_data)
    eligible = list(eligible_data)
    findings: list[dict[str, object]] = []
    for item in selected:
        selected_ids = {source.dataset_name for source in item.sources}
        total = sum(source.count for source in item.sources)
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if (
                info is not None
                and info.synthetic
                and info.derived_from in selected_ids
                and not info.paired_sample_ids_available
            ):
                findings.append({
                    "field": "dataset_name",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "derived_from": info.derived_from,
                    "reason": (
                        "Original and derived datasets cannot be mixed without "
                        "pair identifiers for leakage-safe splitting."
                    ),
                })
    if findings:
        raise DatasetSelectionValidationError(findings)
    return selected


def detection_source_coherence_findings(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
) -> list[dict[str, object]]:
    """Report class/source confounding risk without rejecting an executable plan."""

    selected = list(selected_data)
    eligible = list(eligible_data)
    requested_classes = {item.class_name for item in eligible}
    if len(requested_classes) < 2:
        return []

    eligible_by_family: dict[str, dict[str, int]] = defaultdict(dict)
    for item in eligible:
        for source in item.sources:
            info = resolve_dataset_info(source.dataset_name)
            if info is not None and info.role == DatasetRole.TRAIN:
                family = dataset_family(source.dataset_name)
                current = eligible_by_family[family].get(item.class_name, 0)
                if not info.synthetic or current == 0:
                    eligible_by_family[family][item.class_name] = max(source.count, current)

    common_families = {
        family
        for family, counts in eligible_by_family.items()
        if requested_classes <= set(counts)
        and min(counts[class_name] for class_name in requested_classes)
        >= DETECTION_SHARED_BACKBONE_MIN_COUNT
    }
    if not common_families:
        return []

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
            return []

    return [{
        "code": "SHARED_BACKBONE_NOT_USED",
        "severity": "warning",
        "field": "sources",
        "classes": sorted(requested_classes),
        "common_families": sorted(common_families),
        "reason": (
            "A training family with sufficient data for every requested class is available, "
            "but no such family supplies a meaningful shared backbone across all classes. "
            "This can confound class identity with dataset identity, but does not make the plan inexecutable."
        ),
        "available_options": [
            {
                "option": "use_common_family_as_required_supplement",
                "dataset_family": family,
                "minimum_images_per_class": DETECTION_SHARED_BACKBONE_TARGET_COUNT,
            }
            for family in sorted(common_families)
        ],
    }]


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
            if not is_dataset_downloadable(source.dataset_name):
                findings.append({
                    "field": "dataset_name",
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": "Dataset is currently unavailable for materialization.",
                })
                continue
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
    *,
    use_official_validation: bool = True,
    use_official_test: bool = True,
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
            if (
                role == DatasetRole.VALIDATION and not use_official_validation
            ) or (
                role == DatasetRole.TEST and not use_official_test
            ):
                continue
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


def _meaningful_holdout_counts(
    source_counts: list[int],
    target: int,
    *,
    eligible_indexes: list[int] | None = None,
) -> list[int]:
    """Allocate a representative holdout while preserving source proportions.

    ``eligible_indexes`` remains part of the interface because callers also use it
    to decide which sources belong in primary holdouts. It must not inflate a small
    source beyond its selected-pool share; statistical sufficiency is reported as a
    warning instead of changing the evaluation distribution.
    """

    if eligible_indexes is None:
        return _proportional_holdout_counts(source_counts, target)
    indexes = list(eligible_indexes)
    if not indexes:
        return [0] * len(source_counts)
    eligible_counts = [source_counts[index] for index in indexes]
    eligible_allocations = _proportional_holdout_counts(eligible_counts, target)
    result = [0] * len(source_counts)
    for index, count in zip(indexes, eligible_allocations):
        result[index] = count
    return result


def build_dataset_assignments(
    selected_data: Iterable[ClassDataSelection],
    eligible_data: Iterable[ClassDataSelection],
    *,
    training_only_dataset_ids_by_class: dict[str, set[str]] | None = None,
    allowed_splits_by_class_dataset: dict[tuple[str, str], set[str]] | None = None,
    use_official_validation: bool = True,
    use_official_test: bool = True,
    derive_missing_holdouts: bool = True,
    aggregate_errors: bool = True,
) -> list[ClassDataAssignment]:
    """Create a complete, deterministic split plan from validated source choices."""

    selected_values = list(selected_data)
    eligible_values = list(eligible_data)
    if aggregate_errors and len(selected_values) > 1:
        combined: list[ClassDataAssignment] = []
        findings: list[dict[str, object]] = []
        eligible_by_class = {item.class_name: item for item in eligible_values}
        for item in selected_values:
            try:
                combined.extend(build_dataset_assignments(
                    [item],
                    [eligible_by_class[item.class_name]],
                    training_only_dataset_ids_by_class=training_only_dataset_ids_by_class,
                    allowed_splits_by_class_dataset=allowed_splits_by_class_dataset,
                    use_official_validation=use_official_validation,
                    use_official_test=use_official_test,
                    derive_missing_holdouts=derive_missing_holdouts,
                    aggregate_errors=False,
                ))
            except DatasetSelectionValidationError as exc:
                findings.extend(exc.findings)
        if findings:
            raise DatasetSelectionValidationError(findings)
        return combined

    selected = complete_official_holdout_selection(
        selected_values,
        eligible_values,
        use_official_validation=use_official_validation,
        use_official_test=use_official_test,
    )
    assignments: list[ClassDataAssignment] = []
    available_counts = {
        (item.class_name, source.dataset_name): source.count
        for item in eligible_values
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
        if missing_holdouts and not derive_missing_holdouts:
            raise DatasetSelectionValidationError([{
                "class_name": item.class_name,
                "missing_splits": [split.value for split in missing_holdouts],
                "reason": (
                    "The LLM split strategy forbids derived holdouts, but compatible "
                    "official holdouts are unavailable or disabled."
                ),
            }])
        selected_pool_total = train_total + sum(
            allocation.count
            for source in official_sources
            for allocation in source.allocations
        )
        holdout_count = recommend_holdout_count(selected_pool_total)
        stable_holdout_indexes = [
            index
            for index, count in sorted(
                enumerate(source.count for source in train_sources),
                key=lambda pair: (pair[1], -pair[0]),
                reverse=True,
            )
            if (
                train_sources[index].dataset_name not in (
                    (training_only_dataset_ids_by_class or {}).get(item.class_name, set())
                )
                and
                count >= MIN_SOURCE_POOL_FOR_PRIMARY_HOLDOUT
                and train_total
                and count / train_total >= MIN_SOURCE_POOL_SHARE_FOR_PRIMARY_HOLDOUT
            )
        ][:MAX_PRIMARY_HOLDOUT_IMAGES // MIN_HOLDOUT_IMAGES_PER_CLASS_SOURCE]
        if stable_holdout_indexes:
            holdout_count = min(
                MAX_PRIMARY_HOLDOUT_IMAGES,
                max(
                    holdout_count,
                    len(stable_holdout_indexes) * MIN_HOLDOUT_IMAGES_PER_CLASS_SOURCE,
                ),
            )
        if train_total <= holdout_count * len(missing_holdouts):
            raise DatasetSelectionValidationError([{
                "class_name": item.class_name,
                "reason": "Not enough training images to create disjoint train/validation/test allocations.",
            }])

        remaining_by_source = [source.count for source in train_sources]
        derived_by_split: dict[DatasetSplit, list[int]] = {}
        for split in missing_holdouts:
            split_eligible_indexes = stable_holdout_indexes or None
            if allowed_splits_by_class_dataset is not None:
                split_eligible_indexes = [
                    index for index in stable_holdout_indexes
                    if split.value in allowed_splits_by_class_dataset.get(
                        (item.class_name, train_sources[index].dataset_name),
                        {DatasetSplit.TRAIN.value},
                    )
                ]
                if not split_eligible_indexes:
                    raise DatasetSelectionValidationError([{
                        "class_name": item.class_name,
                        "split": split.value,
                        "reason": (
                            "No sufficiently represented LLM-approved training source "
                            "is permitted to provide this derived holdout."
                        ),
                    }])
            split_counts = _meaningful_holdout_counts(
                remaining_by_source,
                holdout_count,
                eligible_indexes=split_eligible_indexes,
            )
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

    coverage_findings = validate_holdout_source_coverage(
        validated,
        training_only_dataset_ids_by_class=training_only_dataset_ids_by_class,
    )
    if coverage_findings:
        raise DatasetSelectionValidationError(coverage_findings)
    return validated


def validate_holdout_source_coverage(
    assignments: Iterable[ClassDataAssignment],
    *,
    training_only_dataset_ids_by_class: dict[str, set[str]] | None = None,
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
        intentionally_training_only_families = {
            dataset_family(dataset_id)
            for dataset_id in (
                (training_only_dataset_ids_by_class or {}).get(item.class_name, set())
            )
        }
        training_families = training_families - intentionally_training_only_families
        if len(training_families) <= 1:
            continue

        total = sum(family_totals[family] for family in training_families)
        for split in (DatasetSplit.VALIDATION.value, DatasetSplit.TEST.value):
            target = split_totals[split]
            required = {
                family
                for family in sorted(
                    training_families,
                    key=lambda value: (family_totals[value], value),
                    reverse=True,
                )[:MAX_PRIMARY_HOLDOUT_IMAGES // MIN_HOLDOUT_IMAGES_PER_CLASS_SOURCE]
                if (
                    family_totals[family] >= MIN_SOURCE_POOL_FOR_PRIMARY_HOLDOUT
                    and total
                    and family_totals[family] / total
                    >= MIN_SOURCE_POOL_SHARE_FOR_PRIMARY_HOLDOUT
                )
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
    warnings = []
    total_counts = {split.value: 0 for split in DatasetSplit}
    for item in values:
        class_total = sum(
            allocation.count
            for source in item.sources
            for allocation in source.allocations
        )
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
            info = resolve_dataset_info(source.dataset_name)
            if info is not None and info.role == DatasetRole.TRAIN:
                missing_holdouts = [
                    split
                    for split in (DatasetSplit.VALIDATION.value, DatasetSplit.TEST.value)
                    if source_counts[split] == 0
                ]
                if len(missing_holdouts) == 2:
                    warnings.append({
                        "code": "SOURCE_EXCLUDED_FROM_PRIMARY_HOLDOUT",
                        "class_name": item.class_name,
                        "dataset_name": source.dataset_name,
                        "selected_count": sum(source_counts.values()),
                        "selected_share": (
                            sum(source_counts.values()) / class_total if class_total else 0.0
                        ),
                        "minimum_count": MIN_SOURCE_POOL_FOR_PRIMARY_HOLDOUT,
                        "minimum_share": MIN_SOURCE_POOL_SHARE_FOR_PRIMARY_HOLDOUT,
                        "use": "training_only",
                        "excluded_splits": missing_holdouts,
                        "reason": (
                            "The source is not sufficiently represented for reliable "
                            "source-level primary holdouts."
                        ),
                    })
                else:
                    for split in (
                        DatasetSplit.VALIDATION.value,
                        DatasetSplit.TEST.value,
                    ):
                        count = source_counts[split]
                        if 0 < count < MIN_HOLDOUT_IMAGES_PER_CLASS_SOURCE:
                            warnings.append({
                                "code": "SOURCE_METRIC_SAMPLE_SMALL",
                                "class_name": item.class_name,
                                "dataset_name": source.dataset_name,
                                "split": split,
                                "selected_count": count,
                                "minimum_reliable_count": (
                                    MIN_HOLDOUT_IMAGES_PER_CLASS_SOURCE
                                ),
                                "reason": (
                                    "The proportional primary holdout is representative, "
                                    "but source-specific metrics may have high variance."
                                ),
                            })
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
        "multi_family_policy": "derive_holdouts_from_sufficiently_represented_training_sources",
        "single_family_policy": "prefer_compatible_official_holdouts_and_derive_missing_holdouts",
        "total": overall_total,
        "counts": total_counts,
        "ratios": {
            split: count / overall_total if overall_total else 0.0
            for split, count in total_counts.items()
        },
        "classes": classes,
        "warnings": warnings,
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
        target_unique_images=sum(counts),
        minimum_images_by_class=dict(sorted(per_class.items())),
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
