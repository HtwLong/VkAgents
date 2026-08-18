import asyncio
import csv
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cvmodellearning.agents.data_selection_and_augmentation_agents import DataSelectionPatch
from cvmodellearning.agents.data_selection_and_augmentation_agents import (
    detection_dataset_selection_agent,
)
from cvmodellearning.datasets.selection import (
    DatasetSelectionValidationError,
    DetectionSizingFacts,
    DEFAULT_CLASSIFICATION_POOL_PER_CLASS,
    DEFAULT_DETECTION_POOL_PER_CLASS,
    MEDIUM_HIGH_DETECTION_POOL_PER_CLASS,
    MAX_CLASSIFICATION_POOL_PER_CLASS,
    MAX_CLASSIFICATION_SELECTED_IMAGES,
    MAX_DETECTION_SELECTED_IMAGES,
    MAX_DETECTION_POOL_PER_CLASS,
    build_default_dataset_selection,
    build_dataset_assignments,
    build_split_construction_summary,
    canonicalize_selected_dataset_ids,
    determine_detection_dataset_size,
    detection_domain_mix_findings,
    detection_source_coherence_findings,
    filter_dataset_candidates,
    filter_training_candidates,
    limit_selected_source_pools,
    infer_detection_sizing_facts,
    prune_ineligible_optional_sources,
    recommend_holdout_count,
    recommend_detection_pool_per_class,
    validate_detection_domain_mix,
    validate_detection_source_coherence,
    validate_dataset_selection,
)
from cvmodellearning.datasets.availability import (
    DatasetAvailabilityConfigError,
    get_dataset_availability,
)
from cvmodellearning.datasets.registry import (
    DATASET_REGISTRY,
    dataset_family,
    resolve_dataset_info,
)
from cvmodellearning.download import download_data, visionkg_utils
from cvmodellearning.graphrag.build_graph import build_graph
from cvmodellearning.graphrag.dataset_selection_context import (
    aggregate_selected_dataset_properties,
    build_dataset_selection_context,
)
from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    validate_detection_graph_grounded_config,
)
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.schemas.interpretation_schema import ClassDataSelection
from routers import planning


def test_default_detection_selector_uses_simple_selection_contract():
    assert detection_dataset_selection_agent.name == "Detection Data Selector"
    assert detection_dataset_selection_agent.output_type is DataSelectionPatch


def _selection(class_name, *sources):
    return ClassDataSelection(
        class_name=class_name,
        sources=[{"dataset_name": name, "count": count} for name, count in sources],
    )


@pytest.mark.parametrize(
    ("task", "class_name", "dataset_id", "display_name"),
    (
        ("classification", "0", "mnist_cls_train", "MNIST classification train"),
        ("detection", "car", "bdd_100k_det_train", "BDD100K detection train"),
    ),
)
def test_unique_eligible_display_name_is_canonicalized_for_every_cv_task(
    task, class_name, dataset_id, display_name,
):
    eligible = [_selection(class_name, (dataset_id, 2_000))]
    proposed = [_selection(class_name, (display_name, 1_500))]

    normalized, adjustments = canonicalize_selected_dataset_ids(proposed, eligible)
    validated = validate_dataset_selection(normalized, eligible)

    assert validated[0].sources[0].dataset_name == dataset_id
    assert adjustments == [{
        "class_name": class_name,
        "provided_reference": display_name,
        "dataset_id": dataset_id,
    }]


def test_ineligible_optional_sources_are_pruned_only_when_valid_pool_is_sufficient():
    eligible = [
        _selection("car", ("bdd_100k_det_train", 5_000)),
        _selection("bus", ("bdd_100k_det_train", 5_000)),
    ]
    proposed = [
        _selection("car", ("bdd_100k_det_train", 2_000), ("invalid", 500)),
        _selection("bus", ("bdd_100k_det_train", 2_000), ("invalid", 500)),
    ]

    repaired, adjustments = prune_ineligible_optional_sources(
        proposed, eligible, minimum_images_per_class=2_000,
    )

    assert repaired is not None
    assert [source.dataset_name for source in repaired[0].sources] == ["bdd_100k_det_train"]
    assert len(adjustments) == 2


def test_ineligible_sources_are_not_pruned_when_the_remainder_is_too_small():
    eligible = [_selection("car", ("bdd_100k_det_train", 5_000))]
    proposed = [_selection(
        "car", ("bdd_100k_det_train", 1_999), ("invalid", 500),
    )]

    repaired, adjustments = prune_ineligible_optional_sources(
        proposed, eligible, minimum_images_per_class=2_000,
    )

    assert repaired is None
    assert adjustments == []


def test_deployment_coverage_warning_uses_only_requested_unverified_slices():
    state = SimpleNamespace(
        user_query=(
            "Detect objects indoors and outdoors under varied lighting, at different "
            "scales, and when partially occluded."
        )
    )

    assert planning.deployment_coverage_warnings(state) == [{
        "code": "DEPLOYMENT_COVERAGE_UNVERIFIED",
        "severity": "warning",
        "dimensions": ["indoor_outdoor", "lighting", "object_scale", "occlusion"],
        "reason": (
            "The split is source-stratified, but the available sample metadata does "
            "not verify coverage of these requested deployment characteristics."
        ),
    }]


def test_local_registry_matches_visionkg_ontology_rows():
    datasets_csv = (
        Path(__file__).resolve().parents[1]
        / "ontology_data"
        / "nodes"
        / "datasets.csv"
    )
    with datasets_csv.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["id"]: row
            for row in csv.DictReader(handle)
            if "evidence_visionkg_dataset_registry" in row["evidence_ids"]
        }

    role_map = {
        "Training": "train",
        "Validation": "validation",
        "Test": "test",
        "Benchmark": "benchmark",
    }
    task_map = {
        "object_detection": "detection",
        "image_classification": "classification",
        "instance_segmentation": "instance_segmentation",
    }

    assert set(DATASET_REGISTRY) == set(rows)
    for dataset_id, info in DATASET_REGISTRY.items():
        assert info.role.value == role_map[rows[dataset_id]["dataset_role"]]
        assert info.task == task_map[rows[dataset_id]["task_ids"]]


def test_dataset_characteristic_graph_edges_are_complete():
    graph = build_graph(Path(__file__).resolve().parents[1] / "ontology_data")
    characteristic_nodes = {
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("source_csv") == "dataset_characteristics.csv"
    }
    dataset_edges = {
        target
        for _, target, edge in graph.edges(data=True)
        if edge.get("relation") == "has_characteristic"
    }
    property_edges = {
        source
        for source, _, edge in graph.edges(data=True)
        if edge.get("relation") == "characteristic_type"
    }

    assert characteristic_nodes
    assert dataset_edges == characteristic_nodes
    assert property_edges == characteristic_nodes


def test_dataset_characteristic_references_and_activation_policies_are_valid():
    ontology_dir = Path(__file__).resolve().parents[1] / "ontology_data" / "nodes"

    def rows(name):
        with (ontology_dir / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    datasets = {item["id"] for item in rows("datasets.csv")}
    properties = {item["id"]: item for item in rows("dataset_properties.csv")}
    evidence = {item["id"] for item in rows("evidence_sources.csv")}
    characteristics = rows("dataset_characteristics.csv")

    for item in characteristics:
        assert item["dataset_id"] in datasets
        assert item["property_id"] in properties
        assert item["confidence"] in {"Low", "Medium", "High"}
        assert item["value"] in {"true", "false"}
        assert set(filter(None, item["evidence_ids"].split("|"))) <= evidence
    for item in properties.values():
        assert item["aggregation_mode"] in {"any", "all", "weighted_threshold"}
        assert 0.0 <= float(item["activation_threshold"]) <= 1.0
        assert item["minimum_activation_confidence"] in {"Low", "Medium", "High"}


def test_filter_training_candidates_excludes_validation_test_and_benchmark():
    available = [_selection(
        "car",
        ("bdd_100k_det_train", 100),
        ("bdd_100k_det_val", 50),
        ("cars196_det_test", 20),
        ("UA-DETRAC_det", 30),
    )]

    filtered = filter_training_candidates(available, "detection")

    assert [source.dataset_name for source in filtered[0].sources] == ["bdd_100k_det_train"]


def test_filter_excludes_dataset_with_disabled_download():
    available = [_selection(
        "truck",
        ("objects365_det_train", 1000),
        ("openimages_challenge_2019_det_val", 100),
        ("cifar10_cls_train", 500),
        ("KITTI_det", 200),
    )]

    filtered = filter_training_candidates(available, "detection")

    assert filtered[0].sources == []


def test_dataset_availability_can_be_updated_without_code_changes(tmp_path, monkeypatch):
    status_file = tmp_path / "dataset_availability.json"
    status_file.write_text(
        '{"datasets":{"example_det_train":{"downloadable":false,"reason":"API down"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("DATASET_AVAILABILITY_FILE", str(status_file))

    assert get_dataset_availability("example_det_train").downloadable is False
    assert get_dataset_availability("new_det_train").downloadable is True

    status_file.write_text(
        '{"datasets":{"example_det_train":{"downloadable":true,"reason":"Fixed"}}}',
        encoding="utf-8",
    )
    assert get_dataset_availability("example_det_train").downloadable is True


def test_invalid_dataset_availability_config_fails_clearly(tmp_path, monkeypatch):
    status_file = tmp_path / "dataset_availability.json"
    status_file.write_text('{"datasets":{"bad":{"downloadable":"no"}}}', encoding="utf-8")
    monkeypatch.setenv("DATASET_AVAILABILITY_FILE", str(status_file))

    with pytest.raises(DatasetAvailabilityConfigError, match="boolean 'downloadable'"):
        get_dataset_availability("bad")


def test_selection_validation_explicitly_rejects_unavailable_source(tmp_path, monkeypatch):
    status_file = tmp_path / "dataset_availability.json"
    status_file.write_text(
        '{"datasets":{"bdd_100k_det_train":{"downloadable":false}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("DATASET_AVAILABILITY_FILE", str(status_file))
    selected = [_selection("car", ("bdd_100k_det_train", 100))]

    with pytest.raises(DatasetSelectionValidationError) as exc_info:
        validate_dataset_selection(selected, selected)

    assert any(
        finding["reason"] == "Dataset is currently unavailable for materialization."
        for finding in exc_info.value.findings
    )


def test_detection_sizing_reduces_coco_covered_general_photo_request():
    facts = infer_detection_sizing_facts(
        classes=["person", "dog", "cat"],
        model_reference="yolov10n",
        application_domain=None,
        use_case_description="Indoor and outdoor photographs with varied lighting and occlusion.",
        accuracy_category="Medium",
        other_constraints=["per_class_recall>=0.75"],
        robustness_dimensions=["indoor_outdoor", "lighting", "object_scale", "occlusion"],
    )

    recommendation = determine_detection_dataset_size(facts)

    assert facts.pretrained_class_coverage == "full"
    assert facts.domain_shift == "low"
    assert facts.accuracy_demand == "high"
    assert recommendation.target_images_per_class == 1_750
    assert recommendation.maximum_images_per_class == 2_250


def test_detection_sizing_keeps_larger_pool_for_new_classes_and_strong_shift():
    recommendation = determine_detection_dataset_size(DetectionSizingFacts(
        pretrained_class_coverage="none",
        domain_shift="strong",
        accuracy_demand="high",
        robustness_dimensions=("object_scale", "occlusion"),
    ))

    assert recommendation.target_images_per_class == 3_000
    assert recommendation.expansion_requires_evidence is True


def test_explicit_small_object_requirement_raises_detection_pool_to_normal_cap():
    recommendation = determine_detection_dataset_size(DetectionSizingFacts(
        pretrained_class_coverage="partial",
        domain_shift="moderate",
        accuracy_demand="high",
        robustness_dimensions=("lighting",),
        object_size_risk="medium",
        object_size_evidence="explicit_requirement",
    ))

    assert recommendation.target_images_per_class == 3_000
    assert recommendation.facts["object_size_risk"] == "medium"
    assert {
        item["rule"]: item["value"] for item in recommendation.adjustments
    }["explicit_small_object_requirement"] == 750
    assert recommendation.expansion_requires_evidence is True


def test_observed_small_object_prevalence_allows_evidence_backed_expansion():
    recommendation = determine_detection_dataset_size(DetectionSizingFacts(
        pretrained_class_coverage="partial",
        domain_shift="moderate",
        accuracy_demand="high",
        object_size_risk="high",
        object_size_evidence="observed",
        small_object_fraction=0.68,
        median_short_side_px_at_640=24.0,
    ))

    assert recommendation.target_images_per_class == 3_750
    assert recommendation.maximum_images_per_class == 4_250
    assert recommendation.expansion_requires_evidence is False
    assert {
        item["rule"]: item["value"] for item in recommendation.adjustments
    }["observed_small_object_prevalence"] == 1_250


def test_filter_merges_duplicate_class_and_source_availability_rows():
    available = [
        _selection("car", ("coco2017_det_train", 100)),
        _selection(
            "car",
            ("coco2017_det_train", 120),
            ("bdd_100k_det_train", 80),
        ),
    ]

    filtered = filter_training_candidates(available, "detection")

    assert len(filtered) == 1
    assert filtered[0].model_dump() == {
        "class_name": "car",
        "sources": [
            {"dataset_name": "coco2017_det_train", "count": 120},
            {"dataset_name": "bdd_100k_det_train", "count": 80},
        ],
    }


def test_vqa_can_use_training_images_from_visionkg_detection_sources():
    available = [_selection(
        "car",
        ("bdd_100k_det_train", 100),
        ("bdd_100k_det_val", 50),
    )]

    filtered = filter_training_candidates(available, "visual question answering")

    assert [source.dataset_name for source in filtered[0].sources] == ["bdd_100k_det_train"]


def test_filter_training_candidates_infers_detection_train_ids_when_registry_is_missing():
    available = [_selection(
        "car",
        ("example_det_train", 120),
        ("objects365_det_val", 50),
    )]

    filtered = filter_training_candidates(available, "detection")

    assert [source.dataset_name for source in filtered[0].sources] == ["example_det_train"]


def test_filter_training_candidates_requires_classification_sources_for_classification():
    available = [_selection(
        "car",
        ("bdd_100k_det_train", 80),
        ("CUB-200-2011_cls_train", 100),
    )]

    filtered = filter_training_candidates(available, "classification")

    assert [source.dataset_name for source in filtered[0].sources] == ["CUB-200-2011_cls_train"]


def test_explicit_images_per_class_parses_user_request():
    assert planning.explicit_images_per_class("Use 1,200 images per class") == 1200
    assert planning.explicit_images_per_class("Use a balanced dataset") is None


@pytest.mark.parametrize(
    ("accuracy_category", "expected"),
    [
        (None, DEFAULT_DETECTION_POOL_PER_CLASS),
        ("Medium", DEFAULT_DETECTION_POOL_PER_CLASS),
        ("MediumHigh", MEDIUM_HIGH_DETECTION_POOL_PER_CLASS),
        ("High", MEDIUM_HIGH_DETECTION_POOL_PER_CLASS),
        ("VeryHigh", MEDIUM_HIGH_DETECTION_POOL_PER_CLASS),
    ],
)
def test_detection_pool_recommendation_is_accuracy_aware(accuracy_category, expected):
    assert recommend_detection_pool_per_class(accuracy_category) == expected


def test_detection_pool_recommendation_uses_requested_robustness():
    assert recommend_detection_pool_per_class(None, {"lighting", "occlusion"}) == 1_500


def test_translated_dataset_lineage_is_local_and_canonical():
    info = resolve_dataset_info("bdd100k_UNIT_day2night_det_train")

    assert info is not None
    assert info.synthetic is True
    assert info.derived_from == "bdd_100k_det_train"
    assert info.paired_sample_ids_available is False
    assert dataset_family(info.dataset_id) == "bdd_100k"


def test_detection_selection_rejects_unpaired_original_and_derivative():
    selected = [_selection(
        "car",
        ("bdd_100k_det_train", 1_200),
        ("bdd100k_UNIT_day2night_det_train", 800),
    )]

    with pytest.raises(DatasetSelectionValidationError) as exc_info:
        validate_detection_source_coherence(selected, selected)

    assert any(
        "cannot be mixed without pair identifiers" in finding["reason"]
        for finding in exc_info.value.findings
    )


def test_small_detection_supplement_is_kept_training_only_with_warning():
    selected = [_selection(
        "car",
        ("bdd_100k_det_train", 1_900),
        ("coco2017_det_train", 100),
    )]

    coherent = validate_detection_source_coherence(selected, selected)
    plan = build_dataset_assignments(
        coherent,
        selected,
        training_only_dataset_ids_by_class={"car": {"coco2017_det_train"}},
    )
    summary = build_split_construction_summary(plan)

    assert any(
        warning.get("dataset_name") == "coco2017_det_train"
        and warning.get("use") == "training_only"
        for warning in summary["warnings"]
    )


def test_split_builder_enforces_llm_permitted_splits():
    selected = [_selection("car", ("bdd_100k_det_train", 2_000))]

    with pytest.raises(DatasetSelectionValidationError, match="invalid") as exc_info:
        build_dataset_assignments(
            selected,
            selected,
            use_official_validation=False,
            use_official_test=False,
            derive_missing_holdouts=True,
            allowed_splits_by_class_dataset={
                ("car", "bdd_100k_det_train"): {"train", "validation"},
            },
        )

    assert any(
        finding.get("split") == "test"
        and "LLM-approved" in finding["reason"]
        for finding in exc_info.value.findings
    )


def test_split_builder_reports_detectable_errors_for_all_classes_together():
    selected = [
        _selection("car", ("bdd_100k_det_train", 2_000)),
        _selection("bus", ("bdd_100k_det_train", 2_000)),
    ]

    with pytest.raises(DatasetSelectionValidationError) as exc_info:
        build_dataset_assignments(
            selected,
            selected,
            use_official_validation=False,
            use_official_test=False,
            derive_missing_holdouts=True,
            allowed_splits_by_class_dataset={
                ("car", "bdd_100k_det_train"): {"train"},
                ("bus", "bdd_100k_det_train"): {"train"},
            },
        )

    assert {item["class_name"] for item in exc_info.value.findings} == {"car", "bus"}


def test_primary_can_supply_derived_holdouts_while_supplement_stays_training_only():
    selected = [_selection(
        "car",
        ("bdd_100k_det_train", 2_000),
        ("coco2017_det_train", 600),
    )]

    plan = build_dataset_assignments(
        selected,
        selected,
        training_only_dataset_ids_by_class={"car": {"coco2017_det_train"}},
        use_official_validation=False,
        use_official_test=False,
        derive_missing_holdouts=True,
        allowed_splits_by_class_dataset={
            ("car", "bdd_100k_det_train"): {"train", "validation", "test"},
            ("car", "coco2017_det_train"): {"train"},
        },
    )

    by_source = {source.dataset_name: source for source in plan[0].sources}
    assert {str(item.split) for item in by_source["bdd_100k_det_train"].allocations} == {
        "train", "validation", "test",
    }
    assert {str(item.split) for item in by_source["coco2017_det_train"].allocations} == {
        "train",
    }


def test_classification_pool_limits_are_balanced_and_preserve_source_priority():
    limited = limit_selected_source_pools(
        [
            _selection("cat", ("openimages_challenge_2019_det_train", 8_000)),
            _selection(
                "dog",
                ("openimages_challenge_2019_det_train", 3_000),
                ("coco2017_det_train", 5_000),
            ),
        ],
        max_total_images=7_000,
        max_images_per_class=MAX_CLASSIFICATION_POOL_PER_CLASS,
    )

    assert [sum(source.count for source in item.sources) for item in limited] == [3_500, 3_500]
    assert limited[1].model_dump()["sources"] == [
        {"dataset_name": "openimages_challenge_2019_det_train", "count": 3_000},
        {"dataset_name": "coco2017_det_train", "count": 500},
    ]


def test_detection_pool_limit_caps_each_class():
    limited = limit_selected_source_pools(
        [
            _selection("rare", ("openimages_challenge_2019_det_train", 1_000)),
            _selection("common", ("coco2017_det_train", 20_000)),
        ],
        max_total_images=MAX_DETECTION_SELECTED_IMAGES,
        max_images_per_class=MAX_DETECTION_POOL_PER_CLASS,
    )

    assert [sum(source.count for source in item.sources) for item in limited] == [1_000, 10_000]


def test_detection_source_coherence_warns_about_avoidable_class_dataset_coupling():
    eligible = filter_dataset_candidates([
        _selection(
            "person",
            ("cityscapes_det_train", 1_000),
            ("coco2017_det_train", 1_000),
        ),
        _selection(
            "car",
            ("bdd_100k_det_train", 1_000),
            ("coco2017_det_train", 1_000),
        ),
    ], "detection")

    selected = [
        _selection("person", ("cityscapes_det_train", 1_000)),
        _selection("car", ("bdd_100k_det_train", 1_000)),
    ]

    assert validate_detection_source_coherence(selected, eligible) == selected
    findings = detection_source_coherence_findings(selected, eligible)
    assert findings[0]["severity"] == "warning"
    assert findings[0]["common_families"] == ["coco2017"]
    assert findings[0]["available_options"][0]["dataset_family"] == "coco2017"


def test_detection_source_coherence_accepts_meaningful_common_backbone():
    eligible = filter_dataset_candidates([
        _selection("person", ("coco2017_det_train", 1_000)),
        _selection("car", ("coco2017_det_train", 1_000)),
    ], "detection")
    selected = [
        _selection("person", ("coco2017_det_train", 500)),
        _selection("car", ("coco2017_det_train", 500)),
    ]

    assert validate_detection_source_coherence(selected, eligible) == selected


def test_detection_default_selection_prefers_shared_training_family():
    eligible = filter_dataset_candidates([
        _selection(
            "person",
            ("cityscapes_det_train", 2_000),
            ("coco2017_det_train", 1_000),
        ),
        _selection(
            "car",
            ("bdd_100k_det_train", 2_000),
            ("coco2017_det_train", 1_000),
        ),
    ], "detection")

    fallback = build_default_dataset_selection(
        eligible,
        target_images_per_class=800,
        prefer_shared_training_family=True,
    )

    assert [item.model_dump() for item in fallback] == [
        {
            "class_name": "person",
            "sources": [{"dataset_name": "coco2017_det_train", "count": 800}],
        },
        {
            "class_name": "car",
            "sources": [{"dataset_name": "coco2017_det_train", "count": 800}],
        },
    ]


def test_pool_limit_counts_explicit_official_holdouts_in_total_budget():
    limited = limit_selected_source_pools(
        [_selection(
            "car",
            ("bdd_100k_det_train", 100),
            ("bdd_100k_det_val", 100),
        )],
        max_total_images=120,
        max_images_per_class=120,
    )

    assert limited[0].model_dump()["sources"] == [
        {"dataset_name": "bdd_100k_det_train", "count": 100},
        {"dataset_name": "bdd_100k_det_val", "count": 20},
    ]
    plan = build_dataset_assignments(limited, filter_dataset_candidates(
        [_selection(
            "car",
            ("bdd_100k_det_train", 100),
            ("bdd_100k_det_val", 100),
        )],
        "detection",
    ))
    assert sum(
        allocation.count
        for source in plan[0].sources
        for allocation in source.allocations
    ) == 120


def test_split_planning_preserves_official_splits_and_derives_only_missing_test():
    available = [_selection(
        "car",
        ("bdd_100k_det_train", 100),
        ("bdd_100k_det_val", 50),
    )]
    eligible = filter_dataset_candidates(available, "detection")
    plan = build_dataset_assignments(
        [_selection("car", ("bdd_100k_det_train", 80))],
        eligible,
    )

    assert plan[0].model_dump() == {
        "class_name": "car",
        "sources": [
            {
                    "dataset_name": "bdd_100k_det_train",
                    "allocations": [
                        {"split": "train", "count": 56, "assignment_type": "official_split"},
                        {"split": "test", "count": 12, "assignment_type": "derived_from_train"},
                ],
            },
            {
                "dataset_name": "bdd_100k_det_val",
                "allocations": [
                    {"split": "validation", "count": 12, "assignment_type": "official_split"},
                ],
            },
        ],
    }


def test_split_planning_caps_oversized_official_holdout_without_losing_budget():
    available = [_selection(
        "furniture",
        ("openimages_challenge_2019_det_train", 14698),
        ("openimages_challenge_2019_det_val", 1477),
    )]
    eligible = filter_dataset_candidates(available, "detection")

    plan = build_dataset_assignments(
        [_selection(
            "furniture",
            ("openimages_challenge_2019_det_train", 523),
            ("openimages_challenge_2019_det_val", 1477),
        )],
        eligible,
    )

    assert plan[0].model_dump() == {
        "class_name": "furniture",
        "sources": [
            {
                "dataset_name": "openimages_challenge_2019_det_train",
                "allocations": [
                    {"split": "train", "count": 1600, "assignment_type": "official_split"},
                    {"split": "test", "count": 200, "assignment_type": "derived_from_train"},
                ],
            },
            {
                "dataset_name": "openimages_challenge_2019_det_val",
                "allocations": [
                    {"split": "validation", "count": 200, "assignment_type": "official_split"},
                ],
            },
        ],
    }


def test_check_data_filters_available_sources_by_interpreted_task(monkeypatch, tmp_path):
    monkeypatch.setattr(planning, "planning_artifacts_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(
        planning,
        "get_multi_class_stats",
        lambda classes, query_output_path=None, task=None: {
            "car": {
                "bdd_100k_det_train": 120,
                "cars196_cls_train": 80,
                "UA-DETRAC_det": 40,
            },
        },
    )

    response = asyncio.run(planning.check_data(planning.StateRequest(
        context=PipelineState(
            user_query="detect cars",
            task="detection",
            classes=["car"],
        ).model_dump(),
        job_id="task-filter-test",
    )))

    sources = response["context"]["available_data"][0]["sources"]
    assert sources == [{"dataset_name": "bdd_100k_det_train", "count": 120}]


def test_split_planning_preserves_official_test_when_selected():
    available = [_selection(
        "bird",
        ("CUB-200-2011_cls_train", 100),
        ("CUB-200-2011_cls_test", 40),
    )]
    eligible = filter_dataset_candidates(available, "classification")
    plan = build_dataset_assignments(
        [_selection(
            "bird",
            ("CUB-200-2011_cls_train", 80),
            ("CUB-200-2011_cls_test", 10),
        )],
        eligible,
    )

    allocations = {
        source.dataset_name: source.model_dump()["allocations"]
        for source in plan[0].sources
    }
    assert allocations["CUB-200-2011_cls_test"] == [
        {"split": "test", "count": 10, "assignment_type": "official_split"}
    ]
    assert allocations["CUB-200-2011_cls_train"] == [
        {"split": "train", "count": 66, "assignment_type": "official_split"},
        {"split": "validation", "count": 14, "assignment_type": "derived_from_train"},
    ]


def test_unrelated_official_holdout_is_rejected():
    available = [_selection(
        "cat",
        ("openimages_challenge_2019_det_train", 800),
        ("cifar10_cls_test", 100),
    )]
    eligible = filter_dataset_candidates(available, "classification")

    with pytest.raises(DatasetSelectionValidationError) as exc_info:
        validate_dataset_selection(
            [_selection(
                "cat",
                ("openimages_challenge_2019_det_train", 800),
                ("cifar10_cls_test", 100),
            )],
            eligible,
        )

    assert "must belong to a selected training dataset family" in str(
        exc_info.value.findings
    )


def test_missing_holdouts_are_source_stratified_with_adaptive_size():
    available = [_selection(
        "cat",
        ("openimages_challenge_2019_det_train", 600),
        ("coco2017_det_train", 240),
    )]
    eligible = filter_dataset_candidates(available, "classification")
    plan = build_dataset_assignments(available, eligible)
    allocations = {
        source.dataset_name: {
            item["split"]: item["count"]
            for item in source.model_dump()["allocations"]
        }
        for source in plan[0].sources
    }

    assert recommend_holdout_count(840) == 101
    assert sum(item.get("validation", 0) for item in allocations.values()) == 101
    assert sum(item.get("test", 0) for item in allocations.values()) == 101
    assert allocations["openimages_challenge_2019_det_train"]["validation"] > 0
    assert allocations["coco2017_det_train"].get("validation", 0) == 0
    assert allocations["openimages_challenge_2019_det_train"]["test"] > 0
    assert allocations["coco2017_det_train"].get("test", 0) == 0


def test_small_supplement_remains_training_only_and_holdouts_stay_meaningful():
    selected = [_selection(
        "car",
        ("bdd_100k_det_train", 1_500),
        ("ACDC_det_train", 200),
    )]

    plan = build_dataset_assignments(selected, selected)
    allocations = {
        source.dataset_name: {
            allocation.split: allocation.count
            for allocation in source.allocations
        }
        for source in plan[0].sources
    }

    assert allocations["bdd_100k_det_train"] == {
        "train": 1_160,
        "validation": 170,
        "test": 170,
    }
    assert allocations["ACDC_det_train"] == {"train": 200}

    summary = build_split_construction_summary(plan)
    assert len(summary["warnings"]) == 1
    warning = summary["warnings"][0]
    assert warning == {
        "code": "SOURCE_EXCLUDED_FROM_PRIMARY_HOLDOUT",
        "class_name": "car",
        "dataset_name": "ACDC_det_train",
        "selected_count": 200,
        "selected_share": pytest.approx(200 / 1_700),
        "minimum_count": 500,
        "minimum_share": 0.1,
        "use": "training_only",
        "excluded_splits": ["validation", "test"],
        "reason": (
            "The source is not sufficiently represented for reliable "
            "source-level primary holdouts."
        ),
    }


def test_substantial_sources_keep_proportional_primary_holdout_allocations():
    selected = [_selection(
        "car",
        ("bdd_100k_det_train", 1_150),
        ("openimages_challenge_2019_det_train", 750),
    )]

    plan = build_dataset_assignments(selected, selected)
    allocations = {
        source.dataset_name: {
            allocation.split: allocation.count
            for allocation in source.allocations
        }
        for source in plan[0].sources
    }

    assert allocations == {
        "bdd_100k_det_train": {"train": 908, "validation": 121, "test": 121},
        "openimages_challenge_2019_det_train": {
            "train": 592,
            "validation": 79,
            "test": 79,
        },
    }
    warnings = build_split_construction_summary(plan)["warnings"]
    assert [warning["code"] for warning in warnings] == [
        "SOURCE_METRIC_SAMPLE_SMALL",
        "SOURCE_METRIC_SAMPLE_SMALL",
    ]
    assert {warning["split"] for warning in warnings} == {"validation", "test"}
    assert all(warning["selected_count"] == 79 for warning in warnings)


def test_domain_mix_source_ratios_are_preserved_in_primary_holdouts():
    selected = [_selection(
        "traffic light",
        ("bdd_100k_det_train", 2_000),
        ("openimages_challenge_2019_det_train", 500),
    )]

    plan = build_dataset_assignments(selected, selected)
    allocations = {
        source.dataset_name: {
            allocation.split: allocation.count
            for allocation in source.allocations
        }
        for source in plan[0].sources
    }

    assert allocations == {
        "bdd_100k_det_train": {
            "train": 1_600,
            "validation": 200,
            "test": 200,
        },
        "openimages_challenge_2019_det_train": {
            "train": 400,
            "validation": 50,
            "test": 50,
        },
    }


def test_large_but_low_share_supplement_remains_training_only():
    selected = [_selection(
        "car",
        ("bdd_100k_det_train", 9_500),
        ("openimages_challenge_2019_det_train", 500),
    )]

    plan = build_dataset_assignments(selected, selected)
    allocations = {
        source.dataset_name: {
            allocation.split: allocation.count
            for allocation in source.allocations
        }
        for source in plan[0].sources
    }

    assert allocations["openimages_challenge_2019_det_train"] == {"train": 500}
    assert allocations["bdd_100k_det_train"]["validation"] == 500
    assert allocations["bdd_100k_det_train"]["test"] == 500



def test_partial_official_validation_does_not_confound_multi_family_holdouts():
    selected = [_selection(
        "cat",
        ("coco2017_det_train", 800),
        ("openimages_challenge_2019_det_train", 1_000),
        ("LVIS_det_train", 200),
        ("openimages_challenge_2019_det_val", 200),
    )]
    eligible = [_selection(
        "cat",
        ("coco2017_det_train", 4_000),
        ("openimages_challenge_2019_det_train", 12_000),
        ("LVIS_det_train", 1_900),
        ("openimages_challenge_2019_det_val", 300),
    )]

    plan = build_dataset_assignments(selected, eligible)
    allocations = {
        source.dataset_name: {
            allocation.split: allocation
            for allocation in source.allocations
        }
        for source in plan[0].sources
    }

    assert "openimages_challenge_2019_det_val" not in allocations
    assert allocations["openimages_challenge_2019_det_train"][
        "validation"
    ].assignment_type == "derived_from_train"
    assert allocations["openimages_challenge_2019_det_train"][
        "test"
    ].assignment_type == "derived_from_train"
    assert allocations["coco2017_det_train"]["validation"].count == 89
    assert allocations["coco2017_det_train"]["test"].count == 89
    assert allocations["openimages_challenge_2019_det_train"]["validation"].count == 111
    assert allocations["openimages_challenge_2019_det_train"]["test"].count == 111
    assert "validation" not in allocations["LVIS_det_train"]
    assert "test" not in allocations["LVIS_det_train"]

    summary = build_split_construction_summary(plan)
    assert summary["strategy"] == "source_stratified_primary_holdouts"
    assert summary["classes"][0]["families_by_split"] == {
        "train": ["coco2017", "lvis", "openimages_challenge_2019"],
        "validation": ["coco2017", "openimages_challenge_2019"],
        "test": ["coco2017", "openimages_challenge_2019"],
    }


def test_select_datasets_is_safe_without_graphrag(monkeypatch):
    async def fake_run(agent, input):
        payload = __import__("json").loads(input)
        assert "available_data" not in payload
        assert "dataset_catalog" not in payload
        assert "dataset_guidance" not in payload
        assert {
            item["dataset_id"]
            for item in payload["allowed_sources_by_class"]["car"]
        } == {"bdd_100k_det_train", "bdd_100k_det_val"}
        assert payload["training_class_coverage_by_family"] == {"bdd_100k": ["car"]}
        assert payload["target_images_per_class"] == 1_500
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("car", ("bdd_100k_det_train", 80))],
            rationale="Street-domain training data.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    request = planning.StateRequest(
        job_id="dataset-test",
        use_graphrag=False,
        context={
            "task": "detection",
            "classes": ["car"],
            "dataset_selection_graph_context": {"enabled": True, "stale": True},
            "available_data": [_selection(
                "car",
                ("bdd_100k_det_train", 100),
                ("bdd_100k_det_val", 50),
            ).model_dump()],
        },
    )

    result = asyncio.run(planning.select_datasets(request))

    assert result["decision_evidence"]["mode"] == "llm_validated"
    assert result["context"]["selected_data"][0]["sources"][0]["dataset_name"] == "bdd_100k_det_train"
    assert result["context"]["dataset_profile"]["total_selected_images"] == 80
    assert result["context"]["dataset_profile"]["planned_counts"] == {
        "train": 56,
        "validation": 12,
        "test": 12,
    }
    assert result["context"]["dataset_profile"]["official_counts"] == {
        "train": 56,
        "validation": 12,
        "test": 0,
    }
    assert result["context"]["dataset_selection_graph_context"] is None
    assert result["decision_evidence"]["graphrag_used"] is False
    assert result["decision_evidence"]["decision"] == result["context"]["selected_data"]
    assert result["decision_evidence"]["assignments_authoritative"] is True
    assert result["decision_evidence"]["split_policy"]["official_splits_preserved"] is True
    assert result["decision_evidence"]["split_policy"]["test_used_for_model_selection"] is False
    assert result["decision_evidence"]["split_policy"][
        "multi_family_primary_holdouts"
    ] == "derived_from_sufficiently_represented_training_sources"
    assert result["decision_evidence"]["source_selection_rationale"] == (
        "Street-domain training data."
    )
    split_summary = result["decision_evidence"]["split_construction_summary"]
    assert split_summary["counts"] == {"train": 56, "validation": 12, "test": 12}
    assert split_summary["ratios"] == {
        "train": 0.7,
        "validation": 0.15,
        "test": 0.15,
    }
    assert split_summary["classes"][0]["families_by_split"] == {
        "train": ["bdd_100k"],
        "validation": ["bdd_100k"],
        "test": ["bdd_100k"],
    }
    assert result["decision_evidence"]["rationale"] == "Street-domain training data."
    assert result["decision_evidence"]["retrieved_facts"] == []


def test_select_datasets_canonicalizes_display_name_before_split_planning(monkeypatch):
    async def fake_run(agent, input):
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("0", ("MNIST classification train", 1_500))],
            rationale="Use the MNIST training source.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    result = asyncio.run(planning.select_datasets(planning.StateRequest(
        job_id="dataset-display-name-normalization",
        use_graphrag=False,
        context={
            "task": "classification",
            "classes": ["0"],
            "available_data": [_selection(
                "0",
                ("mnist_cls_train", 5_923),
                ("mnist_cls_test", 980),
            ).model_dump()],
        },
    )))

    selected = result["context"]["selected_data"][0]["sources"]
    assert [source["dataset_name"] for source in selected] == [
        "mnist_cls_train",
        "mnist_cls_test",
    ]
    assert result["decision_evidence"]["dataset_identifier_normalizations"] == [{
        "class_name": "0",
        "provided_reference": "MNIST classification train",
        "dataset_id": "mnist_cls_train",
    }]
    assert result["decision_evidence"]["split_construction_summary"]["counts"] == {
        "train": 1_200,
        "validation": 150,
        "test": 150,
    }


def test_classification_endpoint_preserves_full_llm_pool(monkeypatch):
    async def fake_run(agent, input):
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("chair", ("SOP_cls_train", 1000))],
            rationale="Selected the full compatible pool.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    result = asyncio.run(planning.select_datasets(planning.StateRequest(
        job_id="classification-full-pool",
        use_graphrag=False,
            context={
                "task": "classification",
                "classes": ["chair"],
                "available_data": [
                    _selection("chair", ("SOP_cls_train", 1000)).model_dump()
                ],
        },
    )))

    assert result["context"]["dataset_profile"]["planned_counts"] == {
        "train": 800,
        "validation": 100,
        "test": 100,
    }
    assert result["decision_evidence"]["split_policy"][
        "classification_max_pool_per_class"
    ] == MAX_CLASSIFICATION_POOL_PER_CLASS
    assert result["decision_evidence"]["split_policy"][
        "classification_max_selected_images"
    ] == MAX_CLASSIFICATION_SELECTED_IMAGES
    assert result["decision_evidence"]["split_policy"][
        "classification_default_pool_per_class"
    ] == DEFAULT_CLASSIFICATION_POOL_PER_CLASS
    assert result["decision_evidence"]["split_policy"][
        "detection_default_pool_per_class"
    ] == DEFAULT_DETECTION_POOL_PER_CLASS
    assert result["decision_evidence"]["split_policy"][
        "detection_max_pool_per_class"
    ] == MAX_DETECTION_POOL_PER_CLASS
    assert result["decision_evidence"]["split_policy"][
        "detection_max_selected_image_allocations"
    ] == MAX_DETECTION_SELECTED_IMAGES
    assert result["decision_evidence"]["split_policy"][
        "detection_instance_limit_enforced"
    ] is False


def test_imagenet_classification_splits_are_not_downloadable():
    for dataset_id in ("imageNet-1K_cls_train", "imageNet-1K_cls_val"):
        availability = get_dataset_availability(dataset_id)
        assert availability.downloadable is False
        assert "WordNet IDs" in availability.reason


def test_select_datasets_does_not_show_unavailable_dataset_to_llm(monkeypatch):
    async def fake_run(agent, input):
        payload = __import__("json").loads(input)
        candidate_ids = {
            source["dataset_id"]
            for sources in payload["allowed_sources_by_class"].values()
            for source in sources
        }
        assert "objects365_det_train" not in candidate_ids
        assert candidate_ids == {"bdd_100k_det_train"}
        assert payload["allowed_sources_by_class"]["car"][0][
            "available_count"
        ] == 100
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("car", ("bdd_100k_det_train", 80))],
            rationale="Selected a currently downloadable source.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    request = planning.StateRequest(
        job_id="dataset-availability-test",
        use_graphrag=False,
        context={
            "task": "detection",
            "classes": ["car"],
            "available_data": [_selection(
                "car",
                ("objects365_det_train", 100),
                ("bdd_100k_det_train", 100),
            ).model_dump()],
        },
    )

    result = asyncio.run(planning.select_datasets(request))

    assert result["decision_evidence"]["eligible_dataset_ids"] == [
        "bdd_100k_det_train"
    ]


def test_select_datasets_exposes_sources_only_under_their_allowed_class(monkeypatch):
    async def fake_run(agent, input):
        payload = __import__("json").loads(input)
        allowed = payload["allowed_sources_by_class"]
        assert {item["dataset_id"] for item in allowed["car"]} == {
            "ACDC_det_train",
            "bdd_100k_det_train",
        }
        assert {item["dataset_id"] for item in allowed["bus"]} == {
            "bdd_100k_det_train"
        }
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[
                _selection("car", ("bdd_100k_det_train", 80)),
                _selection("bus", ("bdd_100k_det_train", 80)),
            ],
            rationale="Used only sources allowed for each exact class.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    request = planning.StateRequest(
        job_id="dataset-class-specific-eligibility-test",
        use_graphrag=False,
        context={
            "task": "detection",
            "classes": ["car", "bus"],
            "available_data": [
                _selection(
                    "car",
                    ("ACDC_det_train", 100),
                    ("bdd_100k_det_train", 100),
                ).model_dump(),
                _selection("bus", ("bdd_100k_det_train", 100)).model_dump(),
            ],
        },
    )

    result = asyncio.run(planning.select_datasets(request))

    assert result["decision_evidence"]["mode"] == "llm_validated"


def test_select_datasets_uses_graph_guidance_without_changing_eligibility(monkeypatch):
    async def fake_run(agent, input):
        payload = __import__("json").loads(input)
        assert sorted(payload["dataset_guidance"]) == [
            "bdd_100k_det_train",
            "bdd_100k_det_val",
        ]
        assert {
            item["dataset_id"]
            for item in payload["allowed_sources_by_class"]["car"]
        } == set(payload["dataset_guidance"])
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("car", ("bdd_100k_det_train", 80))],
            rationale="Graph-enriched street-domain choice.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    request = planning.StateRequest(
        job_id="dataset-graphrag-test",
        use_graphrag=True,
        context={
            "task": "detection",
            "classes": ["car"],
            "application_domain": "street scenes and autonomous driving",
            "available_data": [_selection(
                "car",
                ("bdd_100k_det_train", 100),
                ("bdd_100k_det_val", 50),
            ).model_dump()],
        },
    )

    result = asyncio.run(planning.select_datasets(request))

    assert result["decision_evidence"]["graphrag_used"] is True
    assert result["decision_evidence"]["guidance_source"] == "dataset_graphrag"
    assert result["decision_evidence"]["eligible_dataset_ids"] == [
        "bdd_100k_det_train",
        "bdd_100k_det_val",
    ]
    assert result["decision_evidence"]["decision_type"] == "dataset_selection"
    assert result["decision_evidence"]["rationale"] == "Graph-enriched street-domain choice."
    assert result["decision_evidence"]["retrieved_facts"]
    assert any(
        fact["type"] == "dataset_characteristic"
        for fact in result["decision_evidence"]["retrieved_facts"]
    )
    assert result["decision_evidence"]["evidence_backed"] is True
    assert any(
        source["id"] == "evidence_bdd100k_paper"
        for source in result["decision_evidence"]["evidence_sources"]
    )
    assert result["context"]["dataset_profile"]["characteristics"] == [
        "ObjectScaleVariation",
        "OutdoorLightingVariation",
    ]


def test_select_datasets_continues_when_optional_graph_guidance_fails(monkeypatch):
    async def fake_run(agent, input):
        payload = __import__("json").loads(input)
        assert "dataset_guidance" not in payload
        assert payload["allowed_sources_by_class"]["car"][0][
            "dataset_id"
        ] == "bdd_100k_det_train"
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("car", ("bdd_100k_det_train", 80))],
            rationale="Locally grounded fallback guidance.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    monkeypatch.setattr(
        planning,
        "build_dataset_selection_context",
        lambda *args: (_ for _ in ()).throw(RuntimeError("graph unavailable")),
    )
    request = planning.StateRequest(
        job_id="dataset-graphrag-failure-test",
        use_graphrag=True,
        context={
            "task": "detection",
            "classes": ["car"],
            "available_data": [
                _selection("car", ("bdd_100k_det_train", 100)).model_dump()
            ],
        },
    )

    result = asyncio.run(planning.select_datasets(request))

    assert result["decision_evidence"]["graphrag_used"] is False
    assert result["decision_evidence"]["guidance_source"] == "local_registry"
    assert result["context"]["selected_data"][0]["sources"][0][
        "dataset_name"
    ] == "bdd_100k_det_train"
    assert "graph unavailable" in result["context"][
        "dataset_selection_graph_context"
    ]["warning"]


def test_select_datasets_repairs_hallucinated_source(monkeypatch):
    calls = []
    async def fake_run(agent, input):
        payload = __import__("json").loads(input)
        calls.append(payload)
        if payload.get("validation_errors"):
            return SimpleNamespace(final_output=DataSelectionPatch(
                selected_data=[_selection("car", ("bdd_100k_det_train", 100))],
                rationale="Repaired to the eligible source.",
            ))
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("car", ("openimages_challenge_2019_det_val", 10))],
            rationale="Invalid source.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    request = planning.StateRequest(
        job_id="dataset-test",
        use_graphrag=False,
        context={
            "task": "detection",
            "classes": ["car"],
            "available_data": [_selection("car", ("bdd_100k_det_train", 100)).model_dump()],
        },
    )

    result = asyncio.run(planning.select_datasets(request))

    assert len(calls) == 2
    assert calls[1]["validation_errors"]
    assert result["decision_evidence"]["mode"] == "llm_validated"
    assert "Repaired to the eligible source" in result["decision_evidence"]["rationale"]
    assert result["context"]["dataset_profile"]["planned_counts"] == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }
    assert any(
        "not an eligible available source" in item["reason"]
        for item in result["decision_evidence"]["validation_findings"]
    )



def test_detection_default_selection_prefers_sufficient_shared_target_domain_family():
    eligible = filter_dataset_candidates([
        _selection(
            "car",
            ("bdd_100k_det_train", 68_000),
            ("openimages_challenge_2019_det_train", 89_000),
            ("coco2017_det_train", 12_000),
        ),
    ], "detection")

    fallback = build_default_dataset_selection(
        eligible,
        target_images_per_class=5_000,
        prefer_shared_training_family=True,
        application_domain="traffic",
    )

    assert fallback[0].model_dump()["sources"] == [
        {"dataset_name": "bdd_100k_det_train", "count": 5_000},
    ]
    assert validate_detection_domain_mix(fallback, eligible, "traffic") == fallback


@pytest.mark.parametrize("target", [750, 1_000, 1_500, 2_000, 2_499, 2_500, 3_000, 5_000])
def test_detection_domain_fallback_is_valid_across_pool_boundaries(target):
    eligible = filter_dataset_candidates([
        _selection(
            "traffic light",
            ("bdd_100k_det_train", 39_401),
            ("openimages_challenge_2019_det_train", 1_773),
        ),
        _selection(
            "traffic sign",
            ("bdd_100k_det_train", 57_188),
            ("openimages_challenge_2019_det_train", 2_813),
        ),
    ], "detection")

    fallback = build_default_dataset_selection(
        eligible,
        target_images_per_class=target,
        prefer_shared_training_family=True,
        application_domain="urban traffic",
    )

    assert all(
        item.model_dump()["sources"] == [
            {"dataset_name": "bdd_100k_det_train", "count": target}
        ]
        for item in fallback
    )
    assert validate_detection_source_coherence(fallback, eligible) == fallback
    assert validate_detection_domain_mix(
        fallback, eligible, "urban traffic"
    ) == fallback
    assignments = build_dataset_assignments(fallback, eligible)
    assert {item.class_name for item in assignments} == {
        "traffic light", "traffic sign"
    }


def test_detection_domain_mix_rejects_general_data_dominance():
    eligible = [_selection(
        "car",
        ("bdd_100k_det_train", 68_000),
        ("openimages_challenge_2019_det_train", 89_000),
        ("coco2017_det_train", 12_000),
    )]
    selected = [_selection(
        "car",
        ("bdd_100k_det_train", 1_500),
        ("openimages_challenge_2019_det_train", 2_500),
        ("coco2017_det_train", 1_000),
    )]

    with pytest.raises(DatasetSelectionValidationError) as exc_info:
        validate_detection_domain_mix(selected, eligible, "traffic")

    assert {finding["code"] for finding in exc_info.value.findings} == {
        "INSUFFICIENT_PRIMARY_DOMAIN_SHARE",
        "TOO_MANY_GENERALIZATION_SOURCES",
    }

    advisory = detection_domain_mix_findings(selected, eligible, "traffic")
    assert {finding["severity"] for finding in advisory} == {"warning"}


def test_select_datasets_repairs_count_above_availability(monkeypatch):
    calls = []
    async def fake_run(agent, input):
        payload = __import__("json").loads(input)
        calls.append(payload)
        count = 100 if payload.get("validation_errors") else 101
        return SimpleNamespace(final_output=DataSelectionPatch(
            selected_data=[_selection("car", ("bdd_100k_det_train", count))],
            rationale="Repaired image count." if count == 100 else "Too many images.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    request = planning.StateRequest(
        job_id="dataset-test",
        context={
            "task": "detection",
            "classes": ["car"],
            "available_data": [_selection("car", ("bdd_100k_det_train", 100)).model_dump()],
        },
    )

    result = asyncio.run(planning.select_datasets(request))

    assert len(calls) == 2
    assert calls[1]["validation_errors"][0]["reason"] == "Selected count exceeds availability."
    assert result["decision_evidence"]["mode"] == "llm_validated"
    assert sum(
        allocation["count"]
        for allocation in result["context"]["selected_data"][0]["sources"][0]["allocations"]
    ) == 100
    assert result["decision_evidence"]["validation_findings"][0]["reason"] == (
        "Selected count exceeds availability."
    )


def test_select_datasets_rejects_missing_availability(monkeypatch):
    async def unexpected_run(agent, input):
        raise AssertionError("The selector must not run without eligible data.")

    monkeypatch.setattr(planning.Runner, "run", unexpected_run)
    request = planning.StateRequest(
        job_id="dataset-test",
        use_graphrag=False,
        context={
            "task": "detection",
            "classes": ["car"],
            "available_data": [],
        },
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(planning.select_datasets(request))

    assert error.value.status_code == 422
    assert error.value.detail["classes"] == ["car"]


def test_select_datasets_keeps_small_inferred_expansions_strict(monkeypatch):
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)
    request = planning.StateRequest(
        job_id="small-expansion-test",
        use_graphrag=False,
        context={
            "task": "detection",
            "classes": ["chair", "table", "carpet"],
            "class_expansions": {"furniture": ["chair", "table", "carpet"]},
            "available_data": [
                _selection("chair", ("coco2017_det_train", 100)).model_dump(),
                _selection("table", ("LVIS_det_train", 100)).model_dump(),
                _selection("carpet", ("objects365_det_val", 100)).model_dump(),
            ],
        },
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(planning.select_datasets(request))

    assert error.value.detail["classes"] == ["carpet"]


def test_select_datasets_rejects_detection_data_for_classification(monkeypatch):
    async def unexpected_run(agent, input):
        raise AssertionError("The selector must not receive detection datasets for classification.")

    monkeypatch.setattr(planning.Runner, "run", unexpected_run)
    request = planning.StateRequest(
        job_id="classification-native-task-test",
        use_graphrag=False,
        context={
            "task": "classification",
            "classes": ["car"],
            "available_data": [
                _selection("car", ("bdd_100k_det_train", 100)).model_dump()
            ],
        },
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(planning.select_datasets(request))

    assert error.value.status_code == 422
    assert error.value.detail == {
        "message": (
            "No compatible image-classification datasets contain enough training "
            "images for all classes in the given user prompt."
        ),
        "classes": ["car"],
        "reason": (
            "Classification requires datasets marked for image classification in "
            "the dataset registry; detection datasets cannot be used as image-level "
            "classification data."
        ),
    }


def test_dataset_profile_does_not_treat_tags_as_distinct_domains():
    from cvmodellearning.datasets.selection import build_dataset_profile

    profile = build_dataset_profile([
        _selection("car", ("bdd_100k_det_train", 80)),
    ])

    assert profile.domains == ["autonomous_driving", "street"]
    assert profile.multi_domain is False


def test_weighted_characteristic_aggregation_uses_selected_counts():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_data=[_selection(
            "car",
            ("bdd_100k_det_train", 20),
            ("voc12_det_train", 80),
        )],
    )

    properties = {
        item["property_id"]: item
        for item in aggregate_selected_dataset_properties(state)
    }

    assert properties["OutdoorLightingVariation"]["support_ratio"] == 0.2
    assert properties["OutdoorLightingVariation"]["active"] is False
    assert properties["ObjectScaleVariation"]["evidence_support_ratio"] == 1.0
    assert properties["ObjectScaleVariation"]["support_ratio"] == 0.2
    assert properties["ObjectScaleVariation"]["active"] is False


def test_medium_confidence_characteristic_guides_but_does_not_require_adjustment():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_data=[_selection("car", ("bdd_100k_det_train", 100))],
    )

    properties = {
        item["property_id"]: item
        for item in aggregate_selected_dataset_properties(state)
    }

    assert properties["FrequentOcclusion"]["evidence_support_ratio"] == 1.0
    assert properties["FrequentOcclusion"]["support_ratio"] == 0.0
    assert properties["FrequentOcclusion"]["active"] is False


def test_weighted_characteristic_activates_at_exact_threshold():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_data=[_selection(
            "car",
            ("bdd_100k_det_train", 30),
            ("voc12_det_train", 70),
        )],
    )

    properties = {
        item["property_id"]: item
        for item in aggregate_selected_dataset_properties(state)
    }

    assert properties["OutdoorLightingVariation"]["support_ratio"] == 0.3
    assert properties["OutdoorLightingVariation"]["active"] is True


def test_dataset_graphrag_context_never_expands_eligible_candidates():
    state = PipelineState(
        task="detection",
        classes=["car"],
        application_domain="street scenes and autonomous driving",
    )
    eligible = [_selection("car", ("bdd_100k_det_train", 100))]

    context = build_dataset_selection_context(state, eligible)

    assert context["eligible_dataset_ids"] == ["bdd_100k_det_train"]
    assert [item["dataset_id"] for item in context["candidate_guidance"]] == [
        "bdd_100k_det_train"
    ]
    candidate = context["candidate_guidance"][0]
    assert candidate["display_name"]
    assert "dataset_name" not in candidate
    assert any(item["matched_user_domain_terms"] for item in candidate["domains"])
    assert {
        item["property_id"] for item in candidate["characteristics"]
    } >= {"OutdoorLightingVariation", "ObjectScaleVariation"}


def test_acdc_training_context_exposes_evidenced_adverse_lighting_coverage():
    state = PipelineState(
        task="detection",
        classes=["car"],
        application_domain="traffic monitoring at night",
    )
    eligible = [_selection("car", ("ACDC_det_train", 1_525))]

    context = build_dataset_selection_context(state, eligible)
    candidate = context["candidate_guidance"][0]

    assert candidate["dataset_id"] == "ACDC_det_train"
    assert candidate["domain_alignment"] == {
        "matched_tags": ["street"],
        "aligned": True,
        "role": "primary",
    }
    lighting = next(
        item
        for item in candidate["characteristics"]
        if item["property_id"] == "OutdoorLightingVariation"
    )
    assert lighting["confidence"] == "High"
    assert lighting["evidence_ids"] == ["evidence_acdc_dataset"]
    assert "nighttime" in lighting["notes"].lower()


def test_dataset_graphrag_context_exposes_local_lineage_without_granting_eligibility():
    state = PipelineState(
        task="detection",
        classes=["car"],
        application_domain="street scenes",
    )
    eligible = [_selection(
        "car",
        ("bdd100k_UNIT_day2night_det_train", 600),
    )]

    context = build_dataset_selection_context(state, eligible)

    assert context["eligible_dataset_ids"] == ["bdd100k_UNIT_day2night_det_train"]
    assert context["candidate_guidance"][0]["lineage"] == {
        "canonical_family": "bdd_100k",
        "derived_from": "bdd_100k_det_train",
        "synthetic": True,
        "paired_sample_ids_available": False,
    }


def test_symbolic_dataset_properties_trigger_hyperparameter_rules():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_model_info={
            "model": [{
                "model_architecture": "yolov8n",
                "architecture_family": "yolov8",
            }]
        },
        selected_data=[_selection(
            "car",
            ("bdd_100k_det_train", 700),
            ("voc12_det_train", 300),
        )],
        available_hardware={"hardware_category": "ConsumerGPU", "vram_gb": 12},
    )

    context = build_hyperparameter_context(state)
    matched_ids = {item["id"] for item in context["matched_adjustment_rules"]}

    assert "rule_yolo_outdoor_lighting_hsv_example" in matched_ids
    assert "rule_yolo_scale_variation_scale_05" in matched_ids
    assert "rule_yolo_multiscale_025" in matched_ids
    assert "rule_yolo_occlusion_cutmix_05" not in matched_ids
    adjustments = {
        field: value for rule in context["matched_adjustment_rules"]
        for field, value in rule["executable_adjustments"].items()
    }
    assert adjustments["hsv_h"] == 0.03
    assert adjustments["multi_scale"] == 0.25
    validate_detection_graph_grounded_config(
        {**context["reference_configuration"], "model_name": "yolov8n"},
        context,
    )


def test_data_check_query_supports_classification_and_detection_annotations(monkeypatch):
    captured = {}

    def fake_query(query_string):
        captured["query"] = query_string
        return []

    monkeypatch.setattr(visionkg_utils, "query", fake_query)

    visionkg_utils.get_multi_class_stats(["car"])

    assert "cv:hasAnnotation" in captured["query"]
    assert "cv:ObjectDetectionAnnotation" not in captured["query"]


def test_data_check_queries_each_class_individually(monkeypatch):
    queries = []

    def fake_query(query_string):
        queries.append(query_string)
        if '"car"' in query_string:
            return [{"datasetName": "bdd_100k_det_train", "count": "12"}]
        return [{"datasetName": "coco2017_det_train", "count": "7"}]

    monkeypatch.setattr(visionkg_utils, "query", fake_query)

    result = visionkg_utils.get_multi_class_stats(["car", "bus"])

    assert len(queries) == 2
    assert '"car"' in queries[0] and '"bus"' not in queries[0]
    assert '"bus"' in queries[1] and '"car"' not in queries[1]
    assert result == {
        "car": {"bdd_100k_det_train": 12},
        "bus": {"coco2017_det_train": 7},
    }


def test_data_check_falls_back_to_exact_official_training_queries(monkeypatch):
    queries = []

    def fake_query(query_string):
        queries.append(query_string)
        if "GROUP BY ?datasetName" in query_string:
            return [{"datasetName": "bdd_100k_det_val", "count": "3261"}]
        if '"bdd_100k_det_train"' in query_string:
            return [{"count": "22469"}]
        return [{"count": "0"}]

    monkeypatch.setattr(visionkg_utils, "query", fake_query)

    result = visionkg_utils.get_multi_class_stats(["pedestrian"], task="detection")

    assert result["pedestrian"]["bdd_100k_det_val"] == 3261
    assert result["pedestrian"]["bdd_100k_det_train"] == 22469
    assert any(
        '"bdd_100k_det_train"' in query_string
        and "GROUP BY ?datasetName" not in query_string
        for query_string in queries
    )


@pytest.mark.parametrize("download_function", [
    download_data.download_visionkg_mixed_datasets_detection,
    download_data.download_visionkg_mixed_datasets_classification,
    download_data.download_visionkg_images_flat,
])
def test_downloaders_accept_selected_data_count_contract(
    monkeypatch,
    download_function,
):
    queries = []
    monkeypatch.setattr(download_data, "query", lambda query: queries.append(query) or [])

    download_function("dataset-test", [{
        "class_name": "car",
        "sources": [{"dataset_name": "bdd_100k_det_train", "count": 7}],
    }])

    assert len(queries) == 1
    limit = int(re.search(r"LIMIT (\d+)", queries[0]).group(1))
    assert limit >= 7
