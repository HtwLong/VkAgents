import pytest

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


def _source(dataset_name, *allocations):
    return DatasetSourceAssignment(
        dataset_name=dataset_name,
        allocations=[
            SplitAllocation(split=split, count=count, assignment_type=assignment_type)
            for split, count, assignment_type in allocations
        ],
    )


def test_legacy_selection_normalizes_to_official_training_only():
    assignments = normalize_dataset_assignments([{
        "class_name": "car",
        "sources": [{"dataset_name": "bdd_100k_det_train", "count": 80}],
    }])

    assert assignments[0].model_dump() == {
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [{
                "split": "train",
                "count": 80,
                "assignment_type": "official_split",
            }],
        }],
    }


def test_valid_assignment_preserves_official_splits_and_derives_from_train():
    plan = [ClassDataAssignment(
        class_name="car",
        sources=[
            _source(
                "bdd_100k_det_train",
                (DatasetSplit.TRAIN, 80, AssignmentType.OFFICIAL_SPLIT),
                (DatasetSplit.TEST, 10, AssignmentType.DERIVED_FROM_TRAIN),
            ),
            _source(
                "bdd_100k_det_val",
                (DatasetSplit.VALIDATION, 10, AssignmentType.OFFICIAL_SPLIT),
            ),
        ],
    )]

    assert validate_dataset_assignments(
        plan,
        {"bdd_100k_det_train": "train", "bdd_100k_det_val": "validation"},
        available_counts={
            ("car", "bdd_100k_det_train"): 90,
            ("car", "bdd_100k_det_val"): 10,
        },
    ) == plan

    planned, official, derived = summarize_dataset_assignments(plan)
    assert planned.model_dump() == {"train": 80, "validation": 10, "test": 10}
    assert official.model_dump() == {"train": 80, "validation": 10, "test": 0}
    assert derived.model_dump() == {"train": 0, "validation": 0, "test": 10}


@pytest.mark.parametrize(
    ("source_role", "split", "assignment_type"),
    [
        ("test", DatasetSplit.TRAIN, AssignmentType.OFFICIAL_SPLIT),
        ("validation", DatasetSplit.TRAIN, AssignmentType.OFFICIAL_SPLIT),
        ("validation", DatasetSplit.TEST, AssignmentType.DERIVED_FROM_TRAIN),
        ("train", DatasetSplit.TRAIN, AssignmentType.DERIVED_FROM_TRAIN),
    ],
)
def test_invalid_role_assignment_is_rejected(source_role, split, assignment_type):
    plan = [ClassDataAssignment(
        class_name="car",
        sources=[_source("source", (split, 10, assignment_type))],
    )]

    with pytest.raises(DatasetAssignmentValidationError):
        validate_dataset_assignments(
            plan,
            {"source": source_role},
            require_all_splits=False,
        )


def test_assignment_rejects_overallocation_and_missing_required_split():
    plan = [ClassDataAssignment(
        class_name="car",
        sources=[_source(
            "train_source",
            (DatasetSplit.TRAIN, 11, AssignmentType.OFFICIAL_SPLIT),
        )],
    )]

    with pytest.raises(DatasetAssignmentValidationError) as exc_info:
        validate_dataset_assignments(
            plan,
            {"train_source": "train"},
            available_counts={("car", "train_source"): 10},
        )

    reasons = {finding["reason"] for finding in exc_info.value.findings}
    assert "Allocated count exceeds availability." in reasons
    assert "Required class has no allocation in the split." in reasons


def test_required_split_coverage_is_checked_per_class():
    plan = [
        ClassDataAssignment(
            class_name="car",
            sources=[_source(
                "car_train",
                (DatasetSplit.TRAIN, 10, AssignmentType.OFFICIAL_SPLIT),
                (DatasetSplit.VALIDATION, 2, AssignmentType.DERIVED_FROM_TRAIN),
                (DatasetSplit.TEST, 2, AssignmentType.DERIVED_FROM_TRAIN),
            )],
        ),
        ClassDataAssignment(
            class_name="truck",
            sources=[_source(
                "truck_train",
                (DatasetSplit.TRAIN, 10, AssignmentType.OFFICIAL_SPLIT),
            )],
        ),
    ]

    with pytest.raises(DatasetAssignmentValidationError) as exc_info:
        validate_dataset_assignments(
            plan,
            {"car_train": "train", "truck_train": "train"},
        )

    missing = {
        (finding.get("class_name"), finding.get("split"))
        for finding in exc_info.value.findings
        if finding["reason"] == "Required class has no allocation in the split."
    }
    assert missing == {("truck", "validation"), ("truck", "test")}
