import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from viewer_backend.data_strategy_contracts import (
    ClassCoverageObjective,
    CompiledDataPlan,
    DataPlanConflict,
    DataPlanReview,
    DataPlanReviewFinding,
    DataSplitStrategy,
    DataStrategy,
    DatasetClassUseOverride,
    DatasetSourceDecision,
)


def test_data_strategy_contract_fields_and_requiredness_match_original():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "backend" / "src"))
    from cvmodellearning.schemas import data_strategy as original  # noqa: PLC0415

    pairs = (
        (ClassCoverageObjective, original.ClassCoverageObjective),
        (DatasetClassUseOverride, original.DatasetClassUseOverride),
        (DatasetSourceDecision, original.DatasetSourceDecision),
        (DataSplitStrategy, original.DataSplitStrategy),
        (DataStrategy, original.DataStrategy),
        (DataPlanConflict, original.DataPlanConflict),
        (CompiledDataPlan, original.CompiledDataPlan),
        (DataPlanReviewFinding, original.DataPlanReviewFinding),
        (DataPlanReview, original.DataPlanReview),
    )
    for viewer, backend in pairs:
        assert viewer.model_fields.keys() == backend.model_fields.keys()
        assert {
            name: field.is_required() for name, field in viewer.model_fields.items()
        } == {
            name: field.is_required() for name, field in backend.model_fields.items()
        }


def test_data_strategy_preserves_original_cross_field_validation():
    with pytest.raises(ValidationError, match="minimum <= preferred <= maximum"):
        ClassCoverageObjective(
            class_name="person",
            minimum_positive_images=500,
            preferred_positive_images=200,
            maximum_positive_images=1000,
            rationale="invalid order",
        )
    with pytest.raises(ValidationError, match="preferred_unique_pool_images"):
        DataStrategy.model_validate({
            "coverage_objectives": [],
            "source_decisions": [],
            "split_strategy": {
                "primary_evaluation_domain": "street",
                "use_official_validation": True,
                "use_official_test": True,
                "derive_missing_holdouts": True,
            },
            "minimum_unique_pool_images": 1000,
            "preferred_unique_pool_images": 500,
            "rationale": "invalid order",
        })


def test_external_evaluation_sources_cannot_derive_holdouts():
    with pytest.raises(ValidationError, match="official split"):
        DatasetSourceDecision(
            dataset_id="benchmark_test",
            role="external_evaluation",
            priority=1,
            activation="required",
            allow_derived_test=True,
            rationale="invalid external evaluation policy",
        )
