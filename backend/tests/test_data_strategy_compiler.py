from cvmodellearning.datasets.selection import build_dataset_assignments
from cvmodellearning.datasets.strategy_compiler import (
    build_strategy_split_permissions,
    compile_data_strategy,
)
from cvmodellearning.schemas.data_strategy import DataStrategy, DatasetClassUseOverride
from cvmodellearning.schemas.interpretation_schema import ClassDataSelection


def _eligible(class_name: str, **sources: int) -> ClassDataSelection:
    return ClassDataSelection(
        class_name=class_name,
        sources=[
            {"dataset_name": dataset_id, "count": count}
            for dataset_id, count in sources.items()
        ],
    )


def _strategy(*, supplement_activation="if_needed_for_preferred") -> DataStrategy:
    return DataStrategy.model_validate({
        "coverage_objectives": [{
            "class_name": "motorcycle",
            "minimum_positive_images": 2_000,
            "preferred_positive_images": 2_500,
            "maximum_positive_images": 3_000,
            "priority": "high",
            "rationale": "Enough positive images for adaptation.",
        }],
        "source_decisions": [{
            "dataset_id": "bdd_100k_det_train",
            "role": "primary",
            "priority": 1,
            "focus_classes": [],
            "activation": "required",
            "minimum_images_when_used": 0,
            "allow_derived_validation": True,
            "allow_derived_test": True,
            "rationale": "Target-domain source.",
        }, {
            "dataset_id": "coco2017_det_train",
            "role": "training_supplement",
            "priority": 2,
            "focus_classes": [],
            "activation": supplement_activation,
            "minimum_images_when_used": 500,
            "allow_derived_validation": False,
            "allow_derived_test": False,
            "rationale": "Optional coverage supplement.",
        }],
        "split_strategy": {
            "primary_evaluation_domain": "urban traffic",
            "use_official_validation": True,
            "use_official_test": True,
            "derive_missing_holdouts": True,
            "group_isolation_keys": [],
            "preserve_natural_evaluation_frequency": True,
            "training_balance_policy": "class_aware_sampling",
        },
        "minimum_unique_pool_images": 2_000,
        "preferred_unique_pool_images": 2_500,
        "preferred_target_is_strict": False,
        "acceptable_compromises": ["Accept less than preferred coverage."],
        "rationale": "Use target-domain data as the base.",
    })


def test_compiler_does_not_create_a_tiny_supplement_to_fill_soft_target():
    plan = compile_data_strategy(
        _strategy(),
        [_eligible("motorcycle", bdd_100k_det_train=2_327, coco2017_det_train=4_000)],
    )

    assert plan.conflicts == []
    assert plan.selected_data[0].model_dump() == {
        "class_name": "motorcycle",
        "sources": [{"dataset_name": "bdd_100k_det_train", "count": 2_327}],
    }
    assert plan.warnings[0]["code"] == "CLASS_PREFERRED_COVERAGE_UNMET"


def test_compiler_uses_meaningful_supplement_when_minimum_needs_it():
    plan = compile_data_strategy(
        _strategy(supplement_activation="if_needed_for_minimum"),
        [_eligible("motorcycle", bdd_100k_det_train=1_700, coco2017_det_train=4_000)],
    )

    assert plan.conflicts == []
    counts = {
        source.dataset_name: source.count
        for source in plan.selected_data[0].sources
    }
    assert counts == {"bdd_100k_det_train": 1_700, "coco2017_det_train": 800}


def test_compiler_ignores_unsupported_focus_and_uses_inventory_eligibility():
    strategy = _strategy()
    strategy.source_decisions[0].focus_classes = ["motorcycle", "person"]

    plan = compile_data_strategy(
        strategy,
        [_eligible("motorcycle", bdd_100k_det_train=1_500, coco2017_det_train=4_000)],
    )

    assert plan.conflicts == []
    assert any(
        warning["code"] == "UNSUPPORTED_SOURCE_FOCUS_IGNORED"
        for warning in plan.warnings
    )


def test_compiler_expands_approved_sources_to_meet_mandatory_minimum():
    strategy = _strategy(supplement_activation="if_needed_for_minimum")
    strategy.coverage_objectives[0].minimum_positive_images = 1_250
    strategy.coverage_objectives[0].preferred_positive_images = 1_500
    strategy.minimum_unique_pool_images = 1_250
    strategy.preferred_unique_pool_images = 1_500

    plan = compile_data_strategy(
        strategy,
        [_eligible("motorcycle", bdd_100k_det_train=1_200, coco2017_det_train=4_000)],
    )

    assert plan.conflicts == []
    assert sum(source.count for source in plan.selected_data[0].sources) >= 1_250


def test_dataset_can_be_supplement_by_default_but_primary_for_one_class():
    strategy = DataStrategy.model_validate({
        "coverage_objectives": [
            {
                "class_name": name,
                "minimum_positive_images": 1_000,
                "preferred_positive_images": 1_200,
                "maximum_positive_images": 1_500,
                "priority": "high",
                "rationale": "Required class coverage.",
            }
            for name in ("car", "bus", "person")
        ],
        "source_decisions": [{
            "dataset_id": "bdd_100k_det_train",
            "role": "primary",
            "priority": 1,
            "focus_classes": [],
            "activation": "required",
            "minimum_images_when_used": 0,
            "allow_derived_validation": True,
            "allow_derived_test": True,
            "class_use_overrides": [],
            "rationale": "Target-domain primary source.",
        }, {
            "dataset_id": "coco2017_det_train",
            "role": "training_supplement",
            "priority": 2,
            "focus_classes": [],
            "activation": "if_needed_for_minimum",
            "minimum_images_when_used": 200,
            "allow_derived_validation": False,
            "allow_derived_test": False,
            "class_use_overrides": [{
                "class_name": "person",
                "role": "primary",
                "activation": "required",
                "minimum_images_when_used": 0,
                "allow_derived_validation": True,
                "allow_derived_test": True,
                "rationale": "Fallback primary because BDD has no person inventory.",
            }],
            "rationale": "General training supplement by default.",
        }],
        "split_strategy": {
            "primary_evaluation_domain": "urban traffic with documented fallback",
            "use_official_validation": False,
            "use_official_test": False,
            "derive_missing_holdouts": True,
            "group_isolation_keys": [],
            "preserve_natural_evaluation_frequency": True,
            "training_balance_policy": "class_aware_sampling",
        },
        "minimum_unique_pool_images": 1_000,
        "preferred_unique_pool_images": 1_200,
        "preferred_target_is_strict": False,
        "acceptable_compromises": ["Person evaluation uses a fallback domain."],
        "rationale": "Prefer BDD and use COCO where target-domain coverage is absent.",
    })

    plan = compile_data_strategy(strategy, [
        _eligible("car", bdd_100k_det_train=2_000, coco2017_det_train=2_000),
        _eligible("bus", bdd_100k_det_train=2_000, coco2017_det_train=2_000),
        _eligible("person", coco2017_det_train=2_000),
    ])

    assert plan.conflicts == []
    assert plan.source_roles["coco2017_det_train"] == {
        "bus": "training_supplement",
        "car": "training_supplement",
        "person": "primary",
    }
    assert strategy.source_decisions[1].use_for_class("car")["allow_derived_test"] is False
    assert strategy.source_decisions[1].use_for_class("person")["allow_derived_test"] is True

    training_only, allowed_splits = build_strategy_split_permissions(
        strategy, ["car", "bus", "person"]
    )
    assignments = build_dataset_assignments(
        plan.selected_data,
        plan.selected_data,
        training_only_dataset_ids_by_class=training_only,
        allowed_splits_by_class_dataset=allowed_splits,
        use_official_validation=False,
        use_official_test=False,
        derive_missing_holdouts=True,
    )
    sources_by_class = {
        item.class_name: {source.dataset_name: source for source in item.sources}
        for item in assignments
    }
    assert {
        str(item.split)
        for item in sources_by_class["person"]["coco2017_det_train"].allocations
    } == {"train", "validation", "test"}


def test_class_override_cannot_invent_dataset_eligibility():
    strategy = _strategy()
    strategy.source_decisions[1].class_use_overrides = [DatasetClassUseOverride.model_validate({
        "class_name": "person",
        "role": "primary",
        "allow_derived_validation": True,
        "allow_derived_test": True,
        "rationale": "Invalid attempted override.",
    })]

    plan = compile_data_strategy(
        strategy,
        [_eligible("motorcycle", bdd_100k_det_train=2_327, coco2017_det_train=4_000)],
    )

    assert plan.conflicts == []
    assert any(
        warning["code"] == "UNSUPPORTED_CLASS_USE_OVERRIDE_IGNORED"
        for warning in plan.warnings
    )


def test_compiler_refuses_to_pretend_group_isolation_metadata_exists():
    strategy = _strategy()
    strategy.split_strategy.group_isolation_keys = ["video_id"]

    plan = compile_data_strategy(
        strategy,
        [_eligible(
            "motorcycle",
            bdd_100k_det_train=2_327,
            coco2017_det_train=4_000,
        )],
        supported_group_isolation_keys=set(),
    )

    assert any(
        conflict.code == "UNSUPPORTED_GROUP_ISOLATION_KEY"
        for conflict in plan.conflicts
    )


def test_compiler_rejects_impossible_unique_image_objective():
    strategy = _strategy()
    strategy.minimum_unique_pool_images = 1_999

    plan = compile_data_strategy(
        strategy,
        [_eligible(
            "motorcycle",
            bdd_100k_det_train=2_327,
            coco2017_det_train=4_000,
        )],
    )

    assert any(
        conflict.code == "UNIQUE_IMAGE_MINIMUM_BELOW_CLASS_MINIMUM"
        for conflict in plan.conflicts
    )
