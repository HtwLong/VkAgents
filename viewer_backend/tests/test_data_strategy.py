from viewer_backend.data_strategy import build_data_strategy


def test_strategy_prioritizes_sources_and_reports_multilabel_uncertainty():
    strategy = build_data_strategy({
        "task": "detection", "classes": ["person", "car"],
        "user_query": "Detect people and cars in street scenes at night.",
        "available_data": [
            {"class_name": "person", "sources": [{"dataset_name": "bdd_100k_det_train", "count": 800}]},
            {"class_name": "car", "sources": [{"dataset_name": "bdd_100k_det_train", "count": 900}]},
        ],
    })
    assert strategy["source_decisions"][0]["dataset_id"] == "bdd_100k_det_train"
    assert strategy["source_decisions"][0]["role"] == "primary"
    assert strategy["split_strategy"]["derive_missing_holdouts"] is True
    assert any(item["code"] == "MULTILABEL_UNIQUE_COUNT_UNVERIFIED" for item in strategy["conflicts"])


def test_strategy_warns_when_class_training_coverage_is_low():
    strategy = build_data_strategy({
        "task": "classification", "classes": ["rare"],
        "available_data": [{"class_name": "rare", "sources": [{"dataset_name": "rare_cls_train", "count": 20}]}],
    })
    assert strategy["conflicts"][0]["code"] == "INSUFFICIENT_TRAINING_AVAILABILITY"
