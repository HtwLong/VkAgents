from viewer_backend.hpo_evaluation import decision_from_findings, evaluate_hpo, repair_hpo


def test_evaluator_repairs_recipe_and_hardware_bounds():
    config = {
        "model_name": "resnet50", "num_epochs": 10, "patience": 10,
        "batch_size": 64, "learning_rate": 2.0, "image_size": 224,
        "criterion_name": "cross_entropy",
        "selected_data": [{"class_name": "cat", "sources": [{"allocations": [
            {"split": "train"}, {"split": "validation"}, {"split": "test"},
        ]}]}],
    }
    context = {
        "task": "classification", "classes": ["cat"],
        "selected_model_info": {"id": "resnet50"},
        "training_hardware": {"max_batch_size": 8},
    }
    recipe = {"id": "recipe", "learning_rate_min": "0.0001", "learning_rate_max": "0.01"}
    findings = evaluate_hpo(config, context, recipe)
    repaired, fields = repair_hpo(config, findings)
    remaining = evaluate_hpo(repaired, context, recipe)
    assert {"learning_rate", "batch_size", "patience"} <= set(fields)
    assert not [item for item in remaining if item["severity"] == "hard_error"]
    assert decision_from_findings(remaining, fields)["accept"] is True


def test_evaluator_does_not_repair_missing_dataset_splits():
    config = {"model_name": "resnet50", "selected_data": []}
    context = {"task": "classification", "classes": ["cat"], "selected_model_info": {"id": "resnet50"}}
    findings = evaluate_hpo(config, context, None)
    repaired, fields = repair_hpo(config, findings)
    assert repaired["criterion_name"] == "cross_entropy"
    assert fields == ["criterion_name"]
    remaining = evaluate_hpo(repaired, context, None)
    assert decision_from_findings(remaining, fields)["accept"] is False
