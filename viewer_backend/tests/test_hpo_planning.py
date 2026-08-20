import sys

from viewer_backend.hpo_planning import materialize_hpo


ASSIGNMENTS = [{
    "class_name": "cat",
    "sources": [{
        "dataset_name": "example_train",
        "allocations": [
            {"split": "train", "count": 80, "assignment_type": "official_split"},
            {"split": "validation", "count": 10, "assignment_type": "derived_from_train"},
            {"split": "test", "count": 10, "assignment_type": "derived_from_train"},
        ],
    }],
}]


def test_materializes_complete_classification_proposal():
    config, provenance = materialize_hpo(
        {"task": "classification", "classes": ["cat"], "selected_data": ASSIGNMENTS,
         "selected_model_info": {"id": "resnet50", "family": "ResNet"}},
        {"model_name": "resnet50", "epochs": 20, "batch_size": 8,
         "learning_rate": 0.001, "optimizer": "adamw", "image_size": 224,
         "rationale": "Fine tune pretrained weights."},
        {"id": "resnet-recipe", "weight_decay_default": "0.0001"},
    )
    assert len(config) == 54
    assert config["criterion_name"] == "cross_entropy"
    assert config["selected_data"] == ASSIGNMENTS
    assert provenance["training_recipe_id"]["source"] == "ontology_recipe"


def test_materializes_complete_vqa_proposal_without_ml_imports():
    config, _ = materialize_hpo(
        {"task": "visual question answering", "classes": ["cat"],
         "selected_data": ASSIGNMENTS, "selected_model_info": {"id": "qwen", "family": "Qwen-VL"}},
        {"model_name": "qwen", "epochs": 2, "batch_size": 2,
         "learning_rate": 0.00002, "optimizer": "adamw", "rationale": "Use LoRA."},
        None,
    )
    assert len(config) == 28
    assert config["use_lora"] is True
    assert {"torch", "torchvision", "ultralytics"}.isdisjoint(sys.modules)
