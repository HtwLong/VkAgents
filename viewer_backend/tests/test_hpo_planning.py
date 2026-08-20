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
         "selected_data": ASSIGNMENTS, "selected_model_info": {"id": "Qwen3-VL-2B-Instruct", "family": "Qwen-VL"}},
        {"model_name": "Qwen3-VL-2B-Instruct", "epochs": 2, "batch_size": 2,
         "learning_rate": 0.00002, "optimizer": "adamw", "rationale": "Use LoRA."},
        None,
    )
    assert len(config) == 28
    assert config["use_lora"] is True
    assert {"torch", "torchvision", "ultralytics"}.isdisjoint(sys.modules)


def test_materializes_original_detection_contract_and_canonicalizes_model_id():
    config, _ = materialize_hpo(
        {"task": "detection", "classes": ["cat"], "selected_data": ASSIGNMENTS,
         "selected_model_info": {"id": "yolov11", "family": "YOLO"}},
        {"model_name": "yolov11", "epochs": 50, "batch_size": -1,
         "learning_rate": 0.01, "optimizer": "auto", "image_size": 640,
         "rationale": "Use the small YOLO variant."},
        None,
    )
    assert len(config) == 70
    assert config["model_name"] == "yolov11_n"
    assert config["batch_size"] == -1
    assert config["scheduler_name"] == "linear"
    assert config["loss_box"] == "ciou"
    assert config["aspect_ratio_range"] is None


def test_materializes_ssd_with_original_backend_specific_sentinels():
    config, _ = materialize_hpo(
        {"task": "detection", "classes": ["cat"], "selected_data": ASSIGNMENTS,
         "selected_model_info": {"id": "ssd300_coco", "family": "SSD"}},
        {"model_name": "ssd300_coco", "epochs": 30, "batch_size": 4,
         "learning_rate": 0.001, "optimizer": "sgd", "image_size": 300,
         "rationale": "Use SSD300."},
        None,
    )
    assert config["model_name"] == "ssd300"
    assert config["model_weights"] == "imagenet_backbone"
    assert config["input_size"] == config["max_size"] == 300
    assert config["augmentation_policy"] == "ssd"
    assert config["mosaic"] == 0
