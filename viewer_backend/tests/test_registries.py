import sys

import pytest

from viewer_backend.registries import get_registry


def test_metadata_registry_resolves_models_without_execution_imports():
    registry = get_registry()
    model = registry.resolve_model("YOLO11n", "detection")
    assert model is not None
    assert model.id == "yolo11n"
    assert model.fine_tuning_supported
    assert registry.recipes_for("detection", model.family)
    assert {"torch", "torchvision", "ultralytics"}.isdisjoint(sys.modules)


def test_metadata_registry_rejects_cross_task_model():
    assert get_registry().resolve_model("yolo11n", "classification") is None


def test_metadata_registry_rejects_ontology_only_detection_model():
    registry = get_registry()
    assert registry.resolve_model("rtdetr_r18", "detection") is None
    assert registry.resolve_model("rtdetr_hgnetv2_l", "detection") is not None


@pytest.mark.parametrize(("reference", "ontology_id"), [
    ("retinanet_resnet50_fpn", "retinanet_resnet50_fpn"),
    ("retinanet_r50_fpn_1x_coco", "retinanet_resnet50_fpn"),
    ("retinanet_r50", "retinanet_resnet50_fpn"),
    ("fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn"),
    ("faster-rcnn_r50_fpn_1x_coco", "fasterrcnn_resnet50_fpn"),
    ("faster_rcnn_r50", "fasterrcnn_resnet50_fpn"),
    ("ssd300_vgg16", "ssd300_vgg16"),
    ("ssd300_coco", "ssd300_vgg16"),
    ("ssd300", "ssd300_vgg16"),
])
def test_detection_registry_resolves_ontology_registry_and_hpo_ids(reference, ontology_id):
    resolved = get_registry().resolve_model(reference, "detection")
    assert resolved is not None
    assert resolved.id == ontology_id
