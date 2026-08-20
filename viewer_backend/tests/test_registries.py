import sys

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
