import pytest

from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.training.resource_guard import (
    MAX_IMAGE_SIDE,
    validate_image_batch_preflight,
)


def test_classification_schema_rejects_corrupted_image_size():
    field = ClassificationConfigModel.model_fields["image_size"]
    assert any(getattr(item, "le", None) == MAX_IMAGE_SIDE for item in field.metadata)


def test_detection_schema_rejects_corrupted_image_sizes():
    for name in ("input_size", "max_size"):
        field = DetectionConfigModel.model_fields[name]
        assert any(getattr(item, "le", None) == MAX_IMAGE_SIDE for item in field.metadata)


def test_preflight_rejects_the_failed_runs_raw_batch():
    with pytest.raises(ValueError, match="must be <= 4096"):
        validate_image_batch_preflight(image_size=28_463, batch_size=4)


def test_preflight_rejects_batch_larger_than_hardware_budget(monkeypatch):
    class Hardware:
        training_memory_budget_gb = 1
        profile_id = "test-profile"

    monkeypatch.setattr(
        "cvmodellearning.training.resource_guard.active_training_hardware_profile",
        lambda: Hardware(),
    )
    with pytest.raises(ValueError, match="exceeding the 1 GiB training-memory budget"):
        validate_image_batch_preflight(image_size=4096, batch_size=16)


def test_preflight_accepts_normal_training_batch():
    validate_image_batch_preflight(image_size=640, batch_size=4)
