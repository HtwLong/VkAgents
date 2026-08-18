import pytest

from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.training.resource_guard import (
    MAX_IMAGE_SIDE,
    estimate_training_memory,
    rank_training_shape_candidates,
    validate_image_batch_preflight,
)
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile


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


def test_rtx6000_rtdetr_batch_16_is_ranked_as_a_fitting_candidate(monkeypatch):
    hardware = get_training_hardware_profile("rtx6000_48gb")
    monkeypatch.setattr(
        "cvmodellearning.training.resource_guard.active_training_hardware_profile",
        lambda: hardware,
    )
    config = {
        "model_name": "rtdetr_hgnetv2_l",
        "optimizer_name": "adamw",
        "input_size": 640,
        "batch_size": 16,
        "amp": False,
    }

    estimate = estimate_training_memory(config)
    assert estimate.assessment == "fits"
    assert estimate.upper_gb is not None
    assert estimate.upper_gb < estimate.budget_gb

    candidates = rank_training_shape_candidates(
        config,
        image_sizes=(640,),
    )
    assert candidates[0]["batch_size"] == 16
