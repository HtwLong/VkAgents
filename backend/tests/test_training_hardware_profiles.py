import pytest

from cvmodellearning.graphrag.hyperparameter_context import build_hyperparameter_context
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.hardware_profiles import (
    DEFAULT_TRAINING_HARDWARE_PROFILE,
    TRAINING_HARDWARE_PROFILE_ENV,
    active_training_hardware_profile,
    get_training_hardware_profile,
)
from routers.planning import get_state


def _yolo_state(**updates) -> PipelineState:
    values = {
        "task": "detection",
        "classes": ["car"],
        "selected_model_info": {"model": {"model_architecture": "yolov8"}},
        "selected_data": [
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    }
    values.update(updates)
    return PipelineState.model_validate(values)


def test_rtx2060_consumer_training_profile_is_the_default(monkeypatch):
    monkeypatch.delenv(TRAINING_HARDWARE_PROFILE_ENV, raising=False)

    assert DEFAULT_TRAINING_HARDWARE_PROFILE == "rtx2060_6gb_ryzen5600x_16gb"
    assert active_training_hardware_profile().profile_id == DEFAULT_TRAINING_HARDWARE_PROFILE


def test_new_state_uses_active_training_profile_without_replacing_inference_hardware(monkeypatch):
    monkeypatch.setenv(TRAINING_HARDWARE_PROFILE_ENV, "macbook_air_m4_16gb")

    state = get_state({
        "available_hardware": {
            "hardware_category": "EdgeDevice",
            "gpu_type": "Deployment accelerator",
            "vram_gb": 2,
        }
    })

    assert state.available_hardware.gpu_type == "Deployment accelerator"
    assert state.training_hardware.profile_id == "macbook_air_m4_16gb"
    assert state.training_hardware.accelerator == "mps"


def test_saved_training_profile_is_stable_when_server_default_changes(monkeypatch):
    saved = get_training_hardware_profile("macbook_air_m4_16gb")
    monkeypatch.setenv(TRAINING_HARDWARE_PROFILE_ENV, "rtx6000_48gb")

    state = get_state({"training_hardware": saved.model_dump(mode="json")})

    assert active_training_hardware_profile().profile_id == "rtx6000_48gb"
    assert state.training_hardware.profile_id == "macbook_air_m4_16gb"


def obsolete_hyperparameters_use_training_hardware_not_inference_hardware():
    inference_only = build_hyperparameter_context(_yolo_state(
        available_hardware={"hardware_category": "ConsumerGPU", "vram_gb": 8},
    ))
    assert "rule_yolo_low_vram_batch" not in {
        rule["id"] for rule in inference_only["matched_adjustment_rules"]
    }

    m4 = build_hyperparameter_context(_yolo_state(
        available_hardware={"hardware_category": "DataCenterGPU", "vram_gb": 48},
        training_hardware=get_training_hardware_profile("macbook_air_m4_16gb"),
    ))
    assert m4["reference_configuration"]["batch_size"] == 4
    assert m4["reference_configuration"]["workers"] == 4
    assert m4["reference_configuration"]["amp"] is False


def test_unknown_training_profile_has_clear_error():
    with pytest.raises(ValueError, match="Unknown training hardware profile"):
        get_training_hardware_profile("missing")


def test_rtx2060_consumer_training_profile_is_registered():
    profile = get_training_hardware_profile("rtx2060_6gb_ryzen5600x_16gb")

    assert profile.accelerator == "cuda"
    assert profile.hardware_category == "ConsumerGPU"
    assert profile.gpu_count == 1
    assert profile.vram_gb == 6
    assert profile.ram_gb == 16
    assert profile.training_memory_budget_gb == 5
    assert profile.max_batch_size == 4
    assert profile.workers == 4
    assert profile.supports_amp is True

