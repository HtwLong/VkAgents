import pytest

from cvmodellearning.training.resource_guard import (
    estimate_training_memory,
    validate_training_resource_config,
)
from routers import execution


def test_training_resource_guard_rejects_batch_above_hardware_limit():
    with pytest.raises(ValueError, match="exceeds the configured training-hardware maximum"):
        validate_training_resource_config({"batch_size": 999, "image_size": 224})


def test_analytical_estimator_reports_components_and_range():
    estimate = estimate_training_memory({
        "model_name": "resnet50",
        "optimizer_name": "adamw",
        "batch_size": 2,
        "image_size": 224,
        "precision": "mixed",
    })

    assert estimate.supported is True
    assert estimate.assessment in {"fits", "borderline", "exceeds"}
    assert estimate.lower_gb < estimate.upper_gb
    assert estimate.components_gb["activations"] > 0
    assert estimate.components_gb["parameters_gradients_optimizer"] > 0


def test_analytical_estimator_scales_with_batch_and_resolution():
    small = estimate_training_memory({
        "model_name": "yolov10_s", "batch_size": 1, "input_size": 640, "amp": True,
    })
    large = estimate_training_memory({
        "model_name": "yolov10_s", "batch_size": 4, "input_size": 960, "amp": True,
    })

    assert large.lower_gb > small.lower_gb
    assert large.components_gb["activations"] > small.components_gb["activations"]


def test_unknown_model_memory_is_explicitly_unverified():
    estimate = estimate_training_memory({
        "model_name": "unknown", "batch_size": 1, "image_size": 224,
    })

    assert estimate.supported is False
    assert estimate.assessment == "unverified"


def test_resource_guard_rejects_analytically_infeasible_configuration():
    with pytest.raises(ValueError, match="Estimated training-memory range"):
        validate_training_resource_config({
            "model_name": "rtdetr_hgnetv2_l",
            "optimizer_name": "adamw",
            "batch_size": 4,
            "input_size": 1280,
            "amp": True,
        })


def test_execution_readiness_report_requires_valid_materialized_splits(monkeypatch, tmp_path):
    class FakePipeline:
        def _require_prepared_data(self, config, job_id):
            return {
                "counts": {"train": 10, "validation": 2, "test": 2},
                "content_isolation": {"cross_split_duplicates": 0},
                "assignment_fingerprint": "abc",
                "manifest_sha256": "def",
            }

    monkeypatch.setattr(execution, "run_dir", lambda _job_id: tmp_path)
    report = execution._build_execution_readiness_report(
        FakePipeline(),
        {"batch_size": 1, "image_size": 224},
        "job",
    )

    assert report["ready"] is True
    assert report["checks"]["prepared_dataset_matches_plan"] is True
    assert (tmp_path / "execution_readiness.json").is_file()
