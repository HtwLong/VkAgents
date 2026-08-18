import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

import routers.execution as execution
from cvmodellearning.download.assignment_manifest import DatasetContentConflict
from cvmodellearning.pipelines.detection_pipe import DetectionPipeline


class _FailingDetectionPipeline:
    def load_model_step(self, _job_id):
        return {"status": "load_failed", "error": "broken checkpoint"}

    def infer_step(self, _job_id, _image):
        return {"status": "inference_failed", "error": "invalid prediction"}


class _CapturingDetectionPipeline:
    def __init__(self):
        self.image = None

    def infer_step(self, _job_id, image):
        self.image = image
        return {
            "status": "success",
            "image_width": image.width,
            "image_height": image.height,
            "detections_count": 0,
            "detections": [],
        }


def test_load_model_endpoint_does_not_wrap_failure_as_success(monkeypatch):
    monkeypatch.setattr(execution, "get_pipeline_by_task", lambda _job_id: _FailingDetectionPipeline())

    with pytest.raises(HTTPException) as error:
        asyncio.run(execution.load_model(execution.LoadModelRequest(job_id="job")))

    assert error.value.status_code == 500
    assert error.value.detail == "broken checkpoint"


def test_inference_endpoint_does_not_wrap_failure_as_success(monkeypatch):
    monkeypatch.setattr(execution, "get_pipeline_by_task", lambda _job_id: _FailingDetectionPipeline())
    monkeypatch.setattr(execution.MODEL_CACHE_MANAGER, "get_model_bundle", lambda _job_id: {"model": object()})
    image = Image.new("RGB", (4, 4))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    with pytest.raises(HTTPException) as error:
        asyncio.run(execution.infer_image("job", UploadFile(file=buffer, filename="image.png")))

    assert error.value.status_code == 500
    assert error.value.detail == "invalid prediction"


def test_inference_normalizes_exif_orientation_before_prediction(monkeypatch):
    pipeline = _CapturingDetectionPipeline()
    monkeypatch.setattr(execution, "get_pipeline_by_task", lambda _job_id: pipeline)
    monkeypatch.setattr(execution.MODEL_CACHE_MANAGER, "get_model_bundle", lambda _job_id: {"model": object()})

    # Orientation 6 means the stored 3x2 pixels must be displayed rotated 90
    # degrees clockwise, producing a normalized 2x3 image.
    image = Image.new("RGB", (3, 2))
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)

    response = asyncio.run(
        execution.infer_image("job", UploadFile(file=buffer, filename="rotated.jpg"))
    )

    assert pipeline.image is not None
    assert pipeline.image.size == (2, 3)
    assert pipeline.image.getexif().get(274) is None
    assert response["result"]["image_width"] == 2
    assert response["result"]["image_height"] == 3


def test_prepare_endpoint_returns_actionable_content_conflict(monkeypatch, tmp_path):
    pipeline = DetectionPipeline()
    conflict = DatasetContentConflict([{
        "first_path": "objects365/train-copy.jpg",
        "first_split": "train",
        "duplicate_path": "objects365/test-copy.jpg",
        "duplicate_split": "test",
    }])

    monkeypatch.setattr(execution, "get_pipeline_by_task", lambda _job_id: pipeline)
    monkeypatch.setattr(execution, "_validate_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(execution, "clear_cancellation", lambda _job_id: None)
    monkeypatch.setattr(execution, "write_run_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(execution, "raise_if_cancelled", lambda _job_id: None)
    monkeypatch.setattr(pipeline, "prepare_data_step", lambda _cfg, _job_id: (_ for _ in ()).throw(conflict))
    manifest = tmp_path / "manifest.json"
    annotations = tmp_path / "annotations.json"
    manifest.write_text("{}")
    annotations.write_text("{}")
    monkeypatch.setattr(execution, "dataset_manifest_path", lambda _job_id: manifest)
    monkeypatch.setattr(execution, "json_labels_path", lambda _job_id: annotations)

    with pytest.raises(HTTPException) as error:
        asyncio.run(execution.step_prepare(execution.StepRequest(job_id="job", chosen_parameters={})))

    assert error.value.status_code == 409
    assert error.value.detail["code"] == "dataset_content_conflict"
    assert error.value.detail["conflicts"] == conflict.conflicts
    assert "download step again" in error.value.detail["recommended_action"]
