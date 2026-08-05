import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

import routers.execution as execution


class _FailingDetectionPipeline:
    def load_model_step(self, _job_id):
        return {"status": "load_failed", "error": "broken checkpoint"}

    def infer_step(self, _job_id, _image):
        return {"status": "inference_failed", "error": "invalid prediction"}


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
