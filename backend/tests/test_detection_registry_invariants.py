import pytest

from cvmodellearning.models.registry import (
    DETECTION_HPO_MODEL_IDS,
    DetectionModelFamily,
    DetectionModelId,
    enabled_models,
    model_ids,
)
from cvmodellearning.pipelines.detection_pipe import _get_trainer_type
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel


def test_base_yolo_models_have_executable_default_hpo_variants() -> None:
    yolo_models = [
        model for model in enabled_models("detection") if model.family == "yolo"
    ]

    assert yolo_models
    assert all(model.hpo_id in DETECTION_HPO_MODEL_IDS for model in yolo_models)


def test_mask_rcnn_is_not_registered_or_routable() -> None:
    assert "mask-rcnn_r50_fpn_1x_coco" not in model_ids("detection")
    assert "mask_rcnn_r50" not in DETECTION_HPO_MODEL_IDS
    assert "mask-rcnn" not in {family.value for family in DetectionModelFamily}
    assert "mask-rcnn_r50_fpn_1x_coco" not in {model.value for model in DetectionModelId}

    with pytest.raises(ValueError, match="Unknown model architecture"):
        _get_trainer_type("mask_rcnn_r50")


def test_detection_hpo_schema_rejects_segmentation() -> None:
    task_schema = DetectionConfigModel.model_json_schema()["properties"]["task_type"]

    assert task_schema["const"] == "detection"
