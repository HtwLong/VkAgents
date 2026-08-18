import pytest

from cvmodellearning.models.registry import (
    DETECTION_HPO_MODEL_IDS,
    DetectionModelFamily,
    DetectionModelId,
    enabled_models,
    model_ids,
    resolve_detection_model_identity,
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


@pytest.mark.parametrize(
    ("reference", "executable_id", "runtime_family"),
    [
        ("yolov10n", "yolov10_n", "yolo"),
        ("yolov10_n", "yolov10_n", "yolo"),
        ("YOLOv10n", "yolov10_n", "yolo"),
        ("YOLOv10", "yolov10_n", "yolo"),
        ("yolo10", "yolov10_n", "yolo"),
        ("RetinaNet R50 FPN", "retinanet_r50", "torchvision"),
        ("rtdetr-l", "rtdetr_hgnetv2_l", "rtdetr"),
    ],
)
def test_detection_identity_resolves_planning_and_execution_references(
    reference,
    executable_id,
    runtime_family,
) -> None:
    identity = resolve_detection_model_identity(reference)

    assert identity is not None
    assert identity.executable_id == executable_id
    assert identity.runtime_family == runtime_family
