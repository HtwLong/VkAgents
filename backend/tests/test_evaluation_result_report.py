import json

from fastapi.testclient import TestClient

from api import app
from cvmodellearning import paths
from cvmodellearning.evaluation.result_report import (
    normalize_report,
    save_classification_report,
    save_detection_report,
)
from cvmodellearning.evaluation.detection_result import dataset_statistics


def test_classification_report_has_frontend_ready_sections(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    report_path = save_classification_report(
        "job",
        {"model_name": "resnet50", "classes": ["cat", "dog"], "batch_size": 8},
        {
            "accuracy": 0.75,
            "macro_f1": 0.7,
            "precision_per_class": [1.0, 0.5],
            "recall_per_class": [0.5, 1.0],
            "f1_per_class": [0.67, 0.67],
            "support_per_class": [2, 2],
            "confusion_matrix": [[1, 1], [0, 2]],
        },
    )

    report = json.loads(report_path.read_text())
    assert report["task"] == "classification"
    assert report["model"]["name"] == "resnet50"
    assert report["metrics"]["accuracy"] == 0.75
    assert report["per_class"][1]["class_name"] == "dog"
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]


def test_detection_report_normalizes_backend_metric_names(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    report_path = save_detection_report(
        "job", {"model_name": "fasterrcnn_resnet50_fpn", "classes": ["car"]},
        {"coco/bbox_mAP": 0.4, "coco/bbox_mAP_50": 0.65,
         "per_class": [{"class_name": "car", "ap": 0.4, "support": 12}],
         "size_metrics": {"ap_small": 0.2}},
    )

    report = json.loads(report_path.read_text())
    assert report["task"] == "detection"
    assert report["metrics"] == {"map": 0.4, "map50": 0.65}
    assert report["schema_version"] == 4
    assert report["per_class"][0]["support"] == 12
    assert report["size_metrics"]["ap_small"] == 0.2


def test_legacy_report_is_normalized_without_rewrite():
    legacy = {"job_id": "old", "classes": ["car"], "metrics": {"map": 0.2}}
    normalized = normalize_report(legacy)
    assert normalized["schema_version"] == 1
    assert normalized["per_class"] == []
    assert normalized["curves"] == {}
    assert legacy.get("schema_version") is None


def test_detection_statistics_fall_back_to_category_order_for_legacy_head_names(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    annotations = paths.test_json_path("job")
    annotations.write_text(json.dumps({
        "images": [{"id": 1}, {"id": 2}],
        "categories": [{"id": 7, "name": "car"}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10]},
            {"image_id": 2, "category_id": 7, "bbox": [0, 0, 20, 20]},
        ],
    }))

    statistics = dataset_statistics("job", ["item"])

    assert statistics["per_class"] == [{"class_name": "item", "instances": 2, "images": 2}]


def test_evaluation_report_endpoint_returns_saved_report(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    saved_path = save_detection_report(
        "endpoint-job", {"model_name": "yolov8_n", "classes": ["car"]},
        {"mAP@.50:.95": 0.42},
    )
    response = TestClient(app).get("/api/v1/evaluate/endpoint-job/report")

    assert response.status_code == 200
    expected = json.loads(saved_path.read_text())
    assert response.json() == expected
