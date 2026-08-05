"""Build the small, task-neutral evaluation payload rendered by the frontend."""

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from cvmodellearning.paths import data_provenance_path, evaluation_report_path, metrics_csv_path


CONFIG_KEYS = (
    "model_weights", "training_mode", "image_size", "input_size", "batch_size",
    "num_epochs", "optimizer_name", "learning_rate", "weight_decay", "track_metric",
    "confidence_threshold", "nms_iou_threshold", "precision",
)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if isinstance(value, int) else float(value)


def _dataset(job_id: str) -> dict[str, Any]:
    path = data_provenance_path(job_id)
    if not path.exists():
        return {"splits": {}, "assignment_fingerprint": None}
    audit = json.loads(path.read_text(encoding="utf-8"))
    official = audit.get("official_counts", {})
    derived = audit.get("derived_counts", {})
    return {
        "splits": {
            split: int(official.get(split, 0)) + int(derived.get(split, 0))
            for split in ("train", "validation", "test")
        },
        "assignment_fingerprint": audit.get("assignment_fingerprint"),
    }


def _history(job_id: str) -> list[dict[str, float | int]]:
    path = metrics_csv_path(job_id)
    if not path.exists():
        return []
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clean: dict[str, float | int] = {}
            for key, value in row.items():
                try:
                    clean[key] = int(value) if key == "epoch" else float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(clean)
    return rows


def _base(job_id: str, task: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "task": task,
        "model": {
            "name": config.get("model_name", "unknown"),
            "weights": config.get("model_weights"),
            "training_mode": config.get("training_mode"),
        },
        "classes": list(config.get("classes") or []),
        "dataset": _dataset(job_id),
        "configuration": {key: config[key] for key in CONFIG_KEYS if key in config},
        "training_history": _history(job_id),
    }


def save_classification_report(
    job_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    report = _base(job_id, "classification", config)
    classes = report["classes"]
    def class_value(key: str, index: int) -> Any:
        values = metrics.get(key) or []
        return values[index] if index < len(values) else None

    per_class = []
    for index, name in enumerate(classes):
        per_class.append({
            "class_name": name,
            "precision": class_value("precision_per_class", index),
            "recall": class_value("recall_per_class", index),
            "f1": class_value("f1_per_class", index),
            "support": class_value("support_per_class", index),
        })
    report.update({
        "metrics": {
            key: _number(metrics.get(key))
            for key in ("accuracy", "loss", "macro_precision", "macro_recall", "macro_f1",
                        "micro_precision", "micro_recall", "micro_f1", "top5_acc")
            if _number(metrics.get(key)) is not None
        },
        "per_class": per_class,
        "confusion_matrix": metrics.get("confusion_matrix", []),
    })
    return _write(job_id, report)


def save_detection_report(
    job_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    report = _base(job_id, "detection", config)
    aliases = {
        "map": ("mAP@.50:.95", "coco/bbox_mAP"),
        "map50": ("mAP@.50", "coco/bbox_mAP_50"),
        "map75": ("mAP@.75", "coco/bbox_mAP_75"),
        "precision": ("precision",),
        "recall": ("recall",),
    }
    normalized = {}
    for name, candidates in aliases.items():
        value = next((_number(metrics.get(key)) for key in candidates if _number(metrics.get(key)) is not None), None)
        if value is not None:
            normalized[name] = value
    report.update({"metrics": normalized, "per_class": [], "confusion_matrix": []})
    return _write(job_id, report)


def _write(job_id: str, report: Mapping[str, Any]) -> Path:
    path = evaluation_report_path(job_id)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
