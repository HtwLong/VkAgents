"""Backend-neutral helpers for rich object-detection evaluation results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from cvmodellearning.paths import artifacts_dir, test_json_path


def _floats(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def dataset_statistics(job_id: str, classes: list[str]) -> dict[str, Any]:
    path = test_json_path(job_id)
    if not path.exists():
        return {}
    try:
        coco = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    categories = {int(row["id"]): str(row["name"]) for row in coco.get("categories", [])}
    images_by_category: dict[int, set[int]] = {key: set() for key in categories}
    instances = {key: 0 for key in categories}
    areas: list[float] = []
    for annotation in coco.get("annotations", []):
        category = int(annotation.get("category_id", -1))
        if category in instances:
            instances[category] += 1
            images_by_category[category].add(int(annotation.get("image_id", -1)))
        bbox = annotation.get("bbox") or []
        if len(bbox) == 4:
            areas.append(max(0.0, float(bbox[2])) * max(0.0, float(bbox[3])))
    by_class = []
    ordered_categories = sorted(categories)
    for index, class_name in enumerate(classes):
        category = next((key for key, name in categories.items() if name == class_name), None)
        # Some old custom-head checkpoints contain generic names such as "item".
        # Category order remains authoritative when class counts match.
        if category is None and len(classes) == len(ordered_categories):
            category = ordered_categories[index]
        by_class.append({
            "class_name": class_name,
            "instances": instances.get(category, 0),
            "images": len(images_by_category.get(category, set())),
        })
    return {
        "images": len(coco.get("images", [])),
        "instances": len(coco.get("annotations", [])),
        "images_without_annotations": max(
            0,
            len(coco.get("images", []))
            - len({int(row.get("image_id", -1)) for row in coco.get("annotations", [])}),
        ),
        "mean_box_area_pixels": float(np.mean(areas)) if areas else 0.0,
        "per_class": by_class,
    }


def _curve(curves_results: Any, key: str) -> dict[str, Any] | None:
    if not isinstance(curves_results, Iterable):
        return None
    for x, y, x_label, y_label in curves_results:
        normalized_name = f"{y_label}{x_label}".upper().replace("-", "").replace("_", "").replace(" ", "")
        if key in normalized_name:
            values = np.asarray(y)
            if values.ndim > 1:
                values = values.mean(axis=0)
            return {"x": _floats(x), "y": _floats(values), "x_label": str(x_label)}
    return None


def _render_authoritative_plots(metrics: Any, classes: list[str]) -> None:
    """Overwrite backend plots using dataset labels instead of checkpoint metadata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(metrics.save_dir)
    box = metrics.box
    curve_specs = {
        ("Recall", "Precision"): ("BoxPR_curve.png", "Precision-Recall Curve"),
        ("Confidence", "F1"): ("BoxF1_curve.png", "F1-Confidence Curve"),
        ("Confidence", "Precision"): ("BoxP_curve.png", "Precision-Confidence Curve"),
        ("Confidence", "Recall"): ("BoxR_curve.png", "Recall-Confidence Curve"),
    }
    for x, y, x_label, y_label in getattr(box, "curves_results", []):
        if (x_label, y_label) not in curve_specs:
            continue
        filename, title = curve_specs[(x_label, y_label)]
        values = np.atleast_2d(np.asarray(y, dtype=float))
        fig, axis = plt.subplots(figsize=(10, 7))
        for index, row in enumerate(values[:len(classes)]):
            axis.plot(x, row, linewidth=1.5, label=classes[index])
        if values.shape[0] > 1:
            axis.plot(x, values[:len(classes)].mean(axis=0), color="blue", linewidth=3, label="all classes")
        axis.set(title=title, xlabel=x_label, ylabel=y_label, xlim=(0, 1), ylim=(0, 1))
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=180)
        plt.close(fig)

    matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    if matrix is None:
        return
    matrix = np.asarray(matrix, dtype=float)
    labels = [*classes, "background"] if matrix.shape[0] == len(classes) + 1 else classes
    for normalized, filename, title in (
        (False, "confusion_matrix.png", "Confusion Matrix"),
        (True, "confusion_matrix_normalized.png", "Confusion Matrix Normalized"),
    ):
        displayed = matrix.copy()
        if normalized:
            displayed /= np.maximum(displayed.sum(axis=0, keepdims=True), 1e-12)
        fig, axis = plt.subplots(figsize=(9, 7))
        image = axis.imshow(displayed, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        for row in range(displayed.shape[0]):
            for column in range(displayed.shape[1]):
                value = displayed[row, column]
                text = f"{value:.2f}" if normalized else f"{int(value)}"
                axis.text(column, row, text, ha="center", va="center",
                          color="white" if value > displayed.max() / 2 else "black")
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
                 yticklabels=labels, xlabel="True", ylabel="Predicted", title=title)
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=180)
        plt.close(fig)


def collect_ultralytics_metrics(metrics: Any, job_id: str, classes: list[str], evaluation_name: str) -> dict[str, Any]:
    box = metrics.box
    maps = _floats(getattr(box, "maps", []))
    precision = _floats(getattr(box, "p", []))
    recall = _floats(getattr(box, "r", []))
    ap50 = _floats(getattr(box, "ap50", []))
    support = {row["class_name"]: row["instances"] for row in dataset_statistics(job_id, classes).get("per_class", [])}
    per_class = []
    for index, name in enumerate(classes):
        p = precision[index] if index < len(precision) else None
        r = recall[index] if index < len(recall) else None
        per_class.append({
            "class_name": name,
            "ap": maps[index] if index < len(maps) else None,
            "ap50": ap50[index] if index < len(ap50) else None,
            "precision": p,
            "recall": r,
            "f1": (2 * p * r / (p + r)) if p is not None and r is not None and p + r else 0.0,
            "support": support.get(name, 0),
        })

    _render_authoritative_plots(metrics, classes)
    destination = artifacts_dir(job_id) / "evaluation" / evaluation_name
    destination.mkdir(parents=True, exist_ok=True)
    visualizations = []
    for source in Path(metrics.save_dir).glob("*.png"):
        target = destination / source.name
        shutil.copy2(source, target)
        visualizations.append({
            "name": source.stem.replace("_", " ").title(),
            "path": f"artifacts/evaluation/{evaluation_name}/{source.name}",
            "url": f"/artifacts/{job_id}/evaluation/{evaluation_name}/{source.name}",
        })
    curves = {}
    curve_results = getattr(box, "curves_results", [])
    for identifier, match in (("precision_recall", "PRECISIONRECALL"), ("f1_confidence", "F1CONFIDENCE"),
                              ("precision_confidence", "PRECISIONCONFIDENCE"), ("recall_confidence", "RECALLCONFIDENCE")):
        value = _curve(curve_results, match)
        if value:
            curves[identifier] = value
    confusion = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    if confusion is not None:
        if hasattr(confusion, "detach"):
            confusion = confusion.detach().cpu()
        if hasattr(confusion, "tolist"):
            confusion = confusion.tolist()
    labels = list(classes)
    if isinstance(confusion, list) and len(confusion) == len(classes) + 1:
        confusion = np.asarray(confusion).T.tolist()  # UI contract: actual rows, predicted columns.
        labels.append("background")
    else:
        confusion = []
    return {
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": labels if confusion else [],
        "curves": curves,
        "dataset_statistics": dataset_statistics(job_id, classes),
        "visualizations": visualizations,
    }
