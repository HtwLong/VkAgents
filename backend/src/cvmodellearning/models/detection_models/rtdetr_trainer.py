"""Minimal Ultralytics RT-DETR-L training and evaluation adapter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from ultralytics import RTDETR

from cvmodellearning.models.detection_models.yolo_trainer import (
    select_ultralytics_device_string,
)
from cvmodellearning.paths import (
    PROJECT_ROOT,
    best_yolo_model_path,
    plots_dir,
    run_dir,
    tool_call_args_path,
    training_log_path,
    yolo_data_yaml_path,
)
from cvmodellearning.jobs.run_control import PipelineCancelled, cancellation_requested, raise_if_cancelled


RTDETR_CHECKPOINTS = {"rtdetr_hgnetv2_l": "rtdetr-l.pt"}


def _checkpoint_path(model_name: str) -> str:
    try:
        checkpoint = RTDETR_CHECKPOINTS[model_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported RT-DETR model_name: {model_name}") from exc
    bundled = PROJECT_ROOT / "src" / checkpoint
    return str(bundled) if bundled.exists() else checkpoint


def _move_training_artifacts(model: RTDETR, job_id: str) -> None:
    save_dir = Path(model.trainer.save_dir)
    weights = save_dir / "weights"
    source_model = save_dir / "weights" / "best.pt"
    if not source_model.exists():
        candidates = sorted(weights.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"RT-DETR produced no checkpoint in {weights}")
        source_model = candidates[0]
    shutil.move(str(source_model), str(best_yolo_model_path(job_id)))

    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        shutil.move(str(results_csv), str(training_log_path(job_id)))
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for image in save_dir.glob(pattern):
            shutil.copy2(image, plots_dir(job_id) / image.name)


def train_rtdetr_from_config(config: Mapping[str, Any], job_id: str) -> str:
    """Fine-tune the registered COCO-pretrained RT-DETR-L checkpoint."""
    flat = dict(config)
    optimizer = flat.get("optimizer")
    if isinstance(optimizer, Mapping):
        flat["optimizer_name"] = optimizer.get("name")
        params = optimizer.get("params") or {}
        if isinstance(params, Mapping):
            flat.update(params)

    model_name = str(flat["model_name"])
    optimizer_name = str(flat.get("optimizer_name", "adamw")).lower()
    optimizer_map = {"adamw": "AdamW", "sgd": "SGD", "rmsprop": "RMSProp"}
    if optimizer_name not in optimizer_map:
        raise ValueError(f"Unsupported RT-DETR optimizer: {optimizer_name}")

    args = {
        "data": str(yolo_data_yaml_path(job_id)),
        "epochs": int(flat["num_epochs"]),
        "patience": int(flat.get("patience", 20)),
        "imgsz": int(flat.get("input_size", 640)),
        "batch": int(flat.get("batch_size", 1)),
        "workers": int(flat.get("workers", 8)),
        "optimizer": optimizer_map[optimizer_name],
        "lr0": float(flat.get("learning_rate", 0.001)),
        "lrf": float(flat.get("final_learning_rate_factor", 0.01)),
        "momentum": float(flat.get("beta1", flat.get("momentum", 0.9))),
        "weight_decay": float(flat.get("weight_decay", 0.0005)),
        "warmup_epochs": float(flat.get("warmup_epochs", 3.0)),
        "warmup_momentum": float(flat.get("warmup_momentum", 0.8)),
        "mosaic": float(flat.get("mosaic", 1.0)),
        "mixup": float(flat.get("mixup", 0.0)),
        "cutmix": float(flat.get("cutmix", 0.0)),
        "degrees": float(flat.get("degrees", 0.0)),
        "translate": float(flat.get("translate", 0.1)),
        "scale": float(flat.get("scale", 0.5)),
        "fliplr": float(flat.get("fliplr", 0.5)),
        "hsv_h": float(flat.get("hsv_h", 0.015)),
        "hsv_s": float(flat.get("hsv_s", 0.7)),
        "hsv_v": float(flat.get("hsv_v", 0.4)),
        "close_mosaic": int(flat.get("close_mosaic", 10)),
        "single_cls": bool(flat.get("single_cls", False)),
        "amp": bool(flat.get("amp", False)),
        "seed": int(flat.get("seed", 0)),
        "deterministic": False,
        "cos_lr": False,
        "device": select_ultralytics_device_string(),
        "project": str(run_dir(job_id)),
        "name": "temp_run",
        "exist_ok": True,
    }
    tool_call_args_path(job_id).write_text(
        json.dumps({"job_id": job_id, "model_name": model_name, **args}, indent=4),
        encoding="utf-8",
    )

    try:
        temporary_run = Path(args["project"]) / str(args["name"])
        if temporary_run.exists():
            shutil.rmtree(temporary_run)
        model = RTDETR(_checkpoint_path(model_name))
        if hasattr(model, "add_callback"):
            model.add_callback(
                "on_train_epoch_end",
                lambda trainer: setattr(trainer, "stop", True)
                if cancellation_requested(job_id)
                else None,
            )
        model.train(**args)
        raise_if_cancelled(job_id)
        _move_training_artifacts(model, job_id)
    except PipelineCancelled:
        raise
    except Exception as exc:
        return f"❌ RT-DETR Training failed for Job ID {job_id}: {exc}"
    return f"✅ Successfully trained {model_name} (Job ID: {job_id})."


def evaluate_rtdetr_model(
    *, batch_size: int, image_size: int, job_id: str
) -> dict[str, float | str]:
    model_path = best_yolo_model_path(job_id)
    data_path = yolo_data_yaml_path(job_id)
    if not model_path.exists():
        return {"error": f"Best RT-DETR model not found at: {model_path}"}
    if not data_path.exists():
        return {"error": f"RT-DETR data YAML not found at: {data_path}"}
    try:
        metrics = RTDETR(str(model_path)).val(
            data=str(data_path),
            split="test",
            batch=batch_size,
            imgsz=image_size,
            project=str(run_dir(job_id)),
            name="test_evaluation",
            exist_ok=True,
        )
        return {
            "mAP@.50:.95": float(metrics.box.map),
            "mAP@.50": float(metrics.box.map50),
            "mAP@.75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "results_dir": str(Path(metrics.save_dir).resolve()),
        }
    except Exception as exc:
        return {"error": f"RT-DETR evaluation failed for Job ID {job_id}: {exc}"}
