import logging
import inspect
import json
import time
import platform
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
import torch.nn.functional as F

from cvmodellearning.agents.agents_utils import save_json
from cvmodellearning.models.model_manager import MODEL_CACHE_MANAGER


# project utils
from cvmodellearning.download.download_data import download_visionkg_mixed_datasets_classification
from cvmodellearning.download.progress import DownloadProgressTracker
from cvmodellearning.jobs.run_control import PipelineCancelled, raise_if_cancelled
from cvmodellearning.datasets.provenance import record_split_access
from cvmodellearning.download.assignment_manifest import (
    assignment_fingerprint,
    file_sha256,
    iter_download_allocations,
    load_dataset_manifest,
    load_preparation_summary,
)
from cvmodellearning.preprocessing.preprocessing import CocoImageDataset
from cvmodellearning.preprocessing.transformations import (
    select_evaluation_transform,
    select_transforms,
)
from cvmodellearning.training.training_utils import (
    RepeatedAugmentationSampler,
    apply_swin_activation_checkpointing,
    classification_parameter_groups,
    classifier_training_module,
    make_epoch_scheduler,
    make_model_ema,
    set_backbone_trainable,
    swin_parameter_groups,
    train_one_epoch,
)
from cvmodellearning.evaluation.evaluation_utils import evaluate
from cvmodellearning.optimization.optimization_utils import make_criterion, make_optimizer
from cvmodellearning.models.classification_model_utils import get_model_weights, make_model
from cvmodellearning.models.classification_lora import (
    CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
    LORA_CHECKPOINT_FORMAT,
    apply_classification_lora,
    classification_lora_metadata,
    classification_lora_state_dict,
    load_classification_lora_state_dict,
    validate_classification_lora_metadata,
)
from cvmodellearning.models.registry import FREEZABLE_CLASSIFICATION_MODEL_IDS
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.training.resource_guard import validate_image_batch_preflight
from cvmodellearning.evaluation.result_report import save_classification_report
from cvmodellearning.paths import (
    run_dir,
    artifacts_dir,
    data_dir,
    csv_labels_path,
    test_report_json_path,
    train_csv_path,
    training_log_path,
    val_csv_path,
    test_csv_path,
    best_model_path,
    metrics_csv_path,
    test_cm_path,
    dataset_manifest_path,
    preparation_summary_path,
)

class ClassificationPipeline:
    """
    Encapsulates the image classification workflow, including data handling, 
    training, evaluation, and in-memory model inference.
    """

    def __init__(self):
        """Initializes the pipeline."""
        pass

    def _require_prepared_data(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        return load_preparation_summary(
            preparation_summary_path(job_id),
            task="classification",
            expected_fingerprint=assignment_fingerprint(config["selected_data"]),
            expected_manifest_sha256=file_sha256(dataset_manifest_path(job_id)),
            required_artifacts={
                "train": train_csv_path(job_id),
                "validation": val_csv_path(job_id),
                "test": test_csv_path(job_id),
            },
        )

    def _metric_value(self, track_metric: str, val_loss: float, val_acc: float, val_metrics: dict) -> float:
        if track_metric == "val_loss":
            return float(val_loss)
        if track_metric == "val_acc":
            return float(val_acc)
        if track_metric in val_metrics:
            return float(val_metrics[track_metric])
        raise ValueError(f"Unknown track_metric: {track_metric}")

    def _metric_is_better(self, track_metric: str, current: float, best: float) -> bool:
        if track_metric in ("val_acc", "macro_f1", "micro_f1"):
            return current > best
        if track_metric == "val_loss":
            return current < best
        raise ValueError(f"Unknown track_metric: {track_metric}")

    @staticmethod
    def _early_stopping_active(config: Dict[str, Any], epoch: int) -> bool:
        """Enable stopping only after a staged backbone has been unfrozen."""
        if int(config.get("patience", 0)) == 0:
            return False
        if config.get("training_mode") != "staged_fine_tune":
            return True
        return epoch > int(config.get("freeze_backbone_epochs", 0))

    def _choose_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _validate_transformed_sample(self, dataset, expected_size: int, split_name: str) -> None:
        """Fail before training if preprocessing does not produce a model-ready image."""
        if len(dataset) == 0:
            raise ValueError(f"The {split_name} dataset is empty.")
        image, _ = dataset[0]
        expected_shape = (3, expected_size, expected_size)
        if not isinstance(image, torch.Tensor) or tuple(image.shape) != expected_shape:
            actual_shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
            raise ValueError(
                f"The {split_name} transform must produce a tensor shaped {expected_shape}; "
                f"received {actual_shape}."
            )

    def _checkpoint_fingerprint(self, checkpoint_path) -> Dict[str, Any]:
        stat = checkpoint_path.stat()
        return {
            "path": str(checkpoint_path.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }

    @staticmethod
    def _restore_checkpoint_model(
        model_name: str,
        num_classes: int,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
    ):
        """Rebuild either a full checkpoint or a pretrained base plus LoRA adapter."""
        is_lora = config.get("training_mode") == "lora"
        checkpoint_format = checkpoint.get("checkpoint_format")
        checkpoint_version = checkpoint.get("checkpoint_format_version")
        if is_lora:
            if checkpoint_format != LORA_CHECKPOINT_FORMAT:
                raise ValueError(
                    f"LoRA configuration requires checkpoint_format='{LORA_CHECKPOINT_FORMAT}', "
                    f"received {checkpoint_format!r}."
                )
            if checkpoint_version != CLASSIFICATION_CHECKPOINT_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported LoRA checkpoint format version: {checkpoint_version!r}."
                )
            validate_classification_lora_metadata(
                model_name,
                config,
                checkpoint.get("adapter_metadata"),
            )
        elif checkpoint_format not in {None, "full_model"}:
            raise ValueError(
                f"Full-model configuration cannot load checkpoint_format={checkpoint_format!r}."
            )
        elif (
            checkpoint_format == "full_model"
            and checkpoint_version != CLASSIFICATION_CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError(
                f"Unsupported full-model checkpoint format version: {checkpoint_version!r}."
            )

        model, _ = make_model(
            model_name,
            "default" if is_lora else "none",
            num_classes=num_classes,
        )
        if is_lora:
            model = apply_classification_lora(model, model_name, config)
            load_classification_lora_state_dict(model, checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def download_data_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Downloads data from VisionKG based on selected_data and creates a labels CSV."""

        # Ensure run directories exist
        data_base = data_dir(job_id)
        data_base.mkdir(parents=True, exist_ok=True)

        total = sum(item.count for item in iter_download_allocations(config["selected_data"]))
        progress = DownloadProgressTracker(job_id, total)
        try:
            kwargs = {}
            if "progress_callback" in inspect.signature(
                download_visionkg_mixed_datasets_classification
            ).parameters:
                def record_progress(**values):
                    raise_if_cancelled(job_id)
                    progress.record(**values)
                kwargs["progress_callback"] = record_progress
            if "cancel_check" in inspect.signature(
                download_visionkg_mixed_datasets_classification
            ).parameters:
                kwargs["cancel_check"] = lambda: raise_if_cancelled(job_id)
            report = download_visionkg_mixed_datasets_classification(
                job_id, config["selected_data"], **kwargs
            )
        except PipelineCancelled:
            progress.finish("stopped")
            raise
        except Exception:
            progress.finish("failed")
            raise
        if not report.get("complete", False):
            progress.record_failed_datasets([
                item["dataset_name"] for item in report["sources"] if item["shortfall"]
            ])
            progress.finish("failed")
            shortfalls = ", ".join(
                f"{item['class_name']} from {item['dataset_name']}: "
                f"{item['downloaded']}/{item['requested']} for {item['assigned_split']}"
                for item in report["sources"]
                if item["shortfall"]
            )
            raise RuntimeError(
                "Classification data download was incomplete; training data was not prepared. "
                f"Shortfalls: {shortfalls}. See artifacts/download_report.json."
            )

        progress.finish("completed")
        return {
            "labels_csv": str(csv_labels_path(job_id)),
            "dataset_manifest": report["manifest_path"],
            "download_report": report["report_path"],
        }

    def prepare_data_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Materialize the manifest's train/validation/test assignments as CSVs."""
        labels_csv = csv_labels_path(job_id)
        if not labels_csv.exists():
            raise FileNotFoundError("Labels CSV not found; call download_data_step first.")
        df = pd.read_csv(
            labels_csv,
            names=["image_filename", "labels"],
            header=0,
            dtype={"labels": str},
        )
        manifest = load_dataset_manifest(
            dataset_manifest_path(job_id),
            task="classification",
            expected_fingerprint=assignment_fingerprint(config["selected_data"]),
        )
        manifest_by_path = {sample["image_path"]: sample for sample in manifest["samples"]}

        allowed_classes = set(config.get("classes") or df["labels"])
        unexpected = sorted(set(df["labels"]) - allowed_classes)
        if unexpected:
            raise ValueError(f"Downloaded labels contain unexpected classes: {unexpected}")
        df = df.drop_duplicates(subset=["image_filename"], keep="first").reset_index(drop=True)
        csv_paths = set(df["image_filename"])
        if csv_paths != set(manifest_by_path):
            raise ValueError("Classification labels CSV and dataset manifest contain different samples.")
        missing_files = [name for name in csv_paths if not (data_dir(job_id) / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} classification samples reference missing image files; "
                f"first missing path: {missing_files[0]}"
            )
        for row in df.itertuples(index=False):
            if row.labels not in manifest_by_path[row.image_filename]["class_names"]:
                raise ValueError(f"Manifest label mismatch for {row.image_filename}.")

        split_frames = {
            split: df[df["image_filename"].map(
                lambda path: manifest_by_path[path]["assigned_split"] == split
            )].reset_index(drop=True)
            for split in ("train", "validation", "test")
        }
        for split, split_df in split_frames.items():
            if split_df.empty:
                raise ValueError(f"Classification {split} split is empty.")
            missing_classes = sorted(allowed_classes - set(split_df["labels"]))
            if missing_classes:
                raise ValueError(f"Classification {split} split is missing classes: {missing_classes}")

        train_csv = train_csv_path(job_id)
        val_csv = val_csv_path(job_id)
        test_csv = test_csv_path(job_id)
        train_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary_splits = {
            train_csv: train_csv.with_suffix(train_csv.suffix + ".tmp"),
            val_csv: val_csv.with_suffix(val_csv.suffix + ".tmp"),
            test_csv: test_csv.with_suffix(test_csv.suffix + ".tmp"),
        }
        split_frames["train"].to_csv(temporary_splits[train_csv], index=False, header=True)
        split_frames["validation"].to_csv(temporary_splits[val_csv], index=False, header=True)
        split_frames["test"].to_csv(temporary_splits[test_csv], index=False, header=True)
        raise_if_cancelled(job_id)
        for final, temporary in temporary_splits.items():
            temporary.replace(final)

        result = {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "counts": {
                "train": int(len(split_frames["train"])),
                "validation": int(len(split_frames["validation"])),
                "test": int(len(split_frames["test"])),
            },
            "assignment_fingerprint": manifest["assignment_fingerprint"],
            "manifest_sha256": file_sha256(dataset_manifest_path(job_id)),
        }
        summary_path = preparation_summary_path(job_id)
        result["preparation_summary"] = str(summary_path)
        temporary_summary = summary_path.with_suffix(".json.tmp")
        temporary_summary.write_text(
            json.dumps({"task": "classification", **result}, indent=2), encoding="utf-8"
        )
        temporary_summary.replace(summary_path)
        return result

    def train_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """The main training loop for the image classification model."""
        preparation = self._require_prepared_data(config, job_id)
        provenance_path = record_split_access(
            job_id,
            task="classification",
            stage="training",
            preparation=preparation,
            split_artifacts={
                "train": train_csv_path(job_id),
                "validation": val_csv_path(job_id),
            },
        )
        config = training_compatible_hpo_config(config)
        validate_image_batch_preflight(
            image_size=int(config.get("image_size", 224)),
            batch_size=int(config.get("batch_size") or 32),
        )
        device = self._choose_device()

        # logging
        logs_base = artifacts_dir(job_id)
        logs_base.mkdir(parents=True, exist_ok=True)
        log_file = training_log_path(job_id)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
            force=True,
        )
        logging.info("Logging initialized")

        # model
        classes = config["classes"]
        which_weights = config.get("model_weights", "default")
        model, weights = make_model(config["model_name"], which_weights, num_classes=len(classes))
        is_swin_v2 = config["model_name"].startswith("swin_v2_")
        is_lora = config.get("training_mode") == "lora"
        supports_backbone_freezing = config["model_name"] in FREEZABLE_CLASSIFICATION_MODEL_IDS
        if is_swin_v2 and config.get("use_activation_checkpointing", False):
            apply_swin_activation_checkpointing(model)
        if is_lora:
            model = apply_classification_lora(model, config["model_name"], config)
        freeze_backbone_epochs = int(config.get("freeze_backbone_epochs", 0))
        if supports_backbone_freezing and freeze_backbone_epochs > 0:
            set_backbone_trainable(model, config["model_name"], False)
        model = model.to(device)

        # transforms
        train_transform, _ = select_transforms(
            config["model_name"],
            image_size=int(config.get("image_size", 256 if is_swin_v2 else 224)),
            weights=weights,
            auto_augment_policy=config.get("auto_augment_policy", "none"),
            random_erasing=float(config.get("random_erasing", 0.0)),
            random_resized_crop_scale_min=float(
                config.get("random_resized_crop_scale_min", 0.6)
            ),
            horizontal_flip_probability=float(
                config.get("horizontal_flip_probability", 0.5)
            ),
        )
        eval_transform = select_evaluation_transform(
            config["model_name"],
            image_size=int(config.get("image_size", 256 if is_swin_v2 else 224)),
            weights=weights,
        )

        # datasets / loaders
        class_to_idx = {name: i for i, name in enumerate(classes)}
        train_dataset = CocoImageDataset(
            csv_file=train_csv_path(job_id),
            root_dir=data_dir(job_id),
            transform=train_transform,
            class_to_idx=class_to_idx,
        )
        val_dataset = CocoImageDataset(
            csv_file=val_csv_path(job_id),
            root_dir=data_dir(job_id),
            transform=eval_transform,
            class_to_idx=class_to_idx,
        )
        configured_image_size = int(config.get("image_size", 256 if is_swin_v2 else 224))
        self._validate_transformed_sample(train_dataset, configured_image_size, "training")
        self._validate_transformed_sample(val_dataset, configured_image_size, "validation")

        is_macos = (platform.system() == "Darwin")
        pin_memory = (device.type == "cuda")
        nw = 0 if (is_macos or device.type == "mps") else 4
        batch_size = config.get("batch_size") or 32
        repetitions = int(config.get("repeated_augmentation_repetitions", 1))
        train_sampler = RepeatedAugmentationSampler(train_dataset, repetitions) if repetitions > 1 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=nw,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin_memory)

        # loss / optimizer
        criterion = make_criterion(config)
        if is_lora:
            optimizer_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        elif is_swin_v2:
            optimizer_params = swin_parameter_groups(model, config)
        elif config["model_name"] in FREEZABLE_CLASSIFICATION_MODEL_IDS:
            optimizer_params = classification_parameter_groups(model, config["model_name"], config)
        else:
            optimizer_params = model.parameters()
        optimizer = make_optimizer(optimizer_params, config)
        scheduler = make_epoch_scheduler(optimizer, config)

        mix_transforms = []
        if float(config.get("mixup_alpha", 0.0)) > 0:
            mix_transforms.append(v2.MixUp(alpha=float(config["mixup_alpha"]), num_classes=len(classes)))
        if float(config.get("cutmix_alpha", 0.0)) > 0:
            mix_transforms.append(v2.CutMix(alpha=float(config["cutmix_alpha"]), num_classes=len(classes)))
        batch_augmentation = None
        if len(mix_transforms) == 1:
            batch_augmentation = mix_transforms[0]
        elif mix_transforms:
            batch_augmentation = v2.RandomChoice(mix_transforms)

        amp_enabled = config.get("precision", "fp32") == "mixed" and device.type == "cuda"
        effective_precision = "mixed" if amp_enabled else "fp32"
        logging.info(
            "Configured precision=%s; effective precision=%s on %s",
            config.get("precision", "fp32"),
            effective_precision,
            device.type,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        ema_model = None
        ema_effective_decay = None
        ema_update_steps = None
        if config.get("use_model_ema", False):
            ema_model, ema_effective_decay, ema_update_steps = make_model_ema(
                model,
                config,
                effective_batch_size=(
                    batch_size * int(config.get("gradient_accumulation_steps", 1))
                ),
                device=device,
            )
        ema_optimizer_step_count = 0

        # training config
        num_epochs = config["num_epochs"]
        patience = config["patience"]
        track_metric = config["track_metric"]

        # metrics CSV header
        metrics_csv = metrics_csv_path(job_id)
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_csv, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,macro_f1,micro_f1,tracked\n")

        best_val = -float("inf") if track_metric in ("val_acc", "macro_f1", "micro_f1") else float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        logging.info("Starting training")
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch - 1)
            if (
                supports_backbone_freezing
                and freeze_backbone_epochs > 0
                and epoch == freeze_backbone_epochs + 1
            ):
                set_backbone_trainable(model, config["model_name"], True)
                logging.info("Unfroze %s backbone at epoch %d", config["model_name"], epoch)
            def maybe_update_ema() -> None:
                nonlocal ema_optimizer_step_count
                if ema_model is None or ema_update_steps is None:
                    return
                if ema_optimizer_step_count % ema_update_steps == 0:
                    ema_model.update_parameters(model)
                    if epoch <= int(config.get("warmup_epochs", 0)):
                        ema_model.n_averaged.zero_()
                ema_optimizer_step_count += 1

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
                gradient_clip_norm=float(config.get("gradient_clip_norm", 0.0)),
                scaler=scaler,
                batch_augmentation=batch_augmentation,
                on_optimizer_step=maybe_update_ema if ema_model is not None else None,
                frozen_backbone=(
                    (model.features if is_swin_v2 else model)
                    if supports_backbone_freezing
                    and freeze_backbone_epochs > 0
                    and epoch <= freeze_backbone_epochs
                    else None
                ),
                trainable_head=(
                    classifier_training_module(model, config["model_name"])
                    if supports_backbone_freezing
                    and freeze_backbone_epochs > 0
                    and epoch <= freeze_backbone_epochs
                    else None
                ),
                cancel_check=lambda: raise_if_cancelled(job_id),
            )
            evaluation_model = (
                ema_model
                if ema_model is not None and int(ema_model.n_averaged.item()) > 0
                else model
            )
            val_loss, val_acc, val_metrics = evaluate(classes, evaluation_model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()

            tracked = self._metric_value(track_metric, val_loss, val_acc, val_metrics)
            macro_f1 = float(val_metrics.get("macro_f1", 0.0))
            micro_f1 = float(val_metrics.get("micro_f1", 0.0))

            logging.info(
                f"Epoch {epoch:02d}/{num_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f} "
                f"{track_metric}={tracked:.4f} elapsed={time.time()-start:.1f}s"
            )

            with open(metrics_csv, "a") as f:
                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{val_loss:.6f},{val_acc:.6f},{macro_f1:.6f},{micro_f1:.6f},{tracked:.6f}\n"
                )

            progress_path = run_dir(job_id) / "progress.json"
            temporary_progress = progress_path.with_suffix(".json.tmp")
            temporary_progress.write_text(json.dumps({
                "status": "running",
                "current_epoch": epoch,
                "total_epochs": int(num_epochs),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "val_macro_f1": macro_f1,
                "val_micro_f1": micro_f1,
                "tracked_metric": track_metric,
                "tracked_value": float(tracked),
                "elapsed_seconds": float(time.time() - start),
            }, indent=2), encoding="utf-8")
            temporary_progress.replace(progress_path)

            is_better = self._metric_is_better(track_metric, tracked, best_val)
            if is_better:
                best_val = tracked
                best_epoch = epoch
                epochs_no_improve = 0
                bm_path = best_model_path(job_id)
                bm_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_model = bm_path.with_suffix(bm_path.suffix + ".tmp")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            classification_lora_state_dict(model)
                            if is_lora
                            else (
                                evaluation_model.module.state_dict()
                                if evaluation_model is ema_model
                                else model.state_dict()
                            )
                        ),
                        "checkpoint_format": LORA_CHECKPOINT_FORMAT if is_lora else "full_model",
                        "checkpoint_format_version": CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
                        "adapter_metadata": (
                            classification_lora_metadata(
                                config["model_name"],
                                config,
                            )
                            if is_lora
                            else None
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "best_metric_name": track_metric,
                        "best_metric_value": best_val,
                        "class_to_idx": class_to_idx,
                        "classes": classes,
                        "config": config,
                        "resolved_preprocessing": {
                            "training": repr(train_transform),
                            "evaluation": repr(eval_transform),
                        },
                        "effective_precision": effective_precision,
                        "ema": {
                            "enabled": ema_model is not None,
                            "configured_decay": float(config.get("model_ema_decay", 0.99998)),
                            "effective_decay": ema_effective_decay,
                            "update_steps": ema_update_steps,
                            "selected_for_checkpoint": evaluation_model is ema_model,
                        },
                    },
                    temporary_model,
                )
                temporary_model.replace(bm_path)
                logging.info(f"New best checkpoint on {track_metric}={best_val:.4f} at epoch {epoch}")
            else:
                early_stopping_active = self._early_stopping_active(config, epoch)
                if early_stopping_active:
                    epochs_no_improve += 1
                if early_stopping_active and epochs_no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch} (no improvement on {track_metric} for {patience} epochs)")
                    break

        return {
            "best_epoch": best_epoch,
            "best_metric": track_metric,
            "best_value": best_val,
            "effective_precision": effective_precision,
            "artifacts": {
                "best_model": str(best_model_path(job_id)),
                "metrics_csv": str(metrics_csv),
                "data_provenance": str(provenance_path),
            },
            "test_batch_size": batch_size,
        }

    def evaluate_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Evaluates the best trained model on the test set."""
        preparation = self._require_prepared_data(config, job_id)
        device = self._choose_device()

        bm_path = best_model_path(job_id)
        if not bm_path.exists():
            raise FileNotFoundError(f"Best model not found at {bm_path}")

        # Load checkpoint and prefer saved training config/classes
        ckpt = torch.load(bm_path, map_location=device)
        cfg_ckpt = ckpt.get("config", {})  # saved during training
        classes = ckpt.get("classes") or config.get("classes") or []
        if not classes:
            raise ValueError("Classes list is missing in both checkpoint and provided config")

        # Resolve model spec from either flat or nested schema
        def resolve_model(cfg: Dict[str, Any]) -> tuple[str, str]:
            if "model_name" in cfg:  # flat schema
                return cfg["model_name"], cfg.get("model_weights", "default")
            m = cfg.get("model")
            if isinstance(m, dict) and "name" in m:  # nested schema
                return m["name"], m.get("weights", "default")
            # Fallback to provided config if needed
            if "model_name" in config:
                return config["model_name"], config.get("model_weights", "default")
            if isinstance(config.get("model"), dict) and "name" in config["model"]:
                return config["model"]["name"], config["model"].get("weights", "default")
            raise ValueError("Cannot resolve model specification from checkpoint or provided config")

        model_name, model_weights = resolve_model(cfg_ckpt)

        # Reproduce the preprocessing saved with the trained checkpoint. Weight
        # metadata is resolved without constructing/downloading another model.
        is_swin_v2 = model_name.startswith("swin_v2_")
        image_size = int(
            cfg_ckpt.get(
                "image_size",
                config.get("image_size", 256 if is_swin_v2 else 224),
            )
        )
        eval_transform = select_evaluation_transform(
            model_name,
            image_size=image_size,
            weights=get_model_weights(model_name, model_weights),
        )

        # DataLoader for test set
        class_to_idx = {name: i for i, name in enumerate(classes)}
        test_dataset = CocoImageDataset(
            csv_file=test_csv_path(job_id),
            root_dir=data_dir(job_id),
            transform=eval_transform,
            class_to_idx=class_to_idx,
        )
        self._validate_transformed_sample(test_dataset, image_size, "test")
        batch_size = cfg_ckpt.get("batch_size") or config.get("batch_size") or 32
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        # Rebuild model (no pretrained weights now), load best state, eval mode
        model = self._restore_checkpoint_model(
            model_name,
            len(classes),
            cfg_ckpt,
            ckpt,
        )
        model = model.to(device)
        model.eval()

        # Criterion from saved config when possible for consistency
        crit_cfg = cfg_ckpt if (
            "criterion_name" in cfg_ckpt
            or "optimizer_name" in cfg_ckpt
            or "criterion" in cfg_ckpt
            or "optimizer" in cfg_ckpt
        ) else config
        criterion = make_criterion(crit_cfg)

        # Evaluate
        test_loss, test_acc, test_metrics = evaluate(classes, model, test_loader, criterion, device)

        json_path = test_report_json_path(job_id)
        cm_path = test_cm_path(job_id)
        report_dict = test_metrics.get("classification_report_dict")
        if not report_dict:
            raise ValueError(
                "Test evaluation produced no classification report; verify that the test split is non-empty."
            )
        save_json(report_dict, json_path.parent, json_path.name)
        if "confusion_matrix" in test_metrics:
            cm = np.array(test_metrics["confusion_matrix"])
            header = ",".join(classes)
            np.savetxt(cm_path, cm, delimiter=",", fmt="%d", header=header, comments="")

        provenance_path = record_split_access(
            job_id,
            task="classification",
            stage="evaluation",
            preparation=preparation,
            split_artifacts={"test": test_csv_path(job_id)},
        )
        report_path = save_classification_report(job_id, cfg_ckpt or config, test_metrics)

        result = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
            "test_micro_f1": float(test_metrics.get("micro_f1", 0.0)),
            "artifacts": {
                "test_report_json": str(json_path),
                "test_confusion_matrix": str(cm_path),
                "evaluation_report": str(report_path),
                "data_provenance": str(provenance_path),
            },
        }
        if "top5_acc" in test_metrics:
            result["test_top5_acc"] = float(test_metrics["top5_acc"])
        return result
    
    # ======================================================================
    # Model Loading
    # ======================================================================

    def load_model_step(self, job_id: str) -> Dict[str, Any]:
        """Loads a trained model (by job_id) into the centralized cache."""
        
        key = f"{job_id}"
        bm_path = best_model_path(job_id)
        if not bm_path.exists():
            raise FileNotFoundError(f"No trained model found for {job_id}")
        checkpoint_fingerprint = self._checkpoint_fingerprint(bm_path)
        cached_bundle = MODEL_CACHE_MANAGER.get_model_bundle(key)

        if cached_bundle and cached_bundle.get("checkpoint_fingerprint") == checkpoint_fingerprint:
            return {
                "status": "loaded from cache", 
                "job_id": job_id, 
                "num_classes": len(cached_bundle["classes"]),
                "classes": cached_bundle["classes"],
                "device": str(cached_bundle["device"]),
            }

        # Proceed with loading from disk if not cached
        device = self._choose_device()

        ckpt = torch.load(bm_path, map_location=device)
        classes = ckpt.get("classes")
        config = ckpt.get("config", {})
        model_name = config.get("model_name")
        
        model_weights = config.get("model_weights", "default")

        # 2. Get the correct transforms
        eval_transform = select_evaluation_transform(
            model_name,
            image_size=int(
                config.get("image_size", 256 if model_name.startswith("swin_v2_") else 224)
            ),
            weights=get_model_weights(model_name, model_weights),
        )

        # 3. Load state dict, set device, and eval mode
        model = self._restore_checkpoint_model(
            model_name,
            len(classes),
            config,
            ckpt,
        )
        model.to(device)
        model.eval()

        # Store the model bundle in the centralized cache
        bundle = {
            "model": model,
            "device": device,
            "classes": classes,
            "transform": eval_transform,
            "model_name": model_name,
            "image_size": int(
                config.get("image_size", 256 if model_name.startswith("swin_v2_") else 224)
            ),
            "checkpoint_fingerprint": checkpoint_fingerprint,
        }
        MODEL_CACHE_MANAGER.set_model_bundle(key, bundle)

        return {
            "status": "loaded from disk",
            "job_id": job_id,
            "num_classes": len(classes),
            "classes": classes,
            "device": str(device),
        }

    # ======================================================================
    # Inference
    # ======================================================================

    def infer_step(self, job_id: str, image: Image.Image) -> Dict[str, Any]:
        """Performs inference on a single image using the model from the centralized cache."""
        
        # Retrieve bundle from the centralized manager
        key = f"{job_id}"
        bundle = MODEL_CACHE_MANAGER.get_model_bundle(key)
        
        if not bundle:
            # If the API calls load_model_step and immediately calls infer_step, 
            # the model should be here. If not, it's an error.
            raise RuntimeError(f"Model not loaded for {job_id}. Call load_model_step first.")

        model = bundle["model"]
        device = bundle["device"]
        transform = bundle["transform"]
        classes = bundle["classes"]

        model.eval()
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        predicted_class = classes[pred_idx.item()]
        confidence = conf.item()
        probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))}

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    # ======================================================================
    # Unloading Utilities
    # ======================================================================

    def unload_all_models(self) -> Dict[str, Any]:
        """
        Delegates the unload-all operation to the centralized manager.
        """
        return MODEL_CACHE_MANAGER.unload_all_models()


    def unload_model(self, job_id: str) -> Dict[str, Any]:
        """
        Delegates the unload operation to the centralized manager.
        """
        return MODEL_CACHE_MANAGER.unload_model(job_id)
