import logging
import time
import platform
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from importlib.resources import files, as_file
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.nn.functional as F

from cvmodellearning.agents.agents_utils import save_json
from cvmodellearning.models.model_manager import MODEL_CACHE_MANAGER


# project utils
from cvmodellearning.download.download_data import download_visionkg_mixed_datasets_classification
from cvmodellearning.preprocessing.preprocessing import CocoImageDataset
from cvmodellearning.preprocessing.transformations import select_transforms
from cvmodellearning.training.training_utils import train_one_epoch
from cvmodellearning.evaluation.evaluation_utils import evaluate
from cvmodellearning.optimization.optimization_utils import make_criterion, make_optimizer
from cvmodellearning.models.classification_model_utils import make_model  
from cvmodellearning.evaluation.report_builder import create_classification_report
from cvmodellearning.paths import (
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
    report_pdf_path,
)

class ClassificationPipeline:
    """
    Encapsulates the image classification workflow, including data handling, 
    training, evaluation, and in-memory model inference.
    """

    def __init__(self):
        """Initializes the pipeline."""
        pass

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

    def _choose_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def download_data_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Downloads data from VisionKG based on selected_data and creates a labels CSV."""

        # Ensure run directories exist
        data_base = data_dir(job_id)
        data_base.mkdir(parents=True, exist_ok=True)

        download_visionkg_mixed_datasets_classification(job_id, config["selected_data"])

        return {
            "labels_csv": str(csv_labels_path(job_id)),
        }

    def prepare_data_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Splits the dataset and prepares train/val/test CSVs."""
        # get path to labels CSV
        labels_csv = csv_labels_path(job_id)
        if not labels_csv.exists():
            raise FileNotFoundError("Labels CSV not found; call download_data_step first.")
        df = pd.read_csv(labels_csv, names=["image_filename", "labels"], header=0)

        # split labels into train/val/test dfs
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["labels"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["labels"], random_state=42)

        # store labels CSVs
        train_csv = train_csv_path(job_id)
        val_csv = val_csv_path(job_id)
        test_csv = test_csv_path(job_id)
        train_csv.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_csv, index=False, header=True)
        val_df.to_csv(val_csv, index=False, header=True)
        test_df.to_csv(test_csv, index=False, header=True)

        return {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "counts": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
        }

    def train_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """The main training loop for the image classification model."""
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
        model = model.to(device)

        # transforms
        train_transform, eval_transform = select_transforms(config["model_name"])
        if weights is not None:
            eval_transform = weights.transforms()

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

        is_macos = (platform.system() == "Darwin")
        pin_memory = (device.type == "cuda")
        nw = 0 if (is_macos or device.type == "mps") else 4
        batch_size = config.get("batch_size") or 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin_memory)

        # loss / optimizer
        criterion = make_criterion(config)
        optimizer = make_optimizer(model.parameters(), config)

        # training config
        num_epochs = config["num_epochs"]
        patience = config["patience"]
        track_metric = config["track_metric"]

        # metrics CSV header
        metrics_csv = metrics_csv_path(job_id)
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_csv.exists():
            with open(metrics_csv, "w") as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,macro_f1,micro_f1,tracked\n")

        best_val = -float("inf") if track_metric in ("val_acc", "macro_f1", "micro_f1") else float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        logging.info("Starting training")
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_metrics = evaluate(classes, model, val_loader, criterion, device)

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

            is_better = self._metric_is_better(track_metric, tracked, best_val)
            if is_better:
                best_val = tracked
                best_epoch = epoch
                epochs_no_improve = 0
                bm_path = best_model_path(job_id)
                bm_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric_name": track_metric,
                        "best_metric_value": best_val,
                        "class_to_idx": class_to_idx,
                        "classes": classes,
                        "config": config,
                    },
                    bm_path,
                )
                logging.info(f"New best checkpoint on {track_metric}={best_val:.4f} at epoch {epoch}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch} (no improvement on {track_metric} for {patience} epochs)")
                    break

        return {
            "best_epoch": best_epoch,
            "best_metric": track_metric,
            "best_value": best_val,
            "artifacts": {
                "best_model": str(best_model_path(job_id)),
                "metrics_csv": str(metrics_csv),
            },
            "test_batch_size": batch_size,
        }

    def evaluate_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Evaluates the best trained model on the test set."""
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

        # Eval transforms: start from defaults and prefer official pretrained transforms if weights were used
        _, eval_transform = select_transforms(model_name)
        if model_weights != "none":
            _m, w = make_model(model_name, model_weights, num_classes=len(classes))
            if w is not None:
                eval_transform = w.transforms()

        # DataLoader for test set
        class_to_idx = {name: i for i, name in enumerate(classes)}
        test_dataset = CocoImageDataset(
            csv_file=test_csv_path(job_id),
            root_dir=data_dir(job_id),
            transform=eval_transform,
            class_to_idx=class_to_idx,
        )
        batch_size = config.get("batch_size") or 32
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        # Rebuild model (no pretrained weights now), load best state, eval mode
        model, _ = make_model(model_name, "none", num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Criterion from saved config when possible for consistency
        crit_cfg = cfg_ckpt if ("criterion_name" in cfg_ckpt or "optimizer_name" in cfg_ckpt) else config
        criterion = make_criterion(crit_cfg)

        # Evaluate
        test_loss, test_acc, test_metrics = evaluate(classes, model, test_loader, criterion, device)

        report_dict = test_metrics.get("classification_report_dict")
        if report_dict:
            # We save the dictionary report to a new JSON file
            json_path = test_report_json_path(job_id)
            save_json(report_dict, json_path.parent, json_path.name)

        cm_path = test_cm_path(job_id)
        if "confusion_matrix" in test_metrics:
            cm = np.array(test_metrics["confusion_matrix"])
            header = ",".join(classes)
            np.savetxt(cm_path, cm, delimiter=",", fmt="%d", header=header, comments="")

        pdf_path = create_classification_report(job_id)

        return {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
            "test_micro_f1": float(test_metrics.get("micro_f1", 0.0)),
            "artifacts": {
                "test_report_json": str(json_path),
                "test_confusion_matrix": str(cm_path),
                "report_pdf": str(pdf_path),
            },
        }
    
    # ======================================================================
    # Model Loading
    # ======================================================================

    def load_model_step(self, job_id: str) -> Dict[str, Any]:
        """Loads a trained model (by job_id) into the centralized cache."""
        
        # --- CHANGE 2a: Check the centralized cache first ---
        key = f"{job_id}"
        cached_bundle = MODEL_CACHE_MANAGER.get_model_bundle(key)
        
        if cached_bundle:
            return {
                "status": "loaded from cache", 
                "job_id": job_id, 
                "num_classes": len(cached_bundle["classes"]),
                "classes": cached_bundle["classes"],
                "device": str(cached_bundle["device"]),
            }

        # Proceed with loading from disk if not cached
        device = self._choose_device()
        bm_path = best_model_path(job_id)

        if not bm_path.exists():
            raise FileNotFoundError(f"No trained model found for {job_id}")

        ckpt = torch.load(bm_path, map_location=device)
        classes = ckpt.get("classes")
        config = ckpt.get("config", {})
        model_name = config.get("model_name")
        
        model_weights = config.get("model_weights", "default")
        model, weights_obj = make_model(model_name, "none", num_classes=len(classes)) # 'none' to avoid downloading again

        # 2. Get the correct transforms
        if weights_obj is not None and model_weights != "none":
            # If training used official weights, use their official transforms
            eval_transform = weights_obj.transforms()
        else:
            # Otherwise, use the generic pipeline transform
            _, eval_transform = select_transforms(model_name)

        # 3. Load state dict, set device, and eval mode
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # Store the model bundle in the centralized cache
        bundle = {
            "model": model,
            "device": device,
            "classes": classes,
            "transform": eval_transform,
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

        with torch.no_grad():
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
