from pathlib import Path
import torch
from ultralytics import YOLO
from typing import List, Literal, Optional, Dict, Any, Union, Mapping
import yaml
import json
import shutil
from cvmodellearning.download.image_cache import link_or_copy
from cvmodellearning.jobs.run_control import PipelineCancelled, cancellation_requested, raise_if_cancelled
from cvmodellearning.training.hardware import detect_training_backend

# --- Import all path functions from your paths.py ---
from cvmodellearning.paths import (
    PROJECT_ROOT,
    run_dir, 
    tool_call_args_path, 
    training_log_path, 
    yolo_data_yaml_path, 
    best_yolo_model_path,
    plots_dir
)

def progress_update_callback(trainer, job_id: str):
    """Writes the current epoch progress and key metrics to a JSON file."""
    progress_file = run_dir(job_id) / "progress.json"
    current_epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    
    # Validation metrics are in trainer.metrics.
    metrics = getattr(trainer, 'metrics', {})
    
    # Try to get training loss.
    train_loss = 0.0
    if hasattr(trainer, 'loss_items') and isinstance(trainer.loss_items, torch.Tensor):
        train_loss = trainer.loss_items[0].item() if trainer.loss_items.numel() > 0 else 0.0

    progress_data = {
        "status": "running",
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "train_loss": train_loss, 
        "val_mAP50": metrics.get('metrics/mAP50(B)', 0.0),
    }

    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not write progress file {progress_file}: {e}")


# --- Training wrapper ---

MODEL_BASE_MAP = {
    'yolo_v8': 'yolov8',
    'yolo_v10': 'yolov10',
    'yolo_v11': 'yolo11', # Correct mapping (no 'v')
    'yolo_v12': 'yolo12', # Correct mapping (no 'v')
}

MPS_CPU_FALLBACK_VERSIONS = {"yolo_v11", "yolo_v12"}

type YoloVersionLiteral = Literal['yolo_v8', 'yolo_v10', 'yolo_v11', 'yolo_v12']
type YoloSizeLiteral = Literal['n', 's', 'm', 'l', 'x']
type OptimizerLiteral = Literal['auto', 'SGD', 'AdamW', 'RMSProp']

def ensure_model_exists(model_name: str) -> str:
    """
    Checks if the model file exists locally. If not, explicitly triggers a download.
    This prevents the 'No such file' error when Ultralytics fails to auto-download custom strings.
    """
    if Path(model_name).exists():
        return model_name
    bundled_model = PROJECT_ROOT / "src" / model_name
    if bundled_model.exists():
        return str(bundled_model)

    print(f"📥 Model '{model_name}' not found locally. Attempting explicit download...")
    try:
        # Initializing YOLO() with a .pt file usually triggers a download if missing
        YOLO(model_name)
        print(f"✅ Download complete: {model_name}")
        return model_name
    except Exception as e:
        print(f"⚠️ Could not auto-download {model_name}. Please ensure internet access or place the file manually.")
        print(f"Error details: {e}")
        raise FileNotFoundError(f"Pretrained checkpoint is unavailable: {model_name}") from e


def _move_best_checkpoint(run_directory: Path, target_path: Path) -> None:
    """Move the best available checkpoint or fail the training run explicitly."""
    weights_directory = run_directory / "weights"
    preferred = weights_directory / "best.pt"
    candidates = [preferred] if preferred.exists() else sorted(weights_directory.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"YOLO training produced no checkpoint in {weights_directory}."
        )
    if candidates[0] != preferred:
        print(f"⚠️ 'best.pt' missing, taking {candidates[0].name} instead.")
    shutil.move(str(candidates[0]), str(target_path))

def _run_yolo_training(
    job_id: str, 
    model_version: YoloVersionLiteral, 
    model_size: YoloSizeLiteral, 
    epochs: int,
    patience: int,
    imgsz: int,
    batch: int,
    workers: int,
    optimizer: OptimizerLiteral,
    lr0: float,
    lrf: float,
    momentum: float, 
    weight_decay: float,
    warmup_epochs: float,
    warmup_momentum: float,
    box: float,
    cls: float,
    dfl: float,
    mosaic: float,
    mixup: float,
    cutmix: float,
    copy_paste: float,
    fliplr: float,
    scale: float,
    degrees: float,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    translate: float,
    close_mosaic: int,
    single_cls: bool,
    rect: bool,
    multi_scale: float,
    amp: bool,
    seed: int,
    freeze: Optional[int] = None, 
):
    
    ULTRALYTICS_PROJECT_ROOT = str(run_dir(job_id)) 
    ULTRALYTICS_RUN_NAME = 'temp_run'
    ultralytics_output_dir = Path(ULTRALYTICS_PROJECT_ROOT) / ULTRALYTICS_RUN_NAME
    
    if ultralytics_output_dir.exists():
        print(f"Cleaning up previous run directory to ensure fresh start: {ultralytics_output_dir}")
        shutil.rmtree(ultralytics_output_dir)
        
    print(f"Ultralytics will save results temporarily to: {ultralytics_output_dir}")
    
    data_yaml_path = str(yolo_data_yaml_path(job_id))
    print(f"Using Data YAML path: {data_yaml_path}")

    try:
        base_name = MODEL_BASE_MAP[model_version]
        final_model_name = f'{base_name}{model_size}.pt'
    except KeyError:
        raise ValueError(f"Invalid model_version: {model_version}. Must be one of {list(MODEL_BASE_MAP.keys())}")
    
    # --- Ensure model exists before training ---
    final_model_name = ensure_model_exists(final_model_name)
    
    print(f"Loading model: {final_model_name}...")
    
    try:
        model = YOLO(final_model_name)

        device = select_yolo_training_device(model_version)
        def on_epoch_end(trainer):
            progress_update_callback(trainer, job_id=job_id)
            if cancellation_requested(job_id):
                trainer.stop = True
        model.add_callback('on_train_epoch_end', on_epoch_end)
        
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=ULTRALYTICS_PROJECT_ROOT,
            name=ULTRALYTICS_RUN_NAME,
            exist_ok=True,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            box=box,
            cls=cls,
            dfl=dfl,
            mosaic=mosaic,
            mixup=mixup,
            cutmix=cutmix,
            copy_paste=copy_paste,
            fliplr=fliplr,
            scale=scale,
            degrees=degrees,
            translate=translate,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            close_mosaic=close_mosaic,
            single_cls=single_cls,
            rect=rect,
            multi_scale=multi_scale,
            freeze=freeze,
            amp=amp,
            seed=seed,
            cos_lr=False,
            deterministic=True,
        )
        raise_if_cancelled(job_id)
        print(f"Training finished successfully for model {final_model_name}.")

        # Post-Training Artifact Relocation
        print("Moving artifacts to required paths...")

        if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
            actual_run_dir = Path(model.trainer.save_dir)
        else:
            actual_run_dir = Path(ULTRALYTICS_PROJECT_ROOT) / ULTRALYTICS_RUN_NAME

        print(f"Detected Ultralytics output directory: {actual_run_dir}")
        
        target_model_path = best_yolo_model_path(job_id)
        target_log_path = training_log_path(job_id)
        target_plots_dir = plots_dir(job_id)

        # A. Move Best Model
        _move_best_checkpoint(actual_run_dir, target_model_path)
        print(f"✅ Best model moved to: {target_model_path}")
        
        # B. Move Training Log
        ultralytics_results_csv = actual_run_dir / 'results.csv'
        if ultralytics_results_csv.exists():
            shutil.move(ultralytics_results_csv, target_log_path)
            print(f"✅ Training log moved to: {target_log_path}")
        else:
            print(f"❌ Warning: Could not find {ultralytics_results_csv}")

        # C. Copy Visualizations
        print(f"Moving training plots to: {target_plots_dir}")
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        files_moved_count = 0
        for ext in image_extensions:
            for img_file in actual_run_dir.glob(ext):
                try:
                    shutil.copy(img_file, target_plots_dir / img_file.name)
                    files_moved_count += 1
                except Exception as e:
                    print(f"Failed to copy plot {img_file.name}: {e}")
        
        print(f"✅ Copied {files_moved_count} visualization plots.")

    except PipelineCancelled:
        raise
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise e # Re-raise to ensure the agent knows it failed

def _safe_flattened_image_name(relative_path: str) -> str:
    normalized = Path(relative_path).as_posix().lstrip("/")
    if not normalized or ".." in Path(normalized).parts:
        raise ValueError(f"Unsafe image path in annotations: {relative_path!r}")
    return normalized.replace("/", "__")


def _source_image_path(data_dir_path: Path, relative_path: str) -> Path:
    path_value = Path(relative_path)
    if path_value.is_absolute() or ".." in path_value.parts:
        raise ValueError(f"Unsafe image path in annotations: {relative_path!r}")
    direct = data_dir_path / path_value
    if direct.is_file():
        return direct

    normalized = Path(relative_path).as_posix().lstrip("/")
    candidates = [
        path
        for path in data_dir_path.rglob(Path(normalized).name)
        if path.is_file()
        and "yolo_dataset" not in path.parts
        and path.as_posix().endswith(normalized)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Image referenced by annotations was not found: {relative_path}")
    raise ValueError(f"Image path is ambiguous: {relative_path}; matches={candidates}")


def convert_json_to_yolo_split(
    json_path: Path,
    data_dir_path: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    categories: List[Dict],
) -> None:
    """Materialize one COCO split as an isolated Ultralytics images/labels pair."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    sorted_cats = sorted(categories, key=lambda x: x['id'])
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(sorted_cats)}
    
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    used_label_names: set[str] = set()
    for img_data in data['images']:
        img_id = img_data['id']
        img_w = float(img_data['width'])
        img_h = float(img_data['height'])
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Image {img_id} has invalid dimensions: {img_w}x{img_h}")

        rel_path_str = img_data.get('image_path') or img_data.get('file_name')
        if not rel_path_str:
            raise ValueError(f"Image {img_id} has no image_path or file_name")
        flattened_name = _safe_flattened_image_name(str(rel_path_str))
        if flattened_name in used_names:
            raise ValueError(f"Flattened YOLO filename collision: {flattened_name}")
        used_names.add(flattened_name)
        label_name = f"{Path(flattened_name).stem}.txt"
        if label_name in used_label_names:
            raise ValueError(f"Flattened YOLO label collision: {label_name}")
        used_label_names.add(label_name)

        source_path = _source_image_path(data_dir_path, str(rel_path_str))
        link_or_copy(source_path, output_images_dir / flattened_name)
        txt_path = output_labels_dir / label_name

        with open(txt_path, 'w') as f_txt:
            for ann in img_annotations.get(img_id, []):
                x_min, y_min, width, height = map(float, ann['bbox'])
                original_cat_id = ann['category_id']
                if original_cat_id not in cat_id_to_index or width <= 0 or height <= 0:
                    continue

                x_max = min(img_w, max(0.0, x_min + width))
                y_max = min(img_h, max(0.0, y_min + height))
                x_min = min(img_w, max(0.0, x_min))
                y_min = min(img_h, max(0.0, y_min))
                width = x_max - x_min
                height = y_max - y_min
                if width <= 0 or height <= 0:
                    continue

                f_txt.write(
                    f"{cat_id_to_index[original_cat_id]} "
                    f"{(x_min + width / 2) / img_w:.6f} "
                    f"{(y_min + height / 2) / img_h:.6f} "
                    f"{width / img_w:.6f} {height / img_h:.6f}\n"
                )

                    
def create_yolo_data_yaml(
    job_id: str, 
    data_dir_path: Path, 
    categories: List[Dict[str, Union[int, str]]], 
) -> Path:
    train_json = data_dir_path / "train_annotations.json"
    val_json = data_dir_path / "val_annotations.json"
    test_json = data_dir_path / "test_annotations.json"

    yolo_root = data_dir_path / "yolo_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)

    split_json = {"train": train_json, "val": val_json, "test": test_json}
    for split_name, split_path in split_json.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split_name} annotations: {split_path}")
        convert_json_to_yolo_split(
            split_path,
            data_dir_path,
            yolo_root / "images" / split_name,
            yolo_root / "labels" / split_name,
            categories,
        )

    print(f"✅ Data preparation complete at {yolo_root}")

    class_names = [c['name'] for c in sorted(categories, key=lambda x: x['id'])]
    names_dict = {i: name for i, name in enumerate(class_names)}

    yaml_data = {
        'path': str(yolo_root.resolve()),
        'train': "images/train",
        'val': "images/val",
        'test': "images/test",
        'nc': len(class_names),
        'names': names_dict
    }

    yaml_path = yolo_data_yaml_path(job_id)
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, indent=4, sort_keys=False)
        
    print(f"✅ YOLO Data YAML created successfully at: {yaml_path}")
    return yaml_path

def select_ultralytics_device_string() -> Optional[Union[str, int]]:
    backend = detect_training_backend()
    if backend == "cuda":
        return "0"
    if backend == "mps":
        return "mps"
    return None


def select_yolo_training_device(model_version: YoloVersionLiteral) -> Optional[Union[str, int]]:
    """Use CPU for YOLO families known to fail during MPS detection training."""
    device = select_ultralytics_device_string()
    if device == "mps" and model_version in MPS_CPU_FALLBACK_VERSIONS:
        print(
            f"⚠️ {model_version} training is unreliable with Apple MPS; "
            "using CPU to avoid detection-loss tensor shape errors."
        )
        return "cpu"
    return device


def _yolo_version_and_size(model_name: str) -> tuple[YoloVersionLiteral, YoloSizeLiteral]:
    family, separator, size = model_name.lower().rpartition("_")
    version_map = {
        "yolov8": "yolo_v8",
        "yolov10": "yolo_v10",
        "yolov11": "yolo_v11",
        "yolov12": "yolo_v12",
    }
    if not separator or family not in version_map or size not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"Unsupported YOLO model_name: {model_name}")
    return version_map[family], size  # type: ignore[return-value]


def train_yolo_from_config(config: Mapping[str, Any], job_id: str) -> str:
    """Execute a validated HPO configuration without an LLM translation step."""
    flat = dict(config)
    optimizer_config = flat.get("optimizer")
    if isinstance(optimizer_config, Mapping):
        flat["optimizer_name"] = optimizer_config.get("name")
        params = optimizer_config.get("params") or {}
        if isinstance(params, Mapping):
            flat.update(params)

    model_version, model_size = _yolo_version_and_size(str(flat["model_name"]))
    optimizer_name = str(flat.get("optimizer_name", "auto")).lower()
    optimizer_map = {
        "auto": "auto",
        "adamw": "AdamW",
        "sgd": "SGD",
        "rmsprop": "RMSProp",
    }
    if optimizer_name not in optimizer_map:
        raise ValueError(f"Unsupported YOLO optimizer: {optimizer_name}")
    momentum = (
        float(flat.get("beta1", 0.9))
        if optimizer_name == "adamw"
        else float(flat.get("momentum", 0.9))
    )

    training_args = dict(
        model_version=model_version,
        model_size=model_size,
        epochs=int(flat["num_epochs"]),
        batch=int(flat.get("batch_size", 16)),
        imgsz=int(flat.get("input_size", 640)),
        optimizer=optimizer_map[optimizer_name],  # type: ignore[arg-type]
        lr0=float(flat.get("learning_rate", 0.01)),
        momentum=momentum,
        weight_decay=float(flat.get("weight_decay", 0.0005)),
        patience=int(flat.get("patience", 20)),
        lrf=float(flat.get("final_learning_rate_factor", 0.01)),
        warmup_epochs=float(flat.get("warmup_epochs", 3.0)),
        warmup_momentum=float(flat.get("warmup_momentum", 0.8)),
        box=float(flat.get("lambda_box", 7.5)),
        cls=float(flat.get("lambda_cls", 0.5)),
        dfl=float(flat.get("lambda_dfl", 1.5)),
        mosaic=float(flat.get("mosaic", 1.0)),
        mixup=float(flat.get("mixup", 0.0)),
        cutmix=float(flat.get("cutmix", 0.0)),
        copy_paste=float(flat.get("copy_paste", 0.0)),
        fliplr=float(flat.get("fliplr", 0.5)),
        scale=float(flat.get("scale", 0.5)),
        degrees=float(flat.get("degrees", 0.0)),
        translate=float(flat.get("translate", 0.1)),
        hsv_h=float(flat.get("hsv_h", 0.015)),
        hsv_s=float(flat.get("hsv_s", 0.7)),
        hsv_v=float(flat.get("hsv_v", 0.4)),
        close_mosaic=int(flat.get("close_mosaic", 10)),
        single_cls=bool(flat.get("single_cls", False)),
        rect=bool(flat.get("rect", False)),
        multi_scale=float(flat.get("multi_scale", 0.0)),
        freeze=flat.get("freeze"),
        workers=int(flat.get("workers", 8)),
        amp=bool(flat.get("amp", True)),
        seed=int(flat.get("seed", 0)),
    )
    audit_file_path = tool_call_args_path(job_id)
    audit_file_path.write_text(
        json.dumps({"job_id": job_id, **training_args}, indent=4),
        encoding="utf-8",
    )
    try:
        _run_yolo_training(job_id=job_id, **training_args)
    except Exception as exc:
        raise RuntimeError(f"YOLO training failed for job {job_id}: {exc}") from exc
    return f"✅ Successfully trained {flat['model_name']} (Job ID: {job_id})."


def evaluate_yolo_model(
    batch_size: int,
    image_size: int,
    job_id: str,
) -> Dict[str, Union[float, str]]:
    """
    Loads the best trained YOLO model and runs evaluation.
    """
    print(f"--- Starting Final Evaluation for Job ID: {job_id} ---")
    
    model_path = best_yolo_model_path(job_id)
    data_yaml_path = yolo_data_yaml_path(job_id)

    if not Path(model_path).exists():
        return {"error": f"❌ Best model not found at: {model_path}. Evaluation cannot proceed."}
    
    if not Path(data_yaml_path).exists():
        return {"error": f"❌ Data YAML not found at: {data_yaml_path}. Evaluation cannot proceed."}

    print(f"Loading model from: {model_path}")
    print(f"Using data config: {data_yaml_path}")
    
    try:
        model = YOLO(model_path)

        metrics = model.val(
            data=str(data_yaml_path),
            split='test',  
            batch=batch_size,
            imgsz=image_size,
            project=str(run_dir(job_id)),
            name="test_evaluation",
            exist_ok=True,
        )
        
        mAP50_95 = metrics.box.map
        mAP50 = metrics.box.map50
        mAP75 = metrics.box.map75
        
        results = {
            "mAP@.50:.95": mAP50_95,
            "mAP@.50": mAP50,
            "mAP@.75": mAP75,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "results_dir": str(Path(metrics.save_dir).resolve()) 
        }
        
        print("\n--- ✅ Evaluation Complete ---")
        print(f"mAP@.50:.95 (Overall): {mAP50_95:.4f}")
        print(f"Results saved to: {results['results_dir']}")
        
        return results

    except Exception as e:
        return {"error": f"❌ An error occurred during test evaluation for Job ID {job_id}: {e}"}
