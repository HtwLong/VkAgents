from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer 
import torch.optim as optim
import torch.nn as nn
from typing import List, Literal, Optional, Dict, Any, Union
import inspect
import yaml
import json
import shutil
import os
from agents import function_tool

# --- Import all path functions from your paths.py ---
from cvmodellearning.paths import (
    run_dir, 
    artifacts_dir, 
    tool_call_args_path, 
    training_log_path, 
    best_model_path, 
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


# --- 1. Custom Trainer Class ---
class FlexibleYOLODetector(DetectionTrainer):
    """
    Overrides the DetectionTrainer to allow passing arbitrary, optimizer-specific arguments 
    for any Ultralytics-based YOLO version (v8, v10, v11, v12).
    """
    def build_optimizer(self, model, name='auto', lr=0.01, momentum=0.937, weight_decay=0.0005, **kwargs):
        g = [], [], []  # weight_decay, no_weight_decay, bias
        for p in model.modules():
            if hasattr(p, 'bias') and p.bias is not None:
                g[2].append(p.bias)
            if isinstance(p, nn.BatchNorm2d):
                g[1].append(p.weight)
            elif hasattr(p, 'weight') and p.weight is not None:
                g[0].append(p.weight)

        OptimizerClass = {
            'sgd': optim.SGD,
            'adamw': optim.AdamW,
            'rmsprop': optim.RMSprop,
            'auto': optim.AdamW
        }.get(name.lower(), optim.AdamW)

        core_args = {'lr': lr, 'weight_decay': weight_decay}
        sig = inspect.signature(OptimizerClass)
        optimizer_args = {}

        if 'momentum' in sig.parameters:
            optimizer_args['momentum'] = momentum
        elif 'betas' in sig.parameters:
            beta1 = momentum
            beta2 = kwargs.pop('beta2', 0.999) 
            optimizer_args['betas'] = (beta1, beta2)

        if name.lower() == 'rmsprop' and 'alpha' in sig.parameters:
            optimizer_args['alpha'] = kwargs.pop('alpha', 0.99)
            
        if 'eps' in sig.parameters:
            optimizer_args['eps'] = kwargs.pop('eps', 1e-8)
            
        final_args = {**core_args, **optimizer_args}
        return OptimizerClass(g[0], **final_args), g

# --- 2. The Flexible Wrapper Function ---

MODEL_BASE_MAP = {
    'yolo_v8': 'yolov8',
    'yolo_v10': 'yolov10',
    'yolo_v11': 'yolo11', # Correct mapping (no 'v')
    'yolo_v12': 'yolo12', # Correct mapping (no 'v')
}

type YoloVersionLiteral = Literal['yolo_v8', 'yolo_v10', 'yolo_v11', 'yolo_v12']
type YoloSizeLiteral = Literal['n', 's', 'm', 'l', 'x']
type OptimizerLiteral = Literal['SGD', 'AdamW', 'RMSprop']

def ensure_model_exists(model_name: str):
    """
    Checks if the model file exists locally. If not, explicitly triggers a download.
    This prevents the 'No such file' error when Ultralytics fails to auto-download custom strings.
    """
    if Path(model_name).exists():
        return

    print(f"📥 Model '{model_name}' not found locally. Attempting explicit download...")
    try:
        # Initializing YOLO() with a .pt file usually triggers a download if missing
        YOLO(model_name)
        print(f"✅ Download complete: {model_name}")
    except Exception as e:
        print(f"⚠️ Could not auto-download {model_name}. Please ensure internet access or place the file manually.")
        print(f"Error details: {e}")

def flexible_yolo_training(
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
    fliplr: float,
    scale: float,
    degrees: float,
    hsv_h: float,
    close_mosaic: int,
    freeze: Optional[int] = None, 
    **optimizer_kwargs: Dict[str, Any]
):
    
    ULTRALYTICS_PROJECT_ROOT = str(run_dir(job_id)) 
    ULTRALYTICS_RUN_NAME = 'temp_run'
    ultralytics_output_dir = Path(ULTRALYTICS_PROJECT_ROOT) / 'detect' / ULTRALYTICS_RUN_NAME
    
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
    ensure_model_exists(final_model_name)
    
    print(f"Loading model: {final_model_name}...")
    
    try:
        model = YOLO(final_model_name)
        model.trainer = FlexibleYOLODetector
        model.args = {**model.args, **optimizer_kwargs}

        device = select_ultralytics_device_string()
        model.add_callback('on_train_epoch_end', lambda trainer: progress_update_callback(trainer, job_id=job_id))
        
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
            fliplr=fliplr,
            scale=scale,
            degrees=degrees,
            hsv_h=hsv_h,
            close_mosaic=close_mosaic,
            freeze=freeze, 
        )
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
        ultralytics_best_pt = actual_run_dir / 'weights' / 'best.pt'
        if ultralytics_best_pt.exists():
            shutil.move(ultralytics_best_pt, target_model_path)
            print(f"✅ Best model moved to: {target_model_path}")
        else:
            print(f"❌ Warning: Could not find {ultralytics_best_pt}")
            weights_dir = actual_run_dir / 'weights'
            if weights_dir.exists():
                pt_files = list(weights_dir.glob("*.pt"))
                if pt_files:
                    print(f"⚠️ 'best.pt' missing, taking {pt_files[0].name} instead.")
                    shutil.move(str(pt_files[0]), str(target_model_path))
        
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

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise e # Re-raise to ensure the agent knows it failed

def convert_json_to_yolo_txt(json_path: Path, output_labels_dir: Path, categories: List[Dict]):
    """
    Converts COCO JSON to YOLO txt. Uses 'image_path' or 'file_name' to generate 
    a flattened label name that perfectly matches the flattened image name.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    images_info = {img['id']: img for img in data['images']}
    
    sorted_cats = sorted(categories, key=lambda x: x['id'])
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(sorted_cats)}
    
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    for img_id, img_data in images_info.items():
        img_w = img_data['width']
        img_h = img_data['height']
        
        # 1. Grab the most complete path available in your JSON
        rel_path_str = img_data.get('image_path', img_data.get('file_name'))
        
        # 2. Flatten it by replacing slashes with underscores
        flattened_name = str(Path(rel_path_str).as_posix()).replace('/', '_')
        txt_name = Path(flattened_name).stem + ".txt"
        
        txt_path = output_labels_dir / txt_name
        
        with open(txt_path, 'w') as f_txt:
            if img_id in img_annotations:
                for ann in img_annotations[img_id]:
                    x_min, y_min, w, h = ann['bbox']
                    original_cat_id = ann['category_id']

                    if original_cat_id not in cat_id_to_index:
                        continue

                    final_class_idx = cat_id_to_index[original_cat_id]

                    x_center = (x_min + w / 2) / img_w
                    y_center = (y_min + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    f_txt.write(f"{final_class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                    
def create_yolo_data_yaml(
    job_id: str, 
    data_dir_path: Path, 
    categories: List[Dict[str, Union[int, str]]], 
) -> Path:
    train_json = data_dir_path / "train_annotations.json"
    val_json = data_dir_path / "val_annotations.json"
    test_json = data_dir_path / "test_annotations.json"

    images_dir = data_dir_path / "images"
    labels_dir = data_dir_path / "labels"
    
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    print("🔄 Reorganizing data with Full Path Flattening to avoid collisions...")
    
    for item in data_dir_path.rglob('*'):
        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            
            # Ensure we aren't moving files already inside the destination directory
            if item.parent.resolve() != images_dir.resolve():
                
                # Get the path relative to the root data folder
                rel_path = item.relative_to(data_dir_path)
                
                # Replace slashes with underscores to create the flattened name
                new_filename = str(rel_path.as_posix()).replace('/', '_')
                new_filepath = images_dir / new_filename
                
                try:
                    shutil.move(str(item), str(new_filepath))
                except shutil.Error as e:
                    print(f"⚠️ Could not move {item.name}: {e}")

    # Remove empty subfolders to clean up
    for dir_path in sorted(data_dir_path.rglob('*'), key=lambda x: len(x.parts), reverse=True):
        if dir_path.is_dir() and dir_path not in [images_dir, labels_dir]:
            try:
                dir_path.rmdir() 
            except OSError:
                pass 

    print("🔄 Converting COCO JSON to YOLO TXT format...")
    if train_json.exists():
        convert_json_to_yolo_txt(train_json, labels_dir, categories)
    if val_json.exists():
        convert_json_to_yolo_txt(val_json, labels_dir, categories)
    if test_json.exists():
        convert_json_to_yolo_txt(test_json, labels_dir, categories)
        
    print(f"✅ Data preparation complete. Images in {images_dir}, Labels in {labels_dir}")

    class_names = [c['name'] for c in sorted(categories, key=lambda x: x['id'])]
    names_dict = {i: name for i, name in enumerate(class_names)}

    yaml_data = {
        'path': str(data_dir_path.resolve()), 
        'train': "images", 
        'val': "images",
        'test': "images",
        'nc': len(class_names),
        'names': names_dict
    }

    yaml_path = yolo_data_yaml_path(job_id)
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, indent=4, sort_keys=False)
        
    print(f"✅ YOLO Data YAML created successfully at: {yaml_path}")
    return yaml_path

def select_ultralytics_device_string() -> Optional[Union[str, int]]:
    if torch.cuda.is_available():
        return '0' 
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return None


@function_tool(strict_mode=True)
def train_yolo_model(
    job_id: str, 
    model_version: YoloVersionLiteral, 
    model_size: YoloSizeLiteral, 
    epochs: int,
    batch: int,
    imgsz: int,
    optimizer: OptimizerLiteral,
    lr0: float,
    momentum: float, 
    weight_decay: float,
    patience: int, 
    lrf: float,
    warmup_epochs: float,
    warmup_momentum: float,
    box: float, 
    cls: float, 
    dfl: float, 
    mosaic: float, 
    mixup: float, 
    fliplr: float, 
    scale: float, 
    degrees: float, 
    hsv_h: float, 
    close_mosaic: int, 
    freeze: Optional[int] = None, 
    workers: int = 8, 
    optimizer_override_json: str = "{}"
) -> str:
    """
    Configures and initiates object detection training for Ultralytics YOLO models.
    """

    call_args = locals()
    audit_file_path = tool_call_args_path(job_id)
    try:
        with open(audit_file_path, 'w') as f:
            json.dump(call_args, f, indent=4)
        print(f"✅ Tool call arguments saved to: {audit_file_path}")
    except Exception as e:
        print(f"❌ Warning: Could not save tool call arguments for audit: {e}")

    try:
        flexible_kwargs: Dict[str, Any] = json.loads(optimizer_override_json)
    except json.JSONDecodeError as e:
        return f"Error: Failed to parse optimizer_override_json. It must be valid JSON: {e}"

    try:
        flexible_yolo_training(
            job_id=job_id,
            model_version=model_version,
            model_size=model_size,
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
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
            fliplr=fliplr,
            scale=scale,
            degrees=degrees,
            hsv_h=hsv_h,
            close_mosaic=close_mosaic,
            freeze=freeze, 
            **flexible_kwargs 
        )
        return f"✅ Successfully initiated YOLO training for {model_version} size {model_size} (Job ID: {job_id}). Artifacts moved."
        
    except Exception as e:
        return f"❌ YOLO Training failed with a runtime error for Job ID {job_id}: {e}"

def evaluate_yolo_model(batch_size: int, job_id: str) -> Dict[str, Union[float, str]]:
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