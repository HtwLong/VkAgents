import os
from agents import function_tool
import torch
import json
import time
from typing import Dict, Any, List, Literal, Tuple, Union
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.datasets import CocoDetection 
from torchvision.transforms import v2 as T
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
from PIL import Image

from cvmodellearning.paths import (
    run_dir, data_dir, metrics_json_path, best_model_path, 
    training_log_path, train_json_path, val_json_path, test_json_path, 
    tool_call_args_path
)

# --- Configuration and Type Definitions ---

TVModel = Literal['retinanet_r50_fpn', 'faster_rcnn_r50_fpn', 'mask_rcnn_r50_fpn']
MonitorMetric = Literal['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']

TV_MODEL_NAME_MAP: Dict[str, TVModel] = {
    "retinanet_r50": 'retinanet_r50_fpn',
    "faster_rcnn_r50": 'faster_rcnn_r50_fpn',
    "mask_rcnn_r50": 'mask_rcnn_r50_fpn',
}

# --- 1. Model Adaptation ---

def get_detection_model(model_name: TVModel, num_classes: int, pre_trained: bool = True) -> torch.nn.Module:
    """Loads and adapts a TorchVision model for N classes."""
    weights = "DEFAULT" if pre_trained else None
    
    if model_name == 'faster_rcnn_r50_fpn':
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    elif model_name == 'mask_rcnn_r50_fpn':
        model = maskrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = model.roi_heads.mask_predictor.mask_fcn5.out_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
    elif model_name == 'retinanet_r50_fpn':
        model = retinanet_resnet50_fpn(weights=weights)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetHead(in_channels=256, num_anchors=num_anchors, num_classes=num_classes)
        
    else:
        # Fallback to general error if an unsupported model name gets through the type check
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

# --- 2. Data Loading (COCO Format) ---

def get_data_loaders(job_id: str, batch_size: int) -> Tuple[DataLoader, DataLoader, COCO]:
    """Sets up PyTorch DataLoaders using the TorchVision CocoDetection class."""
    
    data_root = str(data_dir(job_id))
    
    # Standard transforms for detection
    detection_transforms = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CocoDetection(root=data_root, annFile=str(train_json_path(job_id)), transforms=detection_transforms)
    val_dataset = CocoDetection(root=data_root, annFile=str(val_json_path(job_id)), transforms=detection_transforms)
    
    # Custom collate function to handle variable number of targets per image
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    coco_gt_val = COCO(str(val_json_path(job_id)))
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    return train_loader, val_loader, coco_gt_val

# --- 3. COCO Evaluation ---

def evaluate_coco_metrics(model: torch.nn.Module, data_loader: DataLoader, coco_gt: COCO, device: torch.device) -> Dict[str, float]:
    """Runs prediction and calculates all COCO bounding box metrics using pycocotools."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i_batch, (images, targets_tuple) in enumerate(data_loader):
            
            images = list(img.to(device) for img in images)
            outputs = model(images)

            # targets_list is the list of dictionaries, one per image in the batch
            targets_list = targets_tuple
            
            # Loop through the outputs and corresponding ground truth targets
            for i, output in enumerate(outputs):
                # Extract image_id from the ground truth target dictionary
                try:
                    image_id = targets_list[i]['image_id'].item()
                except Exception:
                    # Fallback if targets structure is unexpected
                    print("Warning: Could not reliably extract image_id from targets.")
                    continue
                
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Convert boxes from [x1, y1, x2, y2] to COCO [x, y, w, h] format
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1] 

                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.001:
                        results.append({
                            "image_id": int(image_id),
                            "category_id": int(label),
                            "bbox": box.tolist(),
                            "score": float(score),
                        })
    
    if not results:
        print("Warning: COCO evaluation failed - No predictions were generated.")
        return {'coco/bbox_mAP': 0.0, 'coco/bbox_mAP_50': 0.0, 'coco/bbox_mAP_75': 0.0}

    # Use COCO API to evaluate
    tmp_results_file = Path("tmp_results_eval.json")
    with open(tmp_results_file, 'w') as f:
        json.dump(results, f)
        
    coco_dt = coco_gt.loadRes(str(tmp_results_file))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    tmp_results_file.unlink()

    metrics = {
        'coco/bbox_mAP': np.float64(coco_eval.stats[0]).item(),
        'coco/bbox_mAP_50': np.float64(coco_eval.stats[1]).item(),
        'coco/bbox_mAP_75': np.float64(coco_eval.stats[2]).item(),
    }
    return metrics

# Helper function for device
def _choose_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

# --- 4. Flexible Training Loop ---

def flexible_torchvision_training(
    model_name: TVModel, num_classes: int, batch_size: int, learning_rate: float,
    epochs: int, monitor_metric: MonitorMetric, patience: int, 
    save_best_model: bool, job_id: str, **kwargs: Dict[str, Any]
):
    """Main training function implementing the full PyTorch workflow with COCO evaluation."""
    
    # 0. Setup Paths and Device
    custom_best_model_path = str(best_model_path(job_id))
    custom_training_log_path = str(training_log_path(job_id))
    Path(custom_best_model_path).parent.mkdir(parents=True, exist_ok=True)
    device = _choose_device()
    print(f"Using device: {device}")

    progress_file_path = run_dir(job_id) / "progress.json"

    # 1. Load Model and Data
    model = get_detection_model(model_name, num_classes, pre_trained=True)
    model.to(device)
    train_loader, val_loader, coco_gt_val = get_data_loaders(job_id, batch_size)
    
    # 2. Setup Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # 3. Early Stopping and Checkpoint Variables
    best_val_metric = -float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    
    # --- Training Loop ---
    with open(custom_training_log_path, 'w') as log_file:
        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            epoch_loss = 0.0
            
            for i, (images_tuple, targets_tuple) in enumerate(train_loader):
                images = list(img.to(device) for img in images_tuple)
                # Targets are extracted from the list tuple returned by collate_fn
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets_tuple]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()
                
            avg_train_loss = epoch_loss / len(train_loader)
            
            # --- Validation Phase ---
            val_metrics = evaluate_coco_metrics(model, val_loader, coco_gt_val, device)
            current_val_metric = val_metrics.get(monitor_metric, -float('inf'))
            
            # --- Logging ---
            log_message = (
                f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | "
                f"Val Metric ({monitor_metric}): {current_val_metric:.4f} | "
                f"mAP@50: {val_metrics.get('coco/bbox_mAP_50', 0.0):.4f} | Time: {time.time() - start_time:.2f}s"
            )
            print(log_message)
            log_file.write(log_message + '\n')

            # --- PROGRESS TRACKING: Write current status to file --- <--- CRITICAL ADDITION
            progress_data = {
                "status": "running",
                "current_epoch": epoch + 1,
                "total_epochs": epochs,
                "train_loss": avg_train_loss,
                "val_mAP": val_metrics.get('coco/bbox_mAP'),
                "val_mAP50": val_metrics.get('coco/bbox_mAP_50'),
            }
            try:
                with open(progress_file_path, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                print(f"Warning: Could not write progress file {progress_file_path}: {e}")
            
            # --- Early Stopping/Checkpointing Logic ---
            if current_val_metric > best_val_metric:
                print(f"🔥 Metric improved from {best_val_metric:.4f} to {current_val_metric:.4f}. Saving checkpoint.")
                best_val_metric = current_val_metric
                epochs_no_improve = 0
                if save_best_model:
                    torch.save(model.state_dict(), custom_best_model_path)
            else:
                epochs_no_improve += 1
            
            if patience > 0 and epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs with no improvement on {monitor_metric}.")
                break
                
    print(f"Training finished. Best model saved to: {custom_best_model_path}")

# --- 5. Agent Callable Functions ---
@function_tool(strict_mode=True)
def train_torchvision_model(
    model_name: str, num_classes: int, batch_size: int, learning_rate: float,
    epochs: int, monitor_metric: MonitorMetric, patience: int, 
    save_best_model: bool, job_id: str, config_override_json: str = "{}" 
) -> str:
    """
    Executes flexible PyTorch/TorchVision object detection training (Faster R-CNN, Mask R-CNN, RetinaNet) 
    using a custom COCO-style workflow and manages checkpointing and early stopping.
    
    This function is the primary entry point for the Agent to launch TorchVision training.
    
    Args:
        model_name (str): 
            The name of the model architecture to train. Must be one of: 
            'retinanet_r50', 'faster_rcnn_r50', or 'mask_rcnn_r50'.
        num_classes (int): 
            The total number of classes in the dataset, which MUST include the background class (N + 1).
        batch_size (int): 
            The number of samples per batch for training and validation.
        learning_rate (float): 
            The initial learning rate for the SGD optimizer (e.g., 0.001).
        epochs (int): 
            The total number of training epochs to run.
        monitor_metric (Literal['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']): 
            The COCO bounding box metric used to track improvement, decide on early stopping, 
            and determine the best checkpoint.
        patience (int): 
            Number of epochs with no improvement on the 'monitor_metric' to wait before 
            triggering early stopping. Use 0 to disable early stopping.
        save_best_model (bool): 
            If True, the model state_dict that achieved the best 'monitor_metric' on 
            the validation set will be saved.
        job_id (str): 
            The unique identifier for the training job, used for resolving data paths and 
            artifact saving locations.
        config_override_json (str, optional): 
            A JSON string containing extra configuration arguments to override defaults 
            in the underlying training logic (e.g., custom optimizer parameters or 
            learning rate scheduler settings). Defaults to "{}".

    Returns:
        str: A message indicating the success of the training initiation or the failure reason.
    """
    
    call_args = locals()
    
    try:
        tv_model_name = TV_MODEL_NAME_MAP[model_name]
    except KeyError:
        return f"Error: Unknown model name: {model_name}. Must be one of {list(TV_MODEL_NAME_MAP.keys())}"
        
    audit_file_path = tool_call_args_path(job_id)
    try:
        with open(audit_file_path, 'w') as f:
            json.dump(call_args, f, indent=4)
        print(f"✅ Tool call arguments saved to: {audit_file_path}")
    except Exception as e:
        print(f"❌ Warning: Could not save tool call arguments for audit: {e}")

    try:
        cfg_overrides: Dict[str, Any] = json.loads(config_override_json)
    except json.JSONDecodeError as e:
        return f"Error: Failed to parse config_override_json. It must be valid JSON: {e}"
        
    try:
        flexible_torchvision_training(
            model_name=tv_model_name, num_classes=num_classes, batch_size=batch_size,
            learning_rate=learning_rate, epochs=epochs, monitor_metric=monitor_metric,
            patience=patience, save_best_model=save_best_model, job_id=job_id,
            **cfg_overrides 
        )
        return (f"Successfully initiated TorchVision training for **{model_name}** (Job ID: {job_id}). "
                f"Monitoring metric: **{monitor_metric}**. Checkpoint saved to: {best_model_path(job_id)}.")
        
    except Exception as e:
        return f"Training failed with a runtime error for {tv_model_name}: {e}"


def evaluate_torchvision_model(
    model_name: str, num_classes: int, job_id: str, batch_size: int = 1
) -> Dict[str, Any]:
    """
    Performs final evaluation on the test dataset using the saved TorchVision checkpoint.
    """
    if not job_id:
        return {"error": "job_id must be provided for evaluation path resolution."}

    try:
        tv_model_name = TV_MODEL_NAME_MAP[model_name]
    except KeyError:
        return {"error": f"Unknown model name: {model_name}"}

    device = _choose_device()
    checkpoint_path = str(best_model_path(job_id))
    test_metrics_file_path = str(metrics_json_path(job_id))
    
    if not Path(checkpoint_path).exists():
        return {"error": f"Model checkpoint not found at: {checkpoint_path}"}
        
    # 1. Load Model and Weights
    model = get_detection_model(tv_model_name, num_classes, pre_trained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # 2. Setup Test DataLoader
    data_root = str(data_dir(job_id))
    test_ann_file = str(test_json_path(job_id))
    
    test_dataset = CocoDetection(
        root=data_root,
        annFile=test_ann_file,
        transforms=T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    )
    def collate_fn(batch):
        return tuple(zip(*batch))
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    coco_gt_test = COCO(test_ann_file)

    # 3. Perform Evaluation
    print(f"Starting final evaluation for job: {job_id}")
    final_metrics = evaluate_coco_metrics(model, test_loader, coco_gt_test, device)

    with open(test_metrics_file_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Final Evaluation Metrics saved to: {test_metrics_file_path}")
    return final_metrics

def load_torchvision_model_for_inference(
    model_name: str, 
    model_path: Path, 
    num_classes: int, 
    device: torch.device
) -> Tuple[torch.nn.Module, T.Compose]:
    """
    Instantiates a TorchVision model, loads weights, sets to eval mode, 
    and returns the model and its required inference transform.
    """
    model_name_lower = model_name.lower()
    
    # 1. Instantiate the base model architecture
    if model_name_lower in ("faster_rcnn", "fasterrcnn_resnet50_fpn"):
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    elif model_name_lower in ("retinanet", "retinanet_resnet50_fpn"):
        model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes)
    elif model_name_lower in ("mask_rcnn", "maskrcnn_resnet50_fpn"):
        # We need to manually ensure the MaskRCNN head is configured with the correct number of classes 
        # as done in the training function (using FastRCNNPredictor and MaskRCNNPredictor)
        # However, for simplicity here, we rely on the num_classes parameter if the function supports it.
        # For TorchVision models that have been customized, full model adaptation may be needed here.
        model_tv_literal = TV_MODEL_NAME_MAP.get(model_name)
        if model_tv_literal:
            model = get_detection_model(model_tv_literal, num_classes, pre_trained=False)
        else:
            raise ValueError(f"Unsupported torchvision model for loading: {model_name}")
    else:
        raise ValueError(f"Unsupported torchvision model for loading: {model_name}")
    
    # 2. Load the trained state_dict and set to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    # 3. Define the minimal inference transform: PIL Image to Tensor
    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform

def run_torchvision_inference(
    model: torch.nn.Module, 
    image: Image.Image, 
    transform: T.Compose,
    device: torch.device
) -> List[List[Union[float, int]]]:
    """
    Runs inference on a single PIL image using a TorchVision model and returns 
    detections in the format: [x_min, y_min, x_max, y_max, score, class_id].
    """
    detections: List[List[Union[float, int]]] = []
    SCORE_THRESHOLD = 0.5 

    # 1. Apply Transform: PIL Image -> Tensor, Add Batch Dim, Move to Device
    image_tensor = transform(image).to(device)
    input_list = [image_tensor]
    
    # 2. Run Inference
    with torch.no_grad():
        # Output is a list of dicts: [{'boxes': tensor, 'labels': tensor, 'scores': tensor}]
        outputs = model(input_list) 
    
    # 3. Parse and Format Output
    if outputs and len(outputs) > 0:
        output = outputs[0]
        
        boxes = output['boxes'].cpu().tolist()
        scores = output['scores'].cpu().tolist()
        labels = output['labels'].cpu().tolist()
        
        # Combine and filter
        for box, score, label in zip(boxes, scores, labels):
            if score >= SCORE_THRESHOLD:
                 # Standard format: [x_min, y_min, x_max, y_max, score, class_id]
                 detections.append([*box, score, float(label)])
             
    return detections