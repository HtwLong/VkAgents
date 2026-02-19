import json
import os
from pathlib import Path
import random
import time
from typing import Dict, Any, List, Literal, Union

from agents import Runner

from PIL import Image
import torch
from ultralytics import YOLO

# project utils
from cvmodellearning.download.download_data import download_visionkg_mixed_datasets_detection
from cvmodellearning.agents.detection_agents import yolo_trainer_agent, torchvision_trainer_agent
from cvmodellearning.evaluation.report_builder import create_detection_report
from cvmodellearning.models.detection_models.yolo_trainer import evaluate_yolo_model
from cvmodellearning.models.detection_models.torchvision_trainer import evaluate_torchvision_model, load_torchvision_model_for_inference, run_torchvision_inference
# from cvmodellearning.models.detection_models.mmdet_trainer import evaluate_mmdet_model
# from cvmodellearning.models.detection_models.mmdet_trainer import MMDET_MODEL_NAME_MAP, MODEL_CONFIG_MAP
from cvmodellearning.inference.inference_utils import _draw_detections


from cvmodellearning.models.detection_models.yolo_trainer import create_yolo_data_yaml
from cvmodellearning.models.model_manager import MODEL_CACHE_MANAGER
from cvmodellearning.paths import (
    artifacts_dir,
    data_dir,
    hpo_config_path, # Used for loading job config
    test_json_path,
    training_log_path,
    best_model_path,
    report_pdf_path,
    json_labels_path,
    train_json_path,
    val_json_path,
    best_yolo_model_path
)


# --- Helper Functions (External to class, as they don't use instance state) ---

def _get_trainer_type(model_name: str) -> Literal["yolo", "torchvision"]:
    """Determine which training utility to use based on model name."""
    yolo_keywords = ('yolo', 'v8', 'v10', 'v11', 'v12')
    if any(k in model_name.lower() for k in yolo_keywords):
        return "yolo"
    
    torchvision_keywords = ('retinanet', 'faster_rcnn', 'mask_rcnn', 'ssd')
    if any(k in model_name.lower() for k in torchvision_keywords):
        return "torchvision"
    
    raise ValueError(f"Unknown model architecture for training: {model_name}")

def _get_device() -> str:
    """Selects the best available device for PyTorch."""
    if torch.cuda.is_available():
        return 'cuda:0'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def _get_category_map(job_id: str) -> Dict[int, str]:
    """Loads the category list from the consolidated JSON file."""
    annotations_path = train_json_path(job_id)  # Can use any split since categories are the same across all
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found at: {annotations_path}. Cannot load categories.")
        
    with open(annotations_path, 'r') as f:
        data = json.load(f)
        
    return {cat['id']: cat['name'] for cat in data.get('categories', [])}

def _get_inference_save_dir(job_id: str) -> Path:
    """Returns the dedicated directory for saving inference results."""
    save_dir = artifacts_dir(job_id) / 'inference_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

# ======================================================================
## Detection Pipeline Class
# ======================================================================

class DetectionPipeline:
    """
    Encapsulates the object detection workflow, including data handling, 
    training via agents, evaluation, and in-memory model inference.
    """

    def __init__(self):
        """Initializes the pipeline."""
        pass
    
    # --- Data Steps ---

    def download_data_step(self, config: Dict[str, Any], job_id: str):
        """Downloads data and create a consolidated COCO-style JSON label file."""

        selected_data: List[Dict[str, Any]] = config["selected_data"]

        data_base = data_dir(job_id)
        data_base.mkdir(parents=True, exist_ok=True)
                
        download_visionkg_mixed_datasets_detection(job_id, selected_data)

        return

    def prepare_data_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Splits the consolidated COCO detection data into train, validation, 
        and test sets, generating new JSON files for each split.
        """
        train_ratio = config.get("train_data_ratio")
        val_ratio = config.get("val_data_ratio")
        
        input_json_path = json_labels_path(job_id)
        
        with open(input_json_path, 'r') as f:
            data = json.load(f)

        uuid_to_int = {}
        for idx, img in enumerate(data['images']):
            # Preserve original ID in a new field just in case
            img['original_id'] = img['id']
            # New Integer ID
            new_id = idx + 1
            uuid_to_int[img['id']] = new_id
            img['id'] = new_id

        valid_annotations = []
        for ann in data.get('annotations', []):
            old_img_id = ann['image_id']
            if old_img_id in uuid_to_int:
                ann['image_id'] = uuid_to_int[old_img_id]
                # Ensure category_id exists
                if 'category_id' not in ann and 'category_id_seq' in ann:
                    ann['category_id'] = ann['category_id_seq']
                # Ensure numeric ID for annotation itself
                if not isinstance(ann['id'], int):
                    ann['id'] = int(ann['id']) if str(ann['id']).isdigit() else hash(str(ann['id'])) % 10000000
                
                valid_annotations.append(ann)

        data['annotations'] = valid_annotations

        # 1. Split the images
        images = data['images']
        random.shuffle(images)
        
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        train_images = images[:train_count]
        val_images = images[train_count : train_count + val_count]
        test_images = images[train_count + val_count :]
        
        
        train_ids = {img['id'] for img in train_images}
        val_ids = {img['id'] for img in val_images}
        test_ids = {img['id'] for img in test_images}
        
        categories = data.get('categories', [])
        info = data.get('info', {})
        
        # 2. Split the annotations
        all_annotations = data['annotations']
        train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_ids]
        val_annotations = [ann for ann in all_annotations if ann['image_id'] in val_ids]
        test_annotations = [ann for ann in all_annotations if ann['image_id'] in test_ids]

        # 3. Write the new JSON files
        def create_split_data(img_list, ann_list):
            return {
                'info': info,
                'categories': categories,
                'images': img_list,
                'annotations': ann_list,
            }

        train_json_path_obj = train_json_path(job_id)
        val_json_path_obj = val_json_path(job_id)
        test_json_path_obj = test_json_path(job_id)

        with open(train_json_path_obj, 'w') as f:
            json.dump(create_split_data(train_images, train_annotations), f, indent=4)
            
        with open(val_json_path_obj, 'w') as f:
            json.dump(create_split_data(val_images, val_annotations), f, indent=4)

        with open(test_json_path_obj, 'w') as f:
            json.dump(create_split_data(test_images, test_annotations), f, indent=4)

        
        # --- Conditional YOLOv8 Data YAML Creation ---
        model_name = config.get("model_name", "").lower()
        yolo_yaml_path = None
        
        if _get_trainer_type(model_name) == "yolo":
            yolo_yaml_path = create_yolo_data_yaml(
                job_id=job_id,
                data_dir_path=data_dir(job_id),
                categories=categories
            )

        result = {
            "train_annotations_json": str(train_json_path_obj),
            "val_annotations_json": str(val_json_path_obj),
            "test_annotations_json": str(test_json_path_obj),
            "classes": [c['name'] for c in categories]
        }
        
        if yolo_yaml_path:
            result["yolo_data_yaml"] = str(yolo_yaml_path)
            
        return result

    # --- Training & Evaluation Steps ---

    async def train_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Trains the specified object detection model using the appropriate LLM Agent.
        """

        # Create a mutable copy of the input config
        agent_config = config.copy()
        
        # Add the job_id parameter to the config dictionary
        agent_config["job_id"] = job_id 
        
        model_name = agent_config["model_name"]
        trainer_type = _get_trainer_type(model_name)
        
        # The agent input is now the updated agent_config (which includes job_id)
        agent_input = json.dumps(agent_config)
        
        if trainer_type == "yolo":
            agent = yolo_trainer_agent
        elif trainer_type == "torchvision":
            agent = torchvision_trainer_agent
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

        try:
            # 1. Run the agent. agent_run_result is now a RunResult object.
            agent_run_result = await Runner.run(agent, input=agent_input)
            
            # 2. Extract the final string output from the RunResult object
            agent_result_message: str = agent_run_result.final_output

            # 3. Check the string message for errors
            if "❌" in agent_result_message or "Error:" in agent_result_message:
                raise RuntimeError(f"Training Agent reported failure: {agent_result_message}")

        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR during Agent execution: {e}")

        if trainer_type == "yolo":
            final_model_path = str(best_yolo_model_path(job_id))
        else:
            final_model_path = str(best_model_path(job_id))

        return {
            "model_name": model_name,
            "trainer_type": trainer_type,
            "agent_output_message": agent_result_message,
            "best_model_path": final_model_path,
            "training_log_path": str(training_log_path(job_id)),
        }


    def evaluate_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Evaluates the best trained model on the test dataset and generates a report.
        """
        
        model_name = config["model_name"]
        trainer_type = _get_trainer_type(model_name)
        num_classes = len(config.get("classes", []))
        batch_size = config.get("batch_size", 1) 
        
        test_metrics: Dict[str, Any]
        
        if trainer_type == "yolo":
            test_metrics = evaluate_yolo_model(
                batch_size=batch_size,
                job_id=job_id
            )
        elif trainer_type == "torchvision":
            test_metrics = evaluate_torchvision_model(
                model_name=model_name,
                num_classes=num_classes,
                job_id=job_id,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

        if "error" in test_metrics:
            raise RuntimeError(f"{trainer_type.upper()} Evaluation failed: {test_metrics['error']}")

        pdf_path = create_detection_report(
            job_id=job_id, 
            results=test_metrics,
            model_name=model_name
        )

        return test_metrics

    # --- Model Loading & Inference ---

    def load_model_step(self, job_id: str) -> Dict[str, Any]:
        """Loads a trained detection model (by job_id) into the centralized cache."""
        
        key = f"{job_id}"
        if MODEL_CACHE_MANAGER.get_model_bundle(key):
            return {"status": "already_loaded", "job_id": job_id}
        
        # 1. Load config FIRST to get model_name
        job_config_path = hpo_config_path(job_id)
        if not job_config_path.exists():
             raise FileNotFoundError(f"Job config not found at: {job_config_path}")
        
        with open(job_config_path, 'r') as f:
            job_config = json.load(f)
        
        model_name = job_config.get("hyperparameter_candidate", "").get("model_name", "")
        
        # Calculate num_classes correctly
        num_classes = len(job_config.get("classes", [])) + 1 
        
        # 2. Determine trainer type and paths AFTER loading config
        trainer_type = _get_trainer_type(model_name)

        if trainer_type == "yolo":
            model_path = best_yolo_model_path(job_id)
        else:
            model_path = best_model_path(job_id)
            
        if not model_path.exists():
            raise FileNotFoundError(f"Best model artifact not found at: {model_path}")
        
        device = _get_device()
        
        model_bundle = {"job_id": job_id, "trainer_type": trainer_type, "device": device}
        
        try:
            if trainer_type == "yolo":
                model = YOLO(str(model_path))
                model_bundle["model"] = model
            elif trainer_type == "torchvision":
                model, transform = load_torchvision_model_for_inference(
                    model_name=model_name, 
                    model_path=model_path, 
                    num_classes=num_classes, 
                    device=device
                )
                model_bundle["model"] = model
                model_bundle["transform"] = transform
            else:
                raise ValueError(f"Unsupported trainer type: {trainer_type}")

            MODEL_CACHE_MANAGER.set_model_bundle(key, model_bundle)
            
            return {"status": "loaded", "job_id": job_id, "trainer_type": trainer_type}

        except Exception as e:
            return {"status": "load_failed", "job_id": job_id, "error": str(e)}


    def infer_step(self, job_id: str, image: Image.Image) -> Dict[str, Any]:
        """Runs inference on a single PIL image using the cached model."""
        
        key = f"{job_id}"
        model_bundle = MODEL_CACHE_MANAGER.get_model_bundle(key)
        if not model_bundle:
            raise ValueError(f"Model for Job ID {job_id} is not loaded. Call load_model_step first.")
            
        model = model_bundle["model"]
        device = _get_device()
        trainer_type = model_bundle["trainer_type"]
        
        categories_map = _get_category_map(job_id)
        image_to_draw = image.copy().convert('RGB')
        timestamp = int(time.time() * 1000)
        output_filename = f"inference_{job_id}_{timestamp}.jpg"
        output_path = _get_inference_save_dir(job_id) / output_filename
        
        detections: List[List[Union[float, int]]] = []
        
        try:
            if trainer_type == "yolo":
                results = model.predict(source=image, verbose=False)
                if len(results) > 0:
                    detections = results[0].boxes.data.cpu().tolist()
            elif trainer_type == "torchvision":
                transform = model_bundle.get("transform")
                detections = run_torchvision_inference(
                    model=model, 
                    image=image, 
                    transform=transform, 
                    device = device,
                )
            else:
                raise ValueError(f"Unsupported trainer type: {trainer_type}")

            if detections:
                saved_path = _draw_detections(image_to_draw, detections, categories_map, output_path)
                
                return {
                    "status": "success", 
                    "detections_count": len(detections),
                    "annotated_image_path": saved_path,
                }
            else:
                 return {
                    "status": "success", 
                    "annotated_image_path": None,
                    "detections_count": 0,
                    "message": "Inference complete, no objects detected."
                }

        except Exception as e:
            return {"status": "inference_failed", "error": str(e)}


    # --- Unloading Utilities (Delegated) ---

    def unload_model(self, job_id: str) -> Dict[str, Any]:
        """Delegates the unload operation to the centralized manager."""
        return MODEL_CACHE_MANAGER.unload_model(job_id)

    def unload_all_models(self) -> Dict[str, Any]:
        """Delegates the unload-all operation to the centralized manager."""
        return MODEL_CACHE_MANAGER.unload_all_models()