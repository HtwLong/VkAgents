import asyncio
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Dict, Any, List, Literal, Union

from PIL import Image
import torch
from ultralytics import RTDETR, YOLO

# project utils
from cvmodellearning.download.download_data import download_visionkg_mixed_datasets_detection
from cvmodellearning.download.progress import DownloadProgressTracker
from cvmodellearning.jobs.run_control import PipelineCancelled, raise_if_cancelled
from cvmodellearning.datasets.provenance import record_split_access
from cvmodellearning.download.assignment_manifest import (
    assignment_fingerprint,
    detection_coverage_requirements,
    evaluate_detection_coverage_acceptance,
    file_sha256,
    iter_download_allocations,
    load_dataset_manifest,
    load_preparation_summary,
    validate_content_isolation,
)
from cvmodellearning.evaluation.result_report import save_detection_report
from cvmodellearning.models.detection_models.yolo_trainer import evaluate_yolo_model, train_yolo_from_config
from cvmodellearning.models.detection_models.rtdetr_trainer import (
    evaluate_rtdetr_model,
    train_rtdetr_from_config,
)
from cvmodellearning.models.detection_models.torchvision_trainer import (
    evaluate_torchvision_model,
    load_torchvision_model_for_inference,
    run_torchvision_inference,
    train_torchvision_from_config,
)
# from cvmodellearning.models.detection_models.mmdet_trainer import evaluate_mmdet_model
# from cvmodellearning.models.detection_models.mmdet_trainer import MMDET_MODEL_NAME_MAP, MODEL_CONFIG_MAP
from cvmodellearning.inference.inference_utils import _draw_detections


from cvmodellearning.models.detection_models.yolo_trainer import create_yolo_data_yaml
from cvmodellearning.models.model_manager import MODEL_CACHE_MANAGER
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.training.resource_guard import validate_image_batch_preflight
from cvmodellearning.schemas.detection_hpo import detection_runtime_family
from cvmodellearning.paths import (
    artifacts_dir,
    data_dir,
    hpo_config_path, # Used for loading job config
    test_json_path,
    training_log_path,
    best_model_path,
    json_labels_path,
    train_json_path,
    val_json_path,
    best_yolo_model_path,
    dataset_manifest_path,
    preparation_summary_path,
)


# --- Helper Functions (External to class, as they don't use instance state) ---

def _get_trainer_type(model_name: str) -> Literal["yolo", "rtdetr", "torchvision"]:
    """Determine which training utility to use based on model name."""
    try:
        return detection_runtime_family(model_name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown model architecture for training: {model_name}"
        ) from exc

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
        
    return {
        index: category['name']
        for index, category in enumerate(
            sorted(data.get('categories', []), key=lambda category: category['id'])
        )
    }

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

    def _require_prepared_data(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        return load_preparation_summary(
            preparation_summary_path(job_id),
            task="detection",
            expected_fingerprint=assignment_fingerprint(config["selected_data"]),
            expected_manifest_sha256=file_sha256(dataset_manifest_path(job_id)),
            required_artifacts={
                "train": train_json_path(job_id),
                "validation": val_json_path(job_id),
                "test": test_json_path(job_id),
            },
        )
    
    # --- Data Steps ---

    def download_data_step(self, config: Dict[str, Any], job_id: str):
        """Downloads data and create a consolidated COCO-style JSON label file."""

        selected_data: List[Dict[str, Any]] = config["selected_data"]

        data_base = data_dir(job_id)
        data_base.mkdir(parents=True, exist_ok=True)
                
        total = sum(item.count for item in iter_download_allocations(selected_data))
        progress = DownloadProgressTracker(job_id, total, resume=True)
        try:
            kwargs = {}
            if "progress_callback" in inspect.signature(
                download_visionkg_mixed_datasets_detection
            ).parameters:
                def record_progress(**values):
                    raise_if_cancelled(job_id)
                    progress.record(**values)
                kwargs["progress_callback"] = record_progress
            if "cancel_check" in inspect.signature(
                download_visionkg_mixed_datasets_detection
            ).parameters:
                kwargs["cancel_check"] = lambda: raise_if_cancelled(job_id)
            if "data_plan_constraints" in inspect.signature(
                download_visionkg_mixed_datasets_detection
            ).parameters:
                kwargs["data_plan_constraints"] = (
                    config.get("data_plan_constraints") or None
                )
            report = download_visionkg_mixed_datasets_detection(
                job_id,
                selected_data,
                **kwargs,
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
            source_shortfalls = [
                f"{item['class_name']} from {item['dataset_name']} "
                f"for {item['assigned_split']}: {item['downloaded']}/{item['requested']}"
                for item in report["sources"]
                if item["shortfall"]
            ]
            coverage = report.get("coverage") or {}
            coverage_shortfalls = []
            requirements = coverage.get("requirements") or {}
            verified_by_split = coverage.get("verified_unique_images_by_split") or {}
            for split, required in (
                requirements.get("target_unique_images_by_split") or {}
            ).items():
                verified = int(verified_by_split.get(split, 0))
                if verified < int(required):
                    coverage_shortfalls.append(
                        f"{split} unique images: {verified}/{required}"
                    )
            for class_name, shortfall in (
                coverage.get("class_image_shortfalls") or {}
            ).items():
                if shortfall:
                    coverage_shortfalls.append(
                        f"{class_name} image coverage shortfall: {shortfall}"
                    )
            shortfalls = ", ".join(source_shortfalls + coverage_shortfalls)
            raise RuntimeError(
                "Detection data download was incomplete. "
                f"Shortfalls: {shortfalls or 'unknown coverage shortfall'}. "
                "See artifacts/download_report.json."
            )
        progress.finish("completed")
        return report

    def prepare_data_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Materialize manifest assignments as train/validation/test COCO files."""
        input_json_path = json_labels_path(job_id)
        with open(input_json_path, 'r') as f:
            data = json.load(f)

        manifest = load_dataset_manifest(
            dataset_manifest_path(job_id),
            task="detection",
            expected_fingerprint=assignment_fingerprint(config["selected_data"]),
        )
        manifest_by_path = {sample["image_path"]: sample for sample in manifest["samples"]}
        image_paths = {
            str(image.get("image_path") or image.get("file_name") or "")
            for image in data.get("images", [])
        }
        if image_paths != set(manifest_by_path):
            raise ValueError("Detection annotations and dataset manifest contain different samples.")
        missing_files = [path for path in image_paths if not (data_dir(job_id) / path).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} detection samples reference missing image files; "
                f"first missing path: {missing_files[0]}"
            )
        image_dimensions: dict[object, tuple[int, int] | None] = {}
        for image in data.get("images", []):
            width, height = image.get("width"), image.get("height")
            image_dimensions[image.get("id")] = (
                (int(width), int(height))
                if isinstance(width, (int, float)) and isinstance(height, (int, float))
                and width > 0 and height > 0
                else None
            )
        content_isolation = validate_content_isolation(manifest, data_dir(job_id))

        uuid_to_int = {}
        for idx, img in enumerate(data['images']):
            # Preserve original ID in a new field just in case
            img['original_id'] = img['id']
            # New Integer ID
            new_id = idx + 1
            uuid_to_int[img['id']] = new_id
            img['id'] = new_id

        valid_annotations = []
        invalid_boxes: list[dict[str, Any]] = []
        projected_short_sides_at_640: list[float] = []
        projected_areas_at_640: list[float] = []
        for ann in data.get('annotations', []):
            old_img_id = ann['image_id']
            if old_img_id in uuid_to_int:
                bbox = ann.get("bbox")
                dimensions = image_dimensions.get(old_img_id)
                if (
                    not isinstance(bbox, (list, tuple))
                    or len(bbox) != 4
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in (bbox or [])
                    )
                    or bbox[2] <= 0
                    or bbox[3] <= 0
                    or bbox[0] < 0
                    or bbox[1] < 0
                    or (dimensions is not None and bbox[0] + bbox[2] > dimensions[0])
                    or (dimensions is not None and bbox[1] + bbox[3] > dimensions[1])
                ):
                    invalid_boxes.append({"annotation_id": ann.get("id"), "bbox": bbox})
                    continue
                if dimensions is not None:
                    projection_scale = 640.0 / max(dimensions)
                    projected_width = float(bbox[2]) * projection_scale
                    projected_height = float(bbox[3]) * projection_scale
                    projected_short_sides_at_640.append(
                        min(projected_width, projected_height)
                    )
                    projected_areas_at_640.append(projected_width * projected_height)
                ann['image_id'] = uuid_to_int[old_img_id]
                # Ensure category_id exists
                if 'category_id' not in ann and 'category_id_seq' in ann:
                    ann['category_id'] = ann['category_id_seq']
                # Ensure numeric ID for annotation itself
                if not isinstance(ann['id'], int):
                    ann['id'] = (
                        int(ann['id'])
                        if str(ann['id']).isdigit()
                        else int(hashlib.sha256(str(ann['id']).encode()).hexdigest()[:12], 16)
                    )
                
                valid_annotations.append(ann)

        if invalid_boxes:
            raise ValueError(
                f"Detection annotations contain {len(invalid_boxes)} invalid or out-of-bounds boxes; "
                f"first invalid annotation: {invalid_boxes[0]}"
            )

        data['annotations'] = valid_annotations
        categories = data.get('categories', [])
        info = data.get('info', {})
        category_names = {category["id"]: category["name"] for category in categories}
        allowed_classes = set(config.get("classes") or category_names.values())
        unexpected = sorted(set(category_names.values()) - allowed_classes)
        if unexpected:
            raise ValueError(f"Downloaded annotations contain unexpected classes: {unexpected}")
        instance_counts: dict[str, int] = {}
        object_size_counts = {"small": 0, "medium": 0, "large": 0}
        for annotation in valid_annotations:
            class_name = category_names.get(annotation.get("category_id"))
            if class_name is None:
                raise ValueError(
                    f"Annotation {annotation.get('id')} refers to an unknown category_id."
                )
            instance_counts[class_name] = instance_counts.get(class_name, 0) + 1
            area = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
            bucket = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            object_size_counts[bucket] += 1

        images_by_split = {split: [] for split in ("train", "validation", "test")}
        for image in data["images"]:
            image_path = str(image.get("image_path") or image.get("file_name"))
            split = manifest_by_path[image_path]["assigned_split"]
            if image.get("assigned_split") not in {None, split}:
                raise ValueError(f"COCO assignment mismatch for {image_path}.")
            image["assigned_split"] = split
            images_by_split[split].append(image)

        annotations_by_split = {}
        class_image_counts = {
            class_name: {} for class_name in sorted(allowed_classes)
        }
        for split, images in images_by_split.items():
            if not images:
                raise ValueError(f"Detection {split} split is empty.")
            image_ids = {image["id"] for image in images}
            annotations = [
                annotation
                for annotation in data["annotations"]
                if annotation["image_id"] in image_ids
            ]
            supported_classes = {
                category_names[annotation["category_id"]]
                for annotation in annotations
                if annotation["category_id"] in category_names
            }
            missing_classes = sorted(allowed_classes - supported_classes)
            if missing_classes:
                raise ValueError(f"Detection {split} split is missing classes: {missing_classes}")
            annotations_by_split[split] = annotations
            image_classes: dict[int, set[str]] = {}
            for annotation in annotations:
                class_name = category_names.get(annotation["category_id"])
                if class_name is not None:
                    image_classes.setdefault(annotation["image_id"], set()).add(class_name)
            for class_name in sorted(allowed_classes):
                class_image_counts[class_name][split] = sum(
                    class_name in names for names in image_classes.values()
                )

        coverage_requirements = detection_coverage_requirements(
            config["selected_data"], config.get("data_plan_constraints") or None,
        )
        verified_images_per_class = {
            class_name: sum(split_counts.values())
            for class_name, split_counts in class_image_counts.items()
        }
        coverage_shortfalls = {
            class_name: max(
                0,
                int(required) - verified_images_per_class.get(class_name, 0),
            )
            for class_name, required in coverage_requirements[
                "minimum_images_per_class"
            ].items()
        }
        unique_image_count = sum(len(images) for images in images_by_split.values())
        unique_image_shortfall = max(
            0,
            coverage_requirements["target_unique_images"] - unique_image_count,
        )
        minimum_unique_image_shortfall = max(
            0,
            coverage_requirements["minimum_unique_images"] - unique_image_count,
        )
        class_split_shortfalls = {
            class_name: {
                split: max(
                    0,
                    int(required)
                    - class_image_counts.get(class_name, {}).get(split, 0),
                )
                for split, required in split_requirements.items()
            }
            for class_name, split_requirements in coverage_requirements[
                "minimum_images_per_class_by_split"
            ].items()
        }
        target_unique_images = coverage_requirements["target_unique_images"]
        unique_coverage_ratio = (
            min(1.0, unique_image_count / target_unique_images)
            if target_unique_images
            else 1.0
        )
        class_split_coverage_satisfied = not any(
            shortfall
            for split_shortfalls in class_split_shortfalls.values()
            for shortfall in split_shortfalls.values()
        )
        coverage = {
            "requirements": coverage_requirements,
            "verified_unique_images": unique_image_count,
            "verified_unique_images_by_split": {
                split: len(images) for split, images in images_by_split.items()
            },
            "verified_images_per_class": verified_images_per_class,
            "verified_images_per_class_by_split": class_image_counts,
            "unique_image_shortfall": unique_image_shortfall,
            "minimum_unique_image_shortfall": minimum_unique_image_shortfall,
            "unique_coverage_ratio": unique_coverage_ratio,
            "class_image_shortfalls": coverage_shortfalls,
            "class_split_image_shortfalls": class_split_shortfalls,
            "class_split_coverage_satisfied": class_split_coverage_satisfied,
            "satisfied": (
                unique_image_shortfall == 0 and class_split_coverage_satisfied
            ),
        }
        acceptance = evaluate_detection_coverage_acceptance(coverage)
        if not acceptance["accepted"]:
            raise ValueError(
                "Prepared detection data does not satisfy its mandatory class-by-split "
                "coverage and minimum unique-image coverage requirements: "
                f"unique coverage={unique_coverage_ratio:.2%} "
                f"(minimum {coverage_requirements['minimum_unique_images']} images), "
                f"class split shortfalls={class_split_shortfalls}."
            )

        preparation_warnings = []
        if acceptance["aspirational_unique_target_accepted"]:
            preparation_warnings.append({
                "type": "aspirational_unique_image_shortfall",
                "message": (
                    "Materialized all mandatory class-by-split coverage from the "
                    "accepted multi-label image pool below its aspirational size."
                ),
                "target_unique_images": target_unique_images,
                "verified_unique_images": unique_image_count,
                "unique_coverage_ratio": unique_coverage_ratio,
                "minimum_unique_coverage_ratio": acceptance[
                    "minimum_unique_coverage_ratio"
                ],
                "planned_unique_images_by_split": coverage_requirements[
                    "target_unique_images_by_split"
                ],
                "adjusted_split_sizes": coverage[
                    "verified_unique_images_by_split"
                ],
            })

        def create_split_data(split):
            return {
                'info': info,
                'categories': categories,
                'images': images_by_split[split],
                'annotations': annotations_by_split[split],
            }

        train_json_path_obj = train_json_path(job_id)
        val_json_path_obj = val_json_path(job_id)
        test_json_path_obj = test_json_path(job_id)

        temporary_splits = {
            train_json_path_obj: train_json_path_obj.with_suffix(".json.tmp"),
            val_json_path_obj: val_json_path_obj.with_suffix(".json.tmp"),
            test_json_path_obj: test_json_path_obj.with_suffix(".json.tmp"),
        }
        for split, final in (
            ("train", train_json_path_obj),
            ("validation", val_json_path_obj),
            ("test", test_json_path_obj),
        ):
            temporary_splits[final].write_text(
                json.dumps(create_split_data(split), indent=4), encoding="utf-8"
            )
        raise_if_cancelled(job_id)
        for final, temporary in temporary_splits.items():
            temporary.replace(final)

        
        # --- Conditional YOLOv8 Data YAML Creation ---
        model_name = config.get("model_name", "").lower()
        yolo_yaml_path = None
        
        if _get_trainer_type(model_name) in {"yolo", "rtdetr"}:
            yolo_yaml_path = create_yolo_data_yaml(
                job_id=job_id,
                data_dir_path=data_dir(job_id),
                categories=categories
            )

        result = {
            "train_annotations_json": str(train_json_path_obj),
            "val_annotations_json": str(val_json_path_obj),
            "test_annotations_json": str(test_json_path_obj),
            "classes": [c['name'] for c in categories],
            "counts": {split: len(images) for split, images in images_by_split.items()},
            "unique_images": {
                split: len(images) for split, images in images_by_split.items()
            },
            "class_image_counts": class_image_counts,
            "instance_counts": instance_counts,
            "object_size_counts": object_size_counts,
            "annotation_statistics": {
                "small_object_fraction_at_640": (
                    sum(area < 32**2 for area in projected_areas_at_640)
                    / len(projected_areas_at_640)
                    if projected_areas_at_640 else None
                ),
                "median_box_short_side_px_at_640": (
                    float(sorted(projected_short_sides_at_640)[
                        len(projected_short_sides_at_640) // 2
                    ])
                    if projected_short_sides_at_640 else None
                ),
                "fraction_below_8px_short_side_at_640": (
                    sum(side < 8 for side in projected_short_sides_at_640)
                    / len(projected_short_sides_at_640)
                    if projected_short_sides_at_640 else None
                ),
            },
            "coverage": coverage,
            "acceptance": acceptance,
            "warnings": preparation_warnings,
            "invalid_box_count": 0,
            "content_isolation": content_isolation,
            "assignment_fingerprint": manifest["assignment_fingerprint"],
            "manifest_sha256": file_sha256(dataset_manifest_path(job_id)),
        }
        
        if yolo_yaml_path:
            result["yolo_data_yaml"] = str(yolo_yaml_path)

        summary_path = preparation_summary_path(job_id)
        result["preparation_summary"] = str(summary_path)
        temporary_summary = summary_path.with_suffix(".json.tmp")
        temporary_summary.write_text(
            json.dumps({"task": "detection", **result}, indent=2), encoding="utf-8"
        )
        temporary_summary.replace(summary_path)
        return result

    # --- Training & Evaluation Steps ---

    async def train_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Train the validated detector with its deterministic backend adapter.
        """

        preparation = self._require_prepared_data(config, job_id)
        provenance_path = record_split_access(
            job_id,
            task="detection",
            stage="training",
            preparation=preparation,
            split_artifacts={
                "train": train_json_path(job_id),
                "validation": val_json_path(job_id),
            },
        )

        # Create a mutable copy of the input config
        agent_config = training_compatible_hpo_config(config)
        validate_image_batch_preflight(
            image_size=max(
                int(agent_config.get("input_size", 640)),
                int(agent_config.get("max_size", agent_config.get("input_size", 640))),
            ),
            batch_size=int(agent_config.get("batch_size", 1)),
        )
        
        # Add the job_id parameter to the config dictionary
        agent_config["job_id"] = job_id 
        
        model_name = agent_config["model_name"]
        trainer_type = _get_trainer_type(model_name)
        
        try:
            if trainer_type == "yolo":
                agent_result_message = await asyncio.to_thread(
                    train_yolo_from_config,
                    agent_config,
                    job_id,
                )
                if "❌" in agent_result_message or "Error:" in agent_result_message:
                    raise RuntimeError(agent_result_message)
            elif trainer_type == "rtdetr":
                agent_result_message = await asyncio.to_thread(
                    train_rtdetr_from_config,
                    agent_config,
                    job_id,
                )
                if "❌" in agent_result_message or "Error:" in agent_result_message:
                    raise RuntimeError(agent_result_message)
            elif trainer_type == "torchvision":
                agent_result_message = await asyncio.to_thread(
                    train_torchvision_from_config,
                    agent_config,
                    job_id,
                )
                if "❌" in agent_result_message or "Error:" in agent_result_message:
                    raise RuntimeError(agent_result_message)
            else:
                raise ValueError(f"Unsupported trainer type: {trainer_type}")

        except PipelineCancelled:
            raise
        except Exception as e:
            raise RuntimeError(f"Detection training failed for job {job_id}: {e}") from e

        if trainer_type in {"yolo", "rtdetr"}:
            final_model_path = str(best_yolo_model_path(job_id))
        else:
            final_model_path = str(best_model_path(job_id))

        return {
            "model_name": model_name,
            "trainer_type": trainer_type,
            "agent_output_message": agent_result_message,
            "best_model_path": final_model_path,
            "training_log_path": str(training_log_path(job_id)),
            "data_provenance": str(provenance_path),
        }


    def evaluate_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Evaluates the best trained model on the test dataset and generates a report.
        """
        
        preparation = self._require_prepared_data(config, job_id)
        model_name = config["model_name"]
        trainer_type = _get_trainer_type(model_name)
        num_classes = len(config.get("classes", [])) + 1
        batch_size = max(1, int(config.get("batch_size", 1)))
        input_size = int(config.get("input_size", 640))
        
        test_metrics: Dict[str, Any]
        
        if trainer_type == "yolo":
            test_metrics = evaluate_yolo_model(
                batch_size=batch_size,
                image_size=input_size,
                job_id=job_id
            )
        elif trainer_type == "rtdetr":
            test_metrics = evaluate_rtdetr_model(
                batch_size=batch_size,
                image_size=input_size,
                job_id=job_id,
            )
        elif trainer_type == "torchvision":
            test_metrics = evaluate_torchvision_model(
                model_name=model_name,
                num_classes=num_classes,
                job_id=job_id,
                batch_size=batch_size,
                input_size=input_size,
                max_size=int(config.get("max_size", 1333)),
                workers=int(config.get("workers", 0)),
                confidence_threshold=float(config.get("confidence_threshold", 0.05)),
                nms_iou_threshold=float(config.get("nms_iou_threshold", 0.5)),
                max_detections=int(config.get("max_detections", 300)),
                topk_candidates=int(config.get("topk_candidates", 400)),
                positive_fraction=float(config.get("positive_fraction", 0.25)),
                matching_iou_threshold=float(config.get("matching_iou_threshold", 0.5)),
            )
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

        if "error" in test_metrics:
            raise RuntimeError(f"{trainer_type.upper()} Evaluation failed: {test_metrics['error']}")

        provenance_path = record_split_access(
            job_id,
            task="detection",
            stage="evaluation",
            preparation=preparation,
            split_artifacts={"test": test_json_path(job_id)},
        )
        report_path = save_detection_report(job_id, config, test_metrics)

        test_metrics["data_provenance"] = str(provenance_path)
        test_metrics["evaluation_report"] = str(report_path)

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
        
        runtime_config = training_compatible_hpo_config(
            job_config.get("hyperparameter_candidate") or job_config
        )
        model_name = str(runtime_config.get("model_name", ""))
        
        # Calculate num_classes correctly
        num_classes = len(runtime_config.get("classes", [])) + 1
        
        # 2. Determine trainer type and paths AFTER loading config
        trainer_type = _get_trainer_type(model_name)

        if trainer_type in {"yolo", "rtdetr"}:
            model_path = best_yolo_model_path(job_id)
        else:
            model_path = best_model_path(job_id)
            
        if not model_path.exists():
            raise FileNotFoundError(f"Best model artifact not found at: {model_path}")
        
        device = _get_device()
        
        model_bundle = {
            "job_id": job_id,
            "trainer_type": trainer_type,
            "device": device,
            "runtime_config": runtime_config,
        }
        
        try:
            if trainer_type == "yolo":
                model = YOLO(str(model_path))
                model_bundle["model"] = model
            elif trainer_type == "rtdetr":
                model_bundle["model"] = RTDETR(str(model_path))
            elif trainer_type == "torchvision":
                model, transform = load_torchvision_model_for_inference(
                    model_name=model_name, 
                    model_path=model_path, 
                    num_classes=num_classes, 
                    device=device,
                    input_size=int(runtime_config.get("input_size", 800)),
                    max_size=int(runtime_config.get("max_size", 1333)),
                    confidence_threshold=float(runtime_config.get("confidence_threshold", 0.05)),
                    nms_iou_threshold=float(runtime_config.get("nms_iou_threshold", 0.5)),
                    max_detections=int(runtime_config.get("max_detections", 300)),
                    topk_candidates=int(runtime_config.get("topk_candidates", 400)),
                    positive_fraction=float(runtime_config.get("positive_fraction", 0.25)),
                    matching_iou_threshold=float(runtime_config.get("matching_iou_threshold", 0.5)),
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
                runtime_config = model_bundle.get("runtime_config", {})
                results = model.predict(
                    source=image,
                    imgsz=int(runtime_config.get("input_size", 640)),
                    conf=float(runtime_config.get("confidence_threshold", 0.25)),
                    iou=float(runtime_config.get("nms_iou_threshold", 0.7)),
                    max_det=int(runtime_config.get("max_detections", 300)),
                    device=model_bundle.get("device"),
                    verbose=False,
                )
                if len(results) > 0:
                    detections = results[0].boxes.data.cpu().tolist()
            elif trainer_type == "rtdetr":
                runtime_config = model_bundle.get("runtime_config", {})
                results = model.predict(
                    source=image,
                    imgsz=int(runtime_config.get("input_size", 640)),
                    conf=float(runtime_config.get("confidence_threshold", 0.25)),
                    max_det=int(runtime_config.get("max_detections", 300)),
                    device=model_bundle.get("device"),
                    verbose=False,
                )
                if results:
                    detections = results[0].boxes.data.cpu().tolist()
            elif trainer_type == "torchvision":
                transform = model_bundle.get("transform")
                detections = run_torchvision_inference(
                    model=model, 
                    image=image, 
                    transform=transform, 
                    device=device,
                    confidence_threshold=float(
                        model_bundle.get("runtime_config", {}).get("confidence_threshold", 0.05)
                    ),
                )
            else:
                raise ValueError(f"Unsupported trainer type: {trainer_type}")

            if detections:
                saved_path = _draw_detections(image_to_draw, detections, categories_map, output_path)
                structured_detections = [
                    {
                        "box": [float(value) for value in detection[:4]],
                        "confidence": float(detection[4]),
                        "class_id": int(detection[5]),
                        "label": categories_map.get(int(detection[5]), f"Class {int(detection[5])}"),
                    }
                    for detection in detections
                ]
                
                return {
                    "status": "success", 
                    "detections_count": len(detections),
                    "annotated_image_path": saved_path,
                    "image_width": image.width,
                    "image_height": image.height,
                    "detections": structured_detections,
                }
            else:
                 return {
                    "status": "success", 
                    "annotated_image_path": None,
                    "detections_count": 0,
                    "image_width": image.width,
                    "image_height": image.height,
                    "detections": [],
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
