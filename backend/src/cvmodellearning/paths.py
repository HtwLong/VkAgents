import os
from pathlib import Path

def find_project_root(markers=("pyproject.toml", ".git")) -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return here.parent

PROJECT_ROOT = find_project_root()
RUNS_ROOT = (PROJECT_ROOT / "runs").resolve()
VISIONKG_CACHE_ROOT = Path(
    os.environ.get("VISIONKG_IMAGE_CACHE_DIR", PROJECT_ROOT / "dataset_cache" / "visionkg")
).expanduser().resolve()

def run_dir(job_id: str) -> Path:
    base = (RUNS_ROOT / job_id).resolve()
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "artifacts").mkdir(parents=True, exist_ok=True)
    return base

def data_dir(job_id: str) -> Path:
    return (run_dir(job_id) / "data").resolve()

def visionkg_cache_dir() -> Path:
    """Return the persistent root for immutable images downloaded from VisionKG."""
    VISIONKG_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return VISIONKG_CACHE_ROOT

def train_data_dir(job_id: str) -> Path:
    return (data_dir(job_id) / "train").resolve()

def val_data_dir(job_id: str) -> Path:
    return (data_dir(job_id) / "val").resolve()

def test_data_dir(job_id: str) -> Path:
    return (data_dir(job_id) / "test").resolve()

def artifacts_dir(job_id: str) -> Path:
    return (run_dir(job_id) / "artifacts").resolve()

def plots_dir(job_id: str) -> Path:
    """Directory for training visualizations (confusion matrix, curves, etc.)."""
    p = artifacts_dir(job_id) / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p

def planning_artifacts_dir(job_id: str) -> Path:
    p = artifacts_dir(job_id) / "planning"
    p.mkdir(parents=True, exist_ok=True)
    return p

def rationales_path(job_id: str) -> Path:
    """Path to the text file logging planning rationales."""
    return planning_artifacts_dir(job_id) / "planning_rationales.txt"

def interpretation_path(job_id: str) -> Path:
    planning_dir = planning_artifacts_dir(job_id)
    current = planning_dir / "STATE_01_INTERPRETATION.json"
    legacy = planning_dir / "RESULT_INTERPRETATION.json"
    return current if current.exists() or not legacy.exists() else legacy

def hpo_config_path(job_id: str) -> Path:
    return planning_artifacts_dir(job_id) / "RESULT_HYPERPARAMETERS.json"

# CSVs in data/
def csv_labels_path(job_id: str) -> Path:
    return data_dir(job_id) / "image_labels.csv"

def train_csv_path(job_id: str) -> Path:
    return data_dir(job_id) / "train_labels.csv"

def val_csv_path(job_id: str) -> Path:
    return data_dir(job_id) / "val_labels.csv"

def test_csv_path(job_id: str) -> Path:
    return data_dir(job_id) / "test_labels.csv"

def dataset_manifest_path(job_id: str) -> Path:
    return data_dir(job_id) / "dataset_manifest.json"

def download_report_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "download_report.json"

def download_progress_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "download_progress.json"

def preparation_summary_path(job_id: str) -> Path:
    return data_dir(job_id) / "preparation_summary.json"

def data_provenance_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "data_provenance.json"





# JSON for either object detection annotations
# OR VQA annotations (we can differentiate based on the content structure)
def json_labels_path(job_id: str) -> Path:
    return data_dir(job_id) / "annotations.json"
 
def train_json_path(job_id: str) -> Path:
    """Path to the training split JSON annotation file."""
    return data_dir(job_id) / "train_annotations.json"

def val_json_path(job_id: str) -> Path:
    """Path to the validation split JSON annotation file."""
    return data_dir(job_id) / "val_annotations.json"

def test_json_path(job_id: str) -> Path:
    """Path to the test split JSON annotation file."""
    return data_dir(job_id) / "test_annotations.json"

def yolo_data_yaml_path(job_id: str) -> Path:
    """Path to the Ultralytics data.yaml configuration file."""
    # This is placed in the data directory alongside the JSON files
    return data_dir(job_id) / "yolo_data.yaml"



# Artifacts
def training_log_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "training_log.txt"

def best_model_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "best_model.pth"

def best_yolo_model_path(job_id: str) -> Path:
    """Path to the best YOLO model file (.pt)."""
    return artifacts_dir(job_id) / "best_model.pt"

def lora_adapter_bundle_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "best_lora_adapter.zip"

def merged_model_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "best_merged_model.pth"

def metrics_csv_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "metrics_log.csv"

def metrics_json_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "metrics_log.json"

def test_cm_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "test_confusion_matrix.csv"

def evaluation_report_path(job_id: str) -> Path:
    """Structured evaluation data consumed by the frontend results view."""
    return artifacts_dir(job_id) / "evaluation_report.json"

def test_report_json_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "test_classification_report.json"

def tool_call_args_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "tool_call_args.json"

def unified_dataset_path(task: str | None = None) -> Path:
    """Path to the legacy or task-specific VisionKG class vocabulary."""
    filenames = {
        None: "unified_dataset.txt",
        "classification": "unified_dataset_classification.txt",
        "detection": "unified_dataset_detection.txt",
    }
    try:
        filename = filenames[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported vocabulary task: {task}") from exc
    return PROJECT_ROOT / "src" / "cvmodellearning" / "download" / filename



__all__ = [
    "PROJECT_ROOT",
    "RUNS_ROOT",
    "VISIONKG_CACHE_ROOT",
    "find_project_root",
    "run_dir",
    "data_dir",
    "visionkg_cache_dir",
    "artifacts_dir",
    "csv_labels_path",
    "train_csv_path",
    "val_csv_path",
    "test_csv_path",
    "dataset_manifest_path",
    "download_report_path",
    "download_progress_path",
    "preparation_summary_path",
    "data_provenance_path",
    "best_model_path",
    "lora_adapter_bundle_path",
    "merged_model_path",
    "metrics_csv_path",
    "test_report_path",
    "test_cm_path",
    "planning_artifacts_dir",
    "interpretation_path",
    "hpo_config_path",
    "rationales_path",
]
