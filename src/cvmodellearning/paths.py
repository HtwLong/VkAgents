from pathlib import Path

def find_project_root(markers=("pyproject.toml", ".git")) -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return here.parent

PROJECT_ROOT = find_project_root()
RUNS_ROOT = (PROJECT_ROOT / "runs").resolve()

def run_dir(job_id: str) -> Path:
    base = (RUNS_ROOT / job_id).resolve()
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "artifacts").mkdir(parents=True, exist_ok=True)
    return base

def data_dir(job_id: str) -> Path:
    return (run_dir(job_id) / "data").resolve()

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
    return planning_artifacts_dir(job_id) / "RESULT_INTERPRETATION.json"

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





# JSON in data/ for object detection annotations
def json_labels_path(job_id: str) -> Path:
    """Path to the consolidated COCO-style JSON annotation file."""
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

def metrics_csv_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "metrics_log.csv"

def metrics_json_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "metrics_log.json"

def test_cm_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "test_confusion_matrix.csv"

def report_pdf_path(job_id: str) -> Path:
    """Path to the consolidated PDF report."""
    return artifacts_dir(job_id) / "report_summary.pdf"

def test_report_json_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "test_classification_report.json"

def tool_call_args_path(job_id: str) -> Path:
    return artifacts_dir(job_id) / "tool_call_args.json"

def unified_dataset_path() -> Path:
    """Path to the unified dataset class list file."""
    return PROJECT_ROOT / "src" / "cvmodellearning" / "download" / "unified_dataset.txt"



__all__ = [
    "PROJECT_ROOT",
    "RUNS_ROOT",
    "find_project_root",
    "run_dir",
    "data_dir",
    "artifacts_dir",
    "csv_labels_path",
    "train_csv_path",
    "val_csv_path",
    "test_csv_path",
    "best_model_path",
    "metrics_csv_path",
    "test_report_path",
    "test_cm_path",
    "planning_artifacts_dir",
    "interpretation_path",
    "hpo_config_path",
    "rationales_path",
]
