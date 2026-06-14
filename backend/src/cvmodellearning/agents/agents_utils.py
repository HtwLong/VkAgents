import asyncio
from pathlib import Path
import sys
import json
from typing import Any, Set, Union
from cvmodellearning.paths import rationales_path, unified_dataset_path
import datetime

async def ask_yes_no(prompt: str) -> bool:
    """
    Ask a yes/no question without blocking the event loop.
    Returns True for yes, False for no.
    """
    while True:
        print(f"{prompt} [y/n]: ", end="", flush=True)
        ans = await asyncio.to_thread(sys.stdin.readline)
        ans = (ans or "").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")

async def ask_user_for_changes() -> str:
    """
    Ask the user what to change in the training configuration (free-form).
    """
    print("\nWhat should be changed in the training configuration? Describe briefly, then press Enter:")
    return (await asyncio.to_thread(sys.stdin.readline)).strip()

def save_json(obj: Any, directory: Union[str, Path], filename: str):
    """
    Saves a Python object as a JSON file at the specified directory.
    
    Args:
        obj: The object to serialize (usually a dict or pydantic model dump).
        directory: The target directory (Path or string).
        filename: The name of the file to save.
    """
    # Ensure the directory exists before attempting to write the file
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = target_dir / filename
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def wrap_input_as_messages(input_dict):
    """
    Helper to wrap a dictionary input as a list of messages for OpenAI GPT agents.
    Converts dict to json string and sets role to 'user'.
    """
    return [
        {
            "role": "user",
            "content": json.dumps(input_dict, indent=2)
        }
    ]

def find_project_root(markers=("pyproject.toml", ".git")) -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return here.parent

def load_unified_dataset_classes() -> Set[str]:
    """
    Loads the unified_dataset.txt file using the centralized path.
    Returns a set of strings for O(1) lookups.
    """
    dataset_path = unified_dataset_path()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find unified_dataset.txt at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        # distinct lines, stripped, lowercase for case-insensitive matching
        classes = {line.strip().lower() for line in f if line.strip()}
    
    return classes

def log_planning_step(job_id: str, step_name: str, input_context: Any, rationale: str, output_summary: Any, round_num: int = None):
    """
    Appends a formatted log entry to the planning rationales file.
    Handles formatting for conversation rounds (e.g., HPO negotiation).
    """
    if not job_id:
        return

    path = rationales_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- Helper to pretty print if dict, fallback to str otherwise ---
    def pretty_fmt(data: Any) -> str:
        if isinstance(data, dict):
            try:
                # default=str handles non-serializable types (like Path or datetime) safely
                return json.dumps(data, indent=2, default=str)
            except Exception:
                return str(data)
        return str(data)

    input_str = pretty_fmt(input_context)
    output_str = pretty_fmt(output_summary)
    # ---------------------------------------------------------------

    step_header = f"STEP: {step_name}"
    if round_num is not None:
        step_header += f" (Round {round_num})"

    log_entry = (
        f"\n{'='*80}\n"
        f"{step_header}\n"
        f"TIMESTAMP: {timestamp}\n"
        f"{'-'*80}\n"
        f"INPUT CONTEXT / CONSTRAINTS:\n{input_str}\n"
        f"{'-'*80}\n"
        f"RATIONALE & REASONING:\n{rationale}\n"
        f"{'-'*80}\n"
        f"OUTPUT / DECISION:\n{output_str}\n"
        f"{'='*80}\n"
    )
    
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed to write rationale log: {e}")