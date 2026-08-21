from pathlib import Path
import ast

from viewer_backend.agent_prompts import (
    DATASET_SELECTION_INSTRUCTIONS,
    HYPERPARAMETER_INSTRUCTIONS,
    MODEL_SELECTION_INSTRUCTIONS,
)
from viewer_backend.skills import load_cv_skill
from viewer_backend.schemas import DatasetPlan, ModelPlan, TaskInterpretation


def test_viewer_planning_skills_are_verbatim_copies_of_original_skills():
    root = Path(__file__).resolve().parents[2]
    original_dir = root / "backend" / "src" / "cvmodellearning" / "skills" / "cv_planning"
    for name in ("diagnose", "model-selection", "dataset-and-split", "data-problems", "recipe-adaptation"):
        assert load_cv_skill(name) == (original_dir / f"{name}.md").read_text(encoding="utf-8").strip()


def test_stage_prompts_include_the_same_relevant_skills_as_original_agents():
    assert load_cv_skill("diagnose") in MODEL_SELECTION_INSTRUCTIONS
    assert load_cv_skill("model-selection") in MODEL_SELECTION_INSTRUCTIONS
    assert load_cv_skill("dataset-and-split") in DATASET_SELECTION_INSTRUCTIONS
    assert load_cv_skill("recipe-adaptation") in HYPERPARAMETER_INSTRUCTIONS
    assert load_cv_skill("data-problems") in HYPERPARAMETER_INSTRUCTIONS


def _declared_fields(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return [
        node.target.id for node in model.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


def test_llm_boundary_patch_field_order_matches_original_agents():
    root = Path(__file__).resolve().parents[2] / "backend" / "src" / "cvmodellearning" / "agents"
    assert list(TaskInterpretation.model_fields) == _declared_fields(
        root / "interpretation_agents.py", "TaskExtractionPatch"
    )
    expected_model = _declared_fields(root / "model_selection_agents.py", "DetectionModelPatch")
    assert list(ModelPlan.model_fields) == expected_model
    assert list(DatasetPlan.model_fields) == _declared_fields(
        root / "data_selection_and_augmentation_agents.py", "DataSelectionPatch"
    )
