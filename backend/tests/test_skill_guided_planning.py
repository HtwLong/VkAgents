from cvmodellearning.graphrag.hyperparameter_context import build_hyperparameter_context
from cvmodellearning.graphrag.model_selection_context import build_model_selection_context
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.skills import load_cv_skill


def test_cv_playbooks_are_loadable_and_procedural():
    model_skill = load_cv_skill("model-selection")
    recipe_skill = load_cv_skill("recipe-adaptation")

    assert "compare" in model_skill.lower()
    assert "not the graph ranking automatically" in model_skill.lower()
    assert "not an immutable configuration" in recipe_skill.lower()


def test_model_graphrag_returns_candidates_without_a_selected_winner():
    context = build_model_selection_context(PipelineState(task="classification"))

    assert context["candidate_models"]
    assert "deterministic_recommendation" not in context
    assert "does not select a winner" in context["instructions_for_selector"]


def test_hyperparameter_graphrag_exposes_reference_recipe():
    state = PipelineState(
        task="classification",
        selected_model_info={
            "model": {
                "model_architecture": "resnet50",
                "description": "test",
            }
        },
        classes=["a", "b"],
    )

    context = build_hyperparameter_context(state)

    assert context["reference_configuration"]
    assert "starting point" in context["instructions_for_generator"]
