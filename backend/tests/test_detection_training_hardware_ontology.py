from pathlib import Path

from cvmodellearning.graphrag.build_graph import build_graph
from cvmodellearning.graphrag.model_selection_context import build_model_selection_context
from cvmodellearning.schemas.interpretation_schema import PipelineState


ONTOLOGY = Path(__file__).parents[1] / "ontology_data"


def _rtx2060_state() -> PipelineState:
    return PipelineState(
        task="detection",
        training_hardware={
            "profile_id": "rtx2060_test",
            "accelerator": "cuda",
            "hardware_category": "ConsumerGPU",
            "gpu_type": "RTX 2060",
            "gpu_count": 1,
            "vram_gb": 6,
            "ram_gb": 16,
            "unified_memory": False,
            "training_memory_budget_gb": 5,
            "max_batch_size": 4,
            "workers": 2,
            "supports_amp": True,
        },
    )


def test_training_requirement_has_model_and_source_edges():
    graph = build_graph(ONTOLOGY)
    requirement = "train_hw_rtdetr_hgnetv2_l"

    assert any(
        target == requirement and edge.get("relation") == "has_training_hardware_requirement"
        for _, target, edge in graph.out_edges("rtdetr_hgnetv2_l", data=True)
    )
    assert any(
        target == "evidence_rtdetr_l_8gb_study"
        and edge.get("relation") == "supported_by_evidence"
        for _, target, edge in graph.out_edges(requirement, data=True)
    )


def test_six_gb_training_gpu_rejects_rtdetr_l_but_keeps_feasible_detectors():
    context = build_model_selection_context(_rtx2060_state(), top_k=100)
    candidates = {item["model"]["id"]: item for item in context["candidate_models"]}

    assert "rtdetr_hgnetv2_l" not in candidates
    assert context["rejected_counts"]["recommended_training_vram_gb"] >= 1
    assert "yolov8s" in candidates
    assert "fasterrcnn_resnet50_fpn" in candidates
    requirement = candidates["fasterrcnn_resnet50_fpn"][
        "model_training_hardware_requirement"
    ]
    assert requirement["recommended_vram_gb"] == 6
    assert "evidence_detectron2_fasterrcnn_model_zoo" in requirement["evidence_ids"]


def test_twelve_gb_training_gpu_exposes_rtdetr_l_requirement_and_sources():
    state = _rtx2060_state()
    state.training_hardware = state.training_hardware.model_copy(
        update={"vram_gb": 12, "training_memory_budget_gb": 10}
    )
    context = build_model_selection_context(state, top_k=100)
    candidates = {item["model"]["id"]: item for item in context["candidate_models"]}
    requirement = candidates["rtdetr_hgnetv2_l"]["model_training_hardware_requirement"]

    assert float(requirement["lowest_observed_success_vram_gb"]) == 8
    assert float(requirement["recommended_vram_gb"]) == 12
    assert requirement["recommendation_status"] == "derived_with_observed_lower_bound"
    source_ids = {source["id"] for source in candidates["rtdetr_hgnetv2_l"]["evidence_sources"]}
    assert "evidence_rtdetr_l_8gb_study" in source_ids
    assert "evidence_rtdetr_training_stability_discussion" in source_ids
