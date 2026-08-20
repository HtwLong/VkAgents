import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cvmodellearning.agents.model_selection_agents import (
    ClassificationModelPatch,
    DetectionModelPatch,
    VQAModelPatch,
)
from cvmodellearning.agents.interpretation_agents import TaskExtractionPatch
from cvmodellearning.graphrag.hyperparameter_context import build_hyperparameter_context
from cvmodellearning.graphrag.model_selection_context import build_model_selection_context
from cvmodellearning.graphrag.decision_evidence import build_model_selection_decision_evidence
from cvmodellearning.models.registry import (
    canonical_model_id,
    family_for_model_reference,
    is_executable_model_reference,
    model_ids_equivalent,
)
from cvmodellearning.schemas.interpretation_schema import PipelineState


@pytest.mark.parametrize(
    ("task", "ontology_task"),
    (
        ("classification", "image_classification"),
        ("detection", "object_detection"),
        ("visual question answering", "visual_question_answering"),
    ),
)
def test_model_selection_graphrag_retrieves_only_the_pipeline_task(task, ontology_task):
    context = build_model_selection_context(PipelineState(task=task), top_k=100)

    assert context["candidate_models"]
    assert context["task_filter"] == ontology_task
    assert {
        candidate["model"]["task"] for candidate in context["candidate_models"]
    } == {ontology_task}
    assert all(
        is_executable_model_reference(task, candidate["model"]["id"])
        for candidate in context["candidate_models"]
    )


def test_very_low_memory_is_a_soft_preference_without_selecting_a_winner():
    context = build_model_selection_context(PipelineState(
        task="classification",
        performance_requirements={
            "primary_metric": "accuracy",
            "accuracy_category": "High",
            "target_is_hard": False,
        },
        deployment_constraints={"memory_category": "VeryLow"},
        available_hardware={"hardware_category": "ConsumerCPU"},
    ))

    candidate_ids = [item["model"]["id"] for item in context["candidate_models"]]
    assert context["filters"]["accuracy_preference"] == "High"
    assert "accuracy_category_at_least" not in context["filters"]
    assert "deterministic_recommendation" not in context
    assert candidate_ids
    assert context["filters"]["memory_category_preference"] == "VeryLow"
    assert "memory_category" not in context["filters"]


def test_low_memory_shortlist_is_neutral_and_does_not_hard_filter_b4():
    context = build_model_selection_context(PipelineState(
        task="classification",
        performance_requirements={
            "primary_metric": "accuracy",
            "accuracy_category": "MediumHigh",
        },
        deployment_constraints={"memory_category": "Low"},
        available_hardware={"hardware_category": "ConsumerCPU"},
    ))

    candidate_ids = [item["model"]["id"] for item in context["candidate_models"]]
    assert len(context["candidate_models"]) <= 7
    assert "deterministic_recommendation" not in context
    assert candidate_ids == sorted(candidate_ids)
    assert candidate_ids
    assert context["filters"]["memory_category_preference"] == "Low"
    assert "memory_category" not in context["rejected_counts"]


def test_default_graphrag_shortlist_is_capped_at_seven():
    context = build_model_selection_context(PipelineState(task="classification"))

    assert len(context["candidate_models"]) == 7
    assert all("rank" not in item for item in context["candidate_models"])
    assert all(item["shortlist_roles"] for item in context["candidate_models"])
    assert [item["model"]["id"] for item in context["candidate_models"]] == sorted(
        item["model"]["id"] for item in context["candidate_models"]
    )


def test_shortlist_presentation_is_independent_of_graph_iteration_order(monkeypatch):
    import cvmodellearning.graphrag.model_selection_context as retrieval

    graph = retrieval.get_model_selection_graph()
    original_nodes = graph.nodes
    baseline = build_model_selection_context(PipelineState(task="classification"))

    class ReversedNodes:
        def __call__(self, data=False):
            return list(reversed(list(original_nodes(data=data))))

        def __getitem__(self, key):
            return original_nodes[key]

    monkeypatch.setattr(graph, "nodes", ReversedNodes())
    reversed_context = build_model_selection_context(PipelineState(task="classification"))

    assert [item["model"]["id"] for item in reversed_context["candidate_models"]] == [
        item["model"]["id"] for item in baseline["candidate_models"]
    ]


def test_apple_silicon_mac_is_not_filtered_as_an_edge_device():
    from routers.planning import ensure_default_hardware_filter

    state = PipelineState(
        task="detection",
        user_query=(
            "Run locally on a MacBook Air with an Apple M4 chip using CPU or Metal acceleration."
        ),
        available_hardware={
            "hardware_category": "EdgeDevice",
            "gpu_type": "Integrated Apple GPU (Metal)",
            "ram_gb": 16,
            "details": "Apple M4, 16 GB unified memory; CPU/Metal acceleration.",
        },
    )

    ensure_default_hardware_filter(state)
    context = build_model_selection_context(state, top_k=100)

    assert state.available_hardware.hardware_category == "ConsumerCPU"
    assert context["filters"]["hardware_category"] == "ConsumerCPU"
    assert context["candidate_models"]


def test_prompt_six_qualitative_constraints_have_deterministic_fallbacks():
    from cvmodellearning.agents.interpretation_agents import TaskExtractionPatch
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(
        task="classification",
        performance_requirements={"primary_metric": "accuracy"},
    )
    apply_qualitative_constraint_fallbacks(
        extracted,
        "CPU inference with a low memory footprint while maintaining reliable accuracy.",
    )

    assert extracted.performance_requirements.accuracy_category == "MediumHigh"
    assert extracted.performance_requirements.target_is_hard is False
    assert extracted.deployment_constraints.memory_category == "Low"


def test_only_explicit_hard_numeric_performance_target_filters_candidates():
    soft = build_model_selection_context(PipelineState(
        task="classification",
        performance_requirements={
            "primary_metric": "accuracy",
            "target_value": 0.82,
            "target_is_hard": False,
        },
    ), top_k=100)
    hard = build_model_selection_context(PipelineState(
        task="classification",
        performance_requirements={
            "primary_metric": "accuracy",
            "target_value": 0.82,
            "target_is_hard": True,
        },
    ), top_k=100)

    assert "benchmark_target" not in soft["filters"]
    assert soft["filters"]["benchmark_preference"]["target_value"] == 0.82
    assert hard["filters"]["benchmark_target"]["target_value"] == 0.82
    assert len(hard["candidate_models"]) < len(soft["candidate_models"])

def test_soft_map_target_ranks_small_detector_ahead_of_larger_variant():
    context = build_model_selection_context(PipelineState(
        task="detection",
        performance_requirements={
            "primary_metric": "mAP@0.5:0.95",
            "target_value": 0.30,
            "target_is_hard": False,
            "accuracy_category": "MediumHigh",
            "latency_category": "Medium",
        },
        available_hardware={"hardware_category": "ConsumerCPU"},
    ), top_k=100)

    candidates = context["candidate_models"]
    candidate_ids = [candidate["model"]["id"] for candidate in candidates]

    assert context["filters"]["benchmark_preference"]["target_value"] == 0.30
    assert "yolo11n" in candidate_ids
    assert candidate_ids.index("yolo11n") < candidate_ids.index("yolo11s")
    assert "deterministic_recommendation" not in context


def test_small_object_requirement_adds_risk_evidence_without_selecting_a_winner():
    context = build_model_selection_context(PipelineState(
        task="detection",
        robustness_requirements={"object_scale": ["small"]},
        performance_requirements={
            "primary_metric": "mAP@0.5:0.95",
            "target_value": 0.30,
            "target_is_hard": False,
            "accuracy_category": "Medium",
            "latency_category": "Medium",
        },
        deployment_constraints={"memory_category": "Low"},
        available_hardware={"hardware_category": "ConsumerCPU"},
    ), top_k=100)

    assert context["filters"]["object_size_risk"] == "medium"
    assert context["candidate_models"]
    assert "deterministic_recommendation" not in context


def test_accuracy_first_rtx2060_shortlist_includes_distinct_detector_architectures():
    context = build_model_selection_context(PipelineState(
        task="detection",
        performance_requirements={
            "primary_metric": "mAP@0.5:0.95",
            "target_value": 0.35,
            "target_is_hard": False,
            "accuracy_category": "MediumHigh",
        },
        constraint_strengths={
            "accuracy": "preference",
            "latency": "unspecified",
            "runtime_memory": "preference",
        },
        robustness_requirements={"object_scale": ["small"], "scene_density": ["dense"]},
        deployment_constraints={
            "memory_category": "Low",
            "max_runtime_memory_mb": 6144,
        },
        available_hardware={
            "hardware_category": "ConsumerGPU",
            "gpu_type": "NVIDIA GeForce RTX 2060",
            "gpu_count": 1,
            "vram_gb": 6,
        },
    ))

    candidates = {item["model"]["id"]: item for item in context["candidate_models"]}
    assert len(candidates) <= 7
    assert "rtdetr_hgnetv2_l" in candidates
    assert "fasterrcnn_resnet50_fpn" in candidates
    assert "retinanet_resnet50_fpn" in candidates
    assert "ssd300_vgg16" in candidates
    assert "yolo12x" in candidates
    assert sum(
        item["model"]["model_family"].lower().startswith("yolo")
        for item in candidates.values()
    ) <= 3
    assert candidates["rtdetr_hgnetv2_l"]["criterion_assessments"][
        "small_object_suitability"
    ]["status"] == "unverified"


def test_soft_memory_headroom_does_not_rank_smaller_model_ahead_of_higher_map():
    import cvmodellearning.graphrag.model_selection_context as retrieval

    state = PipelineState(
        task="detection",
        performance_requirements={
            "primary_metric": "mAP@0.5:0.95",
            "target_value": 0.35,
            "target_is_hard": False,
            "accuracy_category": "MediumHigh",
        },
        deployment_constraints={"max_runtime_memory_mb": 6144},
        available_hardware={"hardware_category": "ConsumerGPU", "vram_gb": 6},
    )
    context = build_model_selection_context(state, top_k=100)
    candidates = {item["model"]["id"]: item for item in context["candidate_models"]}

    assert retrieval._tradeoff_key(
        candidates["rtdetr_hgnetv2_l"], context["filters"]
    ) < retrieval._tradeoff_key(candidates["yolo12s"], context["filters"])


def test_explicit_soft_map_target_overrides_misclassified_primary_latency():
    from cvmodellearning.agents.interpretation_agents import TaskExtractionPatch
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(
        task="detection",
        performance_requirements={
            "primary_metric": "latency",
            "target_value": 0.5,
            "latency_category": "Medium",
            "accuracy_category": "MediumHigh",
        },
    )
    apply_qualitative_constraint_fallbacks(
        extracted,
        "Aim for mAP@0.5:0.95 of around 0.30 or higher; 500 ms latency is desirable.",
    )

    assert extracted.performance_requirements.primary_metric == "mAP@0.5:0.95"
    assert extracted.performance_requirements.target_value == 0.30
    assert extracted.performance_requirements.target_is_hard is False


def test_latency_with_units_uses_dedicated_millisecond_constraint():
    from cvmodellearning.agents.interpretation_agents import TaskExtractionPatch
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(
        task="classification",
        performance_requirements={
            "primary_metric": "latency",
            "target_value": 0.2,
            "target_is_hard": True,
        },
    )
    apply_qualitative_constraint_fallbacks(
        extracted,
        "The CPU model must process an image within 200 milliseconds.",
    )

    assert extracted.deployment_constraints.max_cpu_latency_ms == 200
    assert "max_cpu_latency_ms" in extracted.deployment_constraints.hard_limits
    assert extracted.performance_requirements.target_value is None
    assert extracted.performance_requirements.target_is_hard is False


def test_explicit_accuracy_and_latency_targets_are_preserved():
    from cvmodellearning.agents.interpretation_agents import TaskExtractionPatch
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(
        task="classification",
        performance_requirements={
            "primary_metric": "latency",
            "target_value": 0.0,
        },
    )
    apply_qualitative_constraint_fallbacks(
        extracted,
        (
            "Aim for accuracy of at least 90% and process an image "
            "within 200 milliseconds."
        ),
    )

    assert extracted.performance_requirements.primary_metric == "accuracy"
    assert extracted.performance_requirements.target_value == 0.9
    assert extracted.performance_requirements.target_is_hard is True
    assert extracted.deployment_constraints.max_cpu_latency_ms == 200
    assert "max_cpu_latency_ms" in extracted.deployment_constraints.hard_limits


def test_approximate_latency_is_a_soft_millisecond_preference():
    from cvmodellearning.agents.interpretation_agents import TaskExtractionPatch
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(task="classification")
    apply_qualitative_constraint_fallbacks(
        extracted,
        "Inference below approximately 0.5 seconds would be desirable.",
    )

    assert extracted.deployment_constraints.max_cpu_latency_ms == 500
    assert "max_cpu_latency_ms" not in extracted.deployment_constraints.hard_limits


def test_inference_time_not_important_clears_latency_priority():
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(
        task="detection",
        performance_requirements={"primary_metric": "latency", "latency_category": "Low"},
        constraint_strengths={"latency": "preference"},
    )
    apply_qualitative_constraint_fallbacks(
        extracted,
        "Inference time is not important. Memory should preferably remain below 6 GB.",
    )

    assert extracted.constraint_strengths.latency == "unspecified"
    assert extracted.performance_requirements.latency_category is None


def test_preferred_runtime_memory_limit_is_extracted_as_numeric_soft_constraint():
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(task="detection")
    apply_qualitative_constraint_fallbacks(
        extracted,
        "Memory usage during inference should preferably remain below 6 GB.",
    )

    assert extracted.deployment_constraints.max_runtime_memory_mb == 6144
    assert "max_runtime_memory_mb" not in extracted.deployment_constraints.hard_limits
    assert extracted.constraint_strengths.runtime_memory == "preference"


def test_mandatory_runtime_memory_limit_is_extracted_as_hard_constraint():
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(task="detection")
    apply_qualitative_constraint_fallbacks(
        extracted,
        "Inference memory must remain below 512 MB.",
    )

    assert extracted.deployment_constraints.max_runtime_memory_mb == 512
    assert "max_runtime_memory_mb" in extracted.deployment_constraints.hard_limits


def test_soft_low_memory_keeps_faster_rcnn_eligible_when_runtime_estimate_fits():
    context = build_model_selection_context(PipelineState(
        task="detection",
        performance_requirements={
            "primary_metric": "mAP@0.5:0.95",
            "target_value": 0.35,
            "target_is_hard": False,
        },
        constraint_strengths={"latency": "unspecified", "runtime_memory": "preference"},
        deployment_constraints={
            "memory_category": "Low",
            "max_runtime_memory_mb": 6144,
        },
        available_hardware={"hardware_category": "ConsumerCPU", "ram_gb": 16},
    ), top_k=100)

    candidates = {item["model"]["id"]: item for item in context["candidate_models"]}
    assert "fasterrcnn_resnet50_fpn" in candidates
    assert context["filters"]["memory_category_preference"] == "Low"
    assert context["filters"]["max_runtime_memory_mb_preference"] == 6144
    assert "memory_category" not in context["rejected_counts"]
    assert candidates["fasterrcnn_resnet50_fpn"]["model_inference_memory_estimate"][
        "total_estimated_vram_gb"
    ] == pytest.approx(0.766)


def test_hard_runtime_memory_limit_rejects_faster_rcnn_by_estimated_runtime():
    context = build_model_selection_context(PipelineState(
        task="detection",
        deployment_constraints={
            "max_runtime_memory_mb": 512,
            "hard_limits": ["max_runtime_memory_mb"],
        },
    ), top_k=100)

    candidate_ids = {item["model"]["id"] for item in context["candidate_models"]}
    assert "fasterrcnn_resnet50_fpn" not in candidate_ids
    assert context["rejected_counts"]["max_runtime_memory_mb"] > 0


def test_soft_deployment_targets_rank_candidates_without_rejecting_missing_benchmarks():
    context = build_model_selection_context(PipelineState(
        task="classification",
        performance_requirements={
            "primary_metric": "F1",
            "accuracy_category": "MediumHigh",
            "latency_category": "Medium",
        },
        deployment_constraints={
            "max_runtime_memory_mb": 4096,
            "max_cpu_latency_ms": 500,
        },
        available_hardware={
            "hardware_category": "ConsumerGPU",
            "vram_gb": 16,
        },
    ), top_k=100)

    assert context["candidate_models"]
    assert context["filters"]["max_cpu_latency_ms_preference"] == 500
    assert "max_cpu_latency_ms" not in context["filters"]
    assert "max_cpu_latency_ms" not in context["rejected_counts"]


def test_missing_hard_latency_evidence_warns_instead_of_rejecting_models():
    context = build_model_selection_context(PipelineState(
        task="classification",
        deployment_constraints={
            "max_cpu_latency_ms": 500,
            "hard_limits": ["max_cpu_latency_ms"],
        },
    ), top_k=100)

    assert context["candidate_models"]
    assert context["filters"]["max_cpu_latency_ms"] == 500
    assert context["constraint_warnings"] == [
        "no comparable CPU latency benchmark is available for the requested 500.0ms limit"
    ]


def test_detection_shortlist_excludes_non_executable_ontology_variants():
    context = build_model_selection_context(PipelineState(task="detection"), top_k=100)
    candidate_ids = {
        candidate["model"]["id"] for candidate in context["candidate_models"]
    }

    assert "rtdetr_hgnetv2_l" in candidate_ids
    assert "rtdetr_r50" not in candidate_ids
    assert "maskrcnn_resnet50_fpn" not in candidate_ids
    assert context["rejected_counts"]["not_executable"] > 0


def test_detection_ontology_and_hpo_model_ids_are_equivalent():
    assert canonical_model_id("yolov8n") == "yolov8n"
    assert canonical_model_id("yolov8_n") == "yolov8n"
    assert model_ids_equivalent("yolov8n", "yolov8_n")
    assert not model_ids_equivalent("yolov8n", "yolov8_s")


@pytest.mark.parametrize(
    ("model_id", "expected_family"),
    (
        ("yolov10", "yolo"),
        ("yolo10", "yolo"),
        ("yolov10n", "yolo"),
        ("yolov10_n", "yolo"),
        ("yolov10_x", "yolo"),
        ("faster_rcnn_r50", "faster-rcnn"),
        ("retinanet_r50", "retinanet"),
        ("ssd300", "ssd"),
        ("yolov10unknown", None),
    ),
)
def test_detection_family_resolves_base_models_and_variants(model_id, expected_family):
    assert family_for_model_reference("detection", model_id) == expected_family


def test_training_backend_compatibility_does_not_filter_model_candidates():
    context = build_model_selection_context(PipelineState(task="detection"), top_k=100)
    candidate_ids = {item["model"]["id"] for item in context["candidate_models"]}

    assert {"yolo11n", "yolo11s"}.issubset(candidate_ids)
    assert "yolo12n" in candidate_ids


def test_vqa_retrieval_uses_the_exact_executable_model():
    context = build_model_selection_context(
        PipelineState(
            task="visual question answering",
            performance_requirements={"primary_metric": "latency", "latency_category": "Low"},
        ),
        top_k=100,
    )

    assert [candidate["model"]["id"] for candidate in context["candidate_models"]] == [
        "Qwen3-VL-2B-Instruct"
    ]
    record = build_model_selection_decision_evidence(
        {"model": {"model_architecture": "Qwen3-VL-2B-Instruct"}},
        "Selected the executable VQA model.",
        context,
    )
    assert record["match_scope"] == "exact"
    assert record["retrieved_facts"][0]["support_type"] == "inferred"
    assert record["grounding"]["status"] == "partially_grounded"


def test_classification_selector_returns_one_concrete_model():
    selection = ClassificationModelPatch.model_validate({
        "model": {
            "model_architecture": "resnet50",
            "description": "Balanced executable classifier.",
        },
        "rationale": "Selected from the graph shortlist.",
    })

    assert selection.model.model_architecture == "resnet50"
    assert isinstance(selection.model_dump()["model"], dict)

    with pytest.raises(ValidationError):
        ClassificationModelPatch.model_validate({
            "model": [{"model_architecture": "resnet50", "description": "first"}],
            "rationale": "Lists are no longer accepted.",
        })
    with pytest.raises(ValidationError):
        ClassificationModelPatch.model_validate({
            "model": {"description": "Missing concrete architecture."},
            "rationale": "Invalid selection.",
        })


def test_detection_and_vqa_selectors_also_return_one_model():
    detection = DetectionModelPatch.model_validate({
        "model": {
            "model_architecture": "yolov8",
            "description": "Executable detection family.",
        },
        "rationale": "Detection selection.",
    })
    vqa = VQAModelPatch.model_validate({
        "model": {
            "model_architecture": "Qwen3-VL-2B-Instruct",
            "description": "Executable VQA model.",
        },
        "rationale": "VQA selection.",
    })

    assert detection.model.model_architecture == "yolov8"
    assert vqa.model.model_architecture == "Qwen3-VL-2B-Instruct"


@pytest.mark.parametrize(
    ("patch_type", "architecture", "forbidden_field", "value"),
    (
        (ClassificationModelPatch, "resnet50", "num_epochs", 12),
        (DetectionModelPatch, "yolov11", "num_epochs", 12),
        (DetectionModelPatch, "yolov11", "patience", 2),
        (DetectionModelPatch, "yolov11", "optimizer", "sgd"),
        (DetectionModelPatch, "yolov11", "loss_function_box", "ciou"),
        (VQAModelPatch, "Qwen3-VL-2B-Instruct", "learning_rate", 2e-5),
    ),
)
def test_model_selection_schema_rejects_hpo_owned_fields(
    patch_type, architecture, forbidden_field, value
):
    with pytest.raises(ValidationError):
        patch_type.model_validate({
            "model": {
                "model_architecture": architecture,
                "description": "Architecture-only selection.",
                forbidden_field: value,
            },
            "rationale": "Selection does not own training configuration.",
        })


@pytest.mark.parametrize(
    ("patch_type", "model"),
    (
        (ClassificationModelPatch, {"model_architecture": "resnet50", "description": "Classifier."}),
        (DetectionModelPatch, {"model_architecture": "yolov8", "description": "Detector."}),
        (VQAModelPatch, {"model_architecture": "Qwen3-VL-2B-Instruct", "description": "VQA."}),
    ),
)
def test_agent_model_schema_does_not_accept_architecture_family(patch_type, model):
    assert "architecture_family" not in patch_type.model_json_schema()["$defs"][
        patch_type.model_fields["model"].annotation.__name__
    ]["properties"]
    with pytest.raises(ValidationError):
        patch_type.model_validate({
            "model": {**model, "architecture_family": "efficientnet"},
            "rationale": "Agent output must not contain derived metadata.",
        })


def test_single_saved_model_shape_is_consumed_by_hyperparameter_retrieval():
    state = PipelineState(
        task="classification",
        selected_model_info={
            "model": {
                "model_architecture": "resnet50",
                "architecture_family": "resnet",
                "description": "Single selected model.",
            }
        },
    )

    context = build_hyperparameter_context(state)

    assert context["selected_model_id"] == "resnet50"
    assert context["base_recipe"] is not None


def test_model_decision_evidence_resolves_fact_sources_and_links():
    graph_context = build_model_selection_context(
        PipelineState(task="classification"), top_k=100
    )
    candidate = next(
        item for item in graph_context["candidate_models"]
        if item["model"]["id"] == "resnet50"
    )
    record = build_model_selection_decision_evidence(
        {"model": {"model_architecture": "resnet50"}},
        "Selected for its retrieved trade-offs.",
        graph_context,
    )

    assert record["grounded"] is True
    assert record["grounding"]["status"] == "partially_grounded"
    assert record["grounding"]["evidence_coverage"] == 1.0
    assert record["retrieved_facts"][0]["id"] == candidate["model"]["id"]
    assert record["retrieved_facts"][0]["support_type"] == "direct_evidence"
    memory_facts = [
        fact for fact in record["retrieved_facts"]
        if fact["type"] == "inference_memory_estimate"
    ]
    assert memory_facts
    assert memory_facts[0]["support_type"] == "derived"
    assert memory_facts[0]["derivation"]["method"]
    source_ids = {source["id"] for source in record["evidence_sources"]}
    assert all(
        evidence_id in source_ids
        for fact in record["retrieved_facts"]
        for evidence_id in fact["evidence_ids"]
    )
    assert all(
        source["url"].startswith("https://")
        for source in record["evidence_sources"]
        if source["url"]
    )


def test_model_decision_does_not_attach_unrelated_candidate_evidence():
    graph_context = build_model_selection_context(PipelineState(task="classification"))
    record = build_model_selection_decision_evidence(
        {"model": {"model_architecture": "not_in_retrieved_shortlist"}},
        "An ungrounded choice.",
        graph_context,
    )

    assert record["grounded"] is False
    assert record["retrieved_facts"] == []
    assert record["evidence_sources"] == []


def test_required_initial_model_is_forced_into_limited_shortlist():
    state = PipelineState(
        task="classification",
        user_query="Please use the dinov2 vits14 and LoRA.",
        model_requirements=[{
            "name": "DINOv2 ViT-S/14",
            "backbone": "ViT-S14",
            "requirement_strength": "required",
            "training_mode": "lora",
        }],
    )

    context = build_model_selection_context(state, top_k=3)
    candidate_ids = {
        candidate["model"]["id"] for candidate in context["candidate_models"]
    }

    assert context["required_model_id"] == "dinov2_vits14"
    assert "dinov2_vits14" in candidate_ids
    required = next(
        candidate for candidate in context["candidate_models"]
        if candidate["model"]["id"] == "dinov2_vits14"
    )
    assert "explicit_user_requirement" in required["shortlist_roles"]


def test_select_model_deterministically_applies_initial_required_model(monkeypatch):
    import routers.planning as planning

    output = ClassificationModelPatch.model_validate({
        "model": {
            "model_architecture": "efficientnet_b0",
            "description": "Agent preference that must be overridden.",
        },
        "rationale": "Efficient baseline.",
    })

    async def fake_run(_agent, input):
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={
            "task": "classification",
            "user_query": "Please use DINOv2 ViT-S/14 and LoRA.",
            "model_requirements": [{
                "name": "DINOv2 ViT-S/14",
                "requirement_strength": "required",
                "training_mode": "lora",
            }],
        },
        job_id="required-initial-model",
        use_graphrag=False,
    )))

    selected = result["context"]["selected_model_info"]["model"]
    assert selected["model_architecture"] == "dinov2_vits14"
    assert selected["architecture_family"] == "dinov2"


def test_select_model_repairs_unresolved_required_model_once(monkeypatch):
    import routers.planning as planning

    repaired = TaskExtractionPatch.model_validate({
        "task": "classification",
        "model_requirements": [{
            "name": "DINOv2 ViT-S/14",
            "requirement_strength": "required",
            "training_mode": "lora",
        }],
    })
    selection = ClassificationModelPatch.model_validate({
        "model": {
            "model_architecture": "efficientnet_b0",
            "description": "Agent choice overridden by the required model.",
        },
        "rationale": "Baseline selection.",
    })
    calls = []

    async def fake_run(agent, input):
        calls.append(agent)
        if agent is planning.task_interpretation_agent:
            return SimpleNamespace(final_output=repaired)
        return SimpleNamespace(final_output=selection)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={
            "task": "classification",
            "user_query": "Use the small DINOv2 patch-14 model.",
            "model_requirements": [{
                "name": "DINOv2 ViT-14",
                "requirement_strength": "required",
                "training_mode": "lora",
            }],
        },
        job_id="required-model-repair",
        use_graphrag=False,
    )))

    assert calls[0] is planning.task_interpretation_agent
    assert result["context"]["selected_model_info"]["model"]["model_architecture"] == "dinov2_vits14"
    assert len(result["llm_attempts"]) == 2


def test_select_model_rejects_required_lora_for_unsupported_classifier():
    import routers.planning as planning

    with pytest.raises(planning.HTTPException) as exc_info:
        asyncio.run(planning.select_model(planning.StateRequest(
            context={
                "task": "classification",
                "user_query": "Please use EfficientNet-B0 and LoRA.",
                "model_requirements": [{
                    "name": "EfficientNet-B0",
                    "requirement_strength": "required",
                    "training_mode": "lora",
                }],
            },
            job_id="invalid-required-lora",
        )))

    assert exc_info.value.status_code == 422
    assert "LoRA is not executable" in exc_info.value.detail["message"]


def test_internal_evidence_uses_a_repository_locator_not_a_fake_url():
    context = build_model_selection_context(
        PipelineState(task="classification"), top_k=100
    )
    record = build_model_selection_decision_evidence(
        {"model": {"model_architecture": "mobilenet_v2"}},
        "Selected an executable lightweight model.",
        context,
    )
    internal = next(
        source for source in record["evidence_sources"]
        if source["id"] == "evidence_classification_execution_registry"
    )

    assert internal["locator_type"] == "repository_path"
    assert internal["url"] is None
    assert internal["reference"].startswith("backend/")


def test_detection_family_selection_matches_only_same_generation_variant():
    graph_context = build_model_selection_context(
        PipelineState(task="detection"), top_k=100
    )
    record = build_model_selection_decision_evidence(
        {"model": {"model_architecture": "yolov11"}},
        "Selected the YOLO11 executable family.",
        graph_context,
    )

    assert record["grounded"] is True
    assert record["match_scope"] == "family_variant"
    assert record["retrieved_facts"][0]["id"].startswith("yolo11")

    unrelated = build_model_selection_decision_evidence(
        {"model": {"model_architecture": "yolov7"}},
        "Not represented in the graph shortlist.",
        graph_context,
    )
    assert unrelated["grounded"] is False


def test_select_model_endpoint_returns_and_persists_decision_evidence(monkeypatch):
    import routers.planning as planning

    output = ClassificationModelPatch.model_validate({
        "model": {
            "model_architecture": "efficientnet_b4",
            "description": "Grounded endpoint selection.",
        },
        "rationale": "Selected from retrieved EfficientNet facts.",
        "evaluated_candidates": [
            {
                "candidate_id": "efficientnet_b4",
                "advantages": ["Higher accuracy capacity."],
                "risks": ["Higher resource use."],
                "constraint_status": "feasible",
            },
            {
                "candidate_id": "clip_vit_b16",
                "advantages": ["Lower resource use."],
                "risks": ["Lower accuracy capacity."],
                "constraint_status": "feasible",
            },
        ],
    })

    async def fake_run(_agent, input):
        assert "model_selection_graph_context" in input
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={"task": "classification"},
        job_id="model-evidence",
    )))

    evidence = result["decision_evidence"]
    assert evidence == result["context"]["model_selection_decision_evidence"]
    assert evidence["decision_type"] == "model_selection"
    assert evidence["decision"]["model"]["model_architecture"] == "efficientnet_b4"
    assert evidence["rationale"] == "Selected from retrieved EfficientNet facts."
    assert evidence["grounded"] is True
    assert evidence["evidence_backed"] is True


def _detection_selection_output(*, selected_candidate_id=None):
    payload = {
        "model": {
            "model_architecture": "yolov11",
            "description": "Concrete YOLO11 variant selected from GraphRAG.",
        },
        "rationale": "YOLO11s provides the preferred retrieved overall-accuracy trade-off.",
        "evaluated_candidates": [
            {
                "candidate_id": "yolo11s",
                "advantages": ["Higher retrieved accuracy category."],
                "risks": ["More compute than compact variants."],
                "constraint_status": "feasible",
            },
            {
                "candidate_id": "fasterrcnn_resnet50_fpn",
                "advantages": ["Distinct two-stage detector architecture."],
                "risks": ["Higher inference latency."],
                "constraint_status": "feasible",
            },
        ],
    }
    if selected_candidate_id is not None:
        payload["selected_candidate_id"] = selected_candidate_id
    return DetectionModelPatch.model_validate(payload)


def test_small_object_detection_can_run_without_graphrag(monkeypatch):
    import routers.planning as planning

    async def fake_run(_agent, input):
        return SimpleNamespace(final_output=_detection_selection_output())

    monkeypatch.setattr(
        planning,
        "build_model_selection_context",
        lambda *_args: pytest.fail("GraphRAG context should not be built"),
    )
    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)

    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={
            "task": "detection",
            "robustness_requirements": {"object_scale": ["small"]},
        },
        job_id="small-object-graph-disabled",
        use_graphrag=False,
    )))

    assert result["context"]["use_graphrag"] is False
    assert result["context"]["model_selection_graph_context"] is None
    assert result["decision_evidence"]["grounded"] is False
    assert result["decision_evidence"]["selection_confidence"] == "standard"


def test_detection_endpoint_preserves_exact_graphrag_variant(monkeypatch):
    import routers.planning as planning

    async def fake_run(_agent, input):
        return SimpleNamespace(
            final_output=_detection_selection_output(selected_candidate_id="yolo11s")
        )

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={"task": "detection"},
        job_id="detection-exact-candidate",
    )))

    selected = result["context"]["selected_model_info"]
    assert selected["selected_candidate_id"] == "yolo11s"
    assert selected["model"]["model_architecture"] == "yolov11_s"
    assert result["decision_evidence"]["match_scope"] == "exact"


def test_detection_endpoint_repairs_ambiguous_family_selection(monkeypatch):
    import routers.planning as planning

    outputs = [
        _detection_selection_output(),
        _detection_selection_output(selected_candidate_id="yolo11s"),
    ]
    received_inputs = []

    async def fake_run(_agent, input):
        received_inputs.append(input)
        return SimpleNamespace(final_output=outputs.pop(0))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={"task": "detection"},
        job_id="detection-candidate-repair",
    )))

    assert len(received_inputs) == 2
    assert "selected_candidate_id must exactly equal" in received_inputs[1]
    selected = result["context"]["selected_model_info"]
    assert selected["selected_candidate_id"] == "yolo11s"
    assert selected["model"]["model_architecture"] == "yolov11_s"


def test_small_object_detection_repairs_non_diverse_and_memory_conflating_selection(monkeypatch):
    import routers.planning as planning

    invalid = DetectionModelPatch.model_validate({
        "selected_candidate_id": "yolo11s",
        "model": {
            "model_architecture": "yolov11",
            "description": "Small YOLO detector.",
        },
        "rationale": (
            "Its tiny inference VRAM leaves ample margin for training augmentation and batch size."
        ),
        "evaluated_candidates": [
            {
                "candidate_id": "yolo11s",
                "advantages": ["Highest retrieved overall mAP."],
                "risks": ["AP-small is unavailable."],
                "constraint_status": "feasible",
            },
            {
                "candidate_id": "yolo12n",
                "advantages": ["Strong overall mAP."],
                "risks": ["AP-small is unavailable."],
                "constraint_status": "feasible",
            },
        ],
    })
    valid = DetectionModelPatch.model_validate({
        "selected_candidate_id": "yolo11s",
        "model": {
            "model_architecture": "yolov11",
            "description": "Small YOLO detector.",
        },
        "rationale": (
            "YOLO11s has the strongest retrieved overall mAP. Its separate training-hardware "
            "requirement fits the GPU. Inference memory is not proof of training feasibility."
        ),
        "evaluated_candidates": [
            {
                "candidate_id": "yolo11s",
                "advantages": ["Highest retrieved overall mAP."],
                "risks": ["AP-small is unavailable."],
                "constraint_status": "feasible",
            },
            {
                "candidate_id": "fasterrcnn_resnet50_fpn",
                "advantages": ["Two-stage localization alternative."],
                "risks": ["Lower retrieved overall COCO AP."],
                "constraint_status": "feasible",
            },
            {
                "candidate_id": "retinanet_resnet50_fpn",
                "advantages": ["FPN architecture alternative."],
                "risks": ["Lower retrieved overall COCO AP."],
                "constraint_status": "feasible",
            },
        ],
        "uncertainties": [
            "Comparable AP-small evidence is unavailable for the retrieved candidates."
        ],
    })
    outputs = [invalid, valid]
    received_inputs = []

    async def fake_run(_agent, input):
        received_inputs.append(input)
        return SimpleNamespace(final_output=outputs.pop(0))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={
            "task": "detection",
            "robustness_requirements": {"object_scale": ["small"]},
        },
        job_id="small-object-diverse-comparison",
    )))

    assert len(received_inputs) == 2
    repair = received_inputs[1]
    assert '"inference_memory_used_as_training_evidence": true' in repair
    assert '"TwoStageRegionProposalDetector"' in repair
    evidence = result["decision_evidence"]
    assert len(evidence["evaluated_candidates"]) == 3
    assert evidence["uncertainties"] == [
        "Comparable AP-small evidence is unavailable for the retrieved candidates."
    ]


def test_small_object_detection_accepts_three_candidates_across_two_architectures(monkeypatch):
    """Regression for msrg6tibv6sfolehvq: a third architecture is preferred, not required."""
    import routers.planning as planning

    output = DetectionModelPatch.model_validate({
        "selected_candidate_id": "yolo11s",
        "model": {
            "model_architecture": "yolov11",
            "description": "Feasible YOLO11 small variant.",
        },
        "rationale": (
            "YOLO11s has the strongest retrieved overall mAP. Its inference VRAM leaves ample "
            "margin for training batch size; AP-small remains unverified."
        ),
        "evaluated_candidates": [
            {
                "candidate_id": "yolo11s",
                "advantages": ["Strongest retrieved overall mAP."],
                "risks": ["AP-small is unavailable."],
                "constraint_status": "feasible",
            },
            {
                "candidate_id": "yolov8s",
                "advantages": ["Strong alternative overall mAP."],
                "risks": ["Same broad detector architecture type."],
                "constraint_status": "feasible",
            },
            {
                "candidate_id": "fasterrcnn_resnet50_fpn",
                "advantages": ["Required two-stage localization challenger."],
                "risks": ["Lower retrieved overall COCO AP."],
                "constraint_status": "feasible",
            },
        ],
        "uncertainties": ["Comparable AP-small evidence is unavailable."],
    })
    calls = []

    async def fake_run(_agent, input):
        calls.append(input)
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={
            "task": "detection",
            "robustness_requirements": {"object_scale": ["small"]},
            "performance_requirements": {
                "primary_metric": "mAP@0.5:0.95",
                "target_value": 0.35,
                "target_is_hard": False,
                "accuracy_category": "MediumHigh",
            },
            "available_hardware": {
                "hardware_category": "ConsumerGPU",
                "gpu_type": "RTX 2060",
                "gpu_count": 1,
                "vram_gb": 6,
                "ram_gb": 16,
            },
        },
        job_id="small-object-two-architecture-pass",
    )))

    assert len(calls) == 1
    assert result["context"]["selected_model_info"]["selected_candidate_id"] == "yolo11s"
    evidence = result["decision_evidence"]
    assert evidence["selection_confidence"] == "conditional"
    assert len(evidence["evaluated_candidates"]) == 3
    assert len(evidence["comparison_warnings"]) == 2
    assert "third detector architecture" in evidence["comparison_warnings"][0].lower()
    assert "do not establish training-memory feasibility" in evidence["comparison_warnings"][1].lower()


def test_select_model_sanitizes_accidental_cjk_in_english_rationale(monkeypatch):
    import routers.planning as planning

    output = ClassificationModelPatch.model_validate({
        "model": {
            "model_architecture": "mobilenet_v2",
            "description": "Lightweight classifier.",
        },
        "rationale": "The smallest model meets the硬 targets.",
    })

    async def fake_run(_agent, input):
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={"task": "classification"},
        job_id="rationale-language",
        use_graphrag=False,
    )))

    assert result["decision_evidence"]["rationale"] == (
        "The smallest model meets the targets."
    )


def test_interpretation_postprocessor_derives_strength_accuracy_and_robustness():
    from routers.planning import apply_qualitative_constraint_fallbacks

    extracted = TaskExtractionPatch(
        task="detection",
        classes=["traffic light"],
        performance_requirements={
            "primary_metric": "mAP@0.5:0.95",
            "target_value": 0.30,
        },
    )
    apply_qualitative_constraint_fallbacks(
        extracted,
        "Detect tiny traffic lights at night and in rain; mAP 0.30 should be achieved.",
    )

    assert extracted.performance_requirements.accuracy_category == "MediumHigh"
    assert extracted.constraint_strengths.accuracy == "soft"
    assert extracted.robustness_requirements.object_scale == ["small"]
    assert extracted.robustness_requirements.lighting == ["night"]
    assert extracted.robustness_requirements.weather == ["rain"]


def test_interpretation_postprocessor_distinguishes_preference_and_hard_wording():
    from routers.planning import apply_qualitative_constraint_fallbacks

    preferred = TaskExtractionPatch(task="detection", classes=["car"])
    apply_qualitative_constraint_fallbacks(
        preferred, "Preferably keep latency below 50 ms."
    )
    assert preferred.constraint_strengths.latency == "preference"

    required = TaskExtractionPatch(task="detection", classes=["car"])
    apply_qualitative_constraint_fallbacks(
        required, "Latency must remain below 50 ms."
    )
    assert required.constraint_strengths.latency == "hard"
