import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cvmodellearning.agents.model_selection_agents import (
    ClassificationModelPatch,
    DetectionModelPatch,
    VQAModelPatch,
)
from cvmodellearning.agents.hyperparameter_agents import selected_detection_model_id
from cvmodellearning.graphrag.hyperparameter_context import build_hyperparameter_context
from cvmodellearning.graphrag.model_selection_context import (
    MODEL_SIZE_ORDER,
    build_model_selection_context,
)
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


def test_very_low_memory_selects_smallest_model_meeting_accuracy_preference():
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
    assert context["deterministic_recommendation"]["model_id"] == "efficientnet_b1"
    assert "efficientnet_b0" in candidate_ids
    assert "mobilenet_v3_large" in candidate_ids
    assert "efficientnet_b4" not in candidate_ids
    assert all(
        item["model"]["model_size_category"] in {"Nano", "Small"}
        for item in context["candidate_models"]
    )


def test_prompt_six_policy_selects_b0_and_excludes_b4():
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
    assert context["deterministic_recommendation"]["model_id"] == "efficientnet_b0"
    assert candidate_ids[0] == "efficientnet_b0"
    assert "efficientnet_b1" in candidate_ids
    assert "mobilenet_v3_large" in candidate_ids
    assert "efficientnet_b4" not in candidate_ids
    assert context["rejected_counts"]["memory_category"] > 0


def test_default_graphrag_shortlist_is_capped_at_seven():
    context = build_model_selection_context(PipelineState(task="classification"))

    assert len(context["candidate_models"]) == 7
    assert [item["rank"] for item in context["candidate_models"]] == list(range(1, 8))


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

    smallest_qualified = build_model_selection_context(PipelineState(
        task="classification",
        performance_requirements={
            "primary_metric": "accuracy",
            "target_value": 0.77,
            "target_is_hard": True,
        },
    ))
    assert smallest_qualified["deterministic_recommendation"]["model_id"] == "efficientnet_b0"


def test_soft_map_target_prefers_smallest_qualified_detector_over_accuracy_category():
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
    recommendation_id = context["deterministic_recommendation"]["model_id"]
    recommendation = next(item for item in candidates if item["model"]["id"] == recommendation_id)
    assert recommendation["model"]["model_size_category"] == "Nano"


def test_unverified_hard_cpu_limit_allows_only_one_balanced_family_size_step():
    context = build_model_selection_context(PipelineState(
        task="detection",
        performance_requirements={
            "primary_metric": "mAP@0.5:0.95",
            "target_value": 0.30,
            "target_is_hard": False,
            "accuracy_category": "Medium",
            "latency_category": "Medium",
        },
        deployment_constraints={
            "max_cpu_latency_ms": 500,
            "hard_limits": ["max_cpu_latency_ms"],
        },
        available_hardware={"hardware_category": "ConsumerCPU"},
    ), top_k=100)

    recommendation = context["deterministic_recommendation"]
    selected = context["candidate_models"][0]
    fallback = recommendation["fallback_model"]
    selected_size = MODEL_SIZE_ORDER.index(selected["model"]["model_size_category"])
    fallback_candidate = next(
        item for item in context["candidate_models"]
        if item["model"]["id"] == fallback["model_id"]
    )
    fallback_size = MODEL_SIZE_ORDER.index(
        fallback_candidate["model"]["model_size_category"]
    )

    assert recommendation["policy"] == "balanced_capacity_with_resource_headroom"
    assert selected_size == fallback_size + 1
    assert selected["model"]["model_family"] == fallback_candidate["model"]["model_family"]
    assert recommendation["cpu_latency"]["status"] == "unverified"
    assert recommendation["cpu_latency"]["measured_ms"] is None
    assert recommendation["unverified_constraints"] == ["max_cpu_latency_ms"]
    assert recommendation["deployment_validation_required"] is True


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


def test_select_model_endpoint_overrides_agent_choice_that_violates_memory_policy(monkeypatch):
    import routers.planning as planning

    output = ClassificationModelPatch.model_validate({
        "model": {
            "model_architecture": "efficientnet_b4",
            "description": "Agent preferred the larger model.",
        },
        "rationale": "Agent rationale.",
    })

    async def fake_run(_agent, input):
        assert "deterministic_recommendation" in input
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)
    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={
            "task": "classification",
            "performance_requirements": {
                "primary_metric": "accuracy",
                "accuracy_category": "High",
            },
            "deployment_constraints": {"memory_category": "VeryLow"},
            "available_hardware": {"hardware_category": "ConsumerCPU"},
        },
        job_id="memory-policy",
    )))

    selected = result["context"]["selected_model_info"]["model"]
    assert selected["model_architecture"] == "efficientnet_b1"
    assert selected["architecture_family"] == "efficientnet"
    assert "replaced the agent choice" in result["decision_evidence"]["rationale"]


@pytest.mark.parametrize("use_graphrag", (False, True))
@pytest.mark.parametrize("use_policy_registry", (False, True))
def test_detection_model_resolution_works_for_all_feature_flag_combinations(
    monkeypatch,
    use_graphrag,
    use_policy_registry,
):
    import routers.planning as planning

    output = DetectionModelPatch.model_validate({
        "model": {
            "model_architecture": "yolov10",
            "description": "Executable detector.",
        },
        "rationale": "Selected YOLOv10.",
    })
    graph_context = build_model_selection_context(
        PipelineState(task="detection"), top_k=100
    )
    recommendation = next(
        candidate
        for candidate in graph_context["candidate_models"]
        if candidate["model"]["id"] == "yolov10n"
    )
    graph_context["deterministic_recommendation"] = {
        "model_id": "yolov10n",
        "model_name": recommendation["model"]["model_name"],
        "policy": "smallest_feasible_model_meeting_hard_targets",
        "reason": "Test recommendation.",
    }

    async def fake_run(_agent, input):
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "build_model_selection_context", lambda _state: graph_context)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *_args: None)

    result = asyncio.run(planning.select_model(planning.StateRequest(
        context={"task": "detection"},
        job_id="detection-feature-flags",
        use_graphrag=use_graphrag,
        use_policy_registry=use_policy_registry,
    )))

    selected = result["context"]["selected_model_info"]["model"]
    assert selected["model_architecture"] == (
        "yolov10n" if use_graphrag else "yolov10"
    )
    assert selected["architecture_family"] == "yolo"
    assert selected_detection_model_id(result["context"]) == "yolov10_n"
