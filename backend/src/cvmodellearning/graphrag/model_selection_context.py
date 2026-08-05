from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx

from cvmodellearning.graphrag.build_graph import build_graph
from cvmodellearning.models.registry import is_executable_model_reference
from cvmodellearning.paths import PROJECT_ROOT
from cvmodellearning.schemas.interpretation_schema import PipelineState


USE_MODEL_SELECTION_GRAPHRAG = os.getenv("USE_MODEL_SELECTION_GRAPHRAG", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

TASK_ID_BY_STATE_TASK = {
    "classification": "image_classification",
    "detection": "object_detection",
    "visual question answering": "visual_question_answering",
}

LATENCY_ORDER = ["VeryLow", "Low", "Medium", "MediumHigh", "High", "VeryHigh"]
ACCURACY_ORDER = ["VeryLow", "Low", "Medium", "MediumHigh", "High", "VeryHigh"]

FALLBACK_HARDWARE_CATEGORIES = ["ConsumerCPU", "EdgeDevice"]

HARDWARE_ORDER = {
    "EdgeDevice": 0,
    "ConsumerCPU": 1,
    "ConsumerCPU | EdgeDevice": 1,
    "ConsumerGPU": 2,
    "DataCenterGPU": 3,
}

MODEL_SIZE_ORDER = ["Nano", "Small", "Medium", "Large", "VeryLarge"]

MEMORY_ALLOWED_SIZES = {
    "VeryLow": {"Nano", "Small"},
    "Low": {"Nano", "Small"},
    "Medium": {"Nano", "Small", "Medium"},
    "MediumHigh": {"Nano", "Small", "Medium", "Large"},
    "High": set(MODEL_SIZE_ORDER),
}

MEMORY_PARAMETER_LIMITS_M = {
    "VeryLow": 8.0,
    "Low": 15.0,
    "Medium": 50.0,
}

METRIC_ALIASES = {
    "accuracy": {"accuracy", "top1_acc", "top1_accuracy", "acc"},
    "top1": {"top1_acc", "top1_accuracy"},
    "top-1": {"top1_acc", "top1_accuracy"},
    "map": {"map50_95", "ap50"},
    "map50_95": {"map50_95"},
    "map50-95": {"map50_95"},
    "ap50": {"ap50"},
    "latency": {"latency_ms"},
    "latency_ms": {"latency_ms"},
    "fps": {"fps"},
    "throughput": {"throughput_img_s", "fps"},
    "throughput_img_s": {"throughput_img_s"},
    "parameters": {"params_m"},
    "params": {"params_m"},
    "params_m": {"params_m"},
    "flops": {"flops_b"},
    "flops_b": {"flops_b"},
}

MODEL_FIELDS = [
    "task",
    "model_name",
    "model_family",
    "architecture_type",
    "pretrained_available",
    "fine_tuning_supported",
    "lora_supported",
    "quantization_supported",
    "model_size_category",
    "latency_category",
    "accuracy_category",
    "limitations",
    "incompatible_training_backends",
    "evidence_ids",
]

MEMORY_FIELDS = [
    "model_id",
    "precision_mode",
    "batch_size",
    "context_tokens_for_kv",
    "params_m",
    "flops_b",
    "weight_memory_gb",
    "activation_workspace_gb",
    "kv_cache_gb",
    "runtime_overhead_gb",
    "total_estimated_vram_gb",
    "practical_min_vram_gb",
    "recommended_hardware_category",
    "recommended_hardware_profile",
    "calculation_method",
    "confidence",
    "notes",
    "evidence_ids",
]

BENCHMARK_FIELDS = [
    "model_id",
    "task_id",
    "dataset",
    "metric_id",
    "metric_value",
    "hardware_profile_id",
    "image_size",
    "training_recipe_id",
    "confidence",
    "notes",
    "evidence_ids",
]

METRIC_FIELDS = [
    "task",
    "metric_name",
    "task_id",
    "optimization_direction",
    "primary_or_secondary",
    "description",
    "evidence_ids",
]

HARDWARE_FIELDS = [
    "hardware_name",
    "hardware_category",
    "gpu_memory_gb",
    "num_gpus",
    "cpu_only",
    "latency_category",
    "inference_runtime",
    "precision_mode",
    "benchmark_device",
    "evidence_ids",
]

EVIDENCE_FIELDS = [
    "source_type",
    "title",
    "source_owner",
    "year",
    "url_or_reference",
    "claim_supported",
    "confidence",
    "notes",
]


@lru_cache(maxsize=1)
def get_model_selection_graph() -> nx.MultiDiGraph:
    return build_graph(PROJECT_ROOT / "ontology_data")


def build_model_selection_context(state: PipelineState, top_k: int = 7) -> Dict[str, Any]:
    """Retrieve model-selection facts from the NetworkX knowledge graph."""
    graph = get_model_selection_graph()
    task_id = _normalize_task_id(state.task)
    filters = _active_filters(state, task_id)

    matching_models = []
    rejected_counts: Dict[str, int] = {}
    for model_id, attrs in graph.nodes(data=True):
        if attrs.get("source_csv") != "models.csv":
            continue
        if task_id and _clean(attrs.get("task")) != task_id:
            rejected_counts["task"] = rejected_counts.get("task", 0) + 1
            continue
        if not is_executable_model_reference(state.task, str(model_id)):
            rejected_counts["not_executable"] = rejected_counts.get("not_executable", 0) + 1
            continue

        candidate = _candidate_from_graph(graph, model_id, task_id)
        passes, applied, rejected_by = _passes_filters(candidate, filters)
        if not passes:
            rejected_counts[rejected_by or "unknown"] = rejected_counts.get(rejected_by or "unknown", 0) + 1
            continue

        candidate["matched_filters"] = applied
        candidate["constraint_warnings"] = [
            item.removeprefix("unverified: ")
            for item in applied
            if item.startswith("unverified: ")
        ]
        matching_models.append(candidate)

    matching_models.sort(key=lambda candidate: _candidate_rank_key(candidate, filters))
    recommendation, fallback = _balanced_recommendation(matching_models, filters)
    if recommendation is not None and recommendation is not matching_models[0]:
        matching_models.remove(recommendation)
        matching_models.insert(0, recommendation)
    selected_models = matching_models[:top_k]
    for rank, candidate in enumerate(selected_models, start=1):
        candidate["rank"] = rank
        candidate["cpu_latency"] = _cpu_latency_assessment(candidate, filters)
        candidate.pop("_all_model_benchmark_results", None)

    recommendation = selected_models[0] if selected_models else None
    constraint_warnings = sorted({
        warning
        for candidate in selected_models
        for warning in candidate.get("constraint_warnings", [])
    })
    return {
        "enabled": True,
        "source": "NetworkX knowledge graph from backend/ontology_data",
        "retrieval_strategy": (
            "Sequential graph filtering over models.csv nodes, then traversal to related "
            "model_inference_memory_estimates, model_benchmark_results, evaluation_metrics, "
            "hardware_profiles, datasets, and evidence_sources."
        ),
        "task_filter": task_id,
        "filters": filters,
        "training_hardware": (
            state.training_hardware.model_dump(mode="json")
            if state.training_hardware
            else None
        ),
        "rejected_counts": rejected_counts,
        "constraint_warnings": constraint_warnings,
        "candidate_models": selected_models,
        "deterministic_recommendation": (
            {
                "model_id": recommendation["model"]["id"],
                "model_name": recommendation["model"].get("model_name"),
                "policy": "balanced_capacity_with_resource_headroom",
                "reason": (
                    "Best supported capacity within hard filters, with unverified CPU-only "
                    "latency expansion limited to one model-size tier."
                ),
                "cpu_latency": recommendation.get("cpu_latency"),
                "unverified_constraints": (
                    ["max_cpu_latency_ms"]
                    if (recommendation.get("cpu_latency") or {}).get("status") == "unverified"
                    else []
                ),
                "deployment_validation_required": (
                    (recommendation.get("cpu_latency") or {}).get("status") == "unverified"
                ),
                "fallback_model": (
                    {
                        "model_id": fallback["model"]["id"],
                        "model_name": fallback["model"].get("model_name"),
                        "reason": "Lower-compute feasible fallback for target-hardware latency validation.",
                    }
                    if fallback is not None and fallback is not recommendation
                    else None
                ),
            }
            if recommendation
            else None
        ),
        "instructions_for_selector": (
            "Use these graph-retrieved candidates as grounded model-selection context. "
            "The candidates already satisfy all available filters listed above. "
            "Compare model facts, inference memory estimates, benchmark values, metrics, "
            "limitations, and evidence before choosing the final model. Treat inference-memory "
            "estimates as deployment facts, not proof that full fine-tuning fits the same hardware; "
            "when training hardware is unknown, prefer a lower-resolution variant unless an explicit "
            "accuracy requirement justifies the additional compute. Do not invent graph facts."
        ),
    }


def _cpu_latency_values(candidate: Dict[str, Any]) -> list[float]:
    return [
        value
        for item in candidate.get("_all_model_benchmark_results", [])
        if _clean(item.get("metric_id")) == "latency_ms"
        and (item.get("hardware_profile") or {}).get("cpu_only") in {True, "true", "True"}
        if (value := _float_or_none(item.get("metric_value"))) is not None
    ]


def _cpu_latency_assessment(candidate: Dict[str, Any], filters: Dict[str, Any]) -> dict[str, Any]:
    limit = _float_or_none(
        filters.get("max_cpu_latency_ms", filters.get("max_cpu_latency_ms_preference"))
    )
    values = _cpu_latency_values(candidate)
    if values:
        measured = min(values)
        return {
            "status": "verified" if limit is None or measured <= limit else "failed",
            "measured_ms": measured,
            "limit_ms": limit,
            "feasibility": "measured",
        }
    return {
        "status": "unverified" if limit is not None else "not_requested",
        "measured_ms": None,
        "limit_ms": limit,
        "feasibility": "estimated_from_model_complexity" if limit is not None else "not_assessed",
    }


def _balanced_recommendation(
    candidates: list[Dict[str, Any]],
    filters: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Allow one same-family size step when a hard CPU limit lacks measurements."""
    if not candidates:
        return None, None
    baseline = candidates[0]
    if filters.get("max_cpu_latency_ms") is None or _cpu_latency_values(baseline):
        return baseline, None

    base_model = baseline.get("model") or {}
    base_size = _category_index(base_model.get("model_size_category"), MODEL_SIZE_ORDER)
    base_accuracy = _category_index(base_model.get("accuracy_category"), ACCURACY_ORDER)
    latency_preference = filters.get("latency_preference")
    if base_size is None:
        return baseline, None

    for candidate in candidates[1:]:
        model = candidate.get("model") or {}
        size = _category_index(model.get("model_size_category"), MODEL_SIZE_ORDER)
        accuracy = _category_index(model.get("accuracy_category"), ACCURACY_ORDER)
        if model.get("model_family") != base_model.get("model_family"):
            continue
        if size != base_size + 1:
            continue
        if latency_preference and not _latency_at_most(
            model.get("latency_category"), latency_preference
        ):
            continue
        if accuracy is None or base_accuracy is None or accuracy <= base_accuracy:
            continue
        return candidate, baseline
    return baseline, None


def _candidate_rank_key(candidate: Dict[str, Any], filters: Dict[str, Any]) -> tuple:
    """Choose the smallest feasible model that best meets the requested performance."""
    model = candidate.get("model") or {}
    memory = candidate.get("model_inference_memory_estimate") or {}
    accuracy_index = _category_index(model.get("accuracy_category"), ACCURACY_ORDER)
    latency_index = _category_index(model.get("latency_category"), LATENCY_ORDER)
    minimum_vram = _float_or_none(memory.get("practical_min_vram_gb"))
    preference_penalty = _soft_preference_penalty(candidate, filters)
    benchmark_goal = filters.get("benchmark_target") or filters.get("benchmark_preference")
    if benchmark_goal:
        size_index = _category_index(model.get("model_size_category"), MODEL_SIZE_ORDER)
        total_memory = _float_or_none(memory.get("total_estimated_vram_gb"))
        params_m = _float_or_none(memory.get("params_m"))
        flops_b = _float_or_none(memory.get("flops_b"))
        meets_numeric_target = bool(_benchmarks_that_satisfy_target(
            candidate.get("_all_model_benchmark_results") or candidate.get("model_benchmark_results") or [],
            benchmark_goal.get("primary_metric"),
            benchmark_goal.get("target_value"),
        ))
        return (
            preference_penalty,
            0 if meets_numeric_target else 1,
            float(size_index if size_index is not None else len(MODEL_SIZE_ORDER)),
            float(params_m if params_m is not None else float("inf")),
            float(total_memory if total_memory is not None else float("inf")),
            float(flops_b if flops_b is not None else float("inf")),
            -float(accuracy_index if accuracy_index is not None else -1),
            str(model.get("id", "")),
        )
    if filters.get("memory_category") or any(
        filters.get(field) is not None
        for field in ("max_runtime_memory_mb", "max_model_size_mb", "max_parameters_m")
    ):
        size_index = _category_index(model.get("model_size_category"), MODEL_SIZE_ORDER)
        total_memory = _float_or_none(memory.get("total_estimated_vram_gb"))
        params_m = _float_or_none(memory.get("params_m"))
        flops_b = _float_or_none(memory.get("flops_b"))
        preference_index = _category_index(filters.get("accuracy_preference"), ACCURACY_ORDER)
        meets_preference = (
            preference_index is None
            or (accuracy_index is not None and accuracy_index >= preference_index)
        )
        return (
            preference_penalty,
            0 if meets_preference else 1,
            float(size_index if size_index is not None else len(MODEL_SIZE_ORDER)),
            float(params_m if params_m is not None else float("inf")),
            float(total_memory if total_memory is not None else float("inf")),
            float(flops_b if flops_b is not None else float("inf")),
            -float(accuracy_index if accuracy_index is not None else -1),
            str(model.get("id", "")),
        )
    return (
        preference_penalty,
        -float(accuracy_index if accuracy_index is not None else -1),
        float(latency_index if latency_index is not None else len(LATENCY_ORDER)),
        float(minimum_vram if minimum_vram is not None else float("inf")),
        str(model.get("id", "")),
    )


def _soft_preference_penalty(candidate: Dict[str, Any], filters: Dict[str, Any]) -> int:
    """Count known soft-target violations without rejecting candidates with missing data."""
    model = candidate.get("model") or {}
    memory = candidate.get("model_inference_memory_estimate") or {}
    penalty = 0

    latency_preference = filters.get("latency_preference")
    if latency_preference and not _latency_at_most(model.get("latency_category"), latency_preference):
        penalty += 1

    runtime_limit = _float_or_none(filters.get("max_runtime_memory_mb_preference"))
    runtime_gb = _float_or_none(memory.get("total_estimated_vram_gb"))
    if runtime_limit is not None and runtime_gb is not None and runtime_gb * 1024 > runtime_limit:
        penalty += 1

    model_size_limit = _float_or_none(filters.get("max_model_size_mb_preference"))
    weight_gb = _float_or_none(memory.get("weight_memory_gb"))
    if model_size_limit is not None and weight_gb is not None and weight_gb * 1024 > model_size_limit:
        penalty += 1

    parameter_limit = _float_or_none(filters.get("max_parameters_m_preference"))
    params_m = _float_or_none(memory.get("params_m"))
    if parameter_limit is not None and params_m is not None and params_m > parameter_limit:
        penalty += 1

    cpu_limit = _float_or_none(filters.get("max_cpu_latency_ms_preference"))
    if cpu_limit is not None:
        cpu_latencies = [
            _float_or_none(item.get("metric_value"))
            for item in candidate.get("_all_model_benchmark_results", [])
            if _clean(item.get("metric_id")) == "latency_ms"
            and (item.get("hardware_profile") or {}).get("cpu_only") in {True, "true", "True"}
        ]
        known_latencies = [value for value in cpu_latencies if value is not None]
        if known_latencies and min(known_latencies) > cpu_limit:
            penalty += 1

    return penalty


def disabled_model_selection_context() -> Dict[str, Any]:
    return {
        "enabled": False,
        "source": "NetworkX knowledge graph from backend/ontology_data",
        "candidate_models": [],
        "instructions_for_selector": "GraphRAG model-selection context is disabled globally.",
    }


def format_model_selection_context(context: Dict[str, Any]) -> str:
    """Format graph facts so model-selection agents can read them as additional context."""
    if not context.get("enabled", True):
        return "GraphRAG model-selection context is disabled globally."

    candidates = context.get("candidate_models", [])
    if not candidates:
        return (
            "GraphRAG model-selection context: no model candidates matched the active filters.\n"
            f"Active filters: {_format_mapping(context.get('filters') or {})}\n"
            f"Rejected counts: {_format_mapping(context.get('rejected_counts') or {})}"
        )

    lines = [
        "GraphRAG model-selection context",
        f"Source: {context.get('source')}",
        f"Retrieval: {context.get('retrieval_strategy')}",
        f"Task filter: {context.get('task_filter') or 'none'}",
        f"Active filters: {_format_mapping(context.get('filters') or {})}",
        f"Training hardware (soft feasibility): {_format_mapping(context.get('training_hardware') or {})}",
        "Use: these are the highest-ranked executable graph candidates that satisfy every available filter.",
        "",
    ]

    for index, candidate in enumerate(candidates, start=1):
        model = candidate["model"]
        memory = candidate.get("model_inference_memory_estimate") or {}
        lines.extend(
            [
                f"{index}. {model.get('model_name')} ({model.get('id')})",
                (
                    "   Model facts: "
                    f"task={model.get('task')}, family={model.get('model_family')}, "
                    f"architecture={model.get('architecture_type')}, size={model.get('model_size_category')}, "
                    f"latency_category={model.get('latency_category')}, "
                    f"accuracy_category={model.get('accuracy_category')}, "
                    f"pretrained={model.get('pretrained_available')}, "
                    f"fine_tuning={model.get('fine_tuning_supported')}, "
                    f"lora={model.get('lora_supported')}, quantization={model.get('quantization_supported')}"
                ),
                f"   Matched filters: {'; '.join(candidate.get('matched_filters') or ['none'])}",
            ]
        )
        if model.get("limitations"):
            lines.append(f"   Limitations: {model.get('limitations')}")
        if memory:
            lines.append(
                "   Inference memory estimate: "
                f"precision={memory.get('precision_mode')}, batch_size={memory.get('batch_size')}, "
                f"params_m={memory.get('params_m')}, flops_b={memory.get('flops_b')}, "
                f"total_estimated_vram_gb={memory.get('total_estimated_vram_gb')}, "
                f"practical_min_vram_gb={memory.get('practical_min_vram_gb')}, "
                f"recommended_hardware_category={memory.get('recommended_hardware_category')}, "
                f"recommended_hardware_profile={memory.get('recommended_hardware_profile')}, "
                f"confidence={memory.get('confidence')}"
            )
            if memory.get("notes"):
                lines.append(f"   Memory notes: {memory.get('notes')}")

        if candidate.get("model_benchmark_results"):
            lines.append("   Benchmark results:")
            for benchmark in candidate["model_benchmark_results"]:
                metric = benchmark.get("metric") or {}
                hardware_profile = benchmark.get("hardware_profile") or {}
                metric_label = metric.get("metric_name") or benchmark.get("metric_id")
                hardware_label = hardware_profile.get("hardware_name") or benchmark.get("hardware_profile_id") or "unspecified hardware"
                lines.append(
                    f"   - {benchmark.get('dataset')} / {metric_label}: {benchmark.get('metric_value')} "
                    f"(metric_id={benchmark.get('metric_id')}, direction={metric.get('optimization_direction') or 'unknown'}, "
                    f"hardware={hardware_label}, image_size={benchmark.get('image_size') or 'unspecified'}, "
                    f"confidence={benchmark.get('confidence')})"
                )

        if candidate.get("evaluation_metrics"):
            lines.append("   Relevant evaluation metrics:")
            for metric in candidate["evaluation_metrics"]:
                lines.append(
                    f"   - {metric.get('metric_name')} ({metric.get('id')}): "
                    f"{metric.get('optimization_direction')}, {metric.get('primary_or_secondary')}; "
                    f"{metric.get('description')}"
                )

        if candidate.get("evidence_sources"):
            lines.append("   Evidence sources:")
            for source in candidate["evidence_sources"]:
                lines.append(
                    f"   - {source.get('title')} ({source.get('source_owner')}, {source.get('year')}), "
                    f"{source.get('source_type')}, confidence={source.get('confidence')}: "
                    f"{source.get('url_or_reference')}"
                )
        lines.append("")

    return "\n".join(lines).strip()


def summarize_model_selection_context(context: Dict[str, Any]) -> str:
    """Return a compact summary for the planning step history."""
    if not context.get("enabled", True):
        return "GraphRAG Model Suggestions: disabled globally."

    candidates = context.get("candidate_models", [])
    if not candidates:
        return (
            "GraphRAG Model Suggestions: no matching candidates "
            f"for task={context.get('task_filter')} with filters={_format_mapping(context.get('filters') or {})}."
        )

    labels = []
    for candidate in candidates:
        model = candidate.get("model", {})
        memory = candidate.get("model_inference_memory_estimate") or {}
        suffix = ""
        if memory.get("practical_min_vram_gb") not in (None, ""):
            suffix = f", min_vram={memory.get('practical_min_vram_gb')}GB"
        labels.append(f"{model.get('model_name')} ({model.get('id')}{suffix})")

    warning = " ".join(context.get("constraint_warnings") or [])
    return (
        f"GraphRAG Model Suggestions (task={context.get('task_filter')}, "
        f"filters={_format_mapping(context.get('filters') or {})}): "
        + "; ".join(labels)
        + (f" Warning: {warning}" if warning else "")
    )


def _active_filters(state: PipelineState, task_id: Optional[str]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if task_id:
        filters["task"] = task_id

    performance = state.performance_requirements
    if performance:
        if performance.latency_category:
            filters["latency_preference"] = performance.latency_category
        if performance.accuracy_category:
            filters["accuracy_preference"] = performance.accuracy_category
        if performance.primary_metric and performance.target_value is not None:
            target_filter = "benchmark_target" if performance.target_is_hard else "benchmark_preference"
            filters[target_filter] = {
                "primary_metric": performance.primary_metric,
                "target_value": performance.target_value,
            }

    hardware = state.available_hardware
    if hardware:
        if hardware.vram_gb is not None:
            filters["available_vram_gb"] = hardware.vram_gb
        if hardware.hardware_category:
            filters["hardware_category"] = hardware.hardware_category

    constraints = state.deployment_constraints
    if constraints:
        hard_limits = set(constraints.hard_limits)
        for field in (
            "memory_category",
            "max_runtime_memory_mb",
            "max_model_size_mb",
            "max_parameters_m",
            "max_cpu_latency_ms",
        ):
            value = getattr(constraints, field)
            if value is not None:
                if field == "memory_category" or field in hard_limits:
                    filters[field] = value
                else:
                    filters[f"{field}_preference"] = value

    return filters


def _candidate_from_graph(
    graph: nx.MultiDiGraph,
    model_id: str,
    task_id: Optional[str],
) -> Dict[str, Any]:
    model = _project_attrs(graph.nodes[model_id], MODEL_FIELDS, include_id=model_id)
    memory = _first_related_node(graph, model_id, "has_inference_memory_estimate")

    benchmarks = _related_nodes(graph, model_id, "has_benchmark_result")
    if task_id:
        benchmarks = [benchmark for benchmark in benchmarks if _clean(benchmark.get("task_id")) == task_id]
    benchmark_payloads = _ordered_benchmarks([_benchmark_payload(graph, benchmark) for benchmark in benchmarks])

    metrics = _evaluation_metrics_for_task(graph, task_id, benchmark_payloads)
    evidence = _evidence_sources(graph, [model, memory, *benchmarks, *metrics])

    return {
        "model": model,
        "model_inference_memory_estimate": _project_attrs(
            memory, MEMORY_FIELDS, include_id=memory.get("id")
        ) if memory else None,
        "_all_model_benchmark_results": benchmark_payloads,
        "model_benchmark_results": benchmark_payloads[:10],
        "evaluation_metrics": [_project_attrs(metric, METRIC_FIELDS, include_id=metric.get("id")) for metric in metrics],
        "evidence_sources": evidence,
    }


def _passes_filters(candidate: Dict[str, Any], filters: Dict[str, Any]) -> tuple[bool, List[str], Optional[str]]:
    model = candidate.get("model") or {}
    memory = candidate.get("model_inference_memory_estimate") or {}
    applied = []

    task = filters.get("task")
    if task:
        if _clean(model.get("task")) != task:
            return False, applied, "task"
        applied.append(f"task is {task}")

    latency = filters.get("latency_category_at_most")
    if latency:
        if not _latency_at_most(model.get("latency_category"), latency):
            return False, applied, "latency_category"
        applied.append(f"latency_category {model.get('latency_category')} is at most {latency}")

    accuracy = filters.get("accuracy_category_at_least")
    if accuracy:
        if not _accuracy_at_least(model.get("accuracy_category"), accuracy):
            return False, applied, "accuracy_category"
        applied.append(f"accuracy_category {model.get('accuracy_category')} is at least {accuracy}")

    memory_category = filters.get("memory_category")
    if memory_category:
        allowed_sizes = MEMORY_ALLOWED_SIZES.get(memory_category, set(MODEL_SIZE_ORDER))
        if model.get("model_size_category") not in allowed_sizes:
            return False, applied, "memory_category"
        category_limit = MEMORY_PARAMETER_LIMITS_M.get(memory_category)
        params_m = _float_or_none(memory.get("params_m"))
        if category_limit is not None and (params_m is None or params_m > category_limit):
            return False, applied, "memory_category"
        applied.append(
            f"model size {model.get('model_size_category')} and params_m {params_m} fit memory_category {memory_category}"
        )

    max_parameters_m = _float_or_none(filters.get("max_parameters_m"))
    if max_parameters_m is not None:
        params_m = _float_or_none(memory.get("params_m"))
        if params_m is None or params_m > max_parameters_m:
            return False, applied, "max_parameters_m"
        applied.append(f"params_m {params_m} <= max_parameters_m {max_parameters_m}")

    max_runtime_memory_mb = _float_or_none(filters.get("max_runtime_memory_mb"))
    if max_runtime_memory_mb is not None:
        runtime_gb = _float_or_none(memory.get("total_estimated_vram_gb"))
        if runtime_gb is None or runtime_gb * 1024 > max_runtime_memory_mb:
            return False, applied, "max_runtime_memory_mb"
        applied.append(
            f"estimated runtime memory {runtime_gb * 1024:.1f}MB <= {max_runtime_memory_mb}MB"
        )

    max_model_size_mb = _float_or_none(filters.get("max_model_size_mb"))
    if max_model_size_mb is not None:
        weight_gb = _float_or_none(memory.get("weight_memory_gb"))
        if weight_gb is None or weight_gb * 1024 > max_model_size_mb:
            return False, applied, "max_model_size_mb"
        applied.append(f"estimated FP32 weights {weight_gb * 1024:.1f}MB <= {max_model_size_mb}MB")

    max_cpu_latency_ms = _float_or_none(filters.get("max_cpu_latency_ms"))
    if max_cpu_latency_ms is not None:
        cpu_latencies = [
            _float_or_none(item.get("metric_value"))
            for item in candidate.get("_all_model_benchmark_results", [])
            if _clean(item.get("metric_id")) == "latency_ms"
            and (item.get("hardware_profile") or {}).get("cpu_only") in {True, "true", "True"}
        ]
        cpu_latencies = [value for value in cpu_latencies if value is not None]
        if not cpu_latencies:
            applied.append(
                f"unverified: no comparable CPU latency benchmark is available for the "
                f"requested {max_cpu_latency_ms}ms limit"
            )
        elif min(cpu_latencies) > max_cpu_latency_ms:
            return False, applied, "max_cpu_latency_ms"
        else:
            applied.append(f"measured CPU latency {min(cpu_latencies)}ms <= {max_cpu_latency_ms}ms")

    available_vram = _float_or_none(filters.get("available_vram_gb"))
    if available_vram is not None:
        min_vram = _float_or_none(memory.get("practical_min_vram_gb"))
        if min_vram is None or min_vram > available_vram:
            return False, applied, "practical_min_vram_gb"
        applied.append(f"practical_min_vram_gb {min_vram} <= available_vram_gb {available_vram}")

    hardware_category = filters.get("hardware_category")
    if hardware_category:
        recommended = memory.get("recommended_hardware_category")
        if not _hardware_category_fits(hardware_category, recommended):
            return False, applied, "recommended_hardware_category"
        applied.append(f"hardware_category {_format_hardware_filter(hardware_category)} meets recommended {recommended}")

    benchmark_target = filters.get("benchmark_target")
    if benchmark_target:
        matching = _benchmarks_that_satisfy_target(
            candidate.get("_all_model_benchmark_results") or candidate.get("model_benchmark_results") or [],
            benchmark_target.get("primary_metric"),
            benchmark_target.get("target_value"),
        )
        if not matching:
            return False, applied, "benchmark_target"
        applied.append(
            f"benchmark target {benchmark_target.get('primary_metric')}={benchmark_target.get('target_value')} satisfied"
        )

    return True, applied, None


def _benchmark_payload(graph: nx.MultiDiGraph, benchmark: Dict[str, Any]) -> Dict[str, Any]:
    metric_id = _clean(benchmark.get("metric_id"))
    metric = graph.nodes[metric_id] if metric_id in graph else {}
    hardware_profile_id = _clean(benchmark.get("hardware_profile_id"))
    hardware_profile = graph.nodes[hardware_profile_id] if hardware_profile_id in graph else {}
    payload = _project_attrs(benchmark, BENCHMARK_FIELDS, include_id=benchmark.get("id"))
    payload["metric"] = _project_attrs(metric, METRIC_FIELDS, include_id=metric_id) if metric else None
    payload["hardware_profile"] = (
        _project_attrs(hardware_profile, HARDWARE_FIELDS, include_id=hardware_profile_id)
        if hardware_profile
        else None
    )
    return payload


def _evaluation_metrics_for_task(
    graph: nx.MultiDiGraph,
    task_id: Optional[str],
    benchmarks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    metric_ids = {_clean(benchmark.get("metric_id")) for benchmark in benchmarks if benchmark.get("metric_id")}
    metrics = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("source_csv") != "evaluation_metrics.csv":
            continue
        applies_to_task = bool(task_id) and (
            _clean(attrs.get("task_id")) in {task_id, "all"} or task_id in _split_pipe(attrs.get("task"))
        )
        if node_id in metric_ids or applies_to_task:
            metrics.append({"id": node_id, **attrs})

    metrics.sort(key=lambda metric: (metric.get("primary_or_secondary") != "Primary", metric.get("id", "")))
    return metrics[:10]


def _benchmarks_that_satisfy_target(
    benchmarks: Sequence[Dict[str, Any]],
    primary_metric: Optional[str],
    target_value: Any,
) -> List[Dict[str, Any]]:
    metric_ids = _metric_ids_for_requirement(primary_metric)
    target = _float_or_none(target_value)
    if not metric_ids or target is None:
        return []

    matching = []
    for benchmark in benchmarks:
        metric_id = _clean(benchmark.get("metric_id"))
        if metric_id not in metric_ids:
            continue

        value = _float_or_none(benchmark.get("metric_value"))
        if value is None:
            continue

        metric = benchmark.get("metric") or {}
        direction = _clean(metric.get("optimization_direction")).lower()
        comparable_target = _normalize_target_scale(target, value, direction)
        if direction == "minimize":
            satisfied = value <= comparable_target
        else:
            satisfied = value >= comparable_target

        if satisfied:
            matching.append(benchmark)

    return matching


def _metric_ids_for_requirement(primary_metric: Optional[str]) -> set[str]:
    key = _normalize_token(primary_metric)
    if not key:
        return set()
    return METRIC_ALIASES.get(key, {key})


def _normalize_target_scale(target: float, benchmark_value: float, direction: str) -> float:
    if direction != "minimize" and target <= 1.0 and benchmark_value > 1.0:
        return target * 100.0
    return target


def _latency_at_most(actual: Any, preferred: Any) -> bool:
    actual_index = _category_index(actual, LATENCY_ORDER)
    preferred_index = _category_index(preferred, LATENCY_ORDER)
    return actual_index is not None and preferred_index is not None and actual_index <= preferred_index


def _accuracy_at_least(actual: Any, preferred: Any) -> bool:
    actual_index = _category_index(actual, ACCURACY_ORDER)
    preferred_index = _category_index(preferred, ACCURACY_ORDER)
    return actual_index is not None and preferred_index is not None and actual_index >= preferred_index


def _category_index(value: Any, ordered_values: Sequence[str]) -> Optional[int]:
    normalized = _normalize_token(value)
    for index, category in enumerate(ordered_values):
        if _normalize_token(category) == normalized:
            return index
    return None


def _hardware_category_fits(available: Any, recommended: Any) -> bool:
    if not recommended:
        return False

    available_categories = _expand_hardware_categories(available)
    if len(available_categories) > 1:
        return _clean(recommended) in available_categories

    available_rank = HARDWARE_ORDER.get(_clean(available_categories[0] if available_categories else available), -1)
    recommended_rank = HARDWARE_ORDER.get(_clean(recommended), 999)
    return available_rank >= recommended_rank


def _expand_hardware_categories(value: Any) -> List[str]:
    categories = _split_pipe(value)
    if categories == FALLBACK_HARDWARE_CATEGORIES:
        return FALLBACK_HARDWARE_CATEGORIES.copy()
    return categories


def _format_hardware_filter(value: Any) -> str:
    categories = _expand_hardware_categories(value)
    if len(categories) > 1:
        return " or ".join(categories)
    return _clean(value)


def _ordered_benchmarks(benchmarks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priority = {
        "map50_95": 0,
        "accuracy": 0,
        "top1_acc": 0,
        "latency_ms": 1,
        "fps": 1,
        "params_m": 2,
        "flops_b": 3,
    }
    return sorted(
        benchmarks,
        key=lambda benchmark: (
            priority.get(_clean(benchmark.get("metric_id")), 9),
            benchmark.get("dataset") or "",
            benchmark.get("id") or "",
        ),
    )


def _related_nodes(graph: nx.MultiDiGraph, node_id: str, relation: str) -> List[Dict[str, Any]]:
    related = []
    for _, target_id, attrs in graph.out_edges(node_id, data=True):
        if attrs.get("relation") == relation and target_id in graph:
            related.append({"id": target_id, **graph.nodes[target_id]})
    return related


def _first_related_node(graph: nx.MultiDiGraph, node_id: str, relation: str) -> Optional[Dict[str, Any]]:
    related = _related_nodes(graph, node_id, relation)
    return related[0] if related else None


def _evidence_sources(graph: nx.MultiDiGraph, records: Iterable[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    evidence_ids = []
    for record in records:
        if not record:
            continue
        evidence_ids.extend(_split_pipe(record.get("evidence_ids")))

    seen = set()
    sources = []
    for evidence_id in evidence_ids:
        if evidence_id in seen or evidence_id not in graph:
            continue
        seen.add(evidence_id)
        sources.append(_project_attrs(graph.nodes[evidence_id], EVIDENCE_FIELDS, include_id=evidence_id))
    # Keep every source referenced by a returned fact. Truncating this registry
    # would leave otherwise valid fact.evidence_ids impossible to resolve in
    # the API response and frontend.
    return sources


def _normalize_task_id(task: Optional[str]) -> Optional[str]:
    if not task:
        return None
    clean_task = _clean(task)
    return TASK_ID_BY_STATE_TASK.get(clean_task, clean_task.replace(" ", "_"))


def _project_attrs(attrs: Optional[Dict[str, Any]], fields: Sequence[str], include_id: Optional[str] = None) -> Dict[str, Any]:
    if not attrs:
        return {}

    projected = {}
    if include_id:
        projected["id"] = include_id
    elif attrs.get("id"):
        projected["id"] = attrs.get("id")

    for field in fields:
        value = attrs.get(field)
        if value != "" and value is not None:
            projected[field] = value
    return projected


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_pipe(value: Any) -> List[str]:
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _normalize_token(value: Any) -> str:
    return _clean(value).replace("-", "_").replace(" ", "_").lower()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_mapping(mapping: Dict[str, Any]) -> str:
    if not mapping:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in mapping.items())
