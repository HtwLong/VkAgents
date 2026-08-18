from __future__ import annotations

import re
from typing import Any, Iterable


def _evidence_ids(record: dict[str, Any] | None) -> list[str]:
    if not record:
        return []
    value = record.get("evidence_ids", "")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _source_registry(
    sources: Iterable[dict[str, Any]], allowed_ids: set[str]
) -> list[dict[str, Any]]:
    result = []
    for source in sources:
        source_id = str(source.get("id", "")).strip()
        url = str(source.get("url_or_reference", "")).strip()
        if source_id not in allowed_ids:
            continue
        is_external = url.startswith(("http://", "https://"))
        result.append({
            "id": source_id,
            "title": source.get("title") or source_id,
            "url": url if is_external else None,
            "reference": url,
            "locator_type": "external_url" if is_external else "repository_path",
            "source_type": source.get("source_type"),
            "source_owner": source.get("source_owner"),
            "year": source.get("year"),
            "claim_supported": source.get("claim_supported"),
            "confidence": source.get("confidence"),
        })
    return result


def _fact(
    record: dict[str, Any],
    fact_type: str,
    statement: str,
    *,
    support_type: str | None = None,
    evidence_relationship: str = "directly_states",
    derivation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fact_id = str(record.get("id") or f"retrieved_{fact_type}")
    evidence_ids = _evidence_ids(record)
    if support_type is None:
        support_type = "direct_evidence" if evidence_ids else "internal_assertion"
    return {
        "id": fact_id,
        "type": fact_type,
        "statement": statement,
        "support_type": support_type,
        "confidence": record.get("confidence"),
        "evidence_ids": evidence_ids,
        "evidence_refs": [
            {"evidence_id": evidence_id, "relationship": evidence_relationship}
            for evidence_id in evidence_ids
        ],
        "derivation": derivation,
        "data": {key: value for key, value in record.items() if key != "source_csv"},
    }


def _grounding_summary(facts: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for fact in facts:
        support_type = str(fact.get("support_type") or "unsupported")
        counts[support_type] = counts.get(support_type, 0) + 1
    evidenced = sum(bool(fact.get("evidence_ids")) for fact in facts)
    if not facts:
        status = "ungrounded"
    elif evidenced == len(facts) and set(counts) == {"direct_evidence"}:
        status = "fully_grounded"
    elif evidenced:
        status = "partially_grounded"
    else:
        status = "internally_grounded"
    return {
        "status": status,
        "fact_count": len(facts),
        "support_counts": counts,
        "evidence_coverage": evidenced / len(facts) if facts else 0.0,
    }


def _model_key(value: Any) -> str:
    """Normalize registry/ontology spelling without conflating model generations."""
    key = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    return key.replace("yolov", "yolo")


def _model_support_type(model: dict[str, Any], *, variant: bool = False) -> str | None:
    if variant or "inferred" in str(model.get("limitations", "")).lower():
        return "inferred"
    return None


def _matching_candidate(
    selected: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    selected_key = _model_key(
        selected.get("model_architecture")
        or selected.get("name")
        or selected.get("model_name")
    )
    if not selected_key:
        return None, None

    for candidate in candidates:
        model = candidate.get("model") or {}
        if selected_key in {_model_key(model.get("id")), _model_key(model.get("model_name"))}:
            return candidate, "exact"

    # Detection selection happens at executable family/generation level (for
    # example yolov11), while ontology facts may describe a concrete variant
    # (for example yolo11x). Prefix matching preserves the generation and is
    # deliberately not a generic family fallback (yolov8 must not match yolo11).
    for candidate in candidates:
        model = candidate.get("model") or {}
        candidate_keys = (_model_key(model.get("id")), _model_key(model.get("model_family")))
        if any(
            key and len(selected_key) >= 5
            and (key.startswith(selected_key) or selected_key.startswith(key))
            for key in candidate_keys
        ):
            return candidate, "family_variant"
    return None, None


def build_model_selection_decision_evidence(
    selected_model_info: dict[str, Any],
    rationale: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    selected = selected_model_info.get("model") or {}
    if isinstance(selected, list):
        selected = selected[0] if selected else {}
    candidates = context.get("candidate_models") or []
    candidate, match_scope = _matching_candidate(selected, candidates)

    facts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    if candidate:
        model = candidate.get("model") or {}
        model_statement = (
            f"Retrieved model {model.get('model_name') or model.get('id')} with accuracy category "
            f"{model.get('accuracy_category')} and latency category {model.get('latency_category')}."
        )
        if match_scope == "family_variant":
            model_statement = (
                f"Retrieved variant {model.get('model_name') or model.get('id')} supports the selected "
                "model family/generation; its variant-specific values do not automatically apply to every variant."
            )
        facts.append(_fact(
            model,
            "model",
            model_statement,
            support_type=_model_support_type(model, variant=match_scope == "family_variant"),
            evidence_relationship="corroborates" if match_scope == "family_variant" else "directly_states",
        ))
        memory = candidate.get("model_inference_memory_estimate")
        if memory:
            facts.append(_fact(
                memory,
                "inference_memory_estimate",
                f"Estimated practical minimum VRAM is {memory.get('practical_min_vram_gb')} GB "
                f"for {memory.get('precision_mode') or 'the retrieved precision'}.",
                support_type="derived",
                evidence_relationship="supports_input",
                derivation={
                    "method": memory.get("calculation_method") or "analytical memory estimate",
                    "inputs": {
                        key: memory.get(key)
                        for key in ("params_m", "precision_mode", "batch_size", "context_tokens_for_kv")
                        if memory.get(key) not in (None, "")
                    },
                },
            ))
        training_requirement = candidate.get("model_training_hardware_requirement")
        if training_requirement:
            status = training_requirement.get("recommendation_status") or "derived"
            facts.append(_fact(
                training_requirement,
                "training_hardware_requirement",
                f"Recommended training GPU capacity is "
                f"{training_requirement.get('recommended_vram_gb')} GB for "
                f"{training_requirement.get('training_scope')}; lowest observed successful "
                f"hardware is {training_requirement.get('lowest_observed_success_vram_gb') or 'not reported'} GB.",
                support_type="direct" if status == "evidence_backed_observed_success" else "derived",
                evidence_relationship="supports_input",
                derivation={
                    "method": status,
                    "inputs": {
                        key: training_requirement.get(key)
                        for key in (
                            "input_size", "batch_size", "precision",
                            "lowest_observed_success_vram_gb", "observed_peak_vram_gb",
                        )
                        if training_requirement.get(key) not in (None, "")
                    },
                },
            ))
        for benchmark in candidate.get("model_benchmark_results") or []:
            facts.append(_fact(
                benchmark,
                "benchmark",
                f"{benchmark.get('dataset')} {benchmark.get('metric_id')} benchmark: "
                f"{benchmark.get('metric_value')}.",
            ))
        for metric in candidate.get("evaluation_metrics") or []:
            facts.append(_fact(
                metric,
                "evaluation_metric",
                f"{metric.get('metric_name') or metric.get('id')} is a retrieved "
                f"{metric.get('primary_or_secondary')} metric.",
            ))
        evidence = {eid for fact in facts for eid in fact["evidence_ids"]}
        sources = _source_registry(candidate.get("evidence_sources") or [], evidence)

    return {
        "decision_type": "model_selection",
        "decision": selected_model_info,
        "rationale": rationale,
        "selection_policy": None,
        "retrieved_facts": facts,
        "evidence_sources": sources,
        "grounded": bool(facts),
        "evidence_backed": bool(sources),
        "match_scope": match_scope,
        "grounding": _grounding_summary(facts),
    }


def build_dataset_selection_decision_evidence(
    selected_data: list[dict[str, Any]],
    rationale: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(source.get("dataset_name"))
        for selection in selected_data
        for source in selection.get("sources", [])
        if source.get("dataset_name")
    }
    facts: list[dict[str, Any]] = []

    for candidate in context.get("candidate_guidance") or []:
        dataset_id = str(candidate.get("dataset_id") or "")
        if dataset_id not in selected_ids:
            continue

        dataset_evidence_ids = _evidence_ids(candidate)
        facts.append(_fact(
            {**candidate, "id": dataset_id},
            "dataset",
            f"Retrieved dataset {candidate.get('dataset_name') or dataset_id}: "
            f"{candidate.get('description') or 'no description available.'}",
        ))
        for domain in candidate.get("domains") or []:
            if not domain.get("matched_user_domain_terms"):
                continue
            facts.append(_fact(
                {
                    **domain,
                    "id": f"{dataset_id}:{domain.get('domain_id')}",
                    "evidence_ids": dataset_evidence_ids,
                },
                "dataset_domain",
                f"{candidate.get('dataset_name') or dataset_id} matches the requested domain "
                f"through {domain.get('domain_name') or domain.get('domain_id')} "
                f"({', '.join(domain.get('matched_user_domain_terms') or [])}).",
                support_type="inferred",
            ))
        for characteristic in candidate.get("characteristics") or []:
            facts.append(_fact(
                {
                    **characteristic,
                    "id": f"{dataset_id}:{characteristic.get('property_id')}",
                },
                "dataset_characteristic",
                f"{candidate.get('dataset_name') or dataset_id} has characteristic "
                f"{characteristic.get('property_name') or characteristic.get('property_id')}: "
                f"{characteristic.get('description') or characteristic.get('notes') or 'retrieved from the dataset graph.'}",
            ))

    evidence = {eid for fact in facts for eid in fact["evidence_ids"]}
    sources = _source_registry(context.get("evidence_sources") or [], evidence)
    return {
        "decision_type": "dataset_selection",
        "decision": selected_data,
        "rationale": rationale,
        "retrieved_facts": facts,
        "evidence_sources": sources,
        "grounded": bool(facts),
        "evidence_backed": bool(sources),
        "grounding": _grounding_summary(facts),
    }


def build_hyperparameter_decision_evidence(
    config: dict[str, Any],
    rationale: str,
    context: dict[str, Any],
    field_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    selected_model = context.get("selected_model")
    recipe = context.get("base_recipe")
    if selected_model:
        facts.append(_fact(
            selected_model,
            "model",
            f"Hyperparameters were retrieved for {selected_model.get('model_name') or selected_model.get('id')}.",
            support_type=_model_support_type(selected_model),
        ))
    if recipe:
        facts.append(_fact(
            recipe,
            "training_recipe",
            f"Base recipe {recipe.get('recipe_name') or recipe.get('id')} supplied the grounded defaults.",
        ))
    for parameter in context.get("recipe_parameters") or []:
        facts.append(_fact(
            parameter,
            "recipe_parameter",
            f"Recipe parameter {parameter.get('param_name')}: {parameter.get('param_value')}.",
        ))
    for details in context.get("recipe_details") or []:
        facts.append(_fact(details, "recipe_details", "Retrieved task-specific recipe details."))
    for rule in context.get("matched_adjustment_rules") or []:
        facts.append(_fact(
            rule,
            "adjustment_rule",
            f"Matched rule {rule.get('id')} applied {rule.get('executable_adjustments')}.",
            support_type="inferred",
            evidence_relationship="supports_method",
        ))

    evidence = {eid for fact in facts for eid in fact["evidence_ids"]}
    sources = _source_registry(context.get("evidence_sources") or [], evidence)
    provenance = field_provenance or {}
    return {
        "decision_type": "hyperparameter_selection",
        "decision": config,
        "rationale": rationale,
        "retrieved_facts": facts,
        "evidence_sources": sources,
        "field_provenance": provenance,
        "grounded": bool(facts),
        "evidence_backed": bool(sources),
        "grounding": _grounding_summary(facts),
    }
