"""Original-compatible, metadata-only decision-evidence builders."""

from __future__ import annotations

from typing import Any

from .ontology import get_ontology


def _evidence_ids(record: dict[str, Any]) -> list[str]:
    value = record.get("evidence_ids") or []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _fact(record: dict[str, Any], fact_type: str, statement: str) -> dict[str, Any]:
    evidence_record = record.get("ontology_evidence") or record
    evidence_ids = _evidence_ids(evidence_record)
    return {
        "id": str(record.get("id") or record.get("dataset_id") or f"retrieved_{fact_type}"),
        "type": fact_type,
        "statement": statement,
        "support_type": "direct_evidence" if evidence_ids else "internal_assertion",
        "confidence": evidence_record.get("confidence"),
        "evidence_ids": evidence_ids,
        "evidence_refs": [
            {"evidence_id": evidence_id, "relationship": "directly_states"}
            for evidence_id in evidence_ids
        ],
        "derivation": None,
        "data": dict(evidence_record),
    }


def _grounding_summary(facts: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for fact in facts:
        support_type = str(fact.get("support_type") or "unsupported")
        counts[support_type] = counts.get(support_type, 0) + 1
    evidenced = sum(bool(fact.get("evidence_ids")) for fact in facts)
    status = (
        "ungrounded" if not facts else "fully_grounded" if evidenced == len(facts)
        else "partially_grounded" if evidenced else "internally_grounded"
    )
    return {
        "status": status,
        "fact_count": len(facts),
        "support_counts": counts,
        "evidence_coverage": evidenced / len(facts) if facts else 0.0,
    }


def _source_registry(evidence_ids: set[str]) -> list[dict[str, Any]]:
    result = []
    store = get_ontology()
    for evidence_id in sorted(evidence_ids):
        source = store.by_id.get(evidence_id) or {}
        reference = str(source.get("url_or_reference") or "")
        external = reference.startswith(("http://", "https://"))
        result.append({
            "id": evidence_id,
            "title": source.get("title") or evidence_id,
            "url": reference if external else None,
            "reference": reference,
            "locator_type": "external_url" if external else "repository_path",
            "source_type": source.get("source_type"),
            "source_owner": source.get("source_owner"),
            "year": source.get("year"),
            "claim_supported": source.get("claim_supported"),
            "confidence": source.get("confidence"),
        })
    return result


def _candidate_id(item: dict[str, Any]) -> Any:
    return item.get("id") or item.get("dataset_id") or item.get("dataset_name")


def decision_evidence(
    *,
    decision_type: str,
    selected_id: str | None,
    rationale: str,
    candidates: list[dict[str, Any]],
    graph_context: dict[str, Any] | None,
    uncertainties: list[str] | None = None,
    decision: Any = None,
    field_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the original evidence envelope plus legacy viewer comparison keys."""
    selected = next((item for item in candidates if selected_id == _candidate_id(item)), None)
    relevant = [selected] if selected else candidates[:1]
    fact_type = {
        "model_selection": "model",
        "dataset_selection": "dataset",
        "hyperparameter_selection": "training_recipe",
    }.get(decision_type, decision_type)
    facts = [
        _fact(item, fact_type, f"Retrieved {fact_type.replace('_', ' ')} {_candidate_id(item) or 'candidate'} for this decision.")
        for item in relevant if item
    ]
    evidence_ids = {evidence_id for fact in facts for evidence_id in fact["evidence_ids"]}
    sources = _source_registry(evidence_ids)
    payload = {
        "decision_type": decision_type,
        "decision": decision if decision is not None else selected or {"id": selected_id},
        "rationale": rationale,
        "retrieved_facts": facts,
        "evidence_sources": sources,
        "grounded": bool(facts),
        "evidence_backed": bool(sources),
        "grounding": _grounding_summary(facts),
        "evaluated_candidates": candidates,
        "uncertainties": uncertainties or [],
        "selected_id": selected_id,
        "summary": rationale,
        "selected_candidate": selected,
        "active_filters": (graph_context or {}).get("filters", {}),
        "rejected_counts": (graph_context or {}).get("rejected_counts", {}),
        "matched_constraints": (selected or {}).get("matched_constraints", []),
        "evidence_ids": sorted(evidence_ids),
        "source": (graph_context or {}).get("source"),
    }
    if decision_type == "model_selection":
        payload.update({
            "selection_policy": None,
            "match_scope": "exact" if selected else None,
            "selection_confidence": "grounded" if facts else "ungrounded",
        })
    if field_provenance is not None:
        payload["field_provenance"] = field_provenance
    return payload
