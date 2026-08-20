from __future__ import annotations

from typing import Any


def decision_evidence(
    *,
    decision_type: str,
    selected_id: str | None,
    rationale: str,
    candidates: list[dict[str, Any]],
    graph_context: dict[str, Any] | None,
    uncertainties: list[str] | None = None,
) -> dict[str, Any]:
    """Stable, inspectable evidence envelope shared by planning decisions."""
    selected = next((item for item in candidates if selected_id in {
        item.get("id"), item.get("dataset_id"), item.get("dataset_name")
    }), None)
    evidence_ids = sorted({
        evidence_id
        for item in candidates
        for evidence_id in item.get("evidence_ids", [])
        if evidence_id
    })
    return {
        "decision_type": decision_type,
        "selected_id": selected_id,
        "summary": rationale,
        "rationale": rationale,
        "selected_candidate": selected,
        "evaluated_candidates": candidates,
        "active_filters": (graph_context or {}).get("filters", {}),
        "rejected_counts": (graph_context or {}).get("rejected_counts", {}),
        "matched_constraints": (selected or {}).get("matched_constraints", []),
        "evidence_ids": evidence_ids,
        "uncertainties": uncertainties or [],
        "source": (graph_context or {}).get("source"),
    }
