from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable

import networkx as nx

from cvmodellearning.datasets.registry import resolve_dataset_info
from cvmodellearning.policies.data_selection_policy import matched_domain_tags

from cvmodellearning.graphrag.build_graph import build_graph
from cvmodellearning.paths import PROJECT_ROOT
from cvmodellearning.schemas.interpretation_schema import ClassDataSelection, PipelineState


USE_DATASET_SELECTION_GRAPHRAG = os.getenv(
    "USE_DATASET_SELECTION_GRAPHRAG",
    "true",
).lower() not in {"0", "false", "no", "off"}

CONFIDENCE_RANK = {"Low": 1, "Medium": 2, "High": 3}


@lru_cache(maxsize=1)
def get_dataset_selection_graph() -> nx.MultiDiGraph:
    return build_graph(PROJECT_ROOT / "ontology_data")


def _outgoing(
    graph: nx.MultiDiGraph,
    source: str,
    relation: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if source not in graph:
        return []
    return [
        (str(target), dict(graph.nodes[target]), dict(edge))
        for _, target, edge in graph.out_edges(source, data=True)
        if edge.get("relation") == relation
    ]


def dataset_characteristics(
    graph: nx.MultiDiGraph,
    dataset_id: str,
) -> list[dict[str, Any]]:
    characteristics = []
    for fact_id, fact, fact_edge in _outgoing(
        graph,
        dataset_id,
        "has_characteristic",
    ):
        if str(fact.get("value", "")).lower() not in {"true", "1", "yes"}:
            continue
        for property_id, prop, _ in _outgoing(
            graph,
            fact_id,
            "characteristic_type",
        ):
            characteristics.append({
                "property_id": property_id,
                "property_name": prop.get("property_name", property_id),
                "description": prop.get("description", ""),
                "aggregation_mode": prop.get("aggregation_mode", "any"),
                "activation_threshold": float(prop.get("activation_threshold") or 0.0),
                "minimum_activation_confidence": prop.get(
                    "minimum_activation_confidence",
                    "High",
                ),
                "confidence": fact.get("confidence") or fact_edge.get("confidence", ""),
                "notes": fact.get("notes", ""),
                "evidence_ids": [
                    item
                    for item in str(fact.get("evidence_ids", "")).split("|")
                    if item
                ],
            })
    return characteristics


def _dataset_domains(
    graph: nx.MultiDiGraph,
    dataset_id: str,
    application_domain: str,
) -> list[dict[str, Any]]:
    application_text = application_domain.lower()
    domains = []
    for domain_id, domain, _ in _outgoing(graph, dataset_id, "in_domain"):
        aliases = [
            value.strip()
            for value in str(domain.get("aliases", "")).split("|")
            if value.strip()
        ]
        matched_terms = [term for term in aliases if term.lower() in application_text]
        domains.append({
            "domain_id": domain_id,
            "domain_name": domain.get("domain_name", domain_id),
            "description": domain.get("description", ""),
            "typical_visual_properties": domain.get("typical_visual_properties", ""),
            "matched_user_domain_terms": matched_terms,
        })
    return domains


def build_dataset_selection_context(
    state: PipelineState,
    eligible_data: Iterable[ClassDataSelection],
) -> dict[str, Any]:
    """Enrich locally eligible candidates; never expands the candidate set."""
    graph = get_dataset_selection_graph()
    eligible_ids = sorted({
        source.dataset_name
        for item in eligible_data
        for source in item.sources
    })
    candidates = []
    for dataset_id in eligible_ids:
        dataset = dict(graph.nodes[dataset_id]) if dataset_id in graph else {}
        local_info = resolve_dataset_info(dataset_id)
        domain_matches = (
            matched_domain_tags(local_info.domains, state.application_domain)
            if local_info is not None
            else frozenset()
        )
        candidates.append({
            "dataset_id": dataset_id,
            "display_name": dataset.get("dataset_name", dataset_id),
            "description": dataset.get("description", ""),
            "notes": dataset.get("notes", ""),
            "domains": _dataset_domains(
                graph,
                dataset_id,
                " ".join(filter(None, [
                    state.application_domain,
                    state.use_case_description,
                    state.user_query,
                ])),
            ),
            "characteristics": dataset_characteristics(graph, dataset_id),
            "evidence_ids": [
                item
                for item in str(dataset.get("evidence_ids", "")).split("|")
                if item
            ],
            "lineage": {
                "canonical_family": local_info.canonical_family,
                "derived_from": local_info.derived_from,
                "synthetic": local_info.synthetic,
                "paired_sample_ids_available": local_info.paired_sample_ids_available,
            } if local_info is not None else None,
            "domain_alignment": {
                "matched_tags": sorted(domain_matches),
                "aligned": bool(domain_matches),
                "role": "primary" if domain_matches else "generalization",
            },
        })
    evidence_ids = {
        evidence_id
        for candidate in candidates
        for evidence_id in candidate["evidence_ids"]
    }
    evidence_ids.update(
        evidence_id
        for candidate in candidates
        for characteristic in candidate["characteristics"]
        for evidence_id in characteristic["evidence_ids"]
    )
    return {
        "enabled": True,
        "source": "NetworkX knowledge graph from backend/ontology_data",
        "eligible_dataset_ids": eligible_ids,
        "candidate_guidance": candidates,
        "evidence_sources": [
            {"id": evidence_id, **dict(graph.nodes[evidence_id])}
            for evidence_id in sorted(evidence_ids)
            if evidence_id in graph
            and graph.nodes[evidence_id].get("type") == "EvidenceSource"
        ],
        "instruction": (
            "Use this evidence only to rank and mix locally allowed candidates. "
            "Class-specific eligibility and deterministic domain-mix policy remain "
            "authoritative outside GraphRAG."
        ),
    }


def aggregate_selected_dataset_properties(
    state: PipelineState,
    graph: nx.MultiDiGraph | None = None,
) -> list[dict[str, Any]]:
    """Aggregate source properties in proportion to selected allocations."""
    graph = graph or get_dataset_selection_graph()
    counts: dict[str, int] = {}
    for selection in state.selected_data or []:
        sources = selection.sources if hasattr(selection, "sources") else selection.get("sources", [])
        for source in sources:
            dataset_id = source.dataset_name if hasattr(source, "dataset_name") else source.get("dataset_name")
            if hasattr(source, "allocations"):
                count = sum(allocation.count for allocation in source.allocations)
            elif hasattr(source, "count"):
                count = source.count
            else:
                count = source.get("count", 0)
            if dataset_id:
                counts[str(dataset_id)] = counts.get(str(dataset_id), 0) + max(0, int(count))

    total = sum(counts.values())
    if not total:
        return []

    property_datasets: dict[str, set[str]] = {}
    activation_property_datasets: dict[str, set[str]] = {}
    property_details: dict[str, dict[str, Any]] = {}
    property_evidence: dict[str, set[str]] = {}
    for dataset_id in counts:
        for characteristic in dataset_characteristics(graph, dataset_id):
            property_id = characteristic["property_id"]
            property_datasets.setdefault(property_id, set()).add(dataset_id)
            property_details.setdefault(property_id, characteristic)
            property_evidence.setdefault(property_id, set()).update(
                characteristic.get("evidence_ids", [])
            )
            minimum_confidence = characteristic["minimum_activation_confidence"]
            if CONFIDENCE_RANK.get(characteristic["confidence"], 0) >= CONFIDENCE_RANK.get(
                minimum_confidence,
                3,
            ):
                activation_property_datasets.setdefault(property_id, set()).add(dataset_id)

    aggregated = []
    all_datasets = set(counts)
    for property_id, supporting_datasets in sorted(property_datasets.items()):
        details = property_details[property_id]
        evidence_support_count = sum(counts[item] for item in supporting_datasets)
        evidence_support_ratio = evidence_support_count / total
        activation_datasets = activation_property_datasets.get(property_id, set())
        support_count = sum(counts[item] for item in activation_datasets)
        support_ratio = support_count / total
        mode = details["aggregation_mode"]
        threshold = float(details["activation_threshold"])
        if mode == "all":
            active = activation_datasets == all_datasets
        elif mode == "weighted_threshold":
            active = support_ratio >= threshold
        else:
            active = bool(activation_datasets)
        aggregated.append({
            "property_id": property_id,
            "property_name": details["property_name"],
            "active": active,
            "aggregation_mode": mode,
            "activation_threshold": threshold,
            "support_ratio": round(support_ratio, 6),
            "supporting_selected_count": support_count,
            "evidence_supporting_selected_count": evidence_support_count,
            "total_selected_count": total,
            "supporting_datasets": sorted(supporting_datasets),
            "activation_supporting_datasets": sorted(activation_datasets),
            "evidence_support_ratio": round(evidence_support_ratio, 6),
            "minimum_activation_confidence": details[
                "minimum_activation_confidence"
            ],
            "evidence_ids": sorted(property_evidence.get(property_id, set())),
        })
    return aggregated
