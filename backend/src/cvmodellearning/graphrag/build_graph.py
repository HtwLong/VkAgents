"""
build_graph.py

Initialize a NetworkX knowledge graph from the ontology CSV files.

Expected project layout:

kg_data/
  nodes/
    tasks.csv
    domains.csv
    models.csv
    training_recipes.csv
    adjustment_rules.csv
    dataset_requirements.csv
    evaluation_metrics.csv
    hardware_profiles.csv
    performance_requirements.csv
    evidence_sources.csv
    cv_problems.csv
    model_benchmark_results.csv
  edges/
    edges.csv

Usage:
    python build_graph.py --data-dir kg_data

This script:
1. Loads entity CSV files into pandas DataFrames.
2. Adds every row with an 'id' column as a graph node.
3. Adds manual edges from edges.csv.
4. Calls graph_edge_utils.add_generated_edges(...) to create foreign-key-derived edges.
5. Optionally prints a graph summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import networkx as nx
import pandas as pd

from cvmodellearning.graphrag.graph_edge_utils import add_generated_edges
from cvmodellearning.graphrag.visualize_utils import visualize_interactive_graph


ENTITY_FILE_STEMS = [
    "tasks",
    "domains",
    "models",
    "training_recipes",
    "adjustment_rules",
    "dataset_requirements",
    "evaluation_metrics",
    "hardware_profiles",
    "performance_requirements",
    "evidence_sources",
    "cv_problems",
    "model_benchmark_results",
]


def _clean_value(value: object) -> object:
    """Normalize pandas missing values to empty strings for graph attributes."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return value


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists; otherwise return an empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_entity_dataframes(data_dir: Path, entity_stems: Iterable[str] = ENTITY_FILE_STEMS) -> Dict[str, pd.DataFrame]:
    """Load entity CSV files from data_dir/nodes into a dictionary keyed by file stem."""
    nodes_dir = data_dir / "nodes"
    dfs: Dict[str, pd.DataFrame] = {}
    for stem in entity_stems:
        dfs[stem] = _read_csv_if_exists(nodes_dir / f"{stem}.csv")
    return dfs


def add_nodes_from_dataframes(G: nx.MultiDiGraph, dfs: Dict[str, pd.DataFrame]) -> None:
    """Add every row from every entity DataFrame as a node, using the row's 'id' column."""
    for stem, df in dfs.items():
        if df.empty:
            continue
        if "id" not in df.columns:
            raise ValueError(f"{stem}.csv must contain an 'id' column.")

        for _, row in df.iterrows():
            row_dict = {key: _clean_value(value) for key, value in row.to_dict().items()}
            node_id = str(row_dict.pop("id")).strip()
            if not node_id:
                continue

            # Preserve the explicit ontology type if present; otherwise derive a basic type from file name.
            row_dict.setdefault("type", stem.rstrip("s"))
            row_dict["source_csv"] = f"{stem}.csv"

            G.add_node(node_id, **row_dict)


def add_manual_edges(G: nx.MultiDiGraph, edges_path: Path) -> None:
    """Add curated/manual edges from edges.csv."""
    edges = _read_csv_if_exists(edges_path)
    if edges.empty:
        return

    required_columns = {"source_id", "relation", "target_id"}
    missing = required_columns.difference(edges.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"edges.csv is missing required column(s): {missing_list}")

    for _, row in edges.iterrows():
        source_id = str(row.get("source_id", "")).strip()
        target_id = str(row.get("target_id", "")).strip()
        relation = str(row.get("relation", "")).strip()

        if not source_id or not target_id or not relation:
            continue

        # Fail fast for manual edges because they should be curated and valid.
        if source_id not in G:
            raise ValueError(f"Manual edge source_id not found as node: {source_id}")
        if target_id not in G:
            raise ValueError(f"Manual edge target_id not found as node: {target_id}")

        edge_attrs = {
            key: _clean_value(value)
            for key, value in row.to_dict().items()
            if key not in {"source_id", "target_id"}
        }
        edge_attrs["relation"] = relation
        edge_attrs["generated"] = False

        G.add_edge(source_id, target_id, **edge_attrs)


def build_graph(data_dir: str | Path = "kg_data") -> nx.MultiDiGraph:
    """
    Build and return the complete NetworkX MultiDiGraph.

    Parameters
    ----------
    data_dir:
        Root directory containing 'nodes/' and 'edges/' subdirectories.
    """
    data_dir = Path(data_dir)
    G = nx.MultiDiGraph()

    dfs = load_entity_dataframes(data_dir)
    add_nodes_from_dataframes(G, dfs)
    add_manual_edges(G, data_dir / "edges" / "edges.csv")
    add_generated_edges(G, dfs)

    return G


def summarize_graph(G: nx.MultiDiGraph) -> str:
    """Return a short human-readable graph summary."""
    node_type_counts = {}
    relation_counts = {}

    for _, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    for _, _, attrs in G.edges(data=True):
        relation = attrs.get("relation", "unknown")
        relation_counts[relation] = relation_counts.get(relation, 0) + 1

    lines = [
        f"Graph summary: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        "",
        "Node types:",
    ]
    for node_type, count in sorted(node_type_counts.items()):
        lines.append(f"  - {node_type}: {count}")

    lines.append("")
    lines.append("Edge relations:")
    for relation, count in sorted(relation_counts.items()):
        lines.append(f"  - {relation}: {count}")

    return "\n".join(lines)


def main() -> None:
    ontology_data_dir = Path(__file__).resolve().parents[3] / "ontology_data"
    G = build_graph(ontology_data_dir)

    print(summarize_graph(G))
    visualization_path = visualize_interactive_graph(G, ontology_data_dir / "graph_visualization.html", [], [])
    print(f"\nGraph visualization saved to: {visualization_path}")
    print(summarize_graph(G))


if __name__ == "__main__":
    main()
