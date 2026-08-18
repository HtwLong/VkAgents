from pathlib import Path

import networkx as nx
import pytest

from cvmodellearning.graphrag.visualize_utils import visualize_ontology_schema


def _schema_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("task", type="Task")
    graph.add_node("model", type="Model")
    graph.add_node("source", type="EvidenceSource")
    graph.add_edge("model", "task", relation="supports_task")
    graph.add_edge("model", "source", relation="supported_by_evidence")
    return graph


def test_visualize_ontology_schema_writes_svg(tmp_path: Path) -> None:
    output_path = tmp_path / "ontology-schema.svg"

    result = visualize_ontology_schema(_schema_graph(), output_path)

    assert result == output_path
    assert output_path.is_file()
    assert "<svg" in output_path.read_text(encoding="utf-8")


def test_visualize_ontology_schema_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must use .svg, .pdf, or .png"):
        visualize_ontology_schema(_schema_graph(), tmp_path / "ontology-schema.txt")
