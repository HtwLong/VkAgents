"""
visualize_utils.py

Helpers for creating readable visualizations of NetworkX ontology graphs.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
import re
from pathlib import Path
from textwrap import shorten, wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


TYPE_COLORS = {
    "Task": "#4C78A8",
    "Domain": "#F58518",
    "Model": "#54A24B",
    "TrainingRecipe": "#E45756",
    "AdjustmentRule": "#B279A2",
    "DatasetRequirement": "#72B7B2",
    "EvaluationMetric": "#EECA3B",
    "HardwareProfile": "#9D755D",
    "PerformanceRequirement": "#FF9DA6",
    "EvidenceSource": "#BAB0AC",
    "CVProblem": "#7F7F7F",
    "ModelBenchmarkResult": "#59A14F",
}
DEFAULT_NODE_COLOR = "#8CD17D"
BENCHMARK_SUMMARY_COLOR = "#6C5CE7"


def _contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    text = text.lower()
    return any(needle.lower() in text for needle in needles if needle)


def _searchable_node_text(node_id: object, attrs: dict[str, object]) -> str:
    values = [str(node_id)]
    values.extend(str(value) for value in attrs.values() if value not in (None, ""))
    return " ".join(values)


def _is_blacklisted(node_id: object, attrs: dict[str, object], blacklisted_strings: list[str]) -> bool:
    if not blacklisted_strings:
        return False
    return _contains_any(_searchable_node_text(node_id, attrs), blacklisted_strings)


def _is_benchmark_node(attrs: dict[str, object]) -> bool:
    return str(attrs.get("type", "")).strip() == "ModelBenchmarkResult"


def _html_table(attrs: dict[str, object]) -> str:
    rows = []
    for key, value in attrs.items():
        if value in (None, ""):
            continue
        escaped_value = escape(str(value)).replace("\n", "<br>")
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escaped_value}</td></tr>")
    return "<table>" + "".join(rows) + "</table>" if rows else ""


def _inject_click_details_panel(output_path: Path) -> None:
    """Add a small node-details panel to the generated pyvis HTML."""
    html = output_path.read_text(encoding="utf-8")
    panel = """
<style>
  #node-details-panel {
    position: fixed;
    right: 18px;
    top: 78px;
    z-index: 9999;
    width: min(420px, calc(100vw - 36px));
    max-height: calc(100vh - 110px);
    overflow: auto;
    padding: 14px 16px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d7dce2;
    border-radius: 8px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
    color: #1f2933;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 1.35;
  }
  #node-details-panel h3 {
    margin: 0 0 10px;
    font-size: 15px;
  }
  #node-details-panel table {
    border-collapse: collapse;
    width: 100%;
  }
  #node-details-panel th,
  #node-details-panel td {
    border-top: 1px solid #e5e7eb;
    padding: 6px 4px;
    text-align: left;
    vertical-align: top;
  }
  #node-details-panel th {
    width: 34%;
    color: #4b5563;
    font-weight: 700;
  }
</style>
<div id="node-details-panel">
  <h3>Node details</h3>
  <div>Hover a node for a tooltip, or click one to pin its fields here.</div>
</div>
<script>
  (function () {
    function attachDetailsPanel() {
      if (typeof network === "undefined" || typeof nodes === "undefined") {
        window.setTimeout(attachDetailsPanel, 100);
        return;
      }

      var panel = document.getElementById("node-details-panel");
      network.on("click", function (params) {
        if (!params.nodes || params.nodes.length === 0) {
          panel.innerHTML = "<h3>Node details</h3><div>Click a node to pin its fields here.</div>";
          return;
        }

        var node = nodes.get(params.nodes[0]);
        var title = node && node.title ? node.title : "";
        var label = node && node.label ? String(node.label).replace(/\\n/g, " ") : params.nodes[0];
        panel.innerHTML = "<h3>" + label + "</h3>" + title;
      });
    }
    attachDetailsPanel();
  })();
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", f"{panel}\n</body>")
    else:
        html = f"{html}\n{panel}"
    output_path.write_text(html, encoding="utf-8")


def _dedupe_visible_title(output_path: Path, title: str) -> None:
    """Remove duplicate pyvis heading blocks for the same visible title."""
    html = output_path.read_text(encoding="utf-8")
    title_pattern = re.escape(escape(title))
    heading_pattern = re.compile(rf"\s*<center>\s*<h1>{title_pattern}</h1>\s*</center>\s*", re.IGNORECASE)
    matches = list(heading_pattern.finditer(html))
    if len(matches) <= 1:
        return

    first = matches[0].group(0)
    html = heading_pattern.sub("", html)
    if "<body>" in html:
        html = html.replace("<body>", f"<body>{first}", 1)
    else:
        html = f"{first}{html}"
    output_path.write_text(html, encoding="utf-8")


def _benchmark_summary_label(model_id: object) -> str:
    return f"Benchmarks\n{model_id}"



def _benchmark_summary_id(model_id: object) -> str:
    return f"benchmark_summary::{model_id}"


def _benchmark_summary_for_model(G: nx.MultiDiGraph, model_id: object) -> tuple[str, dict[str, object]] | None:
    benchmark_ids = [
        target
        for _, target, attrs in G.out_edges(model_id, data=True)
        if attrs.get("relation") == "has_benchmark_result" and _is_benchmark_node(G.nodes[target])
    ]
    if not benchmark_ids:
        return None

    metric_values: dict[str, list[str]] = defaultdict(list)
    datasets = set()
    hardware_profiles = set()
    sources = set()

    for benchmark_id in benchmark_ids:
        benchmark_attrs = G.nodes[benchmark_id]
        metric_id = str(benchmark_attrs.get("metric_id", "")).strip()
        metric_value = str(benchmark_attrs.get("metric_value", "")).strip()
        dataset = str(benchmark_attrs.get("dataset", "")).strip()
        hardware_profile_id = str(benchmark_attrs.get("hardware_profile_id", "")).strip()
        source_id = str(benchmark_attrs.get("source_id", "")).strip()

        if metric_id and metric_value:
            metric_values[metric_id].append(metric_value)
        if dataset:
            datasets.add(dataset)
        if hardware_profile_id:
            hardware_profiles.add(hardware_profile_id)
        if source_id:
            sources.add(source_id)

    summary_lines = []
    for metric_id, values in sorted(metric_values.items()):
        unique_values = sorted(set(values))
        preview = ", ".join(unique_values[:6])
        if len(unique_values) > 6:
            preview = f"{preview}, ..."
        summary_lines.append(f"{metric_id}: {preview}")

    summary_attrs = {
        "type": "BenchmarkSummary",
        "model_id": model_id,
        "benchmark_count": len(benchmark_ids),
        "datasets": ", ".join(sorted(datasets)),
        "hardware_profiles": ", ".join(sorted(hardware_profiles)),
        "sources": ", ".join(sorted(sources)),
        "summary": "\n".join(summary_lines),
    }
    return _benchmark_summary_id(model_id), summary_attrs


def _node_label(node_id: object, attrs: dict[str, object]) -> str:
    """Create a compact, readable node label."""
    label = str(
        attrs.get("name")
        or attrs.get("model_name")
        or attrs.get("recipe_name")
        or attrs.get("metric_name")
        or attrs.get("display_name")
        or node_id
    )
    label = shorten(label, width=34, placeholder="...")
    return "\n".join(wrap(label, width=18))


def _edge_labels(G: nx.MultiDiGraph) -> dict[tuple[object, object], str]:
    """Aggregate parallel edge relations into one label per visible edge."""
    relations_by_pair: dict[tuple[object, object], set[str]] = defaultdict(set)
    for source, target, attrs in G.edges(data=True):
        relation = str(attrs.get("relation", "")).strip()
        if relation:
            relations_by_pair[(source, target)].add(relation)

    labels = {}
    for pair, relations in relations_by_pair.items():
        label = ", ".join(sorted(relations))
        labels[pair] = shorten(label, width=28, placeholder="...")
    return labels


def visualize_graph(
    G: nx.MultiDiGraph,
    output_path: str | Path,
    *,
    title: str = "CV Ontology Graph",
    seed: int = 7,
) -> Path:
    """
    Save a readable PNG visualization of the ontology graph.

    Nodes are colored by their ``type`` attribute and labeled with the most useful
    human-readable name available, falling back to the node id.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if G.number_of_nodes() == 0:
        raise ValueError("Cannot visualize an empty graph.")

    node_count = max(G.number_of_nodes(), 1)
    figure_size = (max(12, min(28, node_count * 0.65)), max(8, min(22, node_count * 0.45)))
    k = 1.8 / node_count**0.5
    pos = nx.spring_layout(G, seed=seed, k=k, iterations=200)

    node_types = [str(attrs.get("type", "Unknown")) for _, attrs in G.nodes(data=True)]
    node_colors = [TYPE_COLORS.get(node_type, DEFAULT_NODE_COLOR) for node_type in node_types]
    node_sizes = [900 + min(900, 80 * G.degree(node)) for node in G.nodes()]
    labels = {node: _node_label(node, attrs) for node, attrs in G.nodes(data=True)}

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    ax.axis("off")

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=16,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        edge_color="#9AA0A6",
        width=1.4,
        alpha=0.65,
        min_source_margin=16,
        min_target_margin=16,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#2F3437",
        linewidths=1.0,
        alpha=0.95,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_weight="bold",
        font_color="#1F2933",
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=_edge_labels(G),
        ax=ax,
        font_size=7,
        font_color="#4B5563",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        rotate=False,
    )

    legend_types = sorted(set(node_types))
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TYPE_COLORS.get(node_type, DEFAULT_NODE_COLOR),
            markeredgecolor="#2F3437",
            markersize=10,
            label=node_type,
        )
        for node_type in legend_types
    ]
    ax.legend(
        handles=legend_items,
        title="Node type",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path


def visualize_interactive_graph(
    G: nx.MultiDiGraph,
    output_path: str | Path,
    included_strings: list[str],
    blacklisted_strings: list[str],
    *,
    title: str = "CV Ontology Graph",
) -> Path:
    """
    Save an interactive pyvis HTML graph filtered by node text.

    ``included_strings`` selects seed nodes whose id or attributes contain any of
    the provided strings. The visualization also includes their direct neighbors,
    unless a neighbor matches ``blacklisted_strings``. ModelBenchmarkResult nodes
    are never visualized directly; visible model nodes get one temporary
    BenchmarkSummary node instead.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "pyvis is required for interactive graph visualization. "
            "Install it with `uv add pyvis` or add `pyvis` to the backend dependencies."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_nodes = set()
    benchmark_selected_models = set()
    benchmark_neighbors_by_model = defaultdict(set)

    for node_id, attrs in G.nodes(data=True):
        if _is_blacklisted(node_id, attrs, blacklisted_strings):
            continue
        if not _contains_any(_searchable_node_text(node_id, attrs), included_strings):
            continue

        if _is_benchmark_node(attrs):
            model_id = str(attrs.get("model_id", "")).strip()
            if model_id and model_id in G and not _is_blacklisted(model_id, G.nodes[model_id], blacklisted_strings):
                benchmark_selected_models.add(model_id)
            continue

        selected_nodes.add(node_id)

    for node_id in list(selected_nodes):
        for neighbor_id in set(G.predecessors(node_id)).union(G.successors(node_id)):
            neighbor_attrs = G.nodes[neighbor_id]
            if _is_blacklisted(neighbor_id, neighbor_attrs, blacklisted_strings):
                continue

            if _is_benchmark_node(neighbor_attrs):
                model_id = str(neighbor_attrs.get("model_id", "")).strip()
                if model_id and model_id in G:
                    benchmark_neighbors_by_model[model_id].add(neighbor_id)
                continue

            selected_nodes.add(neighbor_id)

    for model_id in benchmark_selected_models:
        selected_nodes.add(model_id)

    selected_nodes = {
        node_id
        for node_id in selected_nodes
        if node_id in G
        and not _is_benchmark_node(G.nodes[node_id])
        and not _is_blacklisted(node_id, G.nodes[node_id], blacklisted_strings)
    }

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#1F2933",
        heading=title,
        notebook=False,
    )
    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -120,
              "centralGravity": 0.015,
              "springLength": 260,
              "springConstant": 0.035,
              "damping": 0.86,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "stabilization": {
              "enabled": true,
              "iterations": 700,
              "updateInterval": 25,
              "fit": true
            }
          },
          "edges": {
            "smooth": {
              "type": "dynamic"
            },
            "font": {
              "size": 11,
              "align": "middle",
              "background": "white"
            },
            "color": {
              "color": "#9AA0A6",
              "highlight": "#4B5563"
            }
          },
          "nodes": {
            "font": {
              "size": 18,
              "face": "arial"
            },
            "borderWidth": 1.5,
            "margin": 12,
            "shadow": true
          }
        }
        """
    )

    added_nodes = set()
    for node_id in sorted(selected_nodes, key=str):
        attrs = G.nodes[node_id]
        node_type = str(attrs.get("type", "Unknown"))
        degree = G.degree(node_id)
        net.add_node(
            str(node_id),
            label=_node_label(node_id, attrs),
            title=f"<b>{escape(str(node_id))}</b>{_html_table(dict(attrs))}",
            color=TYPE_COLORS.get(node_type, DEFAULT_NODE_COLOR),
            group=node_type,
            size=18 + min(20, degree * 2),
        )
        added_nodes.add(str(node_id))

    visible_model_ids = [
        node_id
        for node_id in selected_nodes
        if str(G.nodes[node_id].get("type", "")) == "Model" or node_id in benchmark_neighbors_by_model
    ]
    for model_id in sorted(set(visible_model_ids), key=str):
        summary = _benchmark_summary_for_model(G, model_id)
        if summary is None:
            continue
        summary_id, summary_attrs = summary
        if _is_blacklisted(summary_id, summary_attrs, blacklisted_strings):
            continue

        net.add_node(
            summary_id,
            label=_benchmark_summary_label(model_id),
            title=f"<b>{escape(summary_id)}</b>{_html_table(summary_attrs)}",
            color=BENCHMARK_SUMMARY_COLOR,
            group="BenchmarkSummary",
            shape="box",
            size=24,
        )
        net.add_edge(
            str(model_id),
            summary_id,
            label="has_benchmark_summary",
            title="has_benchmark_summary",
            arrows="to",
        )
        added_nodes.add(summary_id)

    added_edges = set()
    for source, target, attrs in G.edges(data=True):
        if source not in selected_nodes or target not in selected_nodes:
            continue
        if _is_benchmark_node(G.nodes[source]) or _is_benchmark_node(G.nodes[target]):
            continue

        relation = str(attrs.get("relation", "")).strip()
        edge_key = (str(source), str(target), relation)
        if edge_key in added_edges:
            continue

        net.add_edge(
            str(source),
            str(target),
            label=relation,
            title=_html_table(dict(attrs)) or relation,
            arrows="to",
        )
        added_edges.add(edge_key)

    if not added_nodes:
        raise ValueError("No nodes matched the interactive graph filters.")

    net.save_graph(str(output_path))
    _dedupe_visible_title(output_path, title)
    _inject_click_details_panel(output_path)
    return output_path
