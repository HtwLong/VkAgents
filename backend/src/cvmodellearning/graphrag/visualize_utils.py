"""
visualize_utils.py

Helpers for creating readable visualizations of NetworkX ontology graphs.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
import json
import re
from pathlib import Path
from textwrap import shorten, wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


DEFAULT_NODE_COLOR = "#8CD17D"
BENCHMARK_SUMMARY_COLOR = "#6C5CE7"

TYPE_COLORS = {
    "Task": "#1F77B4",
    "Dataset": "#00A6A6",
    "Model": "#54A24B",
    "TrainingRecipe": "#E45756",
    "TrainingRecipeParameter": "#FF9DA6",
    "ObjectDetectionRecipeDetails": "#F28E2B",
    "ImageClassificationRecipeDetails": "#4E79A7",
    "AdjustmentRule": "#B279A2",
    "EvaluationMetric": "#EECA3B",
    "HardwareProfile": "#9D755D",
    "EvidenceSource": "#BAB0AC",
    "ModelBenchmarkResult": "#59A14F",
    "BenchmarkSummary": BENCHMARK_SUMMARY_COLOR,
}

TYPE_SHAPES = {
    "Task": "diamond",
    "Dataset": "database",
    "Model": "dot",
    "TrainingRecipe": "box",
    "TrainingRecipeParameter": "text",
    "ObjectDetectionRecipeDetails": "box",
    "ImageClassificationRecipeDetails": "box",
    "AdjustmentRule": "hexagon",
    "EvaluationMetric": "triangle",
    "HardwareProfile": "database",
    "EvidenceSource": "ellipse",
    "BenchmarkSummary": "box",
}

TYPE_BASE_SIZES = {
    "Task": 28,
    "Dataset": 22,
    "Model": 30,
    "TrainingRecipe": 24,
    "TrainingRecipeParameter": 16,
    "ObjectDetectionRecipeDetails": 20,
    "ImageClassificationRecipeDetails": 20,
    "BenchmarkSummary": 24,
    "AdjustmentRule": 18,
    "EvaluationMetric": 18,
    "HardwareProfile": 18,
    "EvidenceSource": 14,
}


SCHEMA_GROUP_COLORS = {
    "core": "#DBEAFE",
    "data": "#FEF3C7",
    "model": "#DCFCE7",
    "training": "#F3E8FF",
    "evidence": "#F3F4F6",
}

SCHEMA_NODE_GROUPS = {
    "Task": "core",
    "EvaluationMetric": "core",
    "Dataset": "data",
    "Domain": "data",
    "DatasetCharacteristic": "data",
    "DatasetProperty": "data",
    "Model": "model",
    "ModelBenchmarkResult": "model",
    "ModelInferenceMemoryEstimate": "model",
    "ModelTrainingHardwareRequirement": "model",
    "HardwareProfile": "model",
    "TrainingRecipe": "training",
    "TrainingRecipeParameter": "training",
    "ImageClassificationRecipeDetails": "training",
    "ObjectDetectionRecipeDetails": "training",
    "VQARecipeDetails": "training",
    "AdjustmentRule": "training",
    "EvidenceSource": "evidence",
}

SCHEMA_TYPE_LABELS = {
    "Task": "Task",
    "EvaluationMetric": "Evaluation\nMetric",
    "Dataset": "Dataset",
    "Domain": "Domain",
    "DatasetCharacteristic": "Dataset\nCharacteristic",
    "DatasetProperty": "Dataset\nProperty",
    "Model": "Model",
    "ModelBenchmarkResult": "Model Benchmark\nResult",
    "ModelInferenceMemoryEstimate": "Inference Memory\nEstimate",
    "ModelTrainingHardwareRequirement": "Training Hardware\nRequirement",
    "HardwareProfile": "Hardware\nProfile",
    "TrainingRecipe": "Training\nRecipe",
    "TrainingRecipeParameter": "Recipe\nParameter",
    "ImageClassificationRecipeDetails": "Classification\nRecipe Details",
    "ObjectDetectionRecipeDetails": "Detection\nRecipe Details",
    "VQARecipeDetails": "VQA Recipe\nDetails",
    "AdjustmentRule": "Adjustment\nRule",
    "EvidenceSource": "Evidence\nSource",
}

# Fixed positions make the overview stable across ontology revisions and keep
# the conceptual groups legible in a printed thesis figure.
SCHEMA_POSITIONS = {
    "Task": (0.0, 5.6),
    "EvaluationMetric": (3.0, 5.6),
    "Dataset": (-5.3, 3.6),
    "Domain": (-7.5, 1.7),
    "DatasetCharacteristic": (-5.0, 1.2),
    "DatasetProperty": (-5.0, -1.0),
    "Model": (2.2, 3.5),
    "ModelBenchmarkResult": (5.2, 2.8),
    "HardwareProfile": (7.4, 1.0),
    "ModelInferenceMemoryEstimate": (2.8, 1.0),
    "ModelTrainingHardwareRequirement": (2.4, -1.0),
    "TrainingRecipe": (-0.3, 0.7),
    "TrainingRecipeParameter": (-1.7, -1.4),
    "ImageClassificationRecipeDetails": (-2.7, -3.3),
    "ObjectDetectionRecipeDetails": (-0.3, -3.3),
    "VQARecipeDetails": (2.0, -3.3),
    "AdjustmentRule": (4.6, -2.4),
    "EvidenceSource": (0.0, -5.5),
}

# Only the relationships needed to understand the ontology's design are drawn.
# Provenance is summarized separately because drawing every evidence-bearing
# entity type would dominate the diagram.
SCHEMA_RELATIONSHIPS = (
    ("Model", "supports_task", "Task"),
    ("Dataset", "supports_task", "Task"),
    ("EvaluationMetric", "applies_to_task", "Task"),
    ("Dataset", "in_domain", "Domain"),
    ("Dataset", "has_characteristic", "DatasetCharacteristic"),
    ("DatasetCharacteristic", "characteristic_type", "DatasetProperty"),
    ("Model", "has_benchmark_result", "ModelBenchmarkResult"),
    ("ModelBenchmarkResult", "evaluated_for_task", "Task"),
    ("ModelBenchmarkResult", "evaluated_on_dataset", "Dataset"),
    ("ModelBenchmarkResult", "measures_metric", "EvaluationMetric"),
    ("ModelBenchmarkResult", "measured_on_hardware", "HardwareProfile"),
    ("Model", "has_inference_memory_estimate", "ModelInferenceMemoryEstimate"),
    ("Model", "has_training_hardware_requirement", "ModelTrainingHardwareRequirement"),
    ("Model", "has_training_recipe", "TrainingRecipe"),
    ("TrainingRecipe", "for_task", "Task"),
    ("TrainingRecipe", "has_parameter", "TrainingRecipeParameter"),
    ("TrainingRecipe", "has_recipe_details", "ImageClassificationRecipeDetails"),
    ("TrainingRecipe", "has_recipe_details", "ObjectDetectionRecipeDetails"),
    ("TrainingRecipe", "has_recipe_details", "VQARecipeDetails"),
    ("AdjustmentRule", "applies_to_task", "Task"),
    ("AdjustmentRule", "applies_to_model", "Model"),
    ("AdjustmentRule", "modifies_recipe", "TrainingRecipe"),
)


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


def _inject_filter_controls(output_path: Path) -> None:
    """Add client-side node and edge filters to the generated pyvis HTML."""
    html = output_path.read_text(encoding="utf-8")
    controls = """
<style>
  #graph-filter-panel {
    position: fixed;
    left: 18px;
    top: 78px;
    z-index: 9999;
    width: min(360px, calc(100vw - 36px));
    padding: 12px 14px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d7dce2;
    border-radius: 8px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
    color: #1f2933;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 1.35;
  }
  #graph-filter-panel h3 {
    margin: 0 0 10px;
    font-size: 15px;
  }
  #graph-filter-panel label {
    display: block;
    margin: 8px 0 4px;
    color: #4b5563;
    font-weight: 700;
  }
  #graph-filter-panel input[type="text"] {
    width: 100%;
    box-sizing: border-box;
    padding: 7px 8px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    color: #1f2933;
    font: inherit;
  }
  #graph-filter-panel .graph-filter-options {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 10px 0;
  }
  #graph-filter-panel .graph-filter-options label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin: 0;
    font-weight: 400;
  }
  #graph-filter-panel .graph-filter-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  #graph-filter-panel button {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    background: #f8fafc;
    color: #1f2933;
    cursor: pointer;
    font: inherit;
  }
  #graph-filter-panel button:hover {
    background: #eef2f7;
  }
  #graph-filter-count {
    color: #4b5563;
    font-size: 12px;
  }
</style>
<div id="graph-filter-panel">
  <h3>Filter graph</h3>
  <label for="graph-filter-include">Show only matches</label>
  <input id="graph-filter-include" type="text" placeholder="e.g. YOLO, TrainingRecipe">
  <label for="graph-filter-exclude">Hide matches</label>
  <input id="graph-filter-exclude" type="text" placeholder="e.g. EvidenceSource, supported_by_evidence">
  <div class="graph-filter-options">
    <label><input id="graph-filter-nodes" type="checkbox" checked> Nodes</label>
    <label><input id="graph-filter-edges" type="checkbox" checked> Edges</label>
  </div>
  <div class="graph-filter-actions">
    <button id="graph-filter-reset" type="button">Reset</button>
    <span id="graph-filter-count"></span>
  </div>
</div>
<script>
  (function () {
    function attachGraphFilters() {
      if (typeof network === "undefined" || typeof nodes === "undefined" || typeof edges === "undefined") {
        window.setTimeout(attachGraphFilters, 100);
        return;
      }

      var includeInput = document.getElementById("graph-filter-include");
      var excludeInput = document.getElementById("graph-filter-exclude");
      var nodeToggle = document.getElementById("graph-filter-nodes");
      var edgeToggle = document.getElementById("graph-filter-edges");
      var resetButton = document.getElementById("graph-filter-reset");
      var count = document.getElementById("graph-filter-count");
      var originalNodes = nodes.get();
      var originalEdges = edges.get();

      function terms(value) {
        return String(value || "")
          .toLowerCase()
          .split(",")
          .map(function (term) { return term.trim(); })
          .filter(Boolean);
      }

      function plainText(value) {
        var div = document.createElement("div");
        div.innerHTML = String(value || "");
        return div.textContent || div.innerText || "";
      }

      function searchable(item) {
        return [
          item.id,
          item.label,
          item.group,
          item.from,
          item.to,
          plainText(item.title)
        ].join(" ").toLowerCase();
      }

      function matches(text, includeTerms, excludeTerms) {
        if (includeTerms.length && !includeTerms.some(function (term) { return text.indexOf(term) !== -1; })) {
          return false;
        }
        if (excludeTerms.some(function (term) { return text.indexOf(term) !== -1; })) {
          return false;
        }
        return true;
      }

      function applyFilters() {
        var includeTerms = terms(includeInput.value);
        var excludeTerms = terms(excludeInput.value);
        var filterNodes = nodeToggle.checked;
        var filterEdges = edgeToggle.checked;
        var visibleNodeIds = {};
        var visibleNodeCount = 0;
        var visibleEdgeCount = 0;

        nodes.update(originalNodes.map(function (node) {
          var visible = !filterNodes || matches(searchable(node), includeTerms, excludeTerms);
          visibleNodeIds[node.id] = visible;
          if (visible) {
            visibleNodeCount += 1;
          }
          return { id: node.id, hidden: !visible };
        }));

        edges.update(originalEdges.map(function (edge) {
          var endpointsVisible = visibleNodeIds[edge.from] && visibleNodeIds[edge.to];
          var edgeVisible = !filterEdges || matches(searchable(edge), includeTerms, excludeTerms);
          var visible = endpointsVisible && edgeVisible;
          if (visible) {
            visibleEdgeCount += 1;
          }
          return { id: edge.id, hidden: !visible };
        }));

        count.textContent = visibleNodeCount + " nodes, " + visibleEdgeCount + " edges";
      }

      includeInput.addEventListener("input", applyFilters);
      excludeInput.addEventListener("input", applyFilters);
      nodeToggle.addEventListener("change", applyFilters);
      edgeToggle.addEventListener("change", applyFilters);
      resetButton.addEventListener("click", function () {
        includeInput.value = "";
        excludeInput.value = "";
        nodeToggle.checked = true;
        edgeToggle.checked = true;
        applyFilters();
      });

      applyFilters();
    }
    attachGraphFilters();
  })();
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", f"{controls}\n</body>")
    else:
        html = f"{html}\n{controls}"
    output_path.write_text(html, encoding="utf-8")


def _node_color(attrs: dict[str, object]) -> str:
    return TYPE_COLORS.get(_node_type(attrs), DEFAULT_NODE_COLOR)


def _legend_symbol_svg(shape: str, color: str) -> str:
    """Return an inline symbol that mirrors the pyvis shape used for a node."""
    escaped_color = escape(color)
    common_attrs = f'fill="{escaped_color}" stroke="#2f3437" stroke-width="1.8"'

    if shape == "box":
        symbol = f'<rect x="5" y="7" width="22" height="18" rx="2" {common_attrs} />'
    elif shape == "ellipse":
        symbol = f'<ellipse cx="16" cy="16" rx="12" ry="9" {common_attrs} />'
    elif shape == "diamond":
        symbol = f'<polygon points="16,4 28,16 16,28 4,16" {common_attrs} />'
    elif shape == "triangle":
        symbol = f'<polygon points="16,4 29,27 3,27" {common_attrs} />'
    elif shape == "hexagon":
        symbol = f'<polygon points="10,4 22,4 30,16 22,28 10,28 2,16" {common_attrs} />'
    elif shape == "database":
        symbol = (
            f'<path d="M5 9c0-3 5-5 11-5s11 2 11 5v14c0 3-5 5-11 5S5 26 5 23Z" {common_attrs} />'
            '<path d="M5 9c0 3 5 5 11 5s11-2 11-5" fill="none" stroke="#2f3437" stroke-width="1.8" />'
        )
    else:
        symbol = f'<circle cx="16" cy="16" r="10" {common_attrs} />'

    return (
        '<svg class="graph-legend-symbol" viewBox="0 0 32 32" aria-hidden="true" '
        'focusable="false">'
        f"{symbol}"
        "</svg>"
    )


def _inject_shape_legend(output_path: Path, node_types: list[str]) -> None:
    """Add a bottom-left legend using the actual auto-generated pyvis colors."""
    html = output_path.read_text(encoding="utf-8")
    legend_data = [
        {"type": node_type, "shape": _node_shape({"type": node_type})}
        for node_type in sorted(set(node_types))
    ]

    legend = f"""
<style>
  #graph-shape-legend {{
    position: fixed;
    left: 18px;
    bottom: 18px;
    z-index: 9998;
    width: min(300px, calc(100vw - 36px));
    max-height: min(42vh, 360px);
    overflow: auto;
    padding: 12px 14px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d7dce2;
    border-radius: 8px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
    color: #1f2933;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 1.3;
  }}
  #graph-shape-legend h3 {{
    margin: 0 0 10px;
    font-size: 15px;
  }}
  .graph-legend-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 7px;
  }}
  .graph-legend-item {{
    display: flex;
    align-items: center;
    gap: 9px;
    min-width: 0;
  }}
  .graph-legend-symbol {{
    flex: 0 0 auto;
    width: 14px;
    height: 14px;
    overflow: visible;
  }}
</style>
<div id="graph-shape-legend">
  <h3>Entity legend</h3>
  <div class="graph-legend-grid" id="graph-legend-grid"></div>
</div>
<script>
  (function () {{
    var legendData = {json.dumps(legend_data)};

    function colorFromNode(nodeId) {{
      var renderedNode = network && network.body && network.body.nodes
        ? network.body.nodes[nodeId]
        : null;
      var color = renderedNode && renderedNode.options
        ? renderedNode.options.color
        : null;

      if (typeof color === "string") {{
        return color;
      }}
      if (color && color.background) {{
        return color.background;
      }}
      if (color && color.color) {{
        return color.color;
      }}
      if (color && color.highlight && color.highlight.background) {{
        return color.highlight.background;
      }}
      return "{DEFAULT_NODE_COLOR}";
    }}

    function typeColor(nodeType) {{
      var matchingNodes = nodes.get({{
        filter: function (node) {{
          return node.group === nodeType;
        }}
      }});
      if (!matchingNodes.length) {{
        return "{DEFAULT_NODE_COLOR}";
      }}
      return colorFromNode(matchingNodes[0].id);
    }}

    function symbolSvg(shape, color) {{
      var attrs = 'fill="' + color + '" stroke="#2f3437" stroke-width="1.8"';
      var symbol;
      if (shape === "box") {{
        symbol = '<rect x="5" y="7" width="22" height="18" rx="2" ' + attrs + ' />';
      }} else if (shape === "ellipse") {{
        symbol = '<ellipse cx="16" cy="16" rx="12" ry="9" ' + attrs + ' />';
      }} else if (shape === "diamond") {{
        symbol = '<polygon points="16,4 28,16 16,28 4,16" ' + attrs + ' />';
      }} else if (shape === "triangle") {{
        symbol = '<polygon points="16,4 29,27 3,27" ' + attrs + ' />';
      }} else if (shape === "hexagon") {{
        symbol = '<polygon points="10,4 22,4 30,16 22,28 10,28 2,16" ' + attrs + ' />';
      }} else if (shape === "database") {{
        symbol = '<path d="M5 9c0-3 5-5 11-5s11 2 11 5v14c0 3-5 5-11 5S5 26 5 23Z" ' + attrs + ' />'
          + '<path d="M5 9c0 3 5 5 11 5s11-2 11-5" fill="none" stroke="#2f3437" stroke-width="1.8" />';
      }} else {{
        symbol = '<circle cx="16" cy="16" r="10" ' + attrs + ' />';
      }}
      return '<svg class="graph-legend-symbol" viewBox="0 0 32 32" aria-hidden="true" focusable="false">' + symbol + '</svg>';
    }}

    function attachShapeLegend() {{
      if (typeof network === "undefined" || typeof nodes === "undefined") {{
        window.setTimeout(attachShapeLegend, 100);
        return;
      }}

      var grid = document.getElementById("graph-legend-grid");
      if (!grid) {{
        return;
      }}

      grid.innerHTML = legendData.map(function (item) {{
        return '<div class="graph-legend-item">'
          + symbolSvg(item.shape, typeColor(item.type))
          + '<span>' + item.type + '</span>'
          + '</div>';
      }}).join("");
    }}

    attachShapeLegend();
  }})();
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", f"{legend}\n</body>")
    else:
        html = f"{html}\n{legend}"
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
        or attrs.get("detail_name")
        or attrs.get("metric_name")
        or attrs.get("display_name")
        or node_id
    )
    label = shorten(label, width=34, placeholder="...")
    return "\n".join(wrap(label, width=18))


def _node_type(attrs: dict[str, object]) -> str:
    return str(attrs.get("type", "Unknown")).strip() or "Unknown"


def _node_shape(attrs: dict[str, object]) -> str:
    return TYPE_SHAPES.get(_node_type(attrs), "dot")


def _node_size(node_id: object, attrs: dict[str, object], G: nx.MultiDiGraph) -> int:
    node_type = _node_type(attrs)
    base_size = TYPE_BASE_SIZES.get(node_type, 18)
    degree_bonus = min(18, int(G.degree(node_id) * 1.6))
    return base_size + degree_bonus


def _node_mass(node_id: object, attrs: dict[str, object], G: nx.MultiDiGraph) -> float:
    node_type = _node_type(attrs)
    base_mass = 3.0 if node_type in {"Task", "Model"} else 1.2
    return base_mass + min(4.0, G.degree(node_id) / 8)


def _edge_width(relation: str) -> float:
    if relation in {"has_training_recipe", "has_reference_benchmark_result", "has_benchmark_summary"}:
        return 2.4
    if relation == "has_recipe_details":
        return 2.0
    if relation in {"applies_to_model", "modifies_recipe"}:
        return 1.9
    if relation == "supported_by_evidence":
        return 0.8
    return 1.3


def _edge_color(relation: str) -> str:
    if relation == "has_reference_benchmark_result":
        return "#6C5CE7"
    if relation == "has_training_recipe":
        return "#E45756"
    if relation == "has_recipe_details":
        return "#4E79A7"
    if relation == "has_benchmark_summary":
        return BENCHMARK_SUMMARY_COLOR
    if relation == "supported_by_evidence":
        return "#C8CCD0"
    return "#9AA0A6"


def _edge_smooth_type(relation: str) -> str:
    if relation == "supported_by_evidence":
        return "continuous"
    return "dynamic"


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


def _schema_node_counts(G: nx.MultiDiGraph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _, attrs in G.nodes(data=True):
        counts[_node_type(attrs)] += 1
    return dict(counts)


def _schema_relation_counts(
    G: nx.MultiDiGraph,
) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for source, target, attrs in G.edges(data=True):
        relation = str(attrs.get("relation", "")).strip()
        if not relation:
            continue
        source_type = _node_type(G.nodes[source])
        target_type = _node_type(G.nodes[target])
        counts[(source_type, relation, target_type)] += 1
    return dict(counts)


def _schema_relation_label(relation: str, count: int) -> str:
    readable = relation.replace("_", " ")
    return f"{readable}\n({count})"


def visualize_ontology_schema(
    G: nx.MultiDiGraph,
    output_path: str | Path,
    *,
    title: str = "Computer Vision Planning Ontology",
) -> Path:
    """Render a deterministic, publication-oriented ontology overview.

    Unlike :func:`visualize_graph`, this function does not draw individual
    ontology instances. It aggregates them by entity type, labels each type with
    its instance count, and draws only the principal semantic relationships used
    by planning. The fixed layout is intended for a thesis figure and remains
    stable when individual ontology records are added or removed.

    The output format is selected from ``output_path``. SVG or PDF is recommended
    for publication; PNG is also supported by Matplotlib.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot visualize an empty graph.")
    if output_path.suffix.lower() not in {".svg", ".pdf", ".png"}:
        raise ValueError("Ontology schema output must use .svg, .pdf, or .png.")

    node_counts = _schema_node_counts(G)
    relation_counts = _schema_relation_counts(G)
    visible_types = {
        node_type
        for node_type in SCHEMA_POSITIONS
        if node_counts.get(node_type, 0) > 0
    }

    fig, ax = plt.subplots(figsize=(19, 12), constrained_layout=True)
    ax.set_xlim(-9.0, 9.0)
    ax.set_ylim(-6.5, 6.7)
    ax.axis("off")
    ax.set_title(title, fontsize=22, fontweight="bold", pad=18, color="#172033")
    ax.text(
        0,
        6.25,
        f"Schema-level view of {G.number_of_nodes():,} entities and {G.number_of_edges():,} relationships",
        ha="center",
        va="center",
        fontsize=11,
        color="#4B5563",
    )

    # Draw relationships first so node boxes remain visually dominant.
    drawn_edges: list[tuple[str, str, str, int]] = []
    for source_type, relation, target_type in SCHEMA_RELATIONSHIPS:
        count = relation_counts.get((source_type, relation, target_type), 0)
        if count == 0 or source_type not in visible_types or target_type not in visible_types:
            continue
        drawn_edges.append((source_type, relation, target_type, count))
        source_pos = SCHEMA_POSITIONS[source_type]
        target_pos = SCHEMA_POSITIONS[target_type]
        ax.annotate(
            "",
            xy=target_pos,
            xytext=source_pos,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#374151",
                "linewidth": 1.9,
                "alpha": 0.95,
                "shrinkA": 28,
                "shrinkB": 28,
                "connectionstyle": "arc3,rad=0.05",
            },
            zorder=1,
        )

    # Compact relationship labels are offset perpendicular to their edges.
    pair_occurrences: dict[tuple[str, str], int] = defaultdict(int)
    for source_type, relation, target_type, count in drawn_edges:
        source_x, source_y = SCHEMA_POSITIONS[source_type]
        target_x, target_y = SCHEMA_POSITIONS[target_type]
        pair = tuple(sorted((source_type, target_type)))
        occurrence = pair_occurrences[pair]
        pair_occurrences[pair] += 1
        dx, dy = target_x - source_x, target_y - source_y
        length = max((dx * dx + dy * dy) ** 0.5, 0.1)
        direction = -1 if occurrence % 2 else 1
        offset = 0.16 + 0.13 * occurrence
        label_x = (source_x + target_x) / 2 + direction * (-dy / length) * offset
        label_y = (source_y + target_y) / 2 + direction * (dx / length) * offset
        ax.text(
            label_x,
            label_y,
            _schema_relation_label(relation, count),
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="semibold",
            color="#111827",
            linespacing=0.92,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "linewidth": 0.65,
                "alpha": 0.96,
            },
            zorder=2,
        )

    for node_type in visible_types:
        x, y = SCHEMA_POSITIONS[node_type]
        group = SCHEMA_NODE_GROUPS[node_type]
        label = SCHEMA_TYPE_LABELS.get(node_type, node_type)
        ax.text(
            x,
            y,
            f"{label}\n{node_counts[node_type]} entities",
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#172033",
            linespacing=1.2,
            bbox={
                "boxstyle": "round,pad=0.62,rounding_size=0.15",
                "facecolor": SCHEMA_GROUP_COLORS[group],
                "edgecolor": "#374151",
                "linewidth": 1.25,
            },
            zorder=4,
        )

    total_evidence_edges = sum(
        count
        for (source_type, relation, target_type), count in relation_counts.items()
        if relation == "supported_by_evidence" and target_type == "EvidenceSource"
    )
    if "EvidenceSource" in visible_types and total_evidence_edges:
        note_pos = (-4.1, -5.5)
        evidence_pos = SCHEMA_POSITIONS["EvidenceSource"]
        ax.text(
            *note_pos,
            "Evidence-linked ontology records\n(all supported entity types)",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#374151",
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": "white",
                "edgecolor": "#9CA3AF",
                "linestyle": "--",
            },
            zorder=4,
        )
        ax.annotate(
            "",
            xy=evidence_pos,
            xytext=note_pos,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#374151",
                "linewidth": 1.9,
                "linestyle": "--",
                "shrinkA": 58,
                "shrinkB": 34,
            },
            zorder=2,
        )
        ax.text(
            -2.1,
            -5.22,
            f"supported by evidence\n({total_evidence_edges:,} links)",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="semibold",
            color="#111827",
            bbox={
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "linewidth": 0.65,
                "alpha": 0.96,
            },
            zorder=3,
        )

    legend_items = [
        Patch(
            facecolor=SCHEMA_GROUP_COLORS[group],
            edgecolor="#374151",
            label=label,
        )
        for group, label in (
            ("core", "Core task concepts"),
            ("data", "Dataset knowledge"),
            ("model", "Model and hardware knowledge"),
            ("training", "Training knowledge"),
            ("evidence", "Evidence and provenance"),
        )
    ]
    legend_items.append(
        Line2D([0], [0], color="#374151", linewidth=1.9, label="Semantic relationship")
    )
    legend_items.append(
        Line2D(
            [0], [0], color="#374151", linewidth=1.9, linestyle="--",
            label="Aggregated provenance relationship",
        )
    )
    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        frameon=True,
        framealpha=0.96,
        edgecolor="#D1D5DB",
        fontsize=8.5,
        ncol=1,
    )

    ax.text(
        8.8,
        -6.2,
        "Counts are calculated from the loaded ontology snapshot.",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#6B7280",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


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
    try:
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
    except TypeError:
        # Some NetworkX versions cannot label curved MultiDiGraph edges. The
        # interactive HTML visualization still contains full edge details.
        pass

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
            "keyboard": true,
            "tooltipDelay": 120
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -180,
              "centralGravity": 0.012,
              "springLength": 300,
              "springConstant": 0.03,
              "damping": 0.88,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "maxVelocity": 40,
            "stabilization": {
              "enabled": true,
              "iterations": 500,
              "updateInterval": 25,
              "fit": true
            }
          },
          "edges": {
            "smooth": {
              "type": "dynamic",
              "roundness": 0.25
            },
            "font": {
              "size": 11,
              "align": "middle",
              "background": "white"
            },
            "color": {
              "color": "#9AA0A6",
              "highlight": "#4B5563"
            },
            "selectionWidth": 2
          },
          "nodes": {
            "font": {
              "size": 18,
              "face": "arial"
            },
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
            "margin": 12,
            "shadow": true
          }
        }
        """
    )

    added_nodes = set()
    added_node_types = set()
    for node_id in sorted(selected_nodes, key=str):
        attrs = G.nodes[node_id]
        node_type = _node_type(attrs)
        net.add_node(
            str(node_id),
            label=_node_label(node_id, attrs),
            title=f"<b>{escape(str(node_id))}</b>{_html_table(dict(attrs))}",
            group=node_type,
            shape=_node_shape(attrs),
            mass=_node_mass(node_id, attrs, G),
            size=_node_size(node_id, attrs, G),
        )
        added_nodes.add(str(node_id))
        added_node_types.add(node_type)

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
            group="BenchmarkSummary",
            shape=_node_shape(summary_attrs),
            mass=2.5,
            size=TYPE_BASE_SIZES["BenchmarkSummary"] + 6,
        )
        net.add_edge(
            str(model_id),
            summary_id,
            label="has_benchmark_summary",
            title="has_benchmark_summary",
            arrows="to",
            color=_edge_color("has_benchmark_summary"),
            width=_edge_width("has_benchmark_summary"),
            smooth={"type": _edge_smooth_type("has_benchmark_summary"), "roundness": 0.25},
        )
        added_nodes.add(summary_id)
        added_node_types.add(_node_type(summary_attrs))

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
            color=_edge_color(relation),
            width=_edge_width(relation),
            smooth={"type": _edge_smooth_type(relation), "roundness": 0.25},
        )
        added_edges.add(edge_key)

    if not added_nodes:
        raise ValueError("No nodes matched the interactive graph filters.")

    net.save_graph(str(output_path))
    _dedupe_visible_title(output_path, title)
    _inject_filter_controls(output_path)
    _inject_click_details_panel(output_path)
    _inject_shape_legend(output_path, sorted(added_node_types))
    return output_path
