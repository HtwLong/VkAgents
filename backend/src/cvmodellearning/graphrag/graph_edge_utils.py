"""
graph_edge_utils.py

Utility functions for adding generated ontology/KG edges to a NetworkX graph.

Design used by the cleaned CSV set:
- Entity/fact CSV rows may contain `evidence_ids` as a pipe-delimited list.
- evidence_sources.csv is a source registry only.
- Manual edges in edges.csv should be rare; most obvious edges are generated here.

Important schema conventions:
- models.csv uses singular `model_family`, because each model belongs to one family.
- training_recipes.csv uses plural `model_families`, because one recipe can apply to multiple families.
- model_benchmark_results.csv uses `evidence_ids`; older `source_id` is still supported as a fallback.
"""

from __future__ import annotations

from typing import Dict, Iterable

import networkx as nx
import pandas as pd


DataFrames = Dict[str, pd.DataFrame]


def _has_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    """Return True if every required column exists in the DataFrame."""
    return all(column in df.columns for column in columns)


def _clean(value: object) -> str:
    """Normalize missing values and whitespace to a clean string."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _split_pipe(value: object) -> list[str]:
    """Split pipe-delimited values such as 'YOLOv8|YOLOv10'."""
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _node_exists(G: nx.MultiDiGraph, node_id: str) -> bool:
    return bool(node_id) and node_id in G


def _first_evidence_id(row: pd.Series) -> str:
    """Return the first evidence ID from evidence_ids or legacy source_id."""
    evidence_ids = _split_pipe(row.get("evidence_ids", ""))
    if evidence_ids:
        return evidence_ids[0]
    return _clean(row.get("source_id", ""))


def _row_evidence_ids(row: pd.Series) -> list[str]:
    """Return all evidence IDs from evidence_ids or legacy source_id."""
    evidence_ids = _split_pipe(row.get("evidence_ids", ""))
    if evidence_ids:
        return evidence_ids
    source_id = _clean(row.get("source_id", ""))
    return [source_id] if source_id else []


def _recipe_model_families(recipe: pd.Series) -> list[str]:
    """Support new training_recipes.model_families and old model_family."""
    if "model_families" in recipe.index:
        families = _split_pipe(recipe.get("model_families"))
        if families:
            return families
    family = _clean(recipe.get("model_family", ""))
    return [family] if family else []


def _add_edge_if_nodes_exist(
    G: nx.MultiDiGraph,
    source_id: str,
    target_id: str,
    relation: str,
    *,
    generated: bool = True,
    evidence_id: str = "",
    confidence: str = "",
    notes: str = "",
) -> bool:
    """
    Add an edge only if source and target nodes already exist.

    Returns True if an edge was added, otherwise False.
    """
    source_id = _clean(source_id)
    target_id = _clean(target_id)
    if not (_node_exists(G, source_id) and _node_exists(G, target_id)):
        return False

    attrs = {"relation": relation, "generated": generated}
    if evidence_id:
        attrs["evidence_id"] = evidence_id
    if confidence:
        attrs["confidence"] = confidence
    if notes:
        attrs["notes"] = notes

    G.add_edge(source_id, target_id, **attrs)
    return True


def add_evidence_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate supported_by_evidence edges from `evidence_ids` columns.

    Applies to any loaded DataFrame with an id column and evidence_ids column.
    Also supports legacy model_benchmark_results.source_id as a fallback.
    """
    for stem, df in dfs.items():
        if df.empty or "id" not in df.columns:
            continue
        if stem == "evidence_sources":
            continue

        has_evidence_ids = "evidence_ids" in df.columns
        has_source_id = "source_id" in df.columns
        if not has_evidence_ids and not has_source_id:
            continue

        for _, row in df.iterrows():
            node_id = _clean(row.get("id"))
            confidence = _clean(row.get("confidence"))
            for evidence_id in _row_evidence_ids(row):
                _add_edge_if_nodes_exist(
                    G,
                    node_id,
                    evidence_id,
                    "supported_by_evidence",
                    confidence=confidence,
                    notes=f"Generated from {stem}.evidence_ids." if has_evidence_ids else f"Generated from {stem}.source_id.",
                )


def add_training_recipe_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from training_recipes.csv.

    Generated edges:
    - TrainingRecipe --for_task--> Task, using task_id if Task nodes exist
    - Model --has_training_recipe--> TrainingRecipe, using models.model_family in training_recipes.model_families
    - TrainingRecipe --uses_metric--> EvaluationMetric, using optional metric_ids / primary_metric_ids / secondary_metric_ids
    """
    recipes = dfs.get("training_recipes", pd.DataFrame())
    models = dfs.get("models", pd.DataFrame())

    if recipes.empty or "id" not in recipes.columns:
        return

    for _, recipe in recipes.iterrows():
        recipe_id = _clean(recipe.get("id"))
        task_id = _clean(recipe.get("task_id"))
        recipe_families = set(_recipe_model_families(recipe))
        evidence_id = _first_evidence_id(recipe)
        confidence = _clean(recipe.get("confidence"))

        if task_id:
            _add_edge_if_nodes_exist(
                G,
                recipe_id,
                task_id,
                "for_task",
                evidence_id=evidence_id,
                confidence=confidence,
                notes="Generated from training_recipes.task_id.",
            )

        if recipe_families and not models.empty and _has_columns(models, ["id", "model_family"]):
            for _, model in models.iterrows():
                model_id = _clean(model.get("id"))
                model_family = _clean(model.get("model_family"))
                if model_family in recipe_families:
                    _add_edge_if_nodes_exist(
                        G,
                        model_id,
                        recipe_id,
                        "has_training_recipe",
                        evidence_id=evidence_id,
                        confidence=confidence,
                        notes="Generated by matching models.model_family to training_recipes.model_families.",
                    )

        for metric_column in ["metric_ids", "primary_metric_ids", "secondary_metric_ids"]:
            if metric_column not in recipes.columns:
                continue
            for metric_id in _split_pipe(recipe.get(metric_column)):
                _add_edge_if_nodes_exist(
                    G,
                    recipe_id,
                    metric_id,
                    "uses_metric",
                    evidence_id=evidence_id,
                    confidence=confidence,
                    notes=f"Generated from training_recipes.{metric_column}.",
                )


def add_model_benchmark_result_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from model_benchmark_results.csv.

    Generated edges:
    - Model --has_benchmark_result--> ModelBenchmarkResult, using model_id
    - ModelBenchmarkResult --measures_metric--> EvaluationMetric, using metric_id
    - ModelBenchmarkResult --measured_on_hardware--> HardwareProfile, using hardware_profile_id if present
    - ModelBenchmarkResult --evaluated_for_task--> Task, using task_id if Task nodes exist
    """
    benchmarks = dfs.get("model_benchmark_results", pd.DataFrame())
    if benchmarks.empty or "id" not in benchmarks.columns:
        return

    for _, result in benchmarks.iterrows():
        result_id = _clean(result.get("id"))
        model_id = _clean(result.get("model_id"))
        metric_id = _clean(result.get("metric_id"))
        hardware_profile_id = _clean(result.get("hardware_profile_id"))
        task_id = _clean(result.get("task_id"))
        evidence_id = _first_evidence_id(result)
        confidence = _clean(result.get("confidence"))

        _add_edge_if_nodes_exist(
            G,
            model_id,
            result_id,
            "has_benchmark_result",
            evidence_id=evidence_id,
            confidence=confidence,
            notes="Generated from model_benchmark_results.model_id.",
        )
        _add_edge_if_nodes_exist(
            G,
            result_id,
            metric_id,
            "measures_metric",
            evidence_id=evidence_id,
            confidence=confidence,
            notes="Generated from model_benchmark_results.metric_id.",
        )
        if hardware_profile_id:
            _add_edge_if_nodes_exist(
                G,
                result_id,
                hardware_profile_id,
                "measured_on_hardware",
                evidence_id=evidence_id,
                confidence=confidence,
                notes="Generated from model_benchmark_results.hardware_profile_id.",
            )
        _add_edge_if_nodes_exist(
            G,
            result_id,
            task_id,
            "evaluated_for_task",
            evidence_id=evidence_id,
            confidence=confidence,
            notes="Generated from model_benchmark_results.task_id.",
        )


def add_evaluation_metric_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from evaluation_metrics.csv.

    Generated edges:
    - EvaluationMetric --applies_to_task--> Task, using task_id if Task nodes exist.
    """
    metrics = dfs.get("evaluation_metrics", pd.DataFrame())
    if metrics.empty or not _has_columns(metrics, ["id", "task_id"]):
        return

    for _, metric in metrics.iterrows():
        metric_id = _clean(metric.get("id"))
        task_id = _clean(metric.get("task_id"))
        evidence_id = _first_evidence_id(metric)
        if not task_id or task_id.lower() == "all":
            continue
        _add_edge_if_nodes_exist(
            G,
            metric_id,
            task_id,
            "applies_to_task",
            evidence_id=evidence_id,
            notes="Generated from evaluation_metrics.task_id.",
        )


def add_adjustment_rule_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from adjustment_rules.csv.

    Generated edges:
    - AdjustmentRule --applies_to_task--> Task, using applies_to_task_ids if Task nodes exist
    - AdjustmentRule --applies_to_model--> Model, using applies_to_model_families matched to models.model_family
    - AdjustmentRule --modifies_recipe--> TrainingRecipe, using overlapping task and model family scopes
    """
    rules = dfs.get("adjustment_rules", pd.DataFrame())
    models = dfs.get("models", pd.DataFrame())
    recipes = dfs.get("training_recipes", pd.DataFrame())

    if rules.empty or "id" not in rules.columns:
        return

    for _, rule in rules.iterrows():
        rule_id = _clean(rule.get("id"))
        task_ids = set(_split_pipe(rule.get("applies_to_task_ids")))
        rule_families = set(_split_pipe(rule.get("applies_to_model_families")))
        confidence = _clean(rule.get("confidence"))
        evidence_id = _first_evidence_id(rule)

        for task_id in task_ids:
            _add_edge_if_nodes_exist(
                G,
                rule_id,
                task_id,
                "applies_to_task",
                evidence_id=evidence_id,
                confidence=confidence,
                notes="Generated from adjustment_rules.applies_to_task_ids.",
            )

        if rule_families and not models.empty and _has_columns(models, ["id", "model_family"]):
            for _, model in models.iterrows():
                model_id = _clean(model.get("id"))
                model_family = _clean(model.get("model_family"))
                if model_family in rule_families:
                    _add_edge_if_nodes_exist(
                        G,
                        rule_id,
                        model_id,
                        "applies_to_model",
                        evidence_id=evidence_id,
                        confidence=confidence,
                        notes="Generated by matching adjustment_rules.applies_to_model_families to models.model_family.",
                    )

        if not recipes.empty and _has_columns(recipes, ["id", "task_id"]):
            for _, recipe in recipes.iterrows():
                recipe_id = _clean(recipe.get("id"))
                recipe_task_id = _clean(recipe.get("task_id"))
                recipe_families = set(_recipe_model_families(recipe))

                task_matches = not task_ids or recipe_task_id in task_ids
                family_matches = not rule_families or bool(recipe_families.intersection(rule_families))

                if task_matches and family_matches:
                    _add_edge_if_nodes_exist(
                        G,
                        rule_id,
                        recipe_id,
                        "modifies_recipe",
                        evidence_id=evidence_id,
                        confidence=confidence,
                        notes="Generated by matching rule task/family scope to training_recipes.task_id/model_families.",
                    )


def add_dataset_requirement_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from dataset_requirements.csv.

    Generated edges:
    - DatasetRequirement --for_task--> Task, using task_id if Task nodes exist
    """
    requirements = dfs.get("dataset_requirements", pd.DataFrame())
    if requirements.empty or "id" not in requirements.columns:
        return

    for _, req in requirements.iterrows():
        req_id = _clean(req.get("id"))
        task_id = _clean(req.get("task_id"))
        evidence_id = _first_evidence_id(req)
        _add_edge_if_nodes_exist(
            G,
            req_id,
            task_id,
            "for_task",
            evidence_id=evidence_id,
            notes="Generated from dataset_requirements.task_id.",
        )


def add_generated_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> nx.MultiDiGraph:
    """
    Add all generated edges to the graph.

    Call this after:
    1. Loading all nodes from entity CSVs.
    2. Loading all manual edges from edges.csv.
    """
    add_evidence_edges(G, dfs)
    add_training_recipe_edges(G, dfs)
    add_model_benchmark_result_edges(G, dfs)
    add_evaluation_metric_edges(G, dfs)
    add_adjustment_rule_edges(G, dfs)
    add_dataset_requirement_edges(G, dfs)
    return G
