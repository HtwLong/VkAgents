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
- training_recipe_*_details.csv files store task-specific recipe extension nodes that point
  back to training_recipes.csv through `recipe_id`.
- training_recipe_parameters.csv stores structured parameter facts that point back to
  training_recipes.csv through `recipe_id`.
- datasets.csv stores benchmark dataset nodes that model_benchmark_results.csv can point to
  by matching the result's human-readable `dataset` value.
- model_benchmark_results.csv can optionally use `training_recipe_id` when a benchmark is tied
  to a specific recipe/config; it also uses `evidence_ids`, with older `source_id` supported
  as a fallback.
- model_inference_memory_estimates.csv stores analytical inference-memory estimate nodes that
  point back to models.csv through `model_id`.
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


def _dataset_key(value: object) -> str:
    """Return a normalized key for matching benchmark dataset labels to Dataset nodes."""
    text = _clean(value).lower()
    replacements = {
        "+": " plus ",
        "&": " and ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.replace("-", " ").split())


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


def add_training_recipe_detail_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from task-specific training recipe detail CSVs.

    Generated edges:
    - TrainingRecipe --has_recipe_details--> RecipeDetails, using recipe_id
    """
    detail_stems = [
        "training_recipe_object_detection_details",
        "training_recipe_image_classification_details",
        "training_recipe_vqa_details",
    ]

    for stem in detail_stems:
        details = dfs.get(stem, pd.DataFrame())
        if details.empty or not _has_columns(details, ["id", "recipe_id"]):
            continue

        for _, detail in details.iterrows():
            detail_id = _clean(detail.get("id"))
            recipe_id = _clean(detail.get("recipe_id"))
            evidence_id = _first_evidence_id(detail)
            confidence = _clean(detail.get("confidence"))
            _add_edge_if_nodes_exist(
                G,
                recipe_id,
                detail_id,
                "has_recipe_details",
                evidence_id=evidence_id,
                confidence=confidence,
                notes=f"Generated from {stem}.recipe_id.",
            )


def add_training_recipe_parameter_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from training_recipe_parameters.csv.

    Generated edges:
    - TrainingRecipe --has_parameter--> TrainingRecipeParameter, using recipe_id
    """
    parameters = dfs.get("training_recipe_parameters", pd.DataFrame())
    if parameters.empty or not _has_columns(parameters, ["id", "recipe_id"]):
        return

    for _, parameter in parameters.iterrows():
        parameter_id = _clean(parameter.get("id"))
        recipe_id = _clean(parameter.get("recipe_id"))
        evidence_id = _first_evidence_id(parameter)
        _add_edge_if_nodes_exist(
            G,
            recipe_id,
            parameter_id,
            "has_parameter",
            evidence_id=evidence_id,
            notes="Generated from training_recipe_parameters.recipe_id.",
        )


def add_model_benchmark_result_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from model_benchmark_results.csv.

    Generated edges:
    - Model --has_benchmark_result--> ModelBenchmarkResult, using model_id
    - TrainingRecipe --has_reference_benchmark_result--> ModelBenchmarkResult, using training_recipe_id if present
    - ModelBenchmarkResult --measures_metric--> EvaluationMetric, using metric_id
    - ModelBenchmarkResult --measured_on_hardware--> HardwareProfile, using hardware_profile_id if present
    - ModelBenchmarkResult --evaluated_for_task--> Task, using task_id if Task nodes exist
    """
    benchmarks = dfs.get("model_benchmark_results", pd.DataFrame())
    datasets = dfs.get("datasets", pd.DataFrame())
    if benchmarks.empty or "id" not in benchmarks.columns:
        return

    dataset_ids_by_key = {}
    if not datasets.empty and _has_columns(datasets, ["id", "dataset_name"]):
        for _, dataset in datasets.iterrows():
            dataset_id = _clean(dataset.get("id"))
            dataset_name = _clean(dataset.get("dataset_name"))
            if dataset_id:
                dataset_ids_by_key[_dataset_key(dataset_id)] = dataset_id
            if dataset_name:
                dataset_ids_by_key[_dataset_key(dataset_name)] = dataset_id
        if "aliases" in datasets.columns:
            for _, dataset in datasets.iterrows():
                dataset_id = _clean(dataset.get("id"))
                for alias in _split_pipe(dataset.get("aliases")):
                    dataset_ids_by_key[_dataset_key(alias)] = dataset_id

    for _, result in benchmarks.iterrows():
        result_id = _clean(result.get("id"))
        model_id = _clean(result.get("model_id"))
        training_recipe_id = _clean(result.get("training_recipe_id"))
        metric_id = _clean(result.get("metric_id"))
        hardware_profile_id = _clean(result.get("hardware_profile_id"))
        task_id = _clean(result.get("task_id"))
        dataset_id = dataset_ids_by_key.get(_dataset_key(result.get("dataset")))
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
        if training_recipe_id:
            _add_edge_if_nodes_exist(
                G,
                training_recipe_id,
                result_id,
                "has_reference_benchmark_result",
                evidence_id=evidence_id,
                confidence=confidence,
                notes="Generated from model_benchmark_results.training_recipe_id.",
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
        if dataset_id:
            _add_edge_if_nodes_exist(
                G,
                result_id,
                dataset_id,
                "evaluated_on_dataset",
                evidence_id=evidence_id,
                confidence=confidence,
                notes="Generated by matching model_benchmark_results.dataset to datasets.dataset_name.",
            )


def add_model_inference_memory_estimate_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from model_inference_memory_estimates.csv.

    Generated edges:
    - Model --has_inference_memory_estimate--> ModelInferenceMemoryEstimate, using model_id
    """
    estimates = dfs.get("model_inference_memory_estimates", pd.DataFrame())
    if estimates.empty or not _has_columns(estimates, ["id", "model_id"]):
        return

    for _, estimate in estimates.iterrows():
        estimate_id = _clean(estimate.get("id"))
        model_id = _clean(estimate.get("model_id"))
        evidence_id = _first_evidence_id(estimate)
        confidence = _clean(estimate.get("confidence"))
        _add_edge_if_nodes_exist(
            G,
            model_id,
            estimate_id,
            "has_inference_memory_estimate",
            evidence_id=evidence_id,
            confidence=confidence,
            notes="Generated from model_inference_memory_estimates.model_id.",
        )


def add_dataset_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """
    Generate edges from datasets.csv.

    Generated edges:
    - Dataset --supports_task--> Task, using task_ids if Task nodes exist
    """
    datasets = dfs.get("datasets", pd.DataFrame())
    if datasets.empty or "id" not in datasets.columns:
        return

    for _, dataset in datasets.iterrows():
        dataset_id = _clean(dataset.get("id"))
        evidence_id = _first_evidence_id(dataset)
        for task_id in _split_pipe(dataset.get("task_ids")):
            _add_edge_if_nodes_exist(
                G,
                dataset_id,
                task_id,
                "supports_task",
                evidence_id=evidence_id,
                notes="Generated from datasets.task_ids.",
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


def add_dataset_characteristic_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> None:
    """Connect evidence-backed characteristic facts to datasets and properties."""
    characteristics = dfs.get("dataset_characteristics", pd.DataFrame())
    if characteristics.empty or not _has_columns(
        characteristics,
        ["id", "dataset_id", "property_id"],
    ):
        return

    for _, row in characteristics.iterrows():
        characteristic_id = _clean(row.get("id"))
        dataset_id = _clean(row.get("dataset_id"))
        property_id = _clean(row.get("property_id"))
        evidence_id = _first_evidence_id(row)
        confidence = _clean(row.get("confidence"))
        _add_edge_if_nodes_exist(
            G,
            dataset_id,
            characteristic_id,
            "has_characteristic",
            evidence_id=evidence_id,
            confidence=confidence,
            notes="Generated from dataset_characteristics.dataset_id.",
        )
        _add_edge_if_nodes_exist(
            G,
            characteristic_id,
            property_id,
            "characteristic_type",
            evidence_id=evidence_id,
            confidence=confidence,
            notes="Generated from dataset_characteristics.property_id.",
        )


def add_generated_edges(G: nx.MultiDiGraph, dfs: DataFrames) -> nx.MultiDiGraph:
    """
    Add all generated edges to the graph.

    Call this after:
    1. Loading all nodes from entity CSVs.
    2. Loading all manual edges from edges.csv.
    """
    add_evidence_edges(G, dfs)
    add_dataset_edges(G, dfs)
    add_training_recipe_edges(G, dfs)
    add_training_recipe_detail_edges(G, dfs)
    add_training_recipe_parameter_edges(G, dfs)
    add_model_benchmark_result_edges(G, dfs)
    add_model_inference_memory_estimate_edges(G, dfs)
    add_evaluation_metric_edges(G, dfs)
    add_adjustment_rule_edges(G, dfs)
    add_dataset_requirement_edges(G, dfs)
    add_dataset_characteristic_edges(G, dfs)
    return G
