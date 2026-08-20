from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ..settings import ONTOLOGY_ROOT


TASK_IDS = {
    "classification": "image_classification",
    "detection": "object_detection",
    "visual question answering": "visual_question_answering",
}
CATEGORY_RANK = {
    "VeryLow": 0, "Nano": 0, "Low": 1, "Small": 1, "Medium": 2,
    "MediumHigh": 3, "Large": 3, "High": 4, "VeryLarge": 4, "VeryHigh": 5,
}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float | None:
    try:
        return float(value) if str(value or "").strip() else None
    except (TypeError, ValueError):
        return None


def _split(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


class OntologyStore:
    """Small CSV-backed graph view with no ML/runtime dependencies."""

    def __init__(self, root: Path | None = None):
        self.root = (root or ONTOLOGY_ROOT).resolve()
        self.tables = {
            path.stem: self._read(path)
            for path in sorted((self.root / "nodes").glob("*.csv"))
        }
        self.edges = self._read(self.root / "edges" / "edges.csv")
        self.by_id = {
            row["id"]: row
            for rows in self.tables.values()
            for row in rows
            if row.get("id")
        }
        self.outgoing: dict[str, list[dict[str, str]]] = {}
        for edge in self.edges:
            self.outgoing.setdefault(edge.get("source_id", ""), []).append(edge)

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            return []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def stats(self) -> dict[str, int]:
        return {
            "node_tables": len(self.tables),
            "nodes": len(self.by_id),
            "edges": len(self.edges),
            "models": len(self.tables.get("models", [])),
            "datasets": len(self.tables.get("datasets", [])),
            "training_recipes": len(self.tables.get("training_recipes", [])),
        }

    def model_context(self, context: dict[str, Any], top_k: int = 7) -> dict[str, Any]:
        task_id = TASK_IDS.get(str(context.get("task") or ""))
        performance = context.get("performance_requirements") or {}
        deployment = context.get("deployment_constraints") or {}
        latency_target = performance.get("latency_category")
        accuracy_target = performance.get("accuracy_category")
        memory_mb = _first_number(
            deployment,
            "max_runtime_memory_mb", "runtime_memory_mb", "max_memory_mb",
        )
        memory_gb = memory_mb / 1024 if memory_mb else None
        memory_by_model = _group(self.tables.get("model_inference_memory_estimates", []), "model_id")
        benchmark_by_model = _group(self.tables.get("model_benchmark_results", []), "model_id")
        hardware_by_model = _group(self.tables.get("model_training_hardware_requirements", []), "model_id")
        candidates = []
        rejected = {"task": 0, "fine_tuning": 0, "runtime_memory": 0}
        for model in self.tables.get("models", []):
            if task_id and model.get("task") != task_id:
                rejected["task"] += 1
                continue
            if not _truthy(model.get("fine_tuning_supported")):
                rejected["fine_tuning"] += 1
                continue
            memories = memory_by_model.get(model["id"], [])
            practical_min = min(
                (value for row in memories if (value := _number(row.get("practical_min_vram_gb"))) is not None),
                default=None,
            )
            if memory_gb is not None and practical_min is not None and practical_min > memory_gb:
                rejected["runtime_memory"] += 1
                continue
            score = 0
            reasons = []
            if latency_target and _category_at_most(model.get("latency_category"), latency_target):
                score += 2
                reasons.append(f"latency category {model.get('latency_category')} satisfies {latency_target}")
            if accuracy_target and _category_at_least(model.get("accuracy_category"), accuracy_target):
                score += 2
                reasons.append(f"accuracy category {model.get('accuracy_category')} satisfies {accuracy_target}")
            if model.get("pretrained_available", "").lower() == "true":
                score += 1
                reasons.append("pretrained weights are documented")
            candidates.append({
                "id": model["id"],
                "model_name": model.get("model_name"),
                "model_family": model.get("model_family"),
                "architecture_type": model.get("architecture_type"),
                "model_size_category": model.get("model_size_category"),
                "latency_category": model.get("latency_category"),
                "accuracy_category": model.get("accuracy_category"),
                "fine_tuning_supported": _truthy(model.get("fine_tuning_supported")),
                "lora_supported": model.get("lora_supported"),
                "limitations": model.get("limitations"),
                "inference_memory": [_select(row, (
                    "precision_mode", "params_m", "flops_b", "total_estimated_vram_gb",
                    "practical_min_vram_gb", "confidence", "notes",
                )) for row in memories[:2]],
                "training_hardware": [_select(row, (
                    "framework", "training_scope", "input_size", "batch_size", "precision",
                    "recommended_vram_gb", "confidence", "notes",
                )) for row in hardware_by_model.get(model["id"], [])[:2]],
                "benchmarks": [_select(row, (
                    "dataset", "metric_id", "metric_value", "image_size", "confidence", "notes",
                )) for row in benchmark_by_model.get(model["id"], [])[:4]],
                "matched_constraints": reasons,
                "evidence_ids": _split(model.get("evidence_ids")),
                "_score": score,
            })
        candidates.sort(key=lambda item: (-item["_score"], CATEGORY_RANK.get(item.get("model_size_category"), 99), item["id"]))
        for item in candidates:
            item.pop("_score", None)
        return {
            "enabled": True,
            "source": "viewer_backend/ontology_data CSV knowledge graph",
            "task_filter": task_id,
            "filters": {
                "latency_category": latency_target,
                "accuracy_category": accuracy_target,
                "max_runtime_memory_gb": memory_gb,
                "fine_tuning_required": True,
            },
            "rejected_counts": rejected,
            "candidate_models": candidates[:top_k],
        }

    def dataset_context(self, context: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
        task_id = TASK_IDS.get(str(context.get("task") or ""))
        query = " ".join(str(context.get(key) or "") for key in (
            "application_domain", "use_case_description", "user_query",
        )).lower()
        domain_matches = []
        for domain in self.tables.get("domains", []):
            terms = [domain.get("domain_name", ""), *_split(domain.get("aliases"))]
            matched = [term for term in terms if term and term.lower() in query]
            if matched:
                domain_matches.append({"id": domain["id"], "matched_terms": matched})
        matching_domain_ids = {item["id"] for item in domain_matches}
        domains_by_dataset: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.get("relation") == "in_domain":
                domains_by_dataset.setdefault(edge.get("source_id", ""), []).append(edge.get("target_id", ""))
        characteristics_by_dataset = _group(self.tables.get("dataset_characteristics", []), "dataset_id")
        properties = {row["id"]: row for row in self.tables.get("dataset_properties", [])}
        candidates = []
        for dataset in self.tables.get("datasets", []):
            if task_id and task_id not in _split(dataset.get("task_ids")):
                continue
            domain_ids = domains_by_dataset.get(dataset["id"], [])
            characteristics = []
            for fact in characteristics_by_dataset.get(dataset["id"], []):
                if not _truthy(fact.get("value")):
                    continue
                prop = properties.get(fact.get("property_id", ""), {})
                characteristics.append({
                    "property_id": fact.get("property_id"),
                    "property_name": prop.get("property_name"),
                    "description": prop.get("description"),
                    "confidence": fact.get("confidence"),
                    "notes": fact.get("notes"),
                })
            aligned = bool(set(domain_ids) & matching_domain_ids)
            candidates.append({
                "dataset_id": dataset["id"],
                "display_name": dataset.get("dataset_name"),
                "dataset_role": dataset.get("dataset_role"),
                "description": dataset.get("description"),
                "notes": dataset.get("notes"),
                "domain_ids": domain_ids,
                "domain_aligned": aligned,
                "characteristics": characteristics,
                "evidence_ids": _split(dataset.get("evidence_ids")),
            })
        candidates.sort(key=lambda item: (not item["domain_aligned"], item["dataset_role"] == "Benchmark", item["dataset_id"]))
        return {
            "enabled": True,
            "source": "viewer_backend/ontology_data CSV knowledge graph",
            "task_filter": task_id,
            "matched_domains": domain_matches,
            "candidate_datasets": candidates[:top_k],
            "availability_warning": "Ontology relevance is not proof that images are downloadable or available.",
        }

    def recipe_context(self, context: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        task_id = TASK_IDS.get(str(context.get("task") or ""))
        selected = context.get("selected_model_info") or {}
        family = str(selected.get("family") or selected.get("model_family") or "")
        recipes = []
        parameters = _group(self.tables.get("training_recipe_parameters", []), "recipe_id")
        for recipe in self.tables.get("training_recipes", []):
            families = _split(recipe.get("model_families"))
            if task_id and recipe.get("task_id") != task_id:
                continue
            if family and families and not any(_same_family(family, item) for item in families):
                continue
            recipes.append({
                **_select(recipe, (
                    "id", "recipe_name", "task_id", "model_families", "training_mode",
                    "performance_priority", "hardware_category", "optimizer", "scheduler",
                    "precision", "learning_rate_min", "learning_rate_default", "learning_rate_max",
                    "batch_size_min", "batch_size_default", "batch_size_max",
                    "epochs_min", "epochs_default", "epochs_max",
                    "weight_decay_min", "weight_decay_default", "weight_decay_max",
                    "image_size_min", "image_size_default", "image_size_max", "notes", "evidence_ids",
                )),
                "parameters": [_select(row, ("param_name", "param_value", "notes", "evidence_ids")) for row in parameters.get(recipe["id"], [])],
            })
        return {
            "enabled": True,
            "source": "viewer_backend/ontology_data CSV knowledge graph",
            "task_filter": task_id,
            "model_family": family,
            "candidate_recipes": recipes[:top_k],
        }

    @staticmethod
    def validate_hyperparameters(config: dict[str, Any], recipe: dict[str, Any]) -> list[str]:
        errors = []
        for field, config_name in (
            ("learning_rate", "learning_rate"),
            ("batch_size", "batch_size"),
            ("epochs", "epochs"),
            ("image_size", "image_size"),
        ):
            value = _number(config.get(config_name))
            minimum = _number(recipe.get(f"{field}_min"))
            maximum = _number(recipe.get(f"{field}_max"))
            if value is None:
                continue
            if minimum is not None and value < minimum:
                errors.append(f"{config_name}={value:g} is below ontology minimum {minimum:g}")
            if maximum is not None and value > maximum:
                errors.append(f"{config_name}={value:g} is above ontology maximum {maximum:g}")
        return errors


@lru_cache(maxsize=1)
def get_ontology() -> OntologyStore:
    return OntologyStore()


def _group(rows: Iterable[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        result.setdefault(row.get(field, ""), []).append(row)
    return result


def _select(row: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields if row.get(field) not in (None, "")}


def _first_number(document: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if (value := _number(document.get(key))) is not None:
            return value
    return None


def _category_at_most(actual: Any, target: Any) -> bool:
    return CATEGORY_RANK.get(str(actual), 999) <= CATEGORY_RANK.get(str(target), -1)


def _category_at_least(actual: Any, target: Any) -> bool:
    return CATEGORY_RANK.get(str(actual), -1) >= CATEGORY_RANK.get(str(target), 999)


def _same_family(left: str, right: str) -> bool:
    normalize = lambda value: "".join(character for character in value.lower() if character.isalnum())
    a, b = normalize(left), normalize(right)
    return a == b or a in b or b in a
