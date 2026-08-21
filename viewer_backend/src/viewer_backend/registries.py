"""Metadata-only registries used by viewer planning.

Unlike the full backend registries, entries contain no constructors and cannot
load a model, dataset, optimizer, or training framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .graphrag.ontology import TASK_IDS, OntologyStore, get_ontology
from .hpo_planning import detection_hpo_model_id, supports_detection_hpo_model


EXECUTABLE_CLASSIFICATION_IDS = frozenset({
    "resnet50", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "densenet121", "convnext_tiny", "clip_vit_b16", "dinov2_vits14",
    "dinov2_vitb14", "vit_b_16", "swin_v2_t", "swin_v2_s",
})
EXECUTABLE_VQA_IDS = frozenset({"Qwen3-VL-2B-Instruct"})


def is_planning_model_supported(task: str, value: str) -> bool:
    if task == "detection":
        return supports_detection_hpo_model(value)
    if task == "classification":
        return value in EXECUTABLE_CLASSIFICATION_IDS
    if task == "visual question answering":
        return value in EXECUTABLE_VQA_IDS
    return False


def canonical_reference(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _split(value: Any) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split("|") if item.strip())


@dataclass(frozen=True)
class ModelMetadata:
    id: str
    display_name: str
    family: str
    task_id: str
    aliases: tuple[str, ...]
    fine_tuning_supported: bool
    lora_supported: str
    size_category: str
    latency_category: str
    accuracy_category: str
    limitations: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatasetMetadata:
    id: str
    display_name: str
    task_ids: tuple[str, ...]
    role: str
    aliases: tuple[str, ...]
    description: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class RecipeMetadata:
    id: str
    display_name: str
    task_id: str
    model_families: tuple[str, ...]
    training_mode: str
    values: dict[str, str]
    evidence_ids: tuple[str, ...]


class MetadataRegistry:
    def __init__(self, ontology: OntologyStore | None = None):
        store = ontology or get_ontology()
        self.models = {
            row["id"]: ModelMetadata(
                id=row["id"],
                display_name=row.get("model_name") or row["id"],
                family=row.get("model_family") or "unknown",
                task_id=row.get("task") or "",
                aliases=_split(row.get("aliases")),
                fine_tuning_supported=str(row.get("fine_tuning_supported", "")).lower() == "true",
                lora_supported=row.get("lora_supported") or "unknown",
                size_category=row.get("model_size_category") or "",
                latency_category=row.get("latency_category") or "",
                accuracy_category=row.get("accuracy_category") or "",
                limitations=row.get("limitations") or "",
                evidence_ids=_split(row.get("evidence_ids")),
            )
            for row in store.tables.get("models", []) if row.get("id")
        }
        self.datasets = {
            row["id"]: DatasetMetadata(
                id=row["id"],
                display_name=row.get("dataset_name") or row["id"],
                task_ids=_split(row.get("task_ids")),
                role=row.get("dataset_role") or "",
                aliases=_split(row.get("aliases")),
                description=row.get("description") or "",
                evidence_ids=_split(row.get("evidence_ids")),
            )
            for row in store.tables.get("datasets", []) if row.get("id")
        }
        self.recipes = {
            row["id"]: RecipeMetadata(
                id=row["id"],
                display_name=row.get("recipe_name") or row["id"],
                task_id=row.get("task_id") or "",
                model_families=_split(row.get("model_families")),
                training_mode=row.get("training_mode") or "",
                values=dict(row),
                evidence_ids=_split(row.get("evidence_ids")),
            )
            for row in store.tables.get("training_recipes", []) if row.get("id")
        }

    @staticmethod
    def _resolve(value: str, entries: dict[str, Any], references) -> Any | None:
        target = canonical_reference(value)
        matches = [entry for entry in entries.values() if target in {
            canonical_reference(reference) for reference in references(entry) if reference
        }]
        return matches[0] if len(matches) == 1 else None

    def resolve_model(self, value: str, task: str | None = None) -> ModelMetadata | None:
        task_id = TASK_IDS.get(task or "", task or "")
        entries = {key: item for key, item in self.models.items() if not task_id or item.task_id == task_id}
        if task == "detection":
            executable_id = detection_hpo_model_id(value)
            matches = [
                item for item in entries.values()
                if supports_detection_hpo_model(item.id)
                and detection_hpo_model_id(item.id) == executable_id
            ]
            return matches[0] if len(matches) == 1 else None
        resolved = self._resolve(value, entries, lambda item: (item.id, item.display_name, *item.aliases))
        if resolved is None or (task and not is_planning_model_supported(task, resolved.id)):
            return None
        return resolved

    def resolve_dataset(self, value: str, task: str | None = None) -> DatasetMetadata | None:
        task_id = TASK_IDS.get(task or "", task or "")
        entries = {key: item for key, item in self.datasets.items() if not task_id or task_id in item.task_ids}
        return self._resolve(value, entries, lambda item: (item.id, item.display_name, *item.aliases))

    def recipes_for(self, task: str, model_family: str) -> list[RecipeMetadata]:
        task_id = TASK_IDS.get(task, task)
        family = canonical_reference(model_family)
        return [recipe for recipe in self.recipes.values() if recipe.task_id == task_id and any(
            canonical_reference(candidate) == family for candidate in recipe.model_families
        )]


@lru_cache(maxsize=1)
def get_registry() -> MetadataRegistry:
    return MetadataRegistry()
