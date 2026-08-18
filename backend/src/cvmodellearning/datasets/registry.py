import csv
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DatasetRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BENCHMARK = "benchmark"


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    task: str
    role: DatasetRole
    display_name: str = ""
    aliases: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    description: str = ""
    canonical_family: str | None = None
    derived_from: str | None = None
    synthetic: bool = False
    paired_sample_ids_available: bool = False


_DATASET_LINEAGE: dict[str, dict[str, object]] = {
    "bdd100k_unit_day2night_det_train": {
        "canonical_family": "bdd_100k",
        "derived_from": "bdd_100k_det_train",
        "synthetic": True,
    },
    "bdd100k_fda_day2night_det_train": {
        "canonical_family": "bdd_100k",
        "derived_from": "bdd_100k_det_train",
        "synthetic": True,
    },
    "bdd100k_cyclegan_day2night_det_train": {
        "canonical_family": "bdd_100k",
        "derived_from": "bdd_100k_det_train",
        "synthetic": True,
    },
    "acdc_fda_day2night_det_train": {
        "canonical_family": "acdc",
        "derived_from": "ACDC_det_train",
        "synthetic": True,
    },
}


def _with_lineage(info: DatasetInfo) -> DatasetInfo:
    lineage = _DATASET_LINEAGE.get(info.dataset_id.lower())
    if not lineage:
        return info
    return DatasetInfo(
        dataset_id=info.dataset_id,
        task=info.task,
        role=info.role,
        display_name=info.display_name,
        aliases=info.aliases,
        domains=info.domains,
        description=info.description,
        canonical_family=str(lineage["canonical_family"]),
        derived_from=str(lineage["derived_from"]),
        synthetic=bool(lineage["synthetic"]),
        paired_sample_ids_available=False,
    )


def _dataset(
    dataset_id: str,
    task: str,
    role: DatasetRole,
    *domains: str,
    description: str = "",
) -> DatasetInfo:
    return DatasetInfo(
        dataset_id=dataset_id,
        task=task,
        role=role,
        domains=tuple(domains),
        description=description,
    )


# Runtime correctness must not depend on GraphRAG or on a graph being available.
# The registry is populated from the ontology-backed dataset list whenever it is present,
# and falls back to the hard-coded entries below when the ontology file is unavailable.
_FALLBACK_DATASET_REGISTRY: dict[str, DatasetInfo] = {
    item.dataset_id: item
    for item in (
        _dataset("ACDC_det_train", "detection", DatasetRole.TRAIN, "street", "adverse_weather", "night"),
        _dataset("ACDC_det_val_night", "detection", DatasetRole.VALIDATION, "street", "adverse_weather", "night"),
        _dataset("CUB-200-2011_cls_test", "classification", DatasetRole.TEST, "birds", "fine_grained"),
        _dataset("CUB-200-2011_cls_train", "classification", DatasetRole.TRAIN, "birds", "fine_grained"),
        _dataset("LVIS_det_train", "detection", DatasetRole.TRAIN, "general_objects", "long_tail"),
        _dataset("LVIS_det_val", "detection", DatasetRole.VALIDATION, "general_objects", "long_tail"),
        _dataset("SOP_cls_test", "classification", DatasetRole.TEST, "products", "fine_grained"),
        _dataset("SOP_cls_train", "classification", DatasetRole.TRAIN, "products", "fine_grained"),
        _dataset("UA-DETRAC_det", "detection", DatasetRole.BENCHMARK, "street", "traffic"),
        _dataset("bdd_100k_det_train", "detection", DatasetRole.TRAIN, "street", "autonomous_driving"),
        _dataset("bdd_100k_det_val", "detection", DatasetRole.VALIDATION, "street", "autonomous_driving"),
        _dataset("caltech101_cls", "classification", DatasetRole.BENCHMARK, "general_objects"),
        _dataset("cars196_cls_test", "classification", DatasetRole.TEST, "cars", "fine_grained"),
        _dataset("cars196_cls_train", "classification", DatasetRole.TRAIN, "cars", "fine_grained"),
        _dataset("cars196_det_test", "detection", DatasetRole.TEST, "cars", "fine_grained"),
        _dataset("cifar100_cls_test", "classification", DatasetRole.TEST, "general_objects", "low_resolution"),
        _dataset("cifar10_cls_test", "classification", DatasetRole.TEST, "general_objects", "low_resolution"),
        _dataset("cifar10_cls_train", "classification", DatasetRole.TRAIN, "general_objects", "low_resolution"),
        _dataset("cityscapes_det_val", "detection", DatasetRole.VALIDATION, "street", "autonomous_driving"),
        _dataset("cityscapes_inseg_val", "instance_segmentation", DatasetRole.VALIDATION, "street", "autonomous_driving"),
        _dataset("coco2017_det_val", "detection", DatasetRole.VALIDATION, "general_objects"),
        _dataset("imageNet-1K_cls_train", "classification", DatasetRole.TRAIN, "general_objects"),
        _dataset("imageNet-1K_cls_val", "classification", DatasetRole.VALIDATION, "general_objects"),
        _dataset("mapillary_v1.2_det_train", "detection", DatasetRole.TRAIN, "street", "autonomous_driving"),
        _dataset("mapillary_v1.2_det_val", "detection", DatasetRole.VALIDATION, "street", "autonomous_driving"),
        _dataset("mnist_cls_test", "classification", DatasetRole.TEST, "digits", "grayscale"),
        _dataset("mnist_cls_train", "classification", DatasetRole.TRAIN, "digits", "grayscale"),
        _dataset("objects365_det_val", "detection", DatasetRole.VALIDATION, "general_objects"),
        _dataset("openimages_challenge_2019_det_train", "detection", DatasetRole.TRAIN, "general_objects"),
        _dataset("voc0712_det_val", "detection", DatasetRole.VALIDATION, "general_objects"),
        _dataset("voc07_det_test", "detection", DatasetRole.TEST, "general_objects"),
        _dataset("voc07_det_val", "detection", DatasetRole.VALIDATION, "general_objects"),
        _dataset("voc12_det_train", "detection", DatasetRole.TRAIN, "general_objects"),
        _dataset("voc12_det_val", "detection", DatasetRole.VALIDATION, "general_objects"),
        _dataset("voc12_inseg_train", "instance_segmentation", DatasetRole.TRAIN, "general_objects"),
    )
}


def _build_registry_from_ontology() -> dict[str, DatasetInfo]:
    ontology_path = Path(__file__).resolve().parents[3] / "ontology_data" / "nodes" / "datasets.csv"
    if not ontology_path.exists():
        return {}

    task_map = {
        "object_detection": "detection",
        "image_classification": "classification",
        "instance_segmentation": "instance_segmentation",
        "visual_question_answering": "visual question answering",
    }
    role_map = {
        "Training": DatasetRole.TRAIN,
        "Validation": DatasetRole.VALIDATION,
        "Test": DatasetRole.TEST,
        "Benchmark": DatasetRole.BENCHMARK,
        "Pretraining": DatasetRole.TRAIN,
    }

    registry: dict[str, DatasetInfo] = {}
    with ontology_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            evidence_ids = row.get("evidence_ids", "")
            if "evidence_visionkg_dataset_registry" not in evidence_ids:
                continue
            dataset_id = row.get("id", "").strip()
            task_id = row.get("task_ids", "").strip()
            role_text = row.get("dataset_role", "").strip()
            if not dataset_id or not role_text:
                continue
            task = task_map.get(task_id, task_id)
            role = role_map.get(role_text)
            if role is None:
                continue
            registry[dataset_id] = DatasetInfo(
                dataset_id=dataset_id,
                task=task,
                role=role,
                display_name=row.get("dataset_name", "").strip(),
                aliases=tuple(
                    alias.strip()
                    for alias in row.get("aliases", "").split("|")
                    if alias.strip()
                ),
                description=row.get("description", ""),
            )
    return registry


_ONTOLOGY_DATASET_REGISTRY = _build_registry_from_ontology()

DATASET_REGISTRY: dict[str, DatasetInfo] = {}
for dataset_id, fallback_info in _FALLBACK_DATASET_REGISTRY.items():
    ontology_info = _ONTOLOGY_DATASET_REGISTRY.get(dataset_id)
    if ontology_info is None:
        DATASET_REGISTRY[dataset_id] = fallback_info
    else:
        DATASET_REGISTRY[dataset_id] = DatasetInfo(
            dataset_id=dataset_id,
            task=ontology_info.task or fallback_info.task,
            role=ontology_info.role or fallback_info.role,
            display_name=ontology_info.display_name or fallback_info.display_name,
            aliases=ontology_info.aliases or fallback_info.aliases,
            domains=fallback_info.domains or ontology_info.domains,
            description=(ontology_info.description or fallback_info.description),
        )

for dataset_id, ontology_info in _ONTOLOGY_DATASET_REGISTRY.items():
    if dataset_id not in DATASET_REGISTRY:
        DATASET_REGISTRY[dataset_id] = ontology_info


def infer_dataset_info(dataset_id: str) -> DatasetInfo | None:
    normalized = dataset_id.strip().lower()

    if "det" in normalized or "detection" in normalized:
        task = "detection"
    elif "cls" in normalized or "classification" in normalized:
        task = "classification"
    elif "inseg" in normalized or "instance_segmentation" in normalized:
        task = "instance_segmentation"
    elif "vqa" in normalized or "qa" in normalized or "llava" in normalized:
        task = "visual question answering"
    else:
        return None

    if re.search(r"(_train|\btrain\b)", normalized):
        role = DatasetRole.TRAIN
    elif re.search(r"(_val|_validation|\bval\b)", normalized):
        role = DatasetRole.VALIDATION
    elif re.search(r"(_test|\btest\b)", normalized):
        role = DatasetRole.TEST
    else:
        role = DatasetRole.BENCHMARK

    return _with_lineage(DatasetInfo(
        dataset_id=dataset_id,
        task=task,
        role=role,
        description="Inferred from dataset identifier.",
    ))


def get_dataset_info(dataset_id: str) -> DatasetInfo | None:
    info = DATASET_REGISTRY.get(dataset_id)
    return _with_lineage(info) if info is not None else None


def resolve_dataset_info(dataset_id: str) -> DatasetInfo | None:
    info = DATASET_REGISTRY.get(dataset_id) or infer_dataset_info(dataset_id)
    return _with_lineage(info) if info is not None else None


def dataset_family(dataset_id: str) -> str:
    """Return the stable family shared by a dataset's official split IDs."""

    info = resolve_dataset_info(dataset_id)
    if info is not None and info.canonical_family:
        return info.canonical_family

    return re.sub(
        r"_(?:det|cls|inseg)_(?:train|val|validation|test)(?:_[a-z0-9]+)*$",
        "",
        dataset_id.strip().lower(),
    )
