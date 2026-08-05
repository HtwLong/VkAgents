"""Small shared helpers for assignment-aware dataset downloads."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cvmodellearning.datasets.registry import resolve_dataset_info


@dataclass(frozen=True)
class DownloadAllocation:
    class_name: str
    dataset_name: str
    split: str
    count: int
    assignment_type: str
    source_role: str


def iter_download_allocations(requests: Iterable[Mapping[str, Any]]) -> Iterable[DownloadAllocation]:
    """Yield canonical allocations while accepting the legacy train-only shape."""

    for selection in requests:
        class_name = str(selection.get("class_name") or "").strip()
        if not class_name:
            raise ValueError("Each dataset selection requires a non-empty class_name.")
        for source in selection.get("sources") or []:
            dataset_name = str(source.get("dataset_name") or "").strip()
            if not dataset_name:
                raise ValueError(f"Class {class_name!r} contains a source without dataset_name.")
            info = resolve_dataset_info(dataset_name)
            source_role = info.role.value if info is not None else "unknown"
            allocations = source.get("allocations")
            if allocations is None:
                count = source.get("count", source.get("image_count"))
                allocations = [{
                    "split": "train",
                    "count": count,
                    "assignment_type": "official_split",
                }]
            for allocation in allocations:
                count = allocation.get("count")
                split = str(allocation.get("split") or "").strip()
                assignment_type = str(allocation.get("assignment_type") or "").strip()
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                    raise ValueError(
                        f"Allocation for {class_name!r} from {dataset_name!r} requires a positive integer count."
                    )
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"Unsupported dataset split: {split!r}.")
                if assignment_type not in {"official_split", "derived_from_train"}:
                    raise ValueError(f"Unsupported assignment type: {assignment_type!r}.")
                if assignment_type == "official_split" and source_role not in {"unknown", split}:
                    raise ValueError(
                        f"Official source role {source_role!r} does not match assigned split {split!r}."
                    )
                if assignment_type == "derived_from_train" and (
                    source_role != "train" or split == "train"
                ):
                    raise ValueError(
                        "Derived holdouts must assign an official training source to validation or test."
                    )
                yield DownloadAllocation(
                    class_name=class_name,
                    dataset_name=dataset_name,
                    split=split,
                    count=count,
                    assignment_type=assignment_type,
                    source_role=source_role,
                )


class DatasetManifest:
    """Collect one provenance record per unique downloaded image."""

    def __init__(self, job_id: str, task: str, assignment_fingerprint: str):
        self._data: dict[str, Any] = {
            "version": 1,
            "job_id": job_id,
            "task": task,
            "assignment_fingerprint": assignment_fingerprint,
            "samples": [],
        }
        self._samples: dict[str, dict[str, Any]] = {}

    def add(
        self,
        *,
        sample_id: str,
        image_path: str,
        class_name: str,
        dataset_name: str,
        source_role: str,
        assigned_split: str,
        assignment_type: str,
    ) -> None:
        existing = self._samples.get(sample_id)
        if existing is not None:
            if (
                existing["image_path"] != image_path
                or existing["dataset_name"] != dataset_name
                or existing["source_role"] != source_role
                or existing["assignment_type"] != assignment_type
            ):
                raise ValueError(f"Sample {sample_id!r} has inconsistent provenance metadata.")
            if existing["assigned_split"] != assigned_split:
                raise ValueError(
                    f"Sample {sample_id!r} cannot belong to both "
                    f"{existing['assigned_split']!r} and {assigned_split!r}."
                )
            if class_name not in existing["class_names"]:
                existing["class_names"].append(class_name)
                existing["class_names"].sort()
            return

        sample = {
            "sample_id": sample_id,
            "image_path": image_path,
            "class_names": [class_name],
            "dataset_name": dataset_name,
            "source_role": source_role,
            "assigned_split": assigned_split,
            "assignment_type": assignment_type,
        }
        self._samples[sample_id] = sample
        self._data["samples"].append(sample)

    def split_for(self, sample_id: str) -> str | None:
        sample = self._samples.get(sample_id)
        return str(sample["assigned_split"]) if sample else None

    def as_dict(self) -> dict[str, Any]:
        return self._data


def assignment_fingerprint(requests: Iterable[Mapping[str, Any]]) -> str:
    """Hash the semantic allocation plan independently of input ordering."""

    allocations = [
        {
            "class_name": item.class_name,
            "dataset_name": item.dataset_name,
            "split": item.split,
            "count": item.count,
            "assignment_type": item.assignment_type,
            "source_role": item.source_role,
        }
        for item in iter_download_allocations(requests)
    ]
    allocations.sort(key=lambda item: tuple(str(value) for value in item.values()))
    payload = json.dumps(allocations, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a stable content fingerprint for a persisted dataset artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_content_isolation(
    manifest: Mapping[str, Any],
    data_root: Path,
) -> dict[str, int]:
    """Reject exact image duplicates that cross authoritative split boundaries."""

    first_by_digest: dict[str, tuple[str, str]] = {}
    duplicate_samples = 0
    for sample in manifest.get("samples") or []:
        relative_path = str(sample.get("image_path") or "")
        split = str(sample.get("assigned_split") or "")
        digest = file_sha256(data_root / relative_path)
        previous = first_by_digest.get(digest)
        if previous is not None:
            previous_split, previous_path = previous
            if previous_split != split:
                raise ValueError(
                    "Identical image content occurs in multiple dataset splits: "
                    f"{previous_path!r} ({previous_split}) and "
                    f"{relative_path!r} ({split})."
                )
            duplicate_samples += 1
            continue
        first_by_digest[digest] = (split, relative_path)

    return {
        "unique_content_hashes": len(first_by_digest),
        "duplicate_samples": duplicate_samples,
        "cross_split_duplicates": 0,
    }


def load_dataset_manifest(
    path: Path,
    *,
    task: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Load and validate the manifest used as preparation's source of truth."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("task") != task:
        raise ValueError(f"Dataset manifest task must be {task!r}.")
    if data.get("assignment_fingerprint") != expected_fingerprint:
        raise ValueError("Dataset manifest does not match the current selected_data assignments.")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset manifest contains no samples.")

    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        image_path = str(sample.get("image_path") or "")
        if not sample_id or sample_id in sample_ids:
            raise ValueError("Dataset manifest sample IDs must be non-empty and unique.")
        if not image_path or image_path in image_paths:
            raise ValueError("Dataset manifest image paths must be non-empty and unique.")
        if sample.get("assigned_split") not in {"train", "validation", "test"}:
            raise ValueError("Dataset manifest contains an unsupported split.")
        class_names = sample.get("class_names")
        if (
            not isinstance(class_names, list)
            or not class_names
            or any(not isinstance(name, str) or not name.strip() for name in class_names)
            or len(class_names) != len(set(class_names))
        ):
            raise ValueError("Dataset manifest class names must be non-empty and unique.")
        if not str(sample.get("dataset_name") or "").strip():
            raise ValueError("Dataset manifest samples require a dataset name.")
        split = sample["assigned_split"]
        source_role = sample.get("source_role")
        assignment_type = sample.get("assignment_type")
        if assignment_type == "official_split" and source_role not in {"unknown", split}:
            raise ValueError("Official manifest samples must remain in their source split.")
        if assignment_type == "derived_from_train" and (
            source_role != "train" or split == "train"
        ):
            raise ValueError("Derived manifest samples must be holdouts from a training source.")
        if assignment_type not in {"official_split", "derived_from_train"}:
            raise ValueError("Dataset manifest contains an unsupported assignment type.")
        sample_ids.add(sample_id)
        image_paths.add(image_path)
    return data


def load_preparation_summary(
    path: Path,
    *,
    task: str,
    expected_fingerprint: str,
    expected_manifest_sha256: str,
    required_artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate that preparation completed for the current assignment plan."""

    if not path.exists():
        raise FileNotFoundError(f"Preparation summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("task") != task:
        raise ValueError(f"Preparation summary task must be {task!r}.")
    if data.get("assignment_fingerprint") != expected_fingerprint:
        raise ValueError("Prepared data does not match the current selected_data assignments.")
    if data.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Prepared data does not match the current dataset manifest.")

    counts = data.get("counts")
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(split), int) or counts[split] <= 0
        for split in ("train", "validation", "test")
    ):
        raise ValueError("Preparation summary requires positive train, validation, and test counts.")
    missing = [
        name
        for name, artifact in required_artifacts.items()
        if not artifact.is_file() or artifact.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f"Prepared data artifacts are missing: {', '.join(missing)}")
    return data
