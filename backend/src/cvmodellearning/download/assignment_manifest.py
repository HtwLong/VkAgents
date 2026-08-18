"""Small shared helpers for assignment-aware dataset downloads."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cvmodellearning.datasets.registry import resolve_dataset_info


MINIMUM_DETECTION_UNIQUE_COVERAGE_RATIO = 0.85


@dataclass(frozen=True)
class DownloadAllocation:
    class_name: str
    dataset_name: str
    split: str
    count: int
    assignment_type: str
    source_role: str


class DatasetContentConflict(ValueError):
    """Raised when exact image bytes occur across authoritative splits."""

    def __init__(self, conflicts: list[dict[str, str]]):
        self.conflicts = conflicts
        count = len(conflicts)
        super().__init__(
            "Identical image content occurs in multiple dataset splits "
            f"({count} conflict{'s' if count != 1 else ''})."
        )


def detection_coverage_requirements(
    requests: Iterable[Mapping[str, Any]],
    data_plan_constraints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate allocation counts into independent detection coverage goals.

    Detection allocations historically represented positive images per class.  Their
    sum is also the intended pool size, but multi-label images can satisfy more than
    one class allocation.  Keeping both constraints explicit prevents that overlap
    from silently shrinking the actual training pool.
    """

    per_split = {split: 0 for split in ("train", "validation", "test")}
    per_class: dict[str, int] = {}
    per_class_split: dict[str, dict[str, int]] = {}
    for item in iter_download_allocations(requests):
        per_split[item.split] += item.count
        per_class[item.class_name] = per_class.get(item.class_name, 0) + item.count
        class_splits = per_class_split.setdefault(
            item.class_name,
            {split: 0 for split in ("train", "validation", "test")},
        )
        class_splits[item.split] += item.count
    constraints = dict(data_plan_constraints or {})
    allocation_total = sum(per_split.values())
    preferred_unique = int(
        constraints.get("preferred_unique_pool_images")
        or constraints.get("preferred_unique_images")
        or allocation_total
    )
    minimum_unique = int(
        constraints.get("minimum_unique_pool_images")
        or constraints.get("minimum_unique_images")
        or 0
    )
    if constraints.get("preferred_target_is_strict"):
        minimum_unique = preferred_unique
    if not minimum_unique:
        # Per-class detection allocations overlap: one multi-label image can
        # satisfy several class requests. Without an explicit planning objective,
        # the only guaranteed feasible lower bound is the largest individual
        # class pool, not a percentage of the sum of every class allocation.
        minimum_unique = min(preferred_unique, max(per_class.values(), default=0))
    preferred_unique = max(minimum_unique, preferred_unique)
    if allocation_total:
        exact = {
            split: preferred_unique * count / allocation_total
            for split, count in per_split.items()
        }
        preferred_by_split = {
            split: int(value) for split, value in exact.items()
        }
        remainder = preferred_unique - sum(preferred_by_split.values())
        for split in sorted(
            per_split,
            key=lambda name: (exact[name] - int(exact[name]), per_split[name], name),
            reverse=True,
        )[:remainder]:
            preferred_by_split[split] += 1
    else:
        preferred_by_split = dict(per_split)
    return {
        "target_unique_images": preferred_unique,
        "minimum_unique_images": minimum_unique,
        "target_unique_images_by_split": preferred_by_split,
        "minimum_images_per_class": per_class,
        "minimum_images_per_class_by_split": per_class_split,
    }


def summarize_detection_manifest_coverage(
    manifest: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure unique-image and class-presence coverage from manifest samples."""

    unique_by_split = {split: 0 for split in ("train", "validation", "test")}
    images_per_class: dict[str, int] = {
        str(name): 0 for name in requirements.get("minimum_images_per_class", {})
    }
    images_per_class_split: dict[str, dict[str, int]] = {
        name: {split: 0 for split in unique_by_split}
        for name in images_per_class
    }
    for sample in manifest.get("samples") or []:
        split = str(sample.get("assigned_split") or "")
        if split in unique_by_split:
            unique_by_split[split] += 1
        for class_name in set(sample.get("class_names") or []):
            images_per_class[class_name] = images_per_class.get(class_name, 0) + 1
            images_per_class_split.setdefault(
                class_name, {name: 0 for name in unique_by_split}
            )
            if split in unique_by_split:
                images_per_class_split[class_name][split] += 1

    unique_total = sum(unique_by_split.values())
    unique_shortfall = max(
        0, int(requirements.get("target_unique_images", 0)) - unique_total
    )
    class_shortfalls = {
        name: max(0, int(required) - images_per_class.get(name, 0))
        for name, required in requirements.get("minimum_images_per_class", {}).items()
    }
    class_split_shortfalls = {
        name: {
            split: max(
                0,
                int(required) - images_per_class_split.get(name, {}).get(split, 0),
            )
            for split, required in split_requirements.items()
        }
        for name, split_requirements in requirements.get(
            "minimum_images_per_class_by_split", {}
        ).items()
    }
    target_unique_images = int(requirements.get("target_unique_images", 0))
    minimum_unique_images = int(
        requirements.get("minimum_unique_images", target_unique_images)
    )
    unique_coverage_ratio = (
        min(1.0, unique_total / target_unique_images)
        if target_unique_images
        else 1.0
    )
    class_split_coverage_satisfied = not any(
        shortfall
        for split_shortfalls in class_split_shortfalls.values()
        for shortfall in split_shortfalls.values()
    )
    return {
        "requirements": dict(requirements),
        "verified_unique_images": unique_total,
        "verified_unique_images_by_split": unique_by_split,
        "verified_images_per_class": images_per_class,
        "verified_images_per_class_by_split": images_per_class_split,
        "unique_image_shortfall": unique_shortfall,
        "minimum_unique_images": minimum_unique_images,
        "minimum_unique_image_shortfall": max(0, minimum_unique_images - unique_total),
        "unique_coverage_ratio": unique_coverage_ratio,
        "class_image_shortfalls": class_shortfalls,
        "class_split_image_shortfalls": class_split_shortfalls,
        "class_split_coverage_satisfied": class_split_coverage_satisfied,
        "satisfied": (
            unique_shortfall == 0
            and not any(class_shortfalls.values())
            and class_split_coverage_satisfied
        ),
    }


def evaluate_detection_coverage_acceptance(
    coverage: Mapping[str, Any],
    *,
    unresolved_transfer_failures: Iterable[Mapping[str, Any]] = (),
    cross_split_duplicates: Iterable[Mapping[str, Any]] = (),
    minimum_unique_coverage_ratio: float = MINIMUM_DETECTION_UNIQUE_COVERAGE_RATIO,
) -> dict[str, Any]:
    """Apply the shared strict-or-aspirational detection coverage contract."""

    transfer_failures = [dict(item) for item in unresolved_transfer_failures]
    split_duplicates = [dict(item) for item in cross_split_duplicates]
    unique_shortfall = int(coverage.get("unique_image_shortfall", 0))
    minimum_unique_shortfall = int(
        coverage.get("minimum_unique_image_shortfall", unique_shortfall)
    )
    unique_ratio = float(coverage.get("unique_coverage_ratio", 0.0))
    requirements = coverage.get("requirements") or {}
    target_unique = int(requirements.get("target_unique_images") or 0)
    minimum_unique = int(
        coverage.get("minimum_unique_images")
        or requirements.get("minimum_unique_images")
        or 0
    )
    effective_minimum_ratio = (
        min(1.0, minimum_unique / target_unique)
        if target_unique else minimum_unique_coverage_ratio
    )
    mandatory_coverage_satisfied = bool(
        coverage.get("class_split_coverage_satisfied", False)
        and minimum_unique_shortfall == 0
    )
    # Individual transfer failures are resolved when replacements leave every
    # mandatory class/split objective satisfied. Preserve them as diagnostics,
    # but do not reject an otherwise complete materialized manifest.
    unresolved_failures = [] if mandatory_coverage_satisfied else transfer_failures
    replaced_failures = transfer_failures if mandatory_coverage_satisfied else []
    integrity_satisfied = not unresolved_failures and not split_duplicates
    strict_target_satisfied = bool(
        unique_shortfall == 0 and mandatory_coverage_satisfied
    )
    aspirational_target_accepted = bool(
        unique_shortfall > 0
        and mandatory_coverage_satisfied
        and integrity_satisfied
    )
    return {
        "minimum_unique_coverage_ratio": effective_minimum_ratio,
        "unique_coverage_ratio": unique_ratio,
        "unique_target_satisfied": unique_shortfall == 0,
        "minimum_unique_target_satisfied": minimum_unique_shortfall == 0,
        "strict_target_satisfied": strict_target_satisfied,
        "aspirational_unique_target_accepted": aspirational_target_accepted,
        "class_split_coverage_satisfied": mandatory_coverage_satisfied,
        "transfer_failures": transfer_failures,
        "successfully_replaced_transfer_failures": replaced_failures,
        "unresolved_transfer_failures": unresolved_failures,
        "cross_split_duplicates": split_duplicates,
        "accepted": bool(
            (strict_target_satisfied or aspirational_target_accepted)
            and integrity_satisfied
        ),
    }


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

    def unique_count(self, split: str | None = None) -> int:
        if split is None:
            return len(self._samples)
        return sum(
            sample["assigned_split"] == split for sample in self._samples.values()
        )

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


def files_are_byte_identical(first: Path, second: Path) -> bool:
    """Confirm an exact duplicate after a content-hash lookup.

    SHA-256 is used to find likely duplicates efficiently. Comparing sizes and
    bytes before rejection also makes the behavior correct in the theoretical
    event of a hash collision.
    """

    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as first_handle, second.open("rb") as second_handle:
        while True:
            first_chunk = first_handle.read(1024 * 1024)
            second_chunk = second_handle.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def validate_content_isolation(
    manifest: Mapping[str, Any],
    data_root: Path,
) -> dict[str, int]:
    """Reject exact image duplicates that cross authoritative split boundaries."""

    first_by_digest: dict[str, tuple[str, str]] = {}
    duplicate_samples = 0
    conflicts: list[dict[str, str]] = []
    for sample in manifest.get("samples") or []:
        relative_path = str(sample.get("image_path") or "")
        split = str(sample.get("assigned_split") or "")
        digest = file_sha256(data_root / relative_path)
        previous = first_by_digest.get(digest)
        if previous is not None:
            previous_split, previous_path = previous
            if previous_split != split:
                conflicts.append({
                    "first_path": previous_path,
                    "first_split": previous_split,
                    "duplicate_path": relative_path,
                    "duplicate_split": split,
                })
                continue
            duplicate_samples += 1
            continue
        first_by_digest[digest] = (split, relative_path)

    if conflicts:
        raise DatasetContentConflict(conflicts)

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
