import csv
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import PurePosixPath

from cvmodellearning.download.assignment_manifest import (
    DatasetManifest,
    MINIMUM_DETECTION_UNIQUE_COVERAGE_RATIO,
    assignment_fingerprint,
    detection_coverage_requirements,
    evaluate_detection_coverage_acceptance,
    file_sha256,
    files_are_byte_identical,
    iter_download_allocations,
    summarize_detection_manifest_coverage,
)
from cvmodellearning.download.visionkg_utils import prepare_data, visionkg2cocoDet, query, visionkg_parse_classification, prepare_data_flat
from cvmodellearning.download.image_cache import materialize_cached_images
from cvmodellearning.paths import (
    data_dir,
    dataset_manifest_path,
    download_report_path,
    json_labels_path,
    csv_labels_path,
    visionkg_cache_dir,
)


# Detection queries expand every selected image into all requested-class boxes.
# Keep the image page bounded so large allocations do not create a single,
# resource-heavy VisionKG response.  This is an image limit, not a row limit.
DETECTION_SPARQL_PAGE_SIZE = 100


def _sparql_string(value: str) -> str:
    """Escape a user/config value for use inside a SPARQL string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _detection_source_identity(dataset_name: str, image_name: str) -> str:
    """Identify source photographs shared under different dataset namespaces.

    LVIS annotations are defined on COCO photographs. VisionKG exposes those
    photographs under separate LVIS and COCO paths (and separate RDF image
    URIs), so neither value alone is suitable for enforcing split isolation.
    Their zero-padded COCO filename is the stable shared identity.
    """

    normalized_dataset = dataset_name.casefold()
    filename = PurePosixPath(image_name.replace("\\", "/")).name.casefold()
    if normalized_dataset.startswith(("coco", "lvis")):
        return f"coco-source:{filename}"
    return f"dataset-source:{normalized_dataset}/{filename}"


def _persist_download_artifacts(job_id: str, manifest: DatasetManifest, report: dict) -> None:
    manifest_path = dataset_manifest_path(job_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    report_path = download_report_path(job_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["manifest_path"] = str(manifest_path)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _manifest_cross_split_duplicates(manifest: dict) -> list[dict[str, str]]:
    """Return duplicate persisted image paths assigned to different splits."""
    first_split_by_path: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    for sample in manifest.get("samples") or []:
        image_path = str(sample.get("image_path") or "")
        split = str(sample.get("assigned_split") or "")
        previous_split = first_split_by_path.setdefault(image_path, split)
        if previous_split != split:
            conflicts.append({
                "image_path": image_path,
                "first_split": previous_split,
                "duplicate_split": split,
            })
    return conflicts


def _prepare_with_progress(images: list, job_id: str, progress_callback, cancel_check=None):
    cache_root = visionkg_cache_dir()
    kwargs = {"DATA_ROOT_PATH": str(cache_root)}
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    if cancel_check is not None:
        kwargs["cancel_check"] = cancel_check
    result = prepare_data(images, **kwargs)
    # Real prepare_data results always contain metrics. Keeping legacy/mocked
    # adapters usable makes this boundary backwards compatible for callers.
    if "metrics" in result:
        materialization = materialize_cached_images(
            result.get("successful") or [], cache_root, data_dir(job_id)
        )
        result["metrics"]["materialization"] = materialization
    return result


def _empty_performance() -> dict:
    return {
        "elapsed_seconds": 0.0,
        "sparql_queries": 0,
        "sparql_seconds": 0.0,
        "download_batches": 0,
        "cache_hits": 0,
        "cached_failures": 0,
        "bytes_downloaded": 0,
        "retries": 0,
        "transfer_and_validation_seconds": 0.0,
    }


def _merge_download_metrics(performance: dict, result: dict) -> None:
    performance["download_batches"] += 1
    metrics = result.get("metrics") or {}
    for key in (
        "cache_hits",
        "cached_failures",
        "bytes_downloaded",
        "retries",
        "transfer_and_validation_seconds",
    ):
        performance[key] += metrics.get(key, 0)


def _timed_query(query_string: str, performance: dict, cancel_check=None):
    started = time.perf_counter()
    try:
        if cancel_check is None:
            return query(query_string)
        return query(query_string, cancel_check=cancel_check)
    finally:
        performance["sparql_queries"] += 1
        performance["sparql_seconds"] += time.perf_counter() - started


def _candidate_scan_limit(count: int, already_reserved: int) -> int:
    """Leave room to page past samples reserved by earlier allocations."""
    replacement_buffer = max(count * 3, count + 50)
    return already_reserved + replacement_buffer


@dataclass
class _ClassificationCandidatePool:
    allocation: object
    candidates: list[dict]
    assigned: list[dict]
    next_offset: int = 0
    exhausted: bool = False


def _assign_unique_classification_candidates(
    pools: list[_ClassificationCandidatePool],
    *,
    excluded_paths: set[str] | None = None,
) -> None:
    """Assign unique images, using augmenting moves to protect scarce pools."""
    excluded_paths = excluded_paths or set()
    owner_by_path: dict[str, int] = {}
    for pool in pools:
        pool.assigned.clear()

    def move_one(pool_index: int, blocked_path: str, visited: set[int]) -> bool:
        if pool_index in visited:
            return False
        visited = visited | {pool_index}
        pool = pools[pool_index]
        assigned_paths = {item["image_path"] for item in pool.assigned}
        for candidate in pool.candidates:
            path = candidate["image_path"]
            if path == blocked_path or path in excluded_paths or path in assigned_paths:
                continue
            owner = owner_by_path.get(path)
            if owner is not None and not move_one(owner, path, visited):
                continue
            displaced = next(item for item in pool.assigned if item["image_path"] == blocked_path)
            pool.assigned.remove(displaced)
            pool.assigned.append(candidate)
            owner_by_path.pop(blocked_path, None)
            owner_by_path[path] = pool_index
            return True
        return False

    order = sorted(
        range(len(pools)),
        key=lambda index: (
            len(pools[index].candidates) - pools[index].allocation.count,
            len(pools[index].candidates),
            pools[index].allocation.class_name,
            pools[index].allocation.dataset_name,
            pools[index].allocation.split,
        ),
    )
    for pool_index in order:
        pool = pools[pool_index]
        for candidate in pool.candidates:
            if len(pool.assigned) >= pool.allocation.count:
                break
            path = candidate["image_path"]
            if path in excluded_paths or path in {item["image_path"] for item in pool.assigned}:
                continue
            owner = owner_by_path.get(path)
            if owner is not None and not move_one(owner, path, set()):
                continue
            pool.assigned.append(candidate)
            owner_by_path[path] = pool_index


def _classification_candidate_query(allocation, limit: int, offset: int = 0) -> str:
    escaped_class = _sparql_string(allocation.class_name)
    escaped_dataset = _sparql_string(allocation.dataset_name)
    return f"""
    PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
    PREFIX schema:<http://schema.org/>

    SELECT DISTINCT ?datasetName ?imageName ?image
    WHERE {{
        VALUES ?datasetName {{ "{escaped_dataset}" }}
        VALUES ?labelName {{ "{escaped_class}" }}
        ?image schema:isPartOf / schema:name ?datasetName .
        ?image schema:name ?imageName .
        ?image cv:hasAnnotation ?annotation .
        ?annotation cv:hasLabel/cv:label ?labelName .
    }}
    ORDER BY STR(?image)
    LIMIT {limit}
    OFFSET {offset}
    """

def download_visionkg_mixed_datasets_detection(
    job_id: str, requests: list, *, data_plan_constraints=None,
    progress_callback=None, cancel_check=None
):
    """Download detection images without allowing an image to cross split boundaries."""

    if not isinstance(requests, list):
        raise TypeError(f"Input 'requests' must be a list. Got {type(requests)}.")
    if not requests:
        return {"job_id": job_id, "sources": [], "complete": True, "requested": 0, "downloaded": 0}

    allocations = list(iter_download_allocations(requests))
    coverage_requirements = detection_coverage_requirements(
        requests, data_plan_constraints,
    )
    last_allocation_by_split = {
        split: max(index for index, item in enumerate(allocations) if item.split == split)
        for split in coverage_requirements["target_unique_images_by_split"]
        if any(item.split == split for item in allocations)
    }
    top_up_sources_by_split: dict[str, dict[str, object]] = {}
    for item in allocations:
        top_up_sources_by_split.setdefault(item.split, {}).setdefault(
            item.dataset_name, item
        )
    requested_classes = sorted({
        allocation.class_name
        for allocation in allocations
    })
    requested_label_values = " ".join(
        f'"{_sparql_string(class_name)}"'
        for class_name in requested_classes
    )

    global_image_map: dict[str, int] = {}
    global_category_map: dict[str, int] = {}
    global_anno_id_counter = [0]
    annotation_signatures: set[tuple] = set()
    # Paths control download reuse; source identities enforce split isolation.
    # These differ for datasets such as LVIS, which reuses COCO photographs
    # under a separate dataset namespace and RDF image URI.
    path_reservations: dict[str, str] = {}
    source_reservations: dict[str, str] = {}
    # File names and RDF identifiers are not always unique representations of
    # a photograph. Reserve the downloaded bytes before assigning a candidate.
    content_reservations: dict[str, list[tuple[str, str]]] = {}
    manifest = DatasetManifest(job_id, "detection", assignment_fingerprint(requests))
    report = {"job_id": job_id, "task": "detection", "sources": [], "complete": True}
    performance = _empty_performance()
    download_started = time.perf_counter()
    master_coco_data = {"images": [], "annotations": [], "categories": []}

    for allocation_index, allocation in enumerate(allocations):
        escaped_class = _sparql_string(allocation.class_name)
        escaped_dataset = _sparql_string(allocation.dataset_name)
        accepted_keys: set[str] = set()
        direct_accepted_keys: set[str] = set()
        top_up_accepted_keys: set[str] = set()
        seen_candidates: set[str] = set()
        failures: list[dict] = []
        conflicts = 0
        content_duplicate_conflicts = 0
        content_duplicates: list[dict[str, str]] = []
        offset = 0
        top_up_offset = 0
        top_up_started = False
        dataset_prefix = f"{allocation.dataset_name}/"
        already_reserved = sum(key.startswith(dataset_prefix) for key in path_reservations)
        # Cross-namespace aliases can reject candidates even when no path from
        # this dataset has been seen yet. Include all prior source reservations
        # as conservative pagination headroom so replacements remain reachable.
        reservation_headroom = max(already_reserved, len(source_reservations))
        is_split_top_up = last_allocation_by_split.get(allocation.split) == allocation_index
        split_unique_target = coverage_requirements["target_unique_images_by_split"][allocation.split]
        initial_split_unique = manifest.unique_count(allocation.split)
        required_new_unique = (
            max(0, split_unique_target - initial_split_unique) if is_split_top_up else 0
        )
        scan_target = allocation.count + required_new_unique
        max_candidates = _candidate_scan_limit(scan_target, reservation_headroom)
        top_up_max_candidates = _candidate_scan_limit(
            split_unique_target,
            len(source_reservations),
        )

        while (
            len(accepted_keys) < allocation.count
            or (is_split_top_up and manifest.unique_count(allocation.split) < split_unique_target)
        ):
            remaining_class = max(0, allocation.count - len(accepted_keys))
            remaining_unique = (
                max(0, split_unique_target - manifest.unique_count(allocation.split))
                if is_split_top_up else 0
            )
            top_up_mode = remaining_class == 0 and remaining_unique > 0
            if top_up_mode and not top_up_started:
                top_up_started = True
                top_up_offset = 0
            current_offset = top_up_offset if top_up_mode else offset
            current_limit = top_up_max_candidates if top_up_mode else max_candidates
            if current_offset >= current_limit:
                break
            remaining = max(remaining_class, remaining_unique, 1)
            page_target = (
                max(remaining * 3, remaining + 20)
                if top_up_mode else remaining
            )
            page_limit = min(
                DETECTION_SPARQL_PAGE_SIZE,
                page_target,
                current_limit - current_offset,
            )
            if top_up_mode:
                candidate_dataset_values = " ".join(
                    f'"{_sparql_string(name)}"'
                    for name in top_up_sources_by_split[allocation.split]
                )
                candidate_label_values = requested_label_values
            else:
                candidate_dataset_values = f'"{escaped_dataset}"'
                candidate_label_values = f'"{escaped_class}"'
            query_string = f"""
            PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
            PREFIX schema:<http://schema.org/>
            
            SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image 
                   ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
            WHERE {{
                # --- INNER SUBQUERY (Limits by Image Count) ---
                {{
                    SELECT DISTINCT ?image ?datasetName
                    WHERE {{
                        VALUES ?datasetName {{ {candidate_dataset_values} }}
                        VALUES ?candidateLabel {{ {candidate_label_values} }}
                        ?image schema:isPartOf / schema:name ?datasetName .
                        
                        ?image cv:hasAnnotation ?ann .
                        ?ann cv:hasLabel/cv:label ?candidateLabel .
                    }}
                    ORDER BY STR(?image)
                    LIMIT {page_limit}
                    OFFSET {current_offset}
                }}

                # --- OUTER DATA FETCHING ---
                OPTIONAL {{ ?image schema:name ?imageName }} .
                OPTIONAL {{ 
                    ?image cv:imgWidth ?imageWidth .
                    ?image cv:imgHeight ?imageHeight .
                }}

                ?image cv:hasAnnotation ?annotation .
                ?annotation cv:hasLabel/cv:label ?labelName .
                VALUES ?labelName {{ {requested_label_values} }}
                
                # Keep every requested-class box on a selected image. Restricting
                # this to the allocation class would turn other target objects into
                # false background during detection training.
                ?annotation cv:hasBox ?bbox .
                ?bbox cv:boxHeight ?bbHeight .
                ?bbox cv:boxWidth ?bbWidth .
                ?bbox cv:centerX ?bbCentreX .
                ?bbox cv:centerY ?bbCentreY .
            }}
            """
            raw_result = _timed_query(query_string, performance, cancel_check)
            if not raw_result:
                break
            if top_up_mode:
                top_up_offset += page_limit
            else:
                offset += page_limit

            grouped: OrderedDict[str, list[dict]] = OrderedDict()
            for row in raw_result:
                image_name = row.get("imageName")
                dataset_name = row.get("datasetName")
                if not image_name or not dataset_name:
                    continue
                image_key = f"{dataset_name}/{image_name}"
                grouped.setdefault(image_key, []).append(row)

            candidate_keys: list[str] = []
            for image_key in grouped:
                if image_key in seen_candidates:
                    continue
                seen_candidates.add(image_key)
                row = grouped[image_key][0]
                source_identity = _detection_source_identity(
                    str(row["datasetName"]),
                    str(row["imageName"]),
                )
                reserved_split = source_reservations.get(source_identity)
                if reserved_split is not None and reserved_split != allocation.split:
                    conflicts += 1
                    continue
                candidate_keys.append(image_key)

            if top_up_mode:
                current_coverage = summarize_detection_manifest_coverage(
                    manifest.as_dict(), coverage_requirements
                )
                current_counts = current_coverage[
                    "verified_images_per_class_by_split"
                ]
                required_counts = coverage_requirements[
                    "minimum_images_per_class_by_split"
                ]

                def balance_score(image_key: str) -> tuple:
                    labels = {
                        str(row.get("labelName"))
                        for row in grouped[image_key]
                        if str(row.get("labelName")) in requested_classes
                    }
                    projected = {
                        name: current_counts.get(name, {}).get(allocation.split, 0)
                        + int(name in labels)
                        for name in requested_classes
                    }
                    deficits_served = sum(
                        current_counts.get(name, {}).get(allocation.split, 0)
                        < required_counts.get(name, {}).get(allocation.split, 0)
                        for name in labels
                    )
                    values = list(projected.values())
                    imbalance = max(values) - min(values) if values else 0
                    overage = sum(
                        max(
                            0,
                            projected[name]
                            - required_counts.get(name, {}).get(allocation.split, 0),
                        )
                        for name in requested_classes
                    )
                    return (-deficits_served, imbalance, overage, len(labels), image_key)

                candidate_keys.sort(key=balance_score)
            chosen = candidate_keys[:remaining]

            new_candidates = []
            for image_key in chosen:
                if image_key in path_reservations:
                    continue
                row = grouped[image_key][0]
                new_candidates.append({
                    "file_name": row["imageName"],
                    "image_path": image_key,
                    "url": f"https://vision-api.semkg.org/api/image?image=/{image_key}",
                })
            download_result = (
                _prepare_with_progress(new_candidates, job_id, progress_callback, cancel_check)
                if new_candidates
                else {"successful": [], "failures": []}
            )
            failures.extend(download_result["failures"])
            if new_candidates:
                _merge_download_metrics(performance, download_result)
            successful_new = {item["image_path"] for item in download_result["successful"]}
            content_accepted: set[str] = set()
            for image_key in successful_new:
                digest = file_sha256(data_dir(job_id) / image_key)
                previous = next(
                    (
                        reservation
                        for reservation in content_reservations.get(digest, [])
                        if files_are_byte_identical(
                            data_dir(job_id) / reservation[1],
                            data_dir(job_id) / image_key,
                        )
                    ),
                    None,
                )
                if previous is not None:
                    previous_split, previous_path = previous
                    duplicate = {
                        "image_path": image_key,
                        "assigned_split": allocation.split,
                        "duplicate_of": previous_path,
                        "duplicate_split": previous_split,
                        "sha256": digest,
                    }
                    content_duplicates.append(duplicate)
                    failures.append({
                        **duplicate,
                        "reason": "duplicate_content",
                    })
                    content_duplicate_conflicts += 1
                    continue
                content_reservations.setdefault(digest, []).append(
                    (allocation.split, image_key)
                )
                content_accepted.add(image_key)
            successful_keys = [
                image_key
                for image_key in chosen
                if image_key in path_reservations or image_key in content_accepted
            ]
            if not successful_keys and len(raw_result) < page_limit:
                break

            accepted_keys.update(successful_keys)
            (top_up_accepted_keys if top_up_mode else direct_accepted_keys).update(
                successful_keys
            )
            accepted_rows = [
                row
                for image_key in successful_keys
                for row in grouped[image_key]
            ]
            partial_coco_data = visionkg2cocoDet(
                accepted_rows,
                global_image_map=global_image_map, 
                global_category_map=global_category_map, 
                global_anno_id_counter=global_anno_id_counter
            )

            for image in partial_coco_data.get("images", []):
                image_dataset = str(image.get("image_path") or "").split("/", 1)[0]
                image_allocation = top_up_sources_by_split[allocation.split].get(
                    image_dataset, allocation
                )
                image["assigned_split"] = allocation.split
                image["assignment_type"] = image_allocation.assignment_type
                image["source_role"] = image_allocation.source_role
            master_coco_data["images"].extend(partial_coco_data.get("images", []))
            for annotation in partial_coco_data.get("annotations", []):
                signature = (
                    annotation["image_id"],
                    annotation["category_id"],
                    tuple(annotation["bbox"]),
                )
                if signature not in annotation_signatures:
                    annotation_signatures.add(signature)
                    master_coco_data["annotations"].append(annotation)
            master_coco_data["categories"] = partial_coco_data.get("categories", [])

            for image_key in successful_keys:
                row = grouped[image_key][0]
                image_allocation = top_up_sources_by_split[allocation.split].get(
                    str(row["datasetName"]), allocation
                )
                path_reservations.setdefault(image_key, allocation.split)
                source_identity = _detection_source_identity(
                    str(row["datasetName"]),
                    str(row["imageName"]),
                )
                source_reservations.setdefault(source_identity, allocation.split)
                sample_id = str(row.get("image") or image_key)
                image_labels = sorted({
                    str(item.get("labelName"))
                    for item in grouped[image_key]
                    if str(item.get("labelName")) in requested_classes
                })
                for image_label in image_labels:
                    manifest.add(
                        sample_id=sample_id,
                        image_path=image_key,
                        class_name=image_label,
                        dataset_name=str(row["datasetName"]),
                        source_role=image_allocation.source_role,
                        assigned_split=allocation.split,
                        assignment_type=image_allocation.assignment_type,
                    )

        downloaded = len(accepted_keys)
        direct_downloaded = len(direct_accepted_keys)
        top_up_downloaded = len(top_up_accepted_keys)
        source_report = {
            "class_name": allocation.class_name,
            "dataset_name": allocation.dataset_name,
            "source_role": allocation.source_role,
            "assigned_split": allocation.split,
            "assignment_type": allocation.assignment_type,
            "requested": allocation.count,
            "candidates": len(seen_candidates),
            "downloaded": downloaded,
            "direct_allocation_downloaded": direct_downloaded,
            "global_top_up_downloaded": top_up_downloaded,
            "shortfall": max(0, allocation.count - direct_downloaded),
            "cross_split_conflicts": conflicts,
            "content_duplicate_conflicts": content_duplicate_conflicts,
            "content_duplicates": content_duplicates,
            "failures": failures,
        }
        report["sources"].append(source_report)

    if master_coco_data["images"]:
        json_labels_path(job_id).write_text(
            json.dumps(master_coco_data, indent=2),
            encoding="utf-8",
        )
    coverage = summarize_detection_manifest_coverage(
        manifest.as_dict(), coverage_requirements
    )
    report["coverage"] = coverage
    allocation_shortfalls = [
        {
            "class_name": item["class_name"],
            "dataset_name": item["dataset_name"],
            "assigned_split": item["assigned_split"],
            "requested": item["requested"],
            "direct_allocation_downloaded": item["direct_allocation_downloaded"],
            "shortfall": item["shortfall"],
        }
        for item in report["sources"]
        if item["shortfall"]
    ]
    report["allocation_shortfalls"] = allocation_shortfalls
    unresolved_transfer_failures = [
        failure
        for source in report["sources"]
        for failure in source["failures"]
        if failure.get("reason") != "duplicate_content"
    ]
    cross_split_duplicates = _manifest_cross_split_duplicates(manifest.as_dict())
    acceptance = evaluate_detection_coverage_acceptance(
        coverage,
        unresolved_transfer_failures=unresolved_transfer_failures,
        cross_split_duplicates=cross_split_duplicates,
    )
    report["acceptance"] = acceptance
    warnings = (
        [{
            "type": "allocation_substitution",
            "message": (
                "One or more direct class/source searches were short, but the final "
                "multi-label manifest satisfied all unique-image and class-coverage requirements."
            ),
            "allocation_shortfalls": allocation_shortfalls,
        }]
        if allocation_shortfalls and coverage["satisfied"] else []
    )
    if acceptance["aspirational_unique_target_accepted"]:
        warnings.append({
            "type": "aspirational_unique_image_shortfall",
            "message": (
                "The selected multi-label candidate pools were exhausted after top-up. "
                "All per-class split requirements are satisfied, so the distinct "
                "image pool was accepted below its aspirational target."
            ),
            "target_unique_images": coverage_requirements["target_unique_images"],
            "verified_unique_images": coverage["verified_unique_images"],
            "unique_coverage_ratio": acceptance["unique_coverage_ratio"],
            "minimum_unique_coverage_ratio": MINIMUM_DETECTION_UNIQUE_COVERAGE_RATIO,
            "planned_unique_images_by_split": coverage_requirements[
                "target_unique_images_by_split"
            ],
            "adjusted_split_sizes": coverage["verified_unique_images_by_split"],
        })
    replaced_transfer_failures = acceptance.get(
        "successfully_replaced_transfer_failures"
    ) or []
    if replaced_transfer_failures:
        warnings.append({
            "type": "transfer_failures_replaced",
            "message": (
                "One or more image transfers failed, but replacement candidates "
                "satisfied every mandatory class and split objective."
            ),
            "count": len(replaced_transfer_failures),
            "failures": replaced_transfer_failures,
        })
    report["warnings"] = warnings
    # Detection allocations are retrieval tactics. The verified multi-label
    # manifest is authoritative. The allocation sum remains an aspirational
    # pool-size target once top-up has exhausted the selected candidate pools.
    report["complete"] = acceptance["accepted"]
    report["requested"] = sum(item["requested"] for item in report["sources"])
    report["downloaded"] = sum(item["downloaded"] for item in report["sources"])
    report["unique_downloaded"] = len(manifest.as_dict()["samples"])
    performance["elapsed_seconds"] = time.perf_counter() - download_started
    report["performance"] = performance
    _persist_download_artifacts(job_id, manifest, report)
    return report




def download_visionkg_mixed_datasets_classification(
    job_id: str, requests: list, *, progress_callback=None, cancel_check=None
):
    """Globally allocate unique classification images, then download them."""
    if not isinstance(requests, list):
        raise TypeError(f"Input 'requests' must be a list. Got {type(requests)}.")
    if not requests:
        return {"job_id": job_id, "sources": [], "complete": True, "requested": 0, "downloaded": 0}

    allocations = list(iter_download_allocations(requests))
    manifest = DatasetManifest(job_id, "classification", assignment_fingerprint(requests))
    report = {"job_id": job_id, "task": "classification", "sources": [], "complete": True}
    performance = _empty_performance()
    download_started = time.perf_counter()
    # VisionKG can expose the same photograph under different record IDs and
    # file names. Path-level allocation therefore is not sufficient to keep
    # train, validation, and test content isolated.
    content_reservations: dict[str, list[tuple[str, str, str]]] = {}

    def fetch_candidate_page(
        pool: _ClassificationCandidatePool,
        limit: int,
    ) -> list[dict]:
        """Append one deterministic page and return previously unseen candidates."""

        if pool.exhausted:
            return []
        raw_result = _timed_query(
            _classification_candidate_query(
                pool.allocation,
                limit,
                pool.next_offset,
            ),
            performance,
            cancel_check,
        )
        pool.next_offset += limit
        rows = raw_result or []
        if len(rows) < limit:
            pool.exhausted = True
        seen_paths = {candidate["image_path"] for candidate in pool.candidates}
        added = []
        for row in rows:
            image_name = row.get("imageName")
            returned_dataset = row.get("datasetName")
            if not image_name or not returned_dataset:
                continue
            rel_path = f"{returned_dataset}/{image_name}"
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            candidate = {
                "file_name": image_name,
                "image_path": rel_path,
                "sample_id": str(row.get("image") or rel_path),
                "url": (
                    "https://vision-api.semkg.org/api/image?image="
                    f"/{returned_dataset}/{image_name}"
                ),
            }
            pool.candidates.append(candidate)
            added.append(candidate)
        return added

    pools: list[_ClassificationCandidatePool] = []
    for allocation in allocations:
        scan_limit = _candidate_scan_limit(allocation.count, 0)
        pool = _ClassificationCandidatePool(allocation, [], [])
        fetch_candidate_page(pool, scan_limit)
        pools.append(pool)

    _assign_unique_classification_candidates(pools)
    reserved_paths = {
        image["image_path"]
        for pool in pools
        for image in pool.assigned
    }
    master_csv_rows = []
    for pool in pools:
        allocation = pool.allocation
        failures = []
        successes = []
        content_duplicates: list[dict[str, str]] = []
        pending = list(pool.assigned)
        spare_candidates = [
            candidate
            for candidate in pool.candidates
            if candidate["image_path"] not in reserved_paths
        ]
        while pending:
            result = _prepare_with_progress(pending, job_id, progress_callback, cancel_check)
            _merge_download_metrics(performance, result)
            failures.extend(result["failures"])
            rejected_paths = {
                failed["image_path"]
                for failed in result["failures"]
            }
            for image in result["successful"]:
                rel_path = image["image_path"]
                digest = file_sha256(data_dir(job_id) / rel_path)
                previous = next(
                    (
                        reservation
                        for reservation in content_reservations.get(digest, [])
                        if files_are_byte_identical(
                            data_dir(job_id) / reservation[1],
                            data_dir(job_id) / rel_path,
                        )
                    ),
                    None,
                )
                if previous is not None:
                    previous_split, previous_path, previous_class = previous
                    duplicate = {
                        "image_path": rel_path,
                        "assigned_split": allocation.split,
                        "class_name": allocation.class_name,
                        "duplicate_of": previous_path,
                        "duplicate_split": previous_split,
                        "duplicate_class_name": previous_class,
                        "sha256": digest,
                    }
                    content_duplicates.append(duplicate)
                    failures.append({**duplicate, "reason": "duplicate_content"})
                    rejected_paths.add(rel_path)
                    continue
                content_reservations.setdefault(digest, []).append(
                    (
                        allocation.split,
                        rel_path,
                        allocation.class_name,
                    )
                )
                successes.append(image)

            rejected_count = len(rejected_paths)
            if not rejected_count:
                break
            for rejected_path in rejected_paths:
                reserved_paths.discard(rejected_path)
            while len(spare_candidates) < rejected_count and not pool.exhausted:
                page_limit = max(50, (rejected_count - len(spare_candidates)) * 3)
                added = fetch_candidate_page(pool, page_limit)
                spare_candidates.extend(
                    candidate
                    for candidate in added
                    if candidate["image_path"] not in reserved_paths
                )
            replacements = spare_candidates[:rejected_count]
            spare_candidates = spare_candidates[rejected_count:]
            reserved_paths.update(candidate["image_path"] for candidate in replacements)
            pending = replacements

        for image in successes:
            rel_path = image["image_path"]
            master_csv_rows.append({
                "image_filename": rel_path,
                "labels": allocation.class_name,
            })
            manifest.add(
                sample_id=image["sample_id"],
                image_path=rel_path,
                class_name=allocation.class_name,
                dataset_name=allocation.dataset_name,
                source_role=allocation.source_role,
                assigned_split=allocation.split,
                assignment_type=allocation.assignment_type,
            )

        successful_paths = {image["image_path"] for image in successes}
        allocation_conflicts = sum(
            candidate["image_path"] in reserved_paths
            and candidate["image_path"] not in successful_paths
            for candidate in pool.candidates
        )
        source_report = {
            "class_name": allocation.class_name,
            "dataset_name": allocation.dataset_name,
            "source_role": allocation.source_role,
            "assigned_split": allocation.split,
            "assignment_type": allocation.assignment_type,
            "requested": allocation.count,
            "candidates": len(pool.candidates),
            "downloaded": len(successes),
            "shortfall": allocation.count - len(successes),
            "allocation_conflicts": allocation_conflicts,
            "cross_split_conflicts": allocation_conflicts,
            "content_duplicate_conflicts": len(content_duplicates),
            "content_duplicates": content_duplicates,
            "failures": failures,
        }
        report["sources"].append(source_report)
        if source_report["shortfall"]:
            report["complete"] = False

    if master_csv_rows:
        csv_path = csv_labels_path(job_id)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_filename', 'labels'])
            writer.writeheader()
            writer.writerows(master_csv_rows)

    report["requested"] = sum(item["requested"] for item in report["sources"])
    report["downloaded"] = sum(item["downloaded"] for item in report["sources"])
    report["unique_downloaded"] = len(manifest.as_dict()["samples"])
    performance["elapsed_seconds"] = time.perf_counter() - download_started
    report["performance"] = performance
    _persist_download_artifacts(job_id, manifest, report)
    return report


def download_visionkg_images_flat(job_id: str, requests: list, *, download: bool = True):
    """
    Sequentially queries VisionKG for images and downloads them into a single 
    directory with flattened filenames. No annotations are processed.
    """
    if not isinstance(requests, list):
        raise TypeError(f"Input 'requests' must be a list. Got {type(requests)}.")

    if not requests:
        print("Warning: 'requests' list is empty. Nothing to download.")
        return

    # Track unique images to prevent duplicate downloads
    global_image_set = set()
    images_to_download = []

    for entry in requests:
        if not isinstance(entry, dict):
            continue

        class_name = entry.get("class_name")
        sources = entry.get("sources")

        if not class_name or not sources or not isinstance(sources, list):
            continue

        for source in sources:
            if not isinstance(source, dict):
                continue

            dataset_name = source.get("dataset_name")
            limit = source.get("count", source.get("image_count"))

            if not dataset_name or not isinstance(limit, int):
                continue

            print(f"\n--- Fetching Images: Class '{class_name}' from Dataset '{dataset_name}' (Limit: {limit}) ---")

            # Highly simplified query: We only need the image and dataset names
            query_string = f"""
            PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
            PREFIX schema:<http://schema.org/>
            
            SELECT DISTINCT ?datasetName ?imageName
            WHERE {{
                ?image schema:isPartOf / schema:name ?datasetName .
                FILTER regex(?datasetName, "{dataset_name}", "i")
                
                ?image cv:hasAnnotation ?ann .
                ?ann cv:hasLabel/cv:label ?labelName .
                FILTER regex(?labelName, "{class_name}", "i")
                
                OPTIONAL {{ ?image schema:name ?imageName }} .
            }}
            LIMIT {limit}
            """

            print("  Querying VisionKG...")
            raw_result = query(query_string)
            
            if not raw_result:
                print(f"  No results found for {class_name} in {dataset_name}.")
                continue

            for row in raw_result:
                dataset_n = row.get('datasetName')
                img_name = row.get('imageName')
                
                if not dataset_n or not img_name:
                    continue
                    
                # Creating a clean relative path (standardizing on forward slash for the URL)
                rel_image_path = f"{dataset_n}/{img_name}"
                
                if rel_image_path not in global_image_set:
                    global_image_set.add(rel_image_path)
                    
                    image_url = f"https://vision-api.semkg.org/api/image?image=/{dataset_n}/{img_name}"
                    images_to_download.append({
                        'url': image_url,
                        'image_path': rel_image_path,
                    })

    if images_to_download:
        print(f"\nResolved {len(images_to_download)} unique image URLs:")
        for image in images_to_download:
            print(image["url"])

        if download:
            print(f"\nStarting download for {len(images_to_download)} unique images...")
            prepare_data_flat(images_to_download, DATA_ROOT_PATH=str(data_dir(job_id)))
        else:
            print("\nDownload disabled; no image files were written.")
    else:
        print("\nNo new images to download.")

    return images_to_download
