import csv
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

from cvmodellearning.download.assignment_manifest import (
    DatasetManifest,
    assignment_fingerprint,
    iter_download_allocations,
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


def _sparql_string(value: str) -> str:
    """Escape a user/config value for use inside a SPARQL string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _persist_download_artifacts(job_id: str, manifest: DatasetManifest, report: dict) -> None:
    manifest_path = dataset_manifest_path(job_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    report_path = download_report_path(job_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["manifest_path"] = str(manifest_path)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


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
        "bytes_downloaded": 0,
        "retries": 0,
        "transfer_and_validation_seconds": 0.0,
    }


def _merge_download_metrics(performance: dict, result: dict) -> None:
    performance["download_batches"] += 1
    metrics = result.get("metrics") or {}
    for key in ("cache_hits", "bytes_downloaded", "retries", "transfer_and_validation_seconds"):
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


def _classification_candidate_query(allocation, limit: int) -> str:
    escaped_class = _sparql_string(allocation.class_name)
    escaped_dataset = _sparql_string(allocation.dataset_name)
    return f"""
    PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
    PREFIX schema:<http://schema.org/>

    SELECT DISTINCT ?datasetName ?imageName ?image
    WHERE {{
        ?image schema:isPartOf / schema:name ?datasetName .
        FILTER(LCASE(STR(?datasetName)) = LCASE("{escaped_dataset}"))
        ?image schema:name ?imageName .
        ?image cv:hasAnnotation ?annotation .
        ?annotation cv:hasLabel/cv:label ?labelName .
        FILTER(LCASE(STR(?labelName)) = LCASE("{escaped_class}"))
    }}
    ORDER BY STR(?image)
    LIMIT {limit}
    OFFSET 0
    """

def download_visionkg_mixed_datasets_detection(
    job_id: str, requests: list, *, progress_callback=None, cancel_check=None
):
    """Download detection images without allowing an image to cross split boundaries."""

    if not isinstance(requests, list):
        raise TypeError(f"Input 'requests' must be a list. Got {type(requests)}.")
    if not requests:
        return {"job_id": job_id, "sources": [], "complete": True, "requested": 0, "downloaded": 0}

    requested_classes = sorted({
        allocation.class_name
        for allocation in iter_download_allocations(requests)
    })
    requested_label_filter = ", ".join(
        f'LCASE("{_sparql_string(class_name)}")'
        for class_name in requested_classes
    )

    global_image_map: dict[str, int] = {}
    global_category_map: dict[str, int] = {}
    global_anno_id_counter = [0]
    annotation_signatures: set[tuple] = set()
    reservations: dict[str, str] = {}
    manifest = DatasetManifest(job_id, "detection", assignment_fingerprint(requests))
    report = {"job_id": job_id, "task": "detection", "sources": [], "complete": True}
    performance = _empty_performance()
    download_started = time.perf_counter()
    master_coco_data = {"images": [], "annotations": [], "categories": []}

    for allocation in iter_download_allocations(requests):
        escaped_class = _sparql_string(allocation.class_name)
        escaped_dataset = _sparql_string(allocation.dataset_name)
        accepted_keys: set[str] = set()
        seen_candidates: set[str] = set()
        failures: list[dict] = []
        conflicts = 0
        offset = 0
        dataset_prefix = f"{allocation.dataset_name}/"
        already_reserved = sum(key.startswith(dataset_prefix) for key in reservations)
        max_candidates = _candidate_scan_limit(allocation.count, already_reserved)

        while len(accepted_keys) < allocation.count and offset < max_candidates:
            remaining = allocation.count - len(accepted_keys)
            requested_page = remaining if offset == 0 else max(remaining, 25)
            page_limit = min(requested_page, max_candidates - offset)
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
                        ?image schema:isPartOf / schema:name ?datasetName .
                        FILTER(LCASE(STR(?datasetName)) = LCASE("{escaped_dataset}"))
                        
                        ?image cv:hasAnnotation ?ann .
                        ?ann cv:hasLabel/cv:label ?labelName .
                        FILTER(LCASE(STR(?labelName)) = LCASE("{escaped_class}"))
                    }}
                    ORDER BY STR(?image)
                    LIMIT {page_limit}
                    OFFSET {offset}
                }}

                # --- OUTER DATA FETCHING ---
                OPTIONAL {{ ?image schema:name ?imageName }} .
                OPTIONAL {{ 
                    ?image cv:imgWidth ?imageWidth .
                    ?image cv:imgHeight ?imageHeight .
                }}

                ?image cv:hasAnnotation ?annotation .
                ?annotation cv:hasLabel/cv:label ?labelName .
                
                # Keep every requested-class box on a selected image. Restricting
                # this to the allocation class would turn other target objects into
                # false background during detection training.
                FILTER(LCASE(STR(?labelName)) IN ({requested_label_filter}))
                
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
            offset += page_limit

            grouped: OrderedDict[str, list[dict]] = OrderedDict()
            for row in raw_result:
                image_name = row.get("imageName")
                dataset_name = row.get("datasetName")
                if not image_name or not dataset_name:
                    continue
                image_key = f"{dataset_name}/{image_name}"
                grouped.setdefault(image_key, []).append(row)

            chosen: list[str] = []
            for image_key in grouped:
                if image_key in seen_candidates:
                    continue
                seen_candidates.add(image_key)
                reserved_split = reservations.get(image_key)
                if reserved_split is not None and reserved_split != allocation.split:
                    conflicts += 1
                    continue
                chosen.append(image_key)
                if len(chosen) >= remaining:
                    break

            new_candidates = []
            for image_key in chosen:
                if image_key in reservations:
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
            successful_keys = [
                image_key
                for image_key in chosen
                if image_key in reservations or image_key in successful_new
            ]
            if progress_callback:
                for image_key in successful_keys:
                    if image_key in reservations:
                        progress_callback(successful=True, image_path=image_key)
            if not successful_keys and len(raw_result) < page_limit:
                break

            accepted_keys.update(successful_keys)
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
                image["assigned_split"] = allocation.split
                image["assignment_type"] = allocation.assignment_type
                image["source_role"] = allocation.source_role
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
                reservations.setdefault(image_key, allocation.split)
                row = grouped[image_key][0]
                sample_id = str(row.get("image") or image_key)
                manifest.add(
                    sample_id=sample_id,
                    image_path=image_key,
                    class_name=allocation.class_name,
                    dataset_name=allocation.dataset_name,
                    source_role=allocation.source_role,
                    assigned_split=allocation.split,
                    assignment_type=allocation.assignment_type,
                )

        downloaded = len(accepted_keys)
        source_report = {
            "class_name": allocation.class_name,
            "dataset_name": allocation.dataset_name,
            "source_role": allocation.source_role,
            "assigned_split": allocation.split,
            "assignment_type": allocation.assignment_type,
            "requested": allocation.count,
            "candidates": len(seen_candidates),
            "downloaded": downloaded,
            "shortfall": allocation.count - downloaded,
            "cross_split_conflicts": conflicts,
            "failures": failures,
        }
        report["sources"].append(source_report)
        if source_report["shortfall"]:
            report["complete"] = False

    if master_coco_data["images"]:
        json_labels_path(job_id).write_text(
            json.dumps(master_coco_data, indent=2),
            encoding="utf-8",
        )
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

    pools: list[_ClassificationCandidatePool] = []
    for allocation in allocations:
        scan_limit = _candidate_scan_limit(allocation.count, 0)
        raw_result = _timed_query(
            _classification_candidate_query(allocation, scan_limit),
            performance,
            cancel_check,
        )
        candidates = []
        seen_paths = set()
        for row in raw_result or []:
            image_name = row.get("imageName")
            returned_dataset = row.get("datasetName")
            if not image_name or not returned_dataset:
                continue
            rel_path = os.path.join(returned_dataset, image_name)
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            candidates.append({
                "file_name": image_name,
                "image_path": rel_path,
                "sample_id": str(row.get("image") or rel_path),
                "url": f"https://vision-api.semkg.org/api/image?image=/{returned_dataset}/{image_name}",
            })
        pools.append(_ClassificationCandidatePool(allocation, candidates, []))

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
        pending = list(pool.assigned)
        spare_candidates = [
            candidate
            for candidate in pool.candidates
            if candidate["image_path"] not in reserved_paths
        ]
        while pending:
            result = _prepare_with_progress(pending, job_id, progress_callback, cancel_check)
            _merge_download_metrics(performance, result)
            successes.extend(result["successful"])
            failures.extend(result["failures"])
            failed_count = len(result["failures"])
            if not failed_count:
                break
            for failed in result["failures"]:
                reserved_paths.discard(failed["image_path"])
            replacements = spare_candidates[:failed_count]
            spare_candidates = spare_candidates[failed_count:]
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
