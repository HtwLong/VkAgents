import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import threading
import uuid
import time
from typing import Dict, List
import urllib.request
from pprint import pprint
import requests
from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger(__name__)
DEFAULT_DOWNLOAD_WORKERS = 12
DEFAULT_DOWNLOAD_ATTEMPTS = 5
DEFAULT_DOWNLOAD_REQUESTS_PER_MINUTE = 110
DEFAULT_SPARQL_TIMEOUT = 90
DEFAULT_SPARQL_ATTEMPTS = 3
DOWNLOAD_WORKERS_ENV = "VISIONKG_DOWNLOAD_WORKERS"
DOWNLOAD_REQUESTS_PER_MINUTE_ENV = "VISIONKG_DOWNLOAD_REQUESTS_PER_MINUTE"
SPARQL_TIMEOUT_ENV = "VISIONKG_SPARQL_TIMEOUT"
_download_thread = threading.local()
_executor_lock = threading.Lock()
_shared_executor = None
_shared_executor_workers = None
_rate_limit_lock = threading.Lock()
_rate_limit_until = 0.0
_request_pacing_lock = threading.Lock()
_next_download_request_at = 0.0
_view_fallback_lock = threading.Lock()
_prefer_view_endpoint = False


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("Ignoring invalid %s=%r; using %d.", name, raw, default)
        return default
    if value <= 0:
        LOGGER.warning("Ignoring non-positive %s=%r; using %d.", name, raw, default)
        return default
    return value


def download_worker_count() -> int:
    return _positive_env_int(DOWNLOAD_WORKERS_ENV, DEFAULT_DOWNLOAD_WORKERS)


def download_requests_per_minute() -> int:
    return _positive_env_int(
        DOWNLOAD_REQUESTS_PER_MINUTE_ENV,
        DEFAULT_DOWNLOAD_REQUESTS_PER_MINUTE,
    )


def _download_executor():
    """Reuse the bounded worker pool across pages and allocations."""
    global _shared_executor, _shared_executor_workers
    workers = download_worker_count()
    with _executor_lock:
        if _shared_executor is None or _shared_executor_workers != workers:
            old_executor = _shared_executor
            _shared_executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="visionkg-download",
            )
            _shared_executor_workers = workers
            if old_executor is not None:
                old_executor.shutdown(wait=False)
        return _shared_executor


def _download_session() -> requests.Session:
    """Return one reusable HTTP session per download worker."""
    if not hasattr(_download_thread, "session"):
        session = requests.Session()
        workers = download_worker_count()
        adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _download_thread.session = session
    return _download_thread.session


def _interruptible_wait(seconds: float, cancel_check=None) -> None:
    deadline = time.monotonic() + seconds
    while True:
        if cancel_check is not None:
            cancel_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def _wait_for_rate_limit(cancel_check=None) -> None:
    """Pause every worker while the upstream service is rate-limiting us."""
    while True:
        with _rate_limit_lock:
            remaining = _rate_limit_until - time.monotonic()
        if remaining <= 0:
            return
        _interruptible_wait(remaining, cancel_check)


def _extend_rate_limit(delay: float) -> None:
    global _rate_limit_until
    with _rate_limit_lock:
        _rate_limit_until = max(_rate_limit_until, time.monotonic() + delay)


def _wait_for_download_request_slot(cancel_check=None) -> None:
    """Reserve one globally paced request slot across all download workers."""
    global _next_download_request_at
    interval = 60.0 / download_requests_per_minute()
    with _request_pacing_lock:
        now = time.monotonic()
        request_at = max(now, _next_download_request_at)
        _next_download_request_at = request_at + interval
    _interruptible_wait(max(0.0, request_at - time.monotonic()), cancel_check)


def _response_is_rate_limited(response) -> bool:
    if response is None:
        return False
    return response.status_code == 429 or response.headers.get("X-RateLimit-Remaining") == "0"


def _visionkg_view_fallback_url(url: str) -> str | None:
    raw_prefix = "https://vision-api.semkg.org/api/image?"
    if not url.startswith(raw_prefix):
        return None
    return url.replace(raw_prefix, "https://vision-api.semkg.org/api/view?", 1)


def _preferred_visionkg_url(url: str) -> str:
    with _view_fallback_lock:
        prefer_view = _prefer_view_endpoint
    return (_visionkg_view_fallback_url(url) or url) if prefer_view else url


def _prefer_visionkg_view_endpoint() -> None:
    global _prefer_view_endpoint
    with _view_fallback_lock:
        _prefer_view_endpoint = True


def query(query_string, token="", *, cancel_check=None):
    """
    Executes a SPARQL query against the VisionKG endpoint.
    """

    # endpoint_url = "http://brain.ods.tu-berlin.de:11132/sparql"
    endpoint_url = 'https://vision.semkg.org/sparql'
    last_error = None
    for attempt in range(1, DEFAULT_SPARQL_ATTEMPTS + 1):
        if cancel_check is not None:
            cancel_check()
        try:
            if endpoint_url == 'http://brain.ods.tu-berlin.de:11132/sparql':
                response = _download_session().get(
                    endpoint_url,
                    params={
                        "query": query_string,
                        "format": "application/sparql-results+json",
                    },
                    headers={
                        "Accept": "application/sparql-results+json",
                    },
                    timeout=_positive_env_int(SPARQL_TIMEOUT_ENV, DEFAULT_SPARQL_TIMEOUT),
                )
            else:
                response = _download_session().get(
                    endpoint_url,
                    params={"query": query_string, "token": token},
                    timeout=_positive_env_int(SPARQL_TIMEOUT_ENV, DEFAULT_SPARQL_TIMEOUT),
                )
            try:
                response.raise_for_status()
                payload = response.json()
            finally:
                response.close()
            if cancel_check is not None:
                cancel_check()

            bindings = payload.get("results", {}).get("bindings")
            if not isinstance(bindings, list):
                raise ValueError("VisionKG returned an unexpected SPARQL response")
            return [
                {key: value["value"] for key, value in result.items()}
                for result in bindings
            ]
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt < DEFAULT_SPARQL_ATTEMPTS:
                response = getattr(exc, "response", None)
                retry_after = response.headers.get("Retry-After") if response is not None else None
                try:
                    delay = float(retry_after) if retry_after else 2 ** (attempt - 1)
                except ValueError:
                    delay = 2 ** (attempt - 1)
                _interruptible_wait(min(delay, 30.0), cancel_check)

    raise RuntimeError(
        f"VisionKG SPARQL query failed after {DEFAULT_SPARQL_ATTEMPTS} attempts: {last_error}"
    ) from last_error

def _validated_existing_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        path.unlink(missing_ok=True)
        return False


def _download_one_image(image, root: Path, max_attempts: int, timeout: int, cancel_check=None):
    """Resolve one image and return (successful, result metadata)."""
    from cvmodellearning.download.image_cache import image_path_below, safe_relative_image_path

    relative_path = safe_relative_image_path(image["image_path"])
    destination = image_path_below(root, str(relative_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = image.get("url")
    failure = {**image, "error": "No download URL", "status_code": None}
    if not url:
        return False, failure
    url = _preferred_visionkg_url(url)

    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    started = time.perf_counter()
    retries = 0
    for attempt in range(1, max_attempts + 1):
        if cancel_check is not None:
            cancel_check()
        partial.unlink(missing_ok=True)
        try:
            _wait_for_rate_limit(cancel_check)
            _wait_for_download_request_slot(cancel_check)
            with _download_session().get(url, stream=True, timeout=timeout) as response:
                status_code = response.status_code
                content_type = response.headers.get("Content-Type", "").lower()
                if status_code != 200:
                    fallback_url = _visionkg_view_fallback_url(url)
                    if (
                        status_code == 500
                        and not _response_is_rate_limited(response)
                        and fallback_url is not None
                    ):
                        LOGGER.info(
                            "VisionKG raw-image endpoint failed for %s; retrying via %s",
                            relative_path,
                            fallback_url,
                        )
                        _prefer_visionkg_view_endpoint()
                        url = fallback_url
                    raise requests.HTTPError(f"HTTP {status_code}", response=response)
                if not content_type.startswith("image/"):
                    raise ValueError(f"Unexpected Content-Type {content_type!r}")
                with partial.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=128 * 1024):
                        if cancel_check is not None:
                            cancel_check()
                        if chunk:
                            output.write(chunk)
            with Image.open(partial) as downloaded:
                downloaded.verify()
            partial.replace(destination)
            LOGGER.debug("Downloaded and validated %s", relative_path)
            return True, {**image, "_download_metrics": {
                "bytes_downloaded": destination.stat().st_size,
                "retries": retries,
                "transfer_and_validation_seconds": time.perf_counter() - started,
            }}
        except (requests.RequestException, OSError, ValueError, UnidentifiedImageError) as exc:
            partial.unlink(missing_ok=True)
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            rate_limited = _response_is_rate_limited(response)
            failure = {
                **image,
                "error": str(exc),
                "status_code": status_code,
                "attempts": attempt,
                "rate_limited": rate_limited,
                "rate_limit": response.headers.get("X-RateLimit-Limit") if response is not None else None,
                "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining") if response is not None else None,
                "retry_after": response.headers.get("Retry-After") if response is not None else None,
                "effective_url": url,
            }
            retryable = status_code in {429, 500, 502, 503, 504} or status_code is None
            if attempt < max_attempts and retryable:
                retries += 1
                retry_after = response.headers.get("Retry-After") if response is not None else None
                try:
                    delay = float(retry_after) if retry_after else 2 ** (attempt - 1)
                except ValueError:
                    delay = 2 ** (attempt - 1)
                delay = min(delay, 30.0)
                if rate_limited:
                    _extend_rate_limit(delay)
                    _wait_for_rate_limit(cancel_check)
                else:
                    _interruptible_wait(delay, cancel_check)
                continue
            LOGGER.warning(
                "Failed to download %s: %s after %d attempts "
                "(url=%s, rate_limited=%s, limit=%s, remaining=%s, retry_after=%s)",
                relative_path,
                failure["error"],
                attempt,
                url,
                rate_limited,
                failure["rate_limit"],
                failure["rate_limit_remaining"],
                failure["retry_after"],
            )
            return False, {**failure, "_download_metrics": {
                "bytes_downloaded": 0,
                "retries": retries,
                "transfer_and_validation_seconds": time.perf_counter() - started,
            }}

    return False, failure


def prepare_data(
    images,
    DATA_ROOT_PATH=None,
    *,
    max_attempts=DEFAULT_DOWNLOAD_ATTEMPTS,
    timeout=30,
    max_workers=None,
    progress_callback=None,
    cancel_check=None,
    executor=None,
):
    """Download and validate images concurrently, preserving input order."""
    if not DATA_ROOT_PATH:
        print("DATA path did not set! Path will set default at /tmp")
        DATA_ROOT_PATH = "/tmp"
    root = Path(DATA_ROOT_PATH)
    images = list(images)
    if max_workers is None:
        max_workers = download_worker_count()
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer.")

    cache_path = root / ".download_validation_cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        cache = {}

    cache_hits = 0
    ordered_results = [None] * len(images)
    pending = {}
    for index, image in enumerate(images):
        if cancel_check is not None:
            cancel_check()
        from cvmodellearning.download.image_cache import image_path_below, safe_relative_image_path

        relative_path = safe_relative_image_path(image["image_path"])
        destination = image_path_below(root, str(relative_path))
        cached = cache.get(str(relative_path))
        try:
            stat = destination.stat()
        except OSError:
            stat = None
        if stat is not None and cached == {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            ordered_results[index] = (True, image)
            cache_hits += 1
            if progress_callback:
                progress_callback(successful=True, image_path=str(Path(image["image_path"])))
        elif stat is not None:
            validation_started = time.perf_counter()
            if _validated_existing_image(destination):
                stat = destination.stat()
                cache[str(Path(image["image_path"]))] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
                ordered_results[index] = (True, image)
                cache_hits += 1
                if progress_callback:
                    progress_callback(successful=True, image_path=str(Path(image["image_path"])))
                continue
            LOGGER.debug(
                "Existing image validation failed after %.4fs: %s",
                time.perf_counter() - validation_started,
                destination,
            )
            pending[index] = image
        else:
            pending[index] = image

    # An explicit worker count is used by tests/callers that need an isolated
    # pool. Normal jobs share one service-level pool across all query pages.
    owns_executor = executor is None and max_workers != download_worker_count()
    active_executor = executor or (
        ThreadPoolExecutor(max_workers=max_workers) if owns_executor else _download_executor()
    )
    try:
        futures = {
            active_executor.submit(
                _download_one_image, image, root, max_attempts, timeout
            ) if cancel_check is None else active_executor.submit(
                _download_one_image, image, root, max_attempts, timeout, cancel_check
            ): index
            for index, image in pending.items()
        }
        for future in as_completed(futures):
            index = futures[future]
            successful, result = future.result()
            ordered_results[index] = (successful, result)
            if progress_callback:
                progress_callback(
                    successful=successful,
                    image_path=str(Path(images[index]["image_path"])),
                )
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        if owns_executor:
            active_executor.shutdown(wait=True)

    metrics = {
        "requested": len(images),
        "cache_hits": cache_hits,
        "bytes_downloaded": 0,
        "retries": 0,
        "transfer_and_validation_seconds": 0.0,
    }
    clean_results = []
    for successful, result in ordered_results:
        item = dict(result)
        item_metrics = item.pop("_download_metrics", {})
        metrics["bytes_downloaded"] += int(item_metrics.get("bytes_downloaded", 0))
        metrics["retries"] += int(item_metrics.get("retries", 0))
        metrics["transfer_and_validation_seconds"] += float(
            item_metrics.get("transfer_and_validation_seconds", 0.0)
        )
        clean_results.append((successful, item))
        if successful:
            destination = image_path_below(root, item["image_path"])
            try:
                stat = destination.stat()
            except OSError:
                continue
            cache[str(Path(item["image_path"]))] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    if clean_results:
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
        temporary.replace(cache_path)

    return {
        "successful": [result for successful, result in clean_results if successful],
        "failures": [result for successful, result in clean_results if not successful],
        "metrics": metrics,
    }



def get_multi_class_stats(classes: list, query_output_path=None) -> dict:
    """
    Retrieves dataset statistics for multiple classes using a single 
    SPARQL query optimized with the VALUES clause.
    
    Args:
        classes (list): A list of strings, e.g., ["cat", "dog", "bird"].
        query_output_path: Optional path where the fully rendered SPARQL query
            should be saved before it is executed.

    Returns:
        dict: A nested dictionary where keys are class names and values are 
              dictionaries of dataset counts.
              Example: {'cat': {'coco': 300}, 'dog': {'voc': 100}}
    """
    
    if not classes:
        return {}

    # Initialize the output structure with empty dicts for all requested classes
    all_stats = {cls: {} for cls in classes}

    # Format the classes into a SPARQL VALUES string: "cat" "dog" "bird"
    values_string = " ".join([f'"{cls}"' for cls in classes])

    # 1. Construct the Main Query
    query_string = f"""
    PREFIX cv: <http://vision.semkg.org/onto/v0.1/>
    PREFIX schema: <http://schema.org/>

    SELECT ?targetLabel ?datasetName (COUNT(DISTINCT ?image) AS ?count)
    WHERE {{
        # Inject the list of classes directly into the query engine's execution plan
        VALUES ?targetLabel {{ {values_string} }} 
        
        ?image cv:hasAnnotation ?annotation .
        ?annotation cv:hasLabel ?lbl .
        ?lbl cv:label ?targetLabel .
        ?image schema:isPartOf / schema:name ?datasetName .
    }}
    GROUP BY ?targetLabel ?datasetName
    ORDER BY ?targetLabel DESC(?count)
    """

    if query_output_path is not None:
        query_path = Path(query_output_path)
        query_path.parent.mkdir(parents=True, exist_ok=True)
        query_path.write_text(query_string.strip() + "\n", encoding="utf-8")
    
    print(f"Querying VisionKG for {len(classes)} classes using VALUES...")

    # 2. Execute the Query
    raw_result = query(query_string)

    # 3. Parse Results into Nested Dictionary
    bindings = []
    if isinstance(raw_result, dict) and 'results' in raw_result:
        bindings = raw_result['results']['bindings']
    elif isinstance(raw_result, list):
        bindings = raw_result

    for row in bindings:
        # Helper to extract 'value' if it's a dict, or use the item directly
        def get_val(item):
            return item.get('value') if isinstance(item, dict) else item

        label = get_val(row.get('targetLabel'))
        d_name = get_val(row.get('datasetName'))
        count_val = get_val(row.get('count'))

        if label and d_name and count_val:
            try:
                # Map the results back to the initialized dictionary
                if label in all_stats:
                    all_stats[label][d_name] = int(count_val)
            except ValueError:
                pass

    return all_stats


def get_datasets():
    from cvmodellearning.datasets.registry import DATASET_REGISTRY

    return list(DATASET_REGISTRY)


def visionkg2cocoDet(query_bindings: List[Dict], 
                     global_image_map: Dict = None, 
                     global_category_map: Dict = None,
                     global_anno_id_counter: List[int] = None) -> Dict:
    """
    Converts VisionKG SPARQL bindings to COCO format using global registries
    to maintain consistent IDs across multiple batches.
    """
    
    # Initialize globals if not provided (safe default for single-run use)
    if global_image_map is None:
        global_image_map = {}
    if global_category_map is None:
        global_category_map = {}
    # We use a list for the counter so it can be mutable (passed by reference)
    if global_anno_id_counter is None:
        global_anno_id_counter = [0]

    coco_annotations = []
    coco_images_info = []
    
    # We track images added *in this specific batch* to avoid adding the same image info 
    # twice to the output list, even if it already exists in the global map.
    batch_processed_images = set()

    for anno in query_bindings:
        
        image_name = anno['imageName']
        dataset_name = anno['datasetName']
        label_name = anno['labelName']
        image_key = f"{dataset_name}/{image_name}"
        
        # --- 1. Handle Image ID ---
        if image_key not in global_image_map:
            # Create new ID
            global_image_map[image_key] = len(global_image_map) + 1
            
            # Create Image Info
            image_height = int(anno['imageHeight'])
            image_width = int(anno['imageWidth'])
            image_url = f"https://vision-api.semkg.org/api/image?image=/{dataset_name}/{image_name}"
            
            image_info = {
                'id': global_image_map[image_key],
                'file_name': image_name,
                'dataset': dataset_name,
                'height': image_height,
                'width': image_width,
                'url': image_url,
                'image_path': os.path.join(dataset_name, image_name),
            }
            coco_images_info.append(image_info)
            batch_processed_images.add(image_key)
        
        # If image exists in global map but hasn't been added to this batch's output list yet
        # (OPTIONAL: Depending on if you want the output to contain ALL images or just NEW ones.
        # usually for merging, you only want to append new image dicts).
        
        # --- 2. Handle Category ID ---
        if label_name not in global_category_map:
            global_category_map[label_name] = len(global_category_map) + 1

        # --- 3. Handle Bounding Box ---
        box_center_x = float(anno['bbCentreX'])
        box_center_y = float(anno['bbCentreY'])
        box_height = float(anno['bbHeight'])
        box_width = float(anno['bbWidth'])
        
        image_w = int(anno['imageWidth'])
        image_h = int(anno['imageHeight'])

        # VisionKG stores box centers, while COCO requires the top-left corner.
        # Clamp the converted box to the image so downstream YOLO normalization
        # cannot produce coordinates outside [0, 1].
        x_min = max(0.0, box_center_x - box_width / 2)
        y_min = max(0.0, box_center_y - box_height / 2)
        x_max = min(float(image_w), box_center_x + box_width / 2)
        y_max = min(float(image_h), box_center_y + box_height / 2)
        clipped_width = x_max - x_min
        clipped_height = y_max - y_min
        if clipped_width <= 0 or clipped_height <= 0:
            continue

        global_anno_id_counter[0] += 1
        coco_annotation = {
            'id': global_anno_id_counter[0],
            'image_id': global_image_map[image_key],
            'bbox': [round(x_min, 2), round(y_min, 2),
                     round(clipped_width, 2), round(clipped_height, 2)],
            'category_id': global_category_map[label_name],
            'iscrowd': 0,
            'area': round(clipped_height * clipped_width),
        }
        coco_annotations.append(coco_annotation)

    # Convert global categories map to COCO list format
    # We return the FULL category list every time so the latest update has everything
    coco_categories = [{'id': v, 'name': k, 'supercategory': None} 
                       for k, v in global_category_map.items()]

    return {
        'images': coco_images_info,
        'annotations': coco_annotations,
        'categories': coco_categories
    }

def visionkg_parse_classification(query_bindings: List[Dict], global_image_set: set = None) -> Dict:
    """
    Parses VisionKG SPARQL bindings into flat rows for a classification CSV,
    and returns a list of required images to download.
    """
    if global_image_set is None:
        global_image_set = set()

    images_to_download = []
    csv_rows = []

    for anno in query_bindings:
        image_name = anno['imageName']
        dataset_name = anno['datasetName']
        label_name = anno['labelName']

        # Create the relative path expected by your CocoImageDataset loader
        rel_image_path = os.path.join(dataset_name, image_name)

        # Track unique images so we don't download duplicates across batches
        if rel_image_path not in global_image_set:
            global_image_set.add(rel_image_path)
            
            image_url = f"https://vision-api.semkg.org/api/image?image=/{dataset_name}/{image_name}"
            images_to_download.append({
                'file_name': image_name,
                'url': image_url,
                'image_path': rel_image_path,
            })

        # Append flat row for the CSV
        csv_rows.append({
            'image_filename': rel_image_path,
            'labels': label_name
        })

    return {
        'images_to_download': images_to_download,
        'csv_rows': csv_rows
    }

def prepare_data_flat(images: list, DATA_ROOT_PATH: str = None):
    """
    Downloads images directly into the root directory without creating subfolders.
    The filename is created by flattening the relative image path (replacing '/' with '_').
    """
    if not DATA_ROOT_PATH:
        print("DATA path did not set! Path will set default at /tmp")
        DATA_ROOT_PATH = "/tmp"
        
    if not os.path.isdir(DATA_ROOT_PATH):
        os.makedirs(DATA_ROOT_PATH, exist_ok=True)

    missing_images = []
    
    for image in images:
        # Flatten the path (e.g., "coco2017_det_train/000000102096.jpg" -> "coco2017_det_train_000000102096.jpg")
        # We replace both standard slashes and backslashes just to be safe across OS environments
        flat_filename = image['image_path'].replace('/', '_').replace('\\', '_')
        full_file_path = os.path.join(DATA_ROOT_PATH, flat_filename)
        
        if not os.path.exists(full_file_path):
            is_success = False
            if image.get('url'):
                print(f"Downloading {flat_filename}...")
                try:
                    urllib.request.urlretrieve(image['url'], full_file_path)
                    is_success = True
                except Exception as e:
                    print(f"Download failed for {flat_filename}: {e}")
            
            if not is_success:
                missing_images.append(flat_filename)
                
    if missing_images:
        print(f"\nThe following images could not be downloaded to {DATA_ROOT_PATH}:")
        print(", ".join(missing_images))
    else:
        print("\nAll downloads finished successfully!")
