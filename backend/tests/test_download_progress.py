import json
import inspect
import io
import threading
import time

import pytest
from PIL import Image

from cvmodellearning.download import progress
from cvmodellearning.download import visionkg_utils
from cvmodellearning.download.progress import DownloadProgressTracker
from cvmodellearning.download.visionkg_utils import prepare_data


def test_prepare_data_uses_resilient_download_attempt_default():
    default = inspect.signature(prepare_data).parameters["max_attempts"].default
    assert default == visionkg_utils.DEFAULT_DOWNLOAD_ATTEMPTS == 5


def test_download_progress_tracker_persists_each_result(monkeypatch, tmp_path):
    progress_path = tmp_path / "download_progress.json"
    monkeypatch.setattr(progress, "download_progress_path", lambda _job_id: progress_path)

    tracker = DownloadProgressTracker("job", total=3, persist_interval=0)
    tracker.record(successful=True, image_path="dataset/one.jpg")
    tracker.record(successful=False, image_path="dataset/two.jpg")

    snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
    assert snapshot["status"] == "running"
    assert snapshot["downloaded"] == 1
    assert snapshot["processed"] == 2
    assert snapshot["failed"] == 1
    assert snapshot["failed_datasets"] == []
    assert snapshot["total"] == 3
    assert snapshot["current_image"] == "dataset/two.jpg"

    tracker.record_failed_datasets(["dataset-b", "dataset-a", "dataset-b", ""])
    tracker.finish("completed")
    snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
    assert snapshot["status"] == "completed"
    assert snapshot["current_image"] is None
    assert snapshot["failed_datasets"] == ["dataset-b", "dataset-a"]


def test_download_progress_tracker_throttles_intermediate_writes(monkeypatch, tmp_path):
    progress_path = tmp_path / "download_progress.json"
    clock = iter([0.0, 0.1, 0.2, 0.3])
    monkeypatch.setattr(progress, "download_progress_path", lambda _job_id: progress_path)
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(clock))

    tracker = DownloadProgressTracker("job", total=2, persist_interval=0.5)
    initial = progress_path.read_text(encoding="utf-8")
    tracker.record(successful=True, image_path="dataset/one.jpg")
    tracker.record(successful=True, image_path="dataset/two.jpg")

    assert progress_path.read_text(encoding="utf-8") == initial
    tracker.finish("completed")
    snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
    assert snapshot["downloaded"] == 2
    assert snapshot["status"] == "completed"


def test_prepare_data_reports_reused_valid_image(tmp_path):
    image_path = tmp_path / "dataset" / "one.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (2, 2)).save(image_path)
    updates = []

    result = prepare_data(
        [{"image_path": "dataset/one.jpg", "url": "unused"}],
        DATA_ROOT_PATH=str(tmp_path),
        progress_callback=lambda **update: updates.append(update),
    )

    assert len(result["successful"]) == 1
    assert updates == [{"successful": True, "image_path": "dataset/one.jpg"}]
    assert result["metrics"]["cache_hits"] == 1


def test_prepare_data_uses_validation_metadata_on_second_run(monkeypatch, tmp_path):
    image_path = tmp_path / "dataset" / "one.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (2, 2)).save(image_path)

    first = prepare_data(
        [{"image_path": "dataset/one.jpg", "url": "unused"}],
        DATA_ROOT_PATH=str(tmp_path),
    )
    monkeypatch.setattr(
        visionkg_utils,
        "_validated_existing_image",
        lambda _path: (_ for _ in ()).throw(AssertionError("cache hit should skip decoding")),
    )
    second = prepare_data(
        [{"image_path": "dataset/one.jpg", "url": "unused"}],
        DATA_ROOT_PATH=str(tmp_path),
    )

    assert first["metrics"]["cache_hits"] == 1
    assert second["metrics"]["cache_hits"] == 1


def test_download_worker_count_uses_valid_environment(monkeypatch):
    monkeypatch.setenv("VISIONKG_DOWNLOAD_WORKERS", "12")
    assert visionkg_utils.download_worker_count() == 12

    monkeypatch.setenv("VISIONKG_DOWNLOAD_WORKERS", "invalid")
    assert visionkg_utils.download_worker_count() == visionkg_utils.DEFAULT_DOWNLOAD_WORKERS


def test_download_request_rate_uses_valid_environment(monkeypatch):
    monkeypatch.setenv("VISIONKG_DOWNLOAD_REQUESTS_PER_MINUTE", "40")
    assert visionkg_utils.download_requests_per_minute() == 40

    monkeypatch.setenv("VISIONKG_DOWNLOAD_REQUESTS_PER_MINUTE", "invalid")
    assert (
        visionkg_utils.download_requests_per_minute()
        == visionkg_utils.DEFAULT_DOWNLOAD_REQUESTS_PER_MINUTE
    )


def test_prepare_data_downloads_concurrently_and_preserves_order(monkeypatch, tmp_path):
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_download(image, _root, _max_attempts, _timeout):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return True, image

    monkeypatch.setattr(visionkg_utils, "_download_one_image", fake_download)
    images = [{"image_path": f"dataset/{index}.jpg"} for index in range(4)]

    result = prepare_data(images, DATA_ROOT_PATH=str(tmp_path), max_workers=2)

    assert maximum_active == 2
    assert result["successful"] == images
    assert result["failures"] == []


def test_prepare_data_rejects_invalid_worker_count(tmp_path):
    try:
        prepare_data([], DATA_ROOT_PATH=str(tmp_path), max_workers=0)
    except ValueError as exc:
        assert str(exc) == "max_workers must be a positive integer."
    else:
        raise AssertionError("Expected invalid max_workers to be rejected.")


def test_retry_wait_observes_cancellation_promptly():
    started = time.monotonic()

    def cancel_check():
        if time.monotonic() - started >= 0.05:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        visionkg_utils._interruptible_wait(10, cancel_check)

    assert time.monotonic() - started < 0.5


def test_rate_limit_cooldown_is_shared_between_workers(monkeypatch):
    monkeypatch.setattr(visionkg_utils, "_rate_limit_until", 0.0)
    visionkg_utils._extend_rate_limit(0.05)

    started = time.monotonic()
    visionkg_utils._wait_for_rate_limit()

    assert time.monotonic() - started >= 0.04


def test_download_request_slots_are_globally_paced(monkeypatch):
    monkeypatch.setattr(visionkg_utils, "_next_download_request_at", 0.0)
    monkeypatch.setattr(visionkg_utils, "download_requests_per_minute", lambda: 1200)

    visionkg_utils._wait_for_download_request_slot()
    started = time.monotonic()
    visionkg_utils._wait_for_download_request_slot()

    assert time.monotonic() - started >= 0.04


@pytest.mark.parametrize(
    ("status_code", "headers", "expected"),
    [
        (429, {}, True),
        (500, {"X-RateLimit-Remaining": "0"}, True),
        (500, {"X-RateLimit-Remaining": "12"}, False),
    ],
)
def test_rate_limit_detection_handles_misreported_500(status_code, headers, expected):
    response = type("Response", (), {"status_code": status_code, "headers": headers})()
    assert visionkg_utils._response_is_rate_limited(response) is expected


def test_visionkg_raw_image_url_has_view_fallback():
    raw_url = "https://vision-api.semkg.org/api/image?image=/mnist_cls_train/2/46591.png"
    assert visionkg_utils._visionkg_view_fallback_url(raw_url) == (
        "https://vision-api.semkg.org/api/view?image=/mnist_cls_train/2/46591.png"
    )
    assert visionkg_utils._visionkg_view_fallback_url("https://example.com/image.png") is None


def test_view_endpoint_circuit_breaker_rewrites_later_urls(monkeypatch):
    monkeypatch.setattr(visionkg_utils, "_prefer_view_endpoint", False)
    raw_url = "https://vision-api.semkg.org/api/image?image=/mnist_cls_train/2/46591.png"
    assert visionkg_utils._preferred_visionkg_url(raw_url) == raw_url

    visionkg_utils._prefer_visionkg_view_endpoint()

    assert visionkg_utils._preferred_visionkg_url(raw_url) == (
        "https://vision-api.semkg.org/api/view?image=/mnist_cls_train/2/46591.png"
    )


def test_download_falls_back_to_view_endpoint_after_genuine_raw_500(monkeypatch, tmp_path):
    monkeypatch.setattr(visionkg_utils, "_prefer_view_endpoint", False)
    image_bytes = io.BytesIO()
    Image.new("L", (2, 2)).save(image_bytes, format="PNG")

    class FakeResponse:
        def __init__(self, status_code, headers, body=b""):
            self.status_code = status_code
            self.headers = headers
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_content(self, chunk_size):
            del chunk_size
            yield self.body

    requested = []

    class FakeSession:
        def get(self, url, **_kwargs):
            requested.append(url)
            if "/api/image?" in url:
                return FakeResponse(500, {"X-RateLimit-Remaining": "59"})
            return FakeResponse(200, {"Content-Type": "image/png"}, image_bytes.getvalue())

    monkeypatch.setattr(visionkg_utils, "_download_session", lambda: FakeSession())
    monkeypatch.setattr(visionkg_utils, "_wait_for_rate_limit", lambda *_args: None)
    monkeypatch.setattr(visionkg_utils, "_wait_for_download_request_slot", lambda *_args: None)
    monkeypatch.setattr(visionkg_utils, "_interruptible_wait", lambda *_args: None)

    successful, _result = visionkg_utils._download_one_image(
        {
            "image_path": "mnist_cls_train/2/46591.png",
            "url": "https://vision-api.semkg.org/api/image?image=/mnist_cls_train/2/46591.png",
        },
        tmp_path,
        max_attempts=2,
        timeout=30,
    )

    assert successful is True
    assert requested == [
        "https://vision-api.semkg.org/api/image?image=/mnist_cls_train/2/46591.png",
        "https://vision-api.semkg.org/api/view?image=/mnist_cls_train/2/46591.png",
    ]
