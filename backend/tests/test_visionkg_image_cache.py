import os

import pytest
from PIL import Image

from cvmodellearning.download import download_data
from cvmodellearning.download.image_cache import (
    image_path_below,
    materialize_cached_images,
)


def test_materialize_cached_image_preserves_dataset_path_without_copying(tmp_path):
    cache_root = tmp_path / "cache"
    run_root = tmp_path / "run"
    cached = cache_root / "dataset" / "one.jpg"
    cached.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(cached)

    counts = materialize_cached_images(
        [{"image_path": "dataset/one.jpg"}], cache_root, run_root
    )
    materialized = run_root / "dataset" / "one.jpg"

    assert materialized.is_file()
    assert materialized.read_bytes() == cached.read_bytes()
    assert sum(counts.values()) == 1
    if counts["hardlink"]:
        assert os.path.samefile(cached, materialized)


@pytest.mark.parametrize("path", ["../escape.jpg", "/absolute.jpg", "dataset/../../escape.jpg", ""])
def test_cache_rejects_unsafe_image_paths(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        image_path_below(tmp_path, path)


def test_prepare_with_progress_reuses_cache_and_materializes_each_run(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    runs_root = tmp_path / "runs"
    calls = []

    def fake_prepare(images, DATA_ROOT_PATH, **_kwargs):
        calls.append(DATA_ROOT_PATH)
        cached = cache_root / images[0]["image_path"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists():
            Image.new("RGB", (2, 2)).save(cached)
        return {
            "successful": images,
            "failures": [],
            "metrics": {"cache_hits": int(len(calls) > 1)},
        }

    monkeypatch.setattr(download_data, "visionkg_cache_dir", lambda: cache_root)
    monkeypatch.setattr(download_data, "data_dir", lambda job_id: runs_root / job_id / "data")
    monkeypatch.setattr(download_data, "prepare_data", fake_prepare)
    image = {"image_path": "dataset/one.jpg", "url": "unused"}

    first = download_data._prepare_with_progress([image], "one", None)
    second = download_data._prepare_with_progress([image], "two", None)

    assert calls == [str(cache_root), str(cache_root)]
    assert (runs_root / "one/data/dataset/one.jpg").is_file()
    assert (runs_root / "two/data/dataset/one.jpg").is_file()
    assert first["metrics"]["materialization"]["hardlink"] == 1
    assert second["metrics"]["materialization"]["hardlink"] == 1
