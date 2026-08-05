import csv
import json
import re

import pytest

from cvmodellearning.download import download_data
from cvmodellearning.download.assignment_manifest import (
    DatasetManifest,
    assignment_fingerprint,
    iter_download_allocations,
    load_dataset_manifest,
    validate_content_isolation,
)
from cvmodellearning.pipelines import detection_pipe


def _patch_paths(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts_root.mkdir()
    monkeypatch.setattr(download_data, "data_dir", lambda _job_id: data_root)
    monkeypatch.setattr(download_data, "csv_labels_path", lambda _job_id: data_root / "image_labels.csv")
    monkeypatch.setattr(download_data, "json_labels_path", lambda _job_id: data_root / "annotations.json")
    monkeypatch.setattr(download_data, "dataset_manifest_path", lambda _job_id: data_root / "dataset_manifest.json")
    monkeypatch.setattr(download_data, "download_report_path", lambda _job_id: artifacts_root / "download_report.json")
    return data_root, artifacts_root


def _assignments():
    return [{
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [
                {"split": "train", "count": 1, "assignment_type": "official_split"},
                {"split": "validation", "count": 1, "assignment_type": "derived_from_train"},
                {"split": "test", "count": 1, "assignment_type": "derived_from_train"},
            ],
        }],
    }]


def test_classification_download_reserves_disjoint_samples_per_assignment(monkeypatch, tmp_path):
    data_root, artifacts_root = _patch_paths(monkeypatch, tmp_path)
    rows = [
        {"datasetName": "bdd_100k_det_train", "imageName": f"{index}.jpg", "image": f"image:{index}"}
        for index in range(1, 4)
    ]
    monkeypatch.setattr(download_data, "query", lambda _query: rows)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )

    report = download_data.download_visionkg_mixed_datasets_classification(
        "job",
        _assignments(),
    )

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    assert [sample["assigned_split"] for sample in manifest["samples"]] == [
        "train",
        "validation",
        "test",
    ]
    assert len({sample["sample_id"] for sample in manifest["samples"]}) == 3
    assert report["complete"] is True
    assert report["unique_downloaded"] == 3
    assert sum(item["cross_split_conflicts"] for item in report["sources"]) == 6
    with (data_root / "image_labels.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert json.loads((artifacts_root / "download_report.json").read_text()) == report


def test_classification_download_pages_past_large_prior_allocation(monkeypatch, tmp_path):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    rows = [
        {"datasetName": "bdd_100k_det_train", "imageName": f"{index}.jpg", "image": f"image:{index}"}
        for index in range(8)
    ]

    def paged_query(query):
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        return rows[offset:offset + limit]

    monkeypatch.setattr(download_data, "query", paged_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [{
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [
                {"split": "train", "count": 5, "assignment_type": "official_split"},
                {"split": "test", "count": 2, "assignment_type": "derived_from_train"},
            ],
        }],
    }]

    report = download_data.download_visionkg_mixed_datasets_classification("job", requests)

    assert report["complete"] is True
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    assert [sample["assigned_split"] for sample in manifest["samples"]] == [
        "train", "train", "train", "train", "train", "test", "test",
    ]


def _detection_row(image_index, label="car"):
    return {
        "datasetName": "bdd_100k_det_train",
        "imageName": f"{image_index}.jpg",
        "image": f"image:{image_index}",
        "imageWidth": "100",
        "imageHeight": "80",
        "labelName": label,
        "bbHeight": "20",
        "bbWidth": "30",
        "bbCentreX": "50",
        "bbCentreY": "40",
    }


def test_detection_download_keeps_images_atomic_across_splits(monkeypatch, tmp_path):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    rows = [_detection_row(index) for index in range(1, 4)]
    monkeypatch.setattr(download_data, "query", lambda _query: rows)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )

    report = download_data.download_visionkg_mixed_datasets_detection("job", _assignments())

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    coco = json.loads((data_root / "annotations.json").read_text())
    assert {sample["assigned_split"] for sample in manifest["samples"]} == {
        "train",
        "validation",
        "test",
    }
    assert len(manifest["samples"]) == len(coco["images"]) == 3
    assert {image["assigned_split"] for image in coco["images"]} == {
        "train",
        "validation",
        "test",
    }
    assert report["complete"] is True
    assert report["unique_downloaded"] == 3


def test_detection_same_split_reuse_merges_classes_without_redownload(monkeypatch, tmp_path):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        download_data,
        "query",
        lambda query: [_detection_row(
            1,
            "truck" if '= LCASE("truck")' in query else "car",
        )],
    )
    download_calls = []

    def fake_prepare(images, DATA_ROOT_PATH):
        download_calls.append(list(images))
        return {"successful": images, "failures": []}

    monkeypatch.setattr(download_data, "prepare_data", fake_prepare)
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train",
                    "count": 1,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("car", "truck")
    ]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    coco = json.loads((data_root / "annotations.json").read_text())
    assert len(download_calls) == 1
    assert manifest["samples"][0]["class_names"] == ["car", "truck"]
    assert len(coco["images"]) == 1
    assert len(coco["categories"]) == 2
    assert len(coco["annotations"]) == 2
    assert report["downloaded"] == 2
    assert report["unique_downloaded"] == 1


def test_detection_download_keeps_all_requested_class_boxes_on_selected_image(
    monkeypatch,
    tmp_path,
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)

    def fake_query(query):
        assert 'IN (LCASE("car"), LCASE("truck"))' in query
        if '= LCASE("car")' in query:
            return [_detection_row(1, "car"), _detection_row(1, "truck")]
        return [_detection_row(2, "truck")]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train",
                    "count": 1,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("car", "truck")
    ]

    download_data.download_visionkg_mixed_datasets_detection("job", requests)

    coco = json.loads((data_root / "annotations.json").read_text())
    image_one_id = next(
        image["id"] for image in coco["images"] if image["image_path"].endswith("1.jpg")
    )
    category_names = {item["id"]: item["name"] for item in coco["categories"]}
    assert {
        category_names[annotation["category_id"]]
        for annotation in coco["annotations"]
        if annotation["image_id"] == image_one_id
    } == {"car", "truck"}


def test_manifest_rejects_cross_split_reassignment():
    manifest = DatasetManifest("job", "detection", "fingerprint")
    values = {
        "sample_id": "image:1",
        "image_path": "dataset/1.jpg",
        "class_name": "car",
        "dataset_name": "dataset",
        "source_role": "train",
        "assignment_type": "derived_from_train",
    }
    manifest.add(assigned_split="validation", **values)

    with pytest.raises(ValueError, match="both"):
        manifest.add(assigned_split="test", **values)


def test_manifest_loader_rejects_invalid_provenance(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "task": "classification",
        "assignment_fingerprint": "fingerprint",
        "samples": [{
            "sample_id": "sample",
            "image_path": "dataset/image.jpg",
            "class_names": ["car"],
            "dataset_name": "bdd_100k_det_val",
            "source_role": "validation",
            "assigned_split": "train",
            "assignment_type": "official_split",
        }],
    }))

    with pytest.raises(ValueError, match="remain in their source split"):
        load_dataset_manifest(
            path,
            task="classification",
            expected_fingerprint="fingerprint",
        )


def test_downloader_contract_rejects_invalid_derived_assignment():
    requests = [{
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd_100k_det_val",
            "allocations": [{
                "split": "test",
                "count": 1,
                "assignment_type": "derived_from_train",
            }],
        }],
    }]

    with pytest.raises(ValueError, match="official training source"):
        list(iter_download_allocations(requests))


def test_detection_pipeline_rejects_required_allocation_shortfall(monkeypatch, tmp_path):
    monkeypatch.setattr(detection_pipe, "data_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(
        detection_pipe,
        "download_visionkg_mixed_datasets_detection",
        lambda _job_id, _selected_data: {
            "complete": False,
            "sources": [{
                "class_name": "car",
                "dataset_name": "bdd_100k_det_train",
                "assigned_split": "test",
                "requested": 10,
                "downloaded": 8,
                "shortfall": 2,
            }],
        },
    )

    with pytest.raises(RuntimeError, match="test: 8/10"):
        detection_pipe.DetectionPipeline().download_data_step(
            {"selected_data": _assignments()},
            "job",
        )


def test_detection_preparation_preserves_manifest_assignments(monkeypatch, tmp_path):
    selected_data = _assignments()
    images = []
    annotations = []
    samples = []
    for index, split in enumerate(("train", "validation", "test"), start=1):
        image_path = f"bdd/{index}.jpg"
        (tmp_path / "bdd").mkdir(exist_ok=True)
        (tmp_path / image_path).write_bytes(f"image-{index}".encode())
        images.append({"id": f"image-{index}", "image_path": image_path})
        annotations.append({
            "id": f"annotation-{index}",
            "image_id": f"image-{index}",
            "category_id": 1,
            "bbox": [1, 1, 10, 10],
        })
        samples.append({
            "sample_id": f"image-{index}",
            "image_path": image_path,
            "class_names": ["car"],
            "dataset_name": "bdd_100k_det_train",
            "source_role": "train",
            "assigned_split": split,
            "assignment_type": "official_split" if split == "train" else "derived_from_train",
        })
    annotations_path = tmp_path / "annotations.json"
    manifest_path = tmp_path / "manifest.json"
    annotations_path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car"}],
    }))
    manifest_path.write_text(json.dumps({
        "task": "detection",
        "assignment_fingerprint": assignment_fingerprint(selected_data),
        "samples": samples,
    }))
    monkeypatch.setattr(detection_pipe, "data_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(detection_pipe, "json_labels_path", lambda _job_id: annotations_path)
    monkeypatch.setattr(detection_pipe, "dataset_manifest_path", lambda _job_id: manifest_path)
    monkeypatch.setattr(detection_pipe, "preparation_summary_path", lambda _job_id: tmp_path / "summary.json")
    monkeypatch.setattr(detection_pipe, "train_json_path", lambda _job_id: tmp_path / "train.json")
    monkeypatch.setattr(detection_pipe, "val_json_path", lambda _job_id: tmp_path / "validation.json")
    monkeypatch.setattr(detection_pipe, "test_json_path", lambda _job_id: tmp_path / "test.json")
    monkeypatch.setattr(detection_pipe, "_get_trainer_type", lambda _model_name: "torchvision")

    result = detection_pipe.DetectionPipeline().prepare_data_step(
        {"selected_data": selected_data, "classes": ["car"], "model_name": "detector"},
        "job",
    )

    assert result["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["unique_images"] == result["counts"]
    assert result["class_image_counts"] == {
        "car": {"train": 1, "validation": 1, "test": 1},
    }
    assert result["content_isolation"] == {
        "unique_content_hashes": 3,
        "duplicate_samples": 0,
        "cross_split_duplicates": 0,
    }
    for split in ("train", "validation", "test"):
        prepared = json.loads((tmp_path / f"{split}.json").read_text())
        assert [image["assigned_split"] for image in prepared["images"]] == [split]


def test_content_isolation_rejects_exact_duplicate_across_splits(tmp_path):
    (tmp_path / "train.jpg").write_bytes(b"same pixels")
    (tmp_path / "validation.jpg").write_bytes(b"same pixels")
    manifest = {
        "samples": [
            {"image_path": "train.jpg", "assigned_split": "train"},
            {"image_path": "validation.jpg", "assigned_split": "validation"},
        ]
    }

    with pytest.raises(ValueError, match="Identical image content"):
        validate_content_isolation(manifest, tmp_path)


def test_content_isolation_allows_exact_duplicate_within_one_split(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"same pixels")
    (tmp_path / "two.jpg").write_bytes(b"same pixels")
    manifest = {
        "samples": [
            {"image_path": "one.jpg", "assigned_split": "train"},
            {"image_path": "two.jpg", "assigned_split": "train"},
        ]
    }

    assert validate_content_isolation(manifest, tmp_path) == {
        "unique_content_hashes": 1,
        "duplicate_samples": 1,
        "cross_split_duplicates": 0,
    }
